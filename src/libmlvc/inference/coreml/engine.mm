// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#import "libmlvc/common/logging.hpp"
#import "libmlvc/common/tar.hpp"
#import "libmlvc/common/utils.hpp"
#import "libmlvc/inference/coreml/factory.hpp"
#import "libmlvc/inference/interface.hpp"
#import "libmlvc/inference/model_cache.hpp"

#import <libmlvc/error_codes.hpp>
#import <libmlvc/platform_info.hpp>

#import <CoreML/CoreML.h>
#import <CoreVideo/CVPixelBuffer.h>
#import <Foundation/Foundation.h>
#import <sys/sysctl.h>

#import <cstdio>
#import <map>
#import <mutex>
#import <string>
#import <vector>

namespace libmlvc {

namespace {

inline NSString* ToNSString(std::string_view sv)
{
    return [[[NSString alloc] initWithBytes:sv.data() length:sv.length() encoding:NSUTF8StringEncoding] autorelease];
}

bool IsSafeTarRelativePath(std::string_view path)
{
    if (path.empty()) return false;
    if (path.front() == '/' || path.front() == '\\') return false;

    size_t segmentStart = 0;
    while (segmentStart <= path.size()) {
        size_t segmentEnd = path.find_first_of("/\\", segmentStart);
        if (segmentEnd == std::string_view::npos) segmentEnd = path.size();
        auto segment = path.substr(segmentStart, segmentEnd - segmentStart);

        if (segment.empty() || segment == "." || segment == "..") return false;

        segmentStart = segmentEnd + 1;
        if (segmentEnd == path.size()) break;
    }

    return true;
}

expected<NSString*> BuildSafePackageFilePath(NSString* pkgPath, std::string_view tarEntryPath)
{
    if (!IsSafeTarRelativePath(tarEntryPath)) {
        MLVC_LOG_ERROR("Unsafe TAR path: %s", std::string(tarEntryPath).c_str());
        return make_error_code(Error::io_error);
    }

    NSString* relPath = ToNSString(tarEntryPath);
    if (!relPath) {
        MLVC_LOG_ERROR("Failed to decode TAR path as UTF-8");
        return make_error_code(Error::io_error);
    }

    NSString* base = [pkgPath stringByStandardizingPath];
    NSString* candidate = [[pkgPath stringByAppendingPathComponent:relPath] stringByStandardizingPath];
    if (![candidate hasPrefix:base]
        || ([candidate length] > [base length] && [candidate characterAtIndex:[base length]] != '/')) {
        MLVC_LOG_ERROR("TAR path escapes package root: %s", std::string(tarEntryPath).c_str());
        return make_error_code(Error::io_error);
    }

    return candidate;
}

MLComputeUnits ToMLComputeUnits(ComputeUnit unit)
{
    switch (unit) {
    case ComputeUnit::CPU:
        return MLComputeUnitsCPUOnly;
    case ComputeUnit::GPU:
        return MLComputeUnitsCPUAndGPU;
    case ComputeUnit::NPU:
        return MLComputeUnitsCPUAndNeuralEngine;
    case ComputeUnit::AUTO:
        return MLComputeUnitsAll;
    }
}

MLMultiArrayDataType ToMLDataType(TensorDataType dtype)
{
    switch (dtype) {
    case TensorDataType::FP16:
        return MLMultiArrayDataTypeFloat16;
    case TensorDataType::INT32:
        return MLMultiArrayDataTypeInt32;
    }
    MLVC_ASSERT(false);
    return MLMultiArrayDataTypeFloat16;
}

expected<TensorDataType> FromMLDataType(MLMultiArrayDataType dtype)
{
    switch (dtype) {
    case MLMultiArrayDataTypeFloat16:
        return TensorDataType::FP16;
    case MLMultiArrayDataTypeInt32:
        return TensorDataType::INT32;
    default:
        MLVC_LOG_ERROR("Unsupported MLMultiArray data type: %ld", static_cast<long>(dtype));
        return make_error_code(Error::model_init_error);
    }
}

expected<std::pair<TensorDataType, std::vector<int>>> ExtractTensorInfo(MLFeatureDescription* desc)
{
    if (desc.type != MLFeatureTypeMultiArray) {
        MLVC_LOG_ERROR("Unsupported feature type: %ld", static_cast<long>(desc.type));
        return make_error_code(Error::model_init_error);
    }
    MLMultiArrayConstraint* constraint = desc.multiArrayConstraint;
    if (!constraint) {
        MLVC_LOG_ERROR("No multiArrayConstraint for feature");
        return make_error_code(Error::model_init_error);
    }
    auto dtype = FromMLDataType(constraint.dataType);
    if (!dtype) return dtype.error();

    std::vector<int> shape;
    shape.reserve(constraint.shape.count);
    for (NSNumber* dim in constraint.shape)
        shape.push_back(static_cast<int>(dim.integerValue));
    return std::make_pair(dtype.value(), shape);
}

expected<MLMultiArray*> CreateFP16MLMultiArray(NSArray<NSNumber*>* nsShape, std::span<const int> shape)
{
    size_t totalElements = 1;
    for (int dim : shape)
        totalElements *= dim;
    size_t width = shape.empty() ? 1 : shape.back();
    if (width == 0 || totalElements == 0 || totalElements % width != 0) {
        MLVC_LOG_ERROR("Invalid shape for FP16 MLMultiArray: totalElements=%zu, width=%zu", totalElements, width);
        return make_error_code(Error::model_init_error);
    }

    NSDictionary* attrs = @{(__bridge id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
    CVPixelBufferRef pixelBuffer = nullptr;
    CVReturn result =
        CVPixelBufferCreate(kCFAllocatorDefault, width, totalElements / width, kCVPixelFormatType_OneComponent16Half,
                            (__bridge CFDictionaryRef)attrs, &pixelBuffer);
    if (result != kCVReturnSuccess || !pixelBuffer) {
        MLVC_LOG_ERROR("Failed to create CVPixelBuffer: %d", result);
        return make_error_code(Error::model_init_error);
    }
    MLMultiArray* mlArray = [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:nsShape];
    CVPixelBufferRelease(pixelBuffer);
    if (!mlArray) {
        MLVC_LOG_ERROR("Failed to create MLMultiArray from CVPixelBuffer");
        return make_error_code(Error::model_init_error);
    }
    return mlArray;
}

std::string Fnv1a32(std::string_view s)
{
    uint32_t h = 0x811c9dc5;
    for (auto c : s) {
        h = (h ^ static_cast<uint8_t>(c)) * 0x01000193;
    }
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08x", h);
    return buf;
}

std::string GetOsBuild()
{
    char buf[256] = "";
    size_t size = sizeof(buf);
    if (::sysctlbyname("kern.osversion", buf, &size, nullptr, 0) == 0) {
        return buf;
    }
    MLVC_LOG_WARN("Failed to read the OS build");
    return {};
}

// Model compilation & caching

enum class CompiledModelOrigin { Cache, CompiledToCache, CompiledToTemp };
struct CompiledModel {
    NSURL* url;
    CompiledModelOrigin origin;
};

// Extract TAR model data, write .mlpackage to temp dir, compile, clean up temp dir.
// Returns the compiled .mlmodelc URL (in OS temp location).
expected<CompiledModel> CompileModelFromData(const std::string& name, std::span<const std::byte> modelData)
{
    if (modelData.empty()) {
        MLVC_LOG_ERROR("[%s] Model data is required without a model cache", name.c_str());
        return make_error_code(Error::invalid_argument);
    }

    NSFileManager* fm = [NSFileManager defaultManager];
    NSString* tempDir = [NSTemporaryDirectory() stringByAppendingPathComponent:[[NSUUID UUID] UUIDString]];

    // Ensure temp dir is removed on all exit paths.
    struct TempDirCleanup {
        NSFileManager* fm;
        NSString* dir;
        ~TempDirCleanup() { [fm removeItemAtPath:dir error:nil]; }
    } tempDirCleanup{ fm, tempDir };

    NSError* error = nil;
    if (![fm createDirectoryAtPath:tempDir withIntermediateDirectories:YES attributes:nil error:&error]) {
        MLVC_LOG_ERROR("[%s] Failed to create temp dir: %s", name.c_str(),
                       error.localizedDescription.UTF8String ?: "unknown");
        return make_error_code(Error::io_error);
    }
    auto tarEntries = ExtractTar(modelData);
    if (!tarEntries) {
        MLVC_LOG_ERROR("[%s] Failed to extract TAR: %s", name.c_str(), tarEntries.error().message().c_str());
        return tarEntries.error();
    }
    NSString* pkgPath = [tempDir stringByAppendingPathComponent:@"model.mlpackage"];
    for (const auto& [path, data] : tarEntries.value()) {
        @autoreleasepool {
            auto safeFilePath = BuildSafePackageFilePath(pkgPath, path);
            if (!safeFilePath) {
                MLVC_LOG_ERROR("[%s] Rejected TAR entry path: %s", name.c_str(), path.c_str());
                return safeFilePath.error();
            }
            NSString* filePath = safeFilePath.value();
            if (![fm createDirectoryAtPath:[filePath stringByDeletingLastPathComponent]
                    withIntermediateDirectories:YES
                                     attributes:nil
                                          error:&error]) {
                MLVC_LOG_ERROR("[%s] Failed to create dir: %s", name.c_str(),
                               error.localizedDescription.UTF8String ?: "unknown");
                return make_error_code(Error::io_error);
            }
            if (![[NSData dataWithBytes:data.data() length:data.size()] writeToFile:filePath options:0 error:&error]) {
                MLVC_LOG_ERROR("[%s] Failed to write file: %s", name.c_str(),
                               error.localizedDescription.UTF8String ?: "unknown");
                return make_error_code(Error::io_error);
            }
        }
    }
    error = nil;
    NSURL* compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:pkgPath] error:&error];
    if (error || !compiled) {
        MLVC_LOG_ERROR("[%s] Failed to compile: %s", name.c_str(), error.localizedDescription.UTF8String ?: "unknown");
        return make_error_code(Error::model_init_error);
    }
    return CompiledModel{ compiled, CompiledModelOrigin::CompiledToTemp };
}

// Returns the compiled model from the cache, or compiles it into the cache (into a temp directory without a cache key).
// forceCompile replaces a cached model.
expected<CompiledModel> GetOrCompileModel(const std::string& name, std::span<const std::byte> modelData,
                                          std::string_view cacheKey, bool forceCompile = false)
{
    const bool useCache = !cacheKey.empty();
    if (!useCache) {
        MLVC_LOG_INFO("[%s] Model caching disabled, compiling to temp directory", name.c_str());
        return CompileModelFromData(name, modelData);
    }

    auto& cache = ModelCache::Instance();
    NSFileManager* fm = [NSFileManager defaultManager];

    // 1. Try to load from existing cache, unless forced to compile
    if (auto cacheDirResult = cache.LocateCacheDir(cacheKey); cacheDirResult && !forceCompile) {
        auto cachedModelPath = cacheDirResult.value() / "model.mlmodelc";
        NSString* cachedPath = ToNSString(cachedModelPath.string());
        BOOL isDir = NO;
        if ([fm fileExistsAtPath:cachedPath isDirectory:&isDir] && isDir) {
            MLVC_LOG_INFO("[%s] Found compiled model from cache '%s'", name.c_str(), std::string(cacheKey).c_str());
            cache.CleanupCacheDirs();
            return CompiledModel{ [NSURL fileURLWithPath:cachedPath], CompiledModelOrigin::Cache };
        }
        // Cache dir exists but model.mlmodelc is missing: recompiled below into a new cache directory
        MLVC_LOG_WARN("[%s] Cache directory exists but compiled model missing, recompiling", name.c_str());
    }

    if (modelData.empty()) {
        MLVC_LOG_ERROR("[%s] Compiled model '%s' not found in the model cache", name.c_str(),
                       std::string(cacheKey).c_str());
        return make_error_code(Error::model_init_error);
    }

    // 2. Compile model
    auto compiled = CompileModelFromData(name, modelData);
    if (!compiled) return compiled.error();

    // 3. Move compiled model into cache directory
    auto cacheDirResult = cache.CreateCacheDir(cacheKey);
    if (!cacheDirResult) {
        MLVC_LOG_ERROR("[%s] Failed to create cache directory", name.c_str());
        [fm removeItemAtURL:compiled->url error:nil];
        return cacheDirResult.error();
    }

    auto destPath = cacheDirResult.value() / "model.mlmodelc";
    NSURL* destURL = [NSURL fileURLWithPath:ToNSString(destPath.string())];
    NSError* error = nil;
    if (![fm moveItemAtURL:compiled->url toURL:destURL error:&error]) {
        MLVC_LOG_ERROR("[%s] Failed to move compiled model to cache: %s", name.c_str(),
                       error.localizedDescription.UTF8String ?: "unknown");
        cache.DeleteCacheDir(cacheKey);
        [fm removeItemAtURL:compiled->url error:nil];
        return make_error_code(Error::io_error);
    }

    MLVC_LOG_INFO("[%s] Compiled model cache key: '%s'", name.c_str(), std::string(cacheKey).c_str());
    cache.CleanupCacheDirs();
    return CompiledModel{ destURL, CompiledModelOrigin::CompiledToCache };
}

// Returns a retained model
expected<MLModel*> LoadModel(const std::string& name, NSURL* compiledModelUrl, const CoreMlEngineParams& params,
                             const std::optional<std::string>& functionName)
{
    @autoreleasepool {
        MLModelConfiguration* config = [[[MLModelConfiguration alloc] init] autorelease];
        config.computeUnits = ToMLComputeUnits(params.computeUnit);
        if (@available(macOS 15.0, iOS 18.0, *)) {
            config.optimizationHints.specializationStrategy = MLSpecializationStrategyFastPrediction;
            config.optimizationHints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
            if (functionName) [config setFunctionName:ToNSString(*functionName)];
        }
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:compiledModelUrl configuration:config error:&error];
        if (error || !model) {
            MLVC_LOG_ERROR("[%s] Failed to load model: %s", name.c_str(),
                           error.localizedDescription.UTF8String ?: "unknown");
            return make_error_code(Error::model_init_error);
        }
        return [model retain];
    }
}

expected<std::vector<std::string>> GetFunctionNames(const std::string& name, NSURL* compiledModelUrl)
{
    if (@available(macOS 15.0, iOS 18.0, *)) {
        @autoreleasepool {
            NSError* error = nil;
            MLModelAsset* asset = [MLModelAsset modelAssetWithURL:compiledModelUrl error:&error];
            if (error || !asset) {
                MLVC_LOG_ERROR("[%s] Failed to open the compiled model: %s", name.c_str(),
                               error.localizedDescription.UTF8String ?: "unknown");
                return make_error_code(Error::model_init_error);
            }
            // Outlives a timeout, as the handler may still run
            struct Result {
                std::vector<std::string> names;
                std::optional<std::string> error;
            };
            auto result = std::make_shared<Result>();
            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            [asset functionNamesWithCompletionHandler:^(NSArray<NSString*>* functionNames, NSError* listError) {
              if (listError || !functionNames) {
                  result->error = listError.localizedDescription.UTF8String ?: "unknown";
              } else {
                  for (NSString* functionName in functionNames)
                      result->names.emplace_back(functionName.UTF8String);
              }
              dispatch_semaphore_signal(done);
            }];
            constexpr int64_t kTimeoutSeconds = 60;
            const bool timedOut =
                dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, kTimeoutSeconds * NSEC_PER_SEC)) != 0;
            dispatch_release(done);
            if (timedOut || result->error) {
                MLVC_LOG_ERROR("[%s] Failed to list the model functions: %s", name.c_str(),
                               timedOut ? "timed out" : result->error->c_str());
                return make_error_code(Error::model_init_error);
            }
            return std::move(result->names);
        }
    }
    return std::vector<std::string>{};
}

expected<void> SpecializeModel(const std::string& name, NSURL* compiledModelUrl, const CoreMlEngineParams& params)
{
    auto functionNames = GetFunctionNames(name, compiledModelUrl);
    if (!functionNames) return functionNames.error();
    std::vector<std::optional<std::string>> functions(functionNames->begin(), functionNames->end());
    if (functions.empty()) functions.emplace_back();

    ManualOpTimer timer;
    for (const auto& functionName : functions) {
        auto model = LoadModel(name, compiledModelUrl, params, functionName);
        if (!model) return model.error();
        [model.value() release];
    }
    MLVC_LOG_INFO("[%s] Specialized %zu functions for the device in %.0f ms", name.c_str(), functions.size(), timer.Stop());
    return {};
}

}  // namespace

class CoreMlInferenceTensor : public IInferenceTensor {
public:
    static expected<std::shared_ptr<CoreMlInferenceTensor>> Create(std::string_view name, TensorDataType dataType,
                                                                   std::span<const int> shape)
    {
        auto tensor = std::shared_ptr<CoreMlInferenceTensor>(new CoreMlInferenceTensor(name));
        if (auto ret = tensor->Initialize(dataType, shape); !ret) return ret.error();
        return tensor;
    }

    ~CoreMlInferenceTensor() override
    {
        @autoreleasepool {
            [m_mlMultiArray release];
            m_mlMultiArray = nil;
        }
    }
    expected<std::string_view> Name() const override { return m_name; }
    expected<TensorDataType> DataType() const override { return m_dataType; }
    expected<std::span<const int>> Shape() const override { return std::span<const int>(m_shape); }
    expected<std::span<const int>> Strides() const override { return std::span<const int>(m_strides); }
    expected<std::span<std::byte>> Data() override
    {
        return std::span<std::byte>(reinterpret_cast<std::byte*>(m_mlMultiArray.dataPointer), m_dataSize);
    }
    std::string_view GetName() const { return m_name; }
    MLMultiArray* GetMLMultiArray() const { return m_mlMultiArray; }

private:
    explicit CoreMlInferenceTensor(std::string_view name) : m_name(name) {}

    expected<void> Initialize(TensorDataType dataType, std::span<const int> shape)
    {
        @autoreleasepool {
            NSMutableArray<NSNumber*>* nsShape = [NSMutableArray arrayWithCapacity:shape.size()];
            for (int dim : shape)
                [nsShape addObject:@(dim)];

            if (dataType == TensorDataType::FP16) {
                auto result = CreateFP16MLMultiArray(nsShape, shape);
                if (!result) return result.error();
                m_mlMultiArray = result.value();
            } else {
                NSError* error = nil;
                m_mlMultiArray = [[MLMultiArray alloc] initWithShape:nsShape
                                                            dataType:ToMLDataType(dataType)
                                                               error:&error];
                if (error || !m_mlMultiArray) {
                    MLVC_LOG_ERROR("Failed to create MLMultiArray: %s", error.localizedDescription.UTF8String ?: "unknown");
                    return make_error_code(Error::model_init_error);
                }
            }
            m_dataType = dataType;
            m_shape.assign(shape.begin(), shape.end());
            m_strides.reserve(m_mlMultiArray.strides.count);
            for (NSNumber* s in m_mlMultiArray.strides)
                m_strides.push_back(s.intValue);
            if (!m_shape.empty() && !m_strides.empty()) {
                m_dataSize = static_cast<size_t>(m_shape[0]) * m_strides[0] * TensorDataTypeSize(dataType);
            } else {
                m_dataSize = TensorDataTypeSize(dataType);
                for (NSNumber* dim in m_mlMultiArray.shape)
                    m_dataSize *= dim.unsignedIntegerValue;
            }
        }
        return {};
    }

    std::string m_name;
    MLMultiArray* m_mlMultiArray = nil;
    TensorDataType m_dataType = TensorDataType::FP16;
    std::vector<int> m_shape, m_strides;
    size_t m_dataSize = 0;
};

class CoreMlInferenceSession : public IInferenceSession {
public:
    static expected<std::shared_ptr<CoreMlInferenceSession>> Create(std::string_view name,
                                                                    const CompiledModel& compiledModel,
                                                                    const CoreMlEngineParams& params,
                                                                    const std::optional<std::string>& functionName)
    {
        auto session = std::shared_ptr<CoreMlInferenceSession>(new CoreMlInferenceSession(name));
        if (auto ret = session->Initialize(compiledModel, params, functionName); !ret) return ret.error();
        return session;
    }

    ~CoreMlInferenceSession() override
    {
        @autoreleasepool {
            [m_predictionOptions release];
            m_predictionOptions = nil;
            [m_model release];
            m_model = nil;
            if (m_compiledModelIsTemp && m_compiledModelUrl) {
                [[NSFileManager defaultManager] removeItemAtURL:m_compiledModelUrl error:nil];
            }
            [m_compiledModelUrl release];
            m_compiledModelUrl = nil;
        }
    }

    expected<std::span<const std::string>> GetInputNames() override
    {
        return std::span<const std::string>(m_inputNames);
    }
    expected<std::span<const std::string>> GetOutputNames() override
    {
        return std::span<const std::string>(m_outputNames);
    }

    expected<InferenceTensorPtr> CreateTensor(TensorIoType ioType, std::string_view name,
                                              [[maybe_unused]] bool hostAccessible) override
    {
        const auto& infoMap = (ioType == TensorIoType::INPUT) ? m_inputInfo : m_outputInfo;
        auto it = infoMap.find(std::string(name));
        if (it == infoMap.end()) {
            MLVC_LOG_ERROR("[%s] Tensor %s not found", m_name.c_str(), std::string(name).c_str());
            return make_error_code(Error::invalid_argument);
        }
        return CoreMlInferenceTensor::Create(name, it->second.dataType, it->second.shape);
    }

    expected<void> Run(const std::vector<InferenceTensorPtr>& inputs, const std::vector<InferenceTensorPtr>& outputs) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_model || !m_predictionOptions) {
            MLVC_LOG_ERROR("[%s] Session not initialized", m_name.c_str());
            return make_error_code(Error::model_init_error);
        }
        if (inputs.size() != m_inputNames.size() || outputs.size() != m_outputNames.size()) {
            MLVC_LOG_ERROR("[%s] Tensor count mismatch: inputs %zu/%zu, outputs %zu/%zu", m_name.c_str(), inputs.size(),
                           m_inputNames.size(), outputs.size(), m_outputNames.size());
            return make_error_code(Error::invalid_argument);
        }
        @autoreleasepool {
            NSMutableDictionary<NSString*, MLFeatureValue*>* inputDict = [NSMutableDictionary dictionary];
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto* tensor = static_cast<CoreMlInferenceTensor*>(inputs[i].get());
                if (!tensor->GetMLMultiArray()) {
                    MLVC_LOG_ERROR("[%s] Input tensor has no MLMultiArray", m_name.c_str());
                    return make_error_code(Error::invalid_argument);
                }
                inputDict[ToNSString(m_inputNames[i])] =
                    [MLFeatureValue featureValueWithMultiArray:tensor->GetMLMultiArray()];
            }
            NSError* error = nil;
            auto* inputProvider = [[[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict
                                                                                     error:&error] autorelease];
            if (error || !inputProvider) {
                MLVC_LOG_ERROR("[%s] Failed to create input provider: %s", m_name.c_str(),
                               error.localizedDescription.UTF8String ?: "unknown");
                return make_error_code(Error::model_inference_error);
            }
            NSMutableDictionary<NSString*, id>* outputBackings = [NSMutableDictionary dictionary];
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto* tensor = static_cast<CoreMlInferenceTensor*>(outputs[i].get());
                MLMultiArray* arr = tensor->GetMLMultiArray();
                if (!arr || !arr.pixelBuffer) {
                    MLVC_LOG_ERROR("[%s] Output tensor requires FP16 with pixelBuffer", m_name.c_str());
                    return make_error_code(Error::invalid_argument);
                }
                outputBackings[ToNSString(m_outputNames[i])] =
                    [[[MLMultiArray alloc] initWithPixelBuffer:arr.pixelBuffer shape:arr.shape] autorelease];
            }
            m_predictionOptions.outputBackings = outputBackings;
            auto* result = [m_model predictionFromFeatures:inputProvider options:m_predictionOptions error:&error];
            m_predictionOptions.outputBackings = [NSDictionary dictionary];
            if (error || !result) {
                MLVC_LOG_ERROR("[%s] Inference failed: %s", m_name.c_str(),
                               error.localizedDescription.UTF8String ?: "unknown");
                return make_error_code(Error::model_inference_error);
            }
            return {};
        }
    }

private:
    explicit CoreMlInferenceSession(std::string_view name) : m_name(name) {}

    struct TensorInfo {
        TensorDataType dataType;
        std::vector<int> shape;
    };

    // Model metadata extraction

    expected<void> ExtractModelMetadata()
    {
        auto extract = [](NSDictionary<NSString*, MLFeatureDescription*>* descs, std::vector<std::string>& names,
                          std::map<std::string, TensorInfo>& info) -> expected<void> {
            for (NSString* name in descs) {
                auto ti = ExtractTensorInfo(descs[name]);
                if (!ti) return ti.error();
                std::string n = name.UTF8String;
                names.push_back(n);
                info[n] = TensorInfo{ ti->first, std::move(ti->second) };
            }
            return {};
        };
        MLModelDescription* desc = m_model.modelDescription;
        if (auto r = extract(desc.inputDescriptionsByName, m_inputNames, m_inputInfo); !r) return r.error();
        if (auto r = extract(desc.outputDescriptionsByName, m_outputNames, m_outputInfo); !r) return r.error();
        return {};
    }

    // Session initialization

    expected<void> Initialize(const CompiledModel& compiledModel, const CoreMlEngineParams& params,
                              const std::optional<std::string>& functionName)
    {
        // Take ownership first, so that a temporary compiled model is deleted even if initialization fails
        m_compiledModelUrl = [compiledModel.url retain];
        m_compiledModelIsTemp = compiledModel.origin == CompiledModelOrigin::CompiledToTemp;

        @autoreleasepool {
            auto model = LoadModel(m_name, m_compiledModelUrl, params, functionName);
            if (!model) return model.error();
            m_model = model.value();
            if (auto ret = ExtractModelMetadata(); !ret) return ret.error();
            m_predictionOptions = [[MLPredictionOptions alloc] init];
            return {};
        }
    }

    // Member variables

    std::string m_name;
    std::mutex m_mutex;

    MLModel* m_model = nil;
    MLPredictionOptions* m_predictionOptions = nil;
    NSURL* m_compiledModelUrl = nil;
    bool m_compiledModelIsTemp = false;

    std::vector<std::string> m_inputNames, m_outputNames;
    std::map<std::string, TensorInfo> m_inputInfo, m_outputInfo;
};

class CoreMlInferenceEngine : public IInferenceEngine {
public:
    explicit CoreMlInferenceEngine(const CoreMlEngineParams& params)
        : m_params(params), m_info{ .inferenceBackend = InferenceBackend::COREML, .computeUnit = params.computeUnit }
    {
        const auto& platformInfo = GetPlatformInfo();
        // Core ML drops its device specialization of models on every OS update
        const auto osBuild = GetOsBuild();
        m_envHash = Fnv1a32("coreml|" + std::to_string(platformInfo.osVersion.major) + "|" + osBuild);
        MLVC_LOG_INFO("CoreML initialized: compute_unit=%s, os_ver=%d, os_build=%s, env_hash=%s",
                      ComputeUnitToString(m_info.computeUnit), platformInfo.osVersion.major, osBuild.c_str(),
                      m_envHash.c_str());
    }
    expected<InferenceSessionPtr> CreateSession(std::string_view name, std::span<const std::byte> modelData,
                                                const std::map<std::string, std::span<const std::byte>>& weightsData,
                                                std::string_view cacheKey,
                                                const std::optional<std::string>& functionName = std::nullopt) override
    {
        if (!weightsData.empty()) {
            MLVC_LOG_ERROR("CoreML engine does not support external weights");
            return make_error_code(Error::invalid_argument);
        }
        // Prefix cache key with env hash so OS upgrades invalidate compiled models
        const auto fullCacheKey = FullCacheKey(cacheKey);
        // Ensure the same model is not being compiled concurrently.
        std::lock_guard<std::mutex> lock(m_mutex);
        @autoreleasepool {
            auto compiled = GetOrCompileModel(std::string(name), modelData, fullCacheKey);
            if (!compiled) return compiled.error();
            auto session = CoreMlInferenceSession::Create(name, compiled.value(), m_params, functionName);

            // A cached model that fails to load is recompiled once if there is model data
            if (!session && compiled->origin == CompiledModelOrigin::Cache && !modelData.empty()) {
                MLVC_LOG_WARN("[%s] Failed to load from cache, recompiling", std::string(name).c_str());
                compiled = GetOrCompileModel(std::string(name), modelData, fullCacheKey, /*forceCompile=*/true);
                if (!compiled) {
                    // The cached model doesn't load and can't be replaced: delete it
                    ModelCache::Instance().DeleteCacheDir(fullCacheKey);
                    return compiled.error();
                }
                session = CoreMlInferenceSession::Create(name, compiled.value(), m_params, functionName);
            }
            if (!session) {
                // A freshly compiled model that fails to load is deleted, not compiled again
                if (compiled->origin == CompiledModelOrigin::CompiledToCache) {
                    MLVC_LOG_ERROR("[%s] Failed to load the freshly compiled model, deleting it",
                                   std::string(name).c_str());
                    ModelCache::Instance().DeleteCacheDir(fullCacheKey);
                }
                return session.error();
            }
            return InferenceSessionPtr(session.value());
        }
    }
    expected<bool> PrepareModelCache(std::string_view name, std::span<const std::byte> modelData,
                                     const std::map<std::string, std::span<const std::byte>>& weightsData,
                                     std::string_view cacheKey) override
    {
        if (!weightsData.empty()) {
            MLVC_LOG_ERROR("CoreML engine does not support external weights");
            return make_error_code(Error::invalid_argument);
        }
        if (modelData.empty() || cacheKey.empty()) {
            MLVC_LOG_ERROR("[%s] Model data and cache key are required", std::string(name).c_str());
            return make_error_code(Error::invalid_argument);
        }

        const auto fullCacheKey = FullCacheKey(cacheKey);
        // Ensure the same model is not being compiled concurrently.
        std::lock_guard<std::mutex> lock(m_mutex);
        @autoreleasepool {
            auto compiled = GetOrCompileModel(std::string(name), modelData, fullCacheKey);
            if (!compiled) return compiled.error();
            // Best effort: sessions specialize the model themselves when created
            if (compiled->origin == CompiledModelOrigin::CompiledToCache
                && !SpecializeModel(std::string(name), compiled->url, m_params)) {
                MLVC_LOG_WARN("[%s] Failed to specialize the model for the device", std::string(name).c_str());
            }
            return true;
        }
    }
    const InferenceEngineInfo& GetInfo() const override { return m_info; }

private:
    std::string FullCacheKey(std::string_view cacheKey) const
    {
        return cacheKey.empty() ? std::string{} : m_envHash + "_" + std::string(cacheKey);
    }

    const CoreMlEngineParams m_params;
    const InferenceEngineInfo m_info;
    std::string m_envHash;
    mutable std::mutex m_mutex;
};

expected<std::unique_ptr<IInferenceEngine>> MakeCoreMlEngine(const CoreMlEngineParams& params)
{
    if (@available(macOS 15.0, iOS 18.0, *)) {
        return std::make_unique<CoreMlInferenceEngine>(params);
    }
    MLVC_LOG_ERROR("CoreML engine requires macOS 15.0 or iOS 18.0");
    return make_error_code(Error::unsupported_platform_error);
}

}  // namespace libmlvc
