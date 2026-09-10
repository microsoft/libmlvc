// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#import "libmlvc/common/logging.hpp"
#import "libmlvc/common/tar.hpp"
#import "libmlvc/inference/coreml/factory.hpp"
#import "libmlvc/inference/interface.hpp"
#import "libmlvc/inference/model_cache.hpp"

#import <libmlvc/error_codes.hpp>
#import <libmlvc/platform_info.hpp>

#import <CoreML/CoreML.h>
#import <CoreVideo/CVPixelBuffer.h>
#import <Foundation/Foundation.h>

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
    static expected<std::shared_ptr<CoreMlInferenceSession>>
    Create(std::string_view name, std::span<const std::byte> modelData, const CoreMlEngineParams& params,
           const std::optional<std::string>& functionName, std::string_view cacheKey)
    {
        auto session = std::shared_ptr<CoreMlInferenceSession>(new CoreMlInferenceSession(name));
        if (auto ret = session->Initialize(modelData, params, functionName, cacheKey); !ret) return ret.error();
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

    // Model compilation & caching

    // Extract TAR model data, write .mlpackage to temp dir, compile, clean up temp dir.
    // Returns the compiled .mlmodelc URL (in OS temp location).
    expected<NSURL*> CompileModelFromData(std::span<const std::byte> modelData)
    {
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
            MLVC_LOG_ERROR("[%s] Failed to create temp dir: %s", m_name.c_str(),
                           error.localizedDescription.UTF8String ?: "unknown");
            return make_error_code(Error::io_error);
        }
        auto tarEntries = ExtractTar(modelData);
        if (!tarEntries) {
            MLVC_LOG_ERROR("[%s] Failed to extract TAR: %s", m_name.c_str(), tarEntries.error().message().c_str());
            return tarEntries.error();
        }
        NSString* pkgPath = [tempDir stringByAppendingPathComponent:@"model.mlpackage"];
        for (const auto& [path, data] : tarEntries.value()) {
            @autoreleasepool {
                auto safeFilePath = BuildSafePackageFilePath(pkgPath, path);
                if (!safeFilePath) {
                    MLVC_LOG_ERROR("[%s] Rejected TAR entry path: %s", m_name.c_str(), path.c_str());
                    return safeFilePath.error();
                }
                NSString* filePath = safeFilePath.value();
                if (![fm createDirectoryAtPath:[filePath stringByDeletingLastPathComponent]
                        withIntermediateDirectories:YES
                                         attributes:nil
                                              error:&error]) {
                    MLVC_LOG_ERROR("[%s] Failed to create dir: %s", m_name.c_str(),
                                   error.localizedDescription.UTF8String ?: "unknown");
                    return make_error_code(Error::io_error);
                }
                if (![[NSData dataWithBytes:data.data() length:data.size()] writeToFile:filePath
                                                                                options:0
                                                                                  error:&error]) {
                    MLVC_LOG_ERROR("[%s] Failed to write file: %s", m_name.c_str(),
                                   error.localizedDescription.UTF8String ?: "unknown");
                    return make_error_code(Error::io_error);
                }
            }
        }
        error = nil;
        NSURL* compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:pkgPath] error:&error];
        if (error || !compiled) {
            MLVC_LOG_ERROR("[%s] Failed to compile: %s", m_name.c_str(), error.localizedDescription.UTF8String ?: "unknown");
            return make_error_code(Error::model_init_error);
        }
        return compiled;
    }

    expected<NSURL*> GetOrCompileModel(std::span<const std::byte> modelData, std::string_view cacheKey)
    {
        m_compiledModelIsTemp = true;
        const bool useCache = !cacheKey.empty();
        if (!useCache) {
            MLVC_LOG_INFO("[%s] Model caching disabled, compiling to temp directory", m_name.c_str());
            return CompileModelFromData(modelData);
        }

        auto& cache = ModelCache::Instance();
        NSFileManager* fm = [NSFileManager defaultManager];

        // 1. Try to load from existing cache
        if (auto cacheDirResult = cache.LocateCacheDir(cacheKey)) {
            auto cachedModelPath = cacheDirResult.value() / "model.mlmodelc";
            NSString* cachedPath = ToNSString(cachedModelPath.string());
            BOOL isDir = NO;
            if ([fm fileExistsAtPath:cachedPath isDirectory:&isDir] && isDir) {
                MLVC_LOG_INFO("[%s] Found compiled model from cache '%s'", m_name.c_str(), std::string(cacheKey).c_str());
                cache.CleanupCacheDirs();
                m_compiledModelIsTemp = false;
                return [NSURL fileURLWithPath:cachedPath];
            }
            // Cache dir exists but model.mlmodelc is missing - delete and recompile
            MLVC_LOG_WARN("[%s] Cache directory exists but compiled model missing, recompiling", m_name.c_str());
            cache.DeleteCacheDir(cacheKey);
        }

        // 2. Compile model
        auto compiled = CompileModelFromData(modelData);
        if (!compiled) return compiled.error();

        // 3. Move compiled model into cache directory
        auto cacheDirResult = cache.CreateCacheDir(cacheKey);
        if (!cacheDirResult) {
            MLVC_LOG_WARN("[%s] Failed to create cache directory, using temp compiled model", m_name.c_str());
            return compiled;
        }

        auto destPath = cacheDirResult.value() / "model.mlmodelc";
        NSURL* destURL = [NSURL fileURLWithPath:ToNSString(destPath.string())];
        NSError* error = nil;
        if (![fm moveItemAtURL:compiled.value() toURL:destURL error:&error]) {
            MLVC_LOG_WARN("[%s] Failed to move compiled model to cache: %s", m_name.c_str(),
                          error.localizedDescription.UTF8String ?: "unknown");
            cache.DeleteCacheDir(cacheKey);
            [fm removeItemAtURL:compiled.value() error:nil];
            // Re-compile since we moved the file unsuccessfully (source may be gone)
            return CompileModelFromData(modelData);
        }

        MLVC_LOG_INFO("[%s] Compiled model cache key: '%s'", m_name.c_str(), std::string(cacheKey).c_str());
        cache.CleanupCacheDirs();
        m_compiledModelIsTemp = false;
        return destURL;
    }

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

    expected<void> Initialize(std::span<const std::byte> modelData, const CoreMlEngineParams& params,
                              const std::optional<std::string>& functionName, std::string_view cacheKey)
    {
        if (@available(macOS 15.0, iOS 18.0, *)) { /* OK */
        } else {
            MLVC_LOG_ERROR("[%s] Requires macOS 15.0 or iOS 18.0", m_name.c_str());
            return make_error_code(Error::model_init_error);
        }
        @autoreleasepool {
            auto modelUrl = GetOrCompileModel(modelData, cacheKey);
            if (!modelUrl) return modelUrl.error();
            m_compiledModelUrl = [modelUrl.value() retain];

            MLModelConfiguration* config = [[[MLModelConfiguration alloc] init] autorelease];
            config.computeUnits = ToMLComputeUnits(params.computeUnit);
            if (@available(macOS 15.0, iOS 18.0, *)) {
                config.optimizationHints.specializationStrategy = MLSpecializationStrategyFastPrediction;
                config.optimizationHints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
                if (functionName) [config setFunctionName:ToNSString(*functionName)];
            }
            NSError* error = nil;
            m_model = [MLModel modelWithContentsOfURL:m_compiledModelUrl configuration:config error:&error];
            if (error || !m_model) {
                m_model = nil;
                MLVC_LOG_ERROR("[%s] Failed to load model: %s", m_name.c_str(),
                               error.localizedDescription.UTF8String ?: "unknown");
                return make_error_code(Error::model_init_error);
            }
            [m_model retain];
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
        m_envHash = Fnv1a32("coreml|" + std::to_string(platformInfo.osVersion.major));
        MLVC_LOG_INFO("CoreML initialized: compute_unit=%s, os_ver=%d, env_hash=%s",
                      ComputeUnitToString(m_info.computeUnit), platformInfo.osVersion.major, m_envHash.c_str());
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
        std::string fullCacheKey;
        if (!cacheKey.empty()) {
            fullCacheKey = m_envHash + "_" + std::string(cacheKey);
        }
        // Ensure the same model is not being compiled concurrently.
        std::lock_guard<std::mutex> lock(m_mutex);
        return CoreMlInferenceSession::Create(name, modelData, m_params, functionName, fullCacheKey);
    }
    const InferenceEngineInfo& GetInfo() const override { return m_info; }

private:
    const CoreMlEngineParams m_params;
    const InferenceEngineInfo m_info;
    std::string m_envHash;
    mutable std::mutex m_mutex;
};

expected<std::unique_ptr<IInferenceEngine>> MakeCoreMlEngine(const CoreMlEngineParams& params)
{
    return std::make_unique<CoreMlInferenceEngine>(params);
}

}  // namespace libmlvc
