// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/bundle.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/platform_names.hpp"
#include "libmlvc/common/tar.hpp"
#include "libmlvc/inference/interface.hpp"
#include "libmlvc/inference/model_cache.hpp"
#include "support/test_config.hpp"
#if defined(MLVC_PLATFORM_WINCLASSIC)
    #include "libmlvc/inference/winml/factory.hpp"
#elif defined(MLVC_PLATFORM_APPLE)
    #include "libmlvc/inference/coreml/factory.hpp"
#endif

#include <libmlvc/libmlvc.hpp>
#include <libmlvc/platform_info.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace libmlvc;

namespace {

constexpr auto kCacheKey = "unit_test_model";
constexpr auto kAccessTimeFilename = "last_access.time";
const auto kInvalidModel = std::vector<std::byte>(1024, std::byte{ 0x5a });

struct TestModel {
    std::vector<std::byte> m_bundle;
    std::string m_name;
    std::span<const std::byte> m_modelData{};
    std::map<std::string, std::span<const std::byte>> m_weightsData;
    std::optional<std::string> m_functionName;
};

// Use the lowest-resolution decoder from the engine-compatible bundle.
expected<TestModel> LoadTestModel(const InferenceEngineInfo& info)
{
    const auto versions = std::array{ GetTestConfig().mlvcVersion };
    auto bundles = LoadModelBundles(GetDefaultModelBundlesDir(), versions, info);
    if (!bundles) return bundles.error();

    auto model = TestModel{};
    model.m_bundle = std::move(bundles->front());
    auto files = ExtractTar(model.m_bundle);
    if (!files) return files.error();
    auto manifest = BundleManifest::FromJsonBuffer(files->at("bundle_manifest.json"));
    if (!manifest) return manifest.error();
    if (manifest->modelMetadata.empty()) {
        MLVC_LOG_ERROR("Test bundle has no model resolutions");
        return make_error_code(Error::invalid_argument);
    }
    for (const auto& metadata : manifest->modelMetadata | std::views::values) {
        if (metadata->modelWidth <= 0 || metadata->modelHeight <= 0) {
            MLVC_LOG_ERROR("Test bundle has invalid model dimensions");
            return make_error_code(Error::invalid_argument);
        }
    }
    const auto smallest = std::ranges::min_element(manifest->modelMetadata, [](const auto& a, const auto& b) {
        return static_cast<int64_t>(a.second->modelWidth) * a.second->modelHeight
               < static_cast<int64_t>(b.second->modelWidth) * b.second->modelHeight;
    });
    const auto& part = manifest->modelManifests.at(smallest->first).modelParts.at(ModelPartId::DECODER);
    const auto& entry = manifest->modelRegistry.at(part.registryId);
    model.m_name = smallest->first + "/" + ModelPartIdToString(ModelPartId::DECODER);
    model.m_modelData = files->at(entry.modelPath);
    model.m_functionName = part.functionName;
    if (entry.weightsPath) {
        model.m_weightsData[*entry.weightsPath] = files->at(*entry.weightsPath);
    }
    MLVC_LOG_INFO("Inference engine tests use %s", model.m_name.c_str());
    return model;
}

expected<std::unique_ptr<IInferenceEngine>> MakeEngine(ComputeUnit computeUnit)
{
    if (computeUnit == ComputeUnit::AUTO) {
        computeUnit = HasSupportedNpu() ? ComputeUnit::NPU : ComputeUnit::GPU;
    }
#if defined(MLVC_PLATFORM_WINCLASSIC)
    const bool isQualcomm = GetPlatformInfo().cpuVendor == VENDOR_QUALCOMM;
    WinMlEngineParams params;
    params.initMode = GetTestConfig().winmlInitMode;
    params.computeUnit = computeUnit;
    if (computeUnit == ComputeUnit::CPU) {
        params.onnxExecutionProvider = OnnxExecutionProvider::CPU;
    } else if (isQualcomm) {
        params.onnxExecutionProvider = OnnxExecutionProvider::QNN;
    } else {
        params.onnxExecutionProvider =
            computeUnit == ComputeUnit::NPU ? OnnxExecutionProvider::OPENVINO : OnnxExecutionProvider::DIRECTML;
    }
    return MakeWinMlEngine(params);
#elif defined(MLVC_PLATFORM_APPLE)
    return MakeCoreMlEngine(CoreMlEngineParams{ computeUnit });
#endif
}

bool SupportsModelCache(const InferenceEngineInfo& info)
{
    if (info.inferenceBackend == InferenceBackend::COREML) return true;
    return info.computeUnit == ComputeUnit::NPU
           && (info.onnxExecutionProvider == OnnxExecutionProvider::OPENVINO
               || info.onnxExecutionProvider == OnnxExecutionProvider::QNN);
}

bool HasComputeUnit(ComputeUnit computeUnit)
{
    switch (computeUnit) {
    case ComputeUnit::CPU:
        return true;
    case ComputeUnit::NPU:
        return HasSupportedNpu();
    case ComputeUnit::GPU:
#if defined(MLVC_PLATFORM_APPLE)
        return true;
#else
        return std::ranges::any_of(GetPlatformInfo().accelerators,
                                   [](const auto& accelerator) { return accelerator.type == AcceleratorType::GPU; });
#endif
    case ComputeUnit::AUTO:
        return HasComputeUnit(ComputeUnit::NPU) || HasComputeUnit(ComputeUnit::GPU);
    }
    return false;
}

// Runs the session once on freshly allocated tensors
std::error_code RunSession(IInferenceSession& session)
{
    auto inputNames = session.GetInputNames();
    if (!inputNames) return inputNames.error();
    auto outputNames = session.GetOutputNames();
    if (!outputNames) return outputNames.error();

    std::vector<InferenceTensorPtr> inputs, outputs;
    for (const auto& name : inputNames.value()) {
        auto tensor = session.CreateTensor(TensorIoType::INPUT, name);
        if (!tensor) return tensor.error();
        inputs.push_back(std::move(tensor.value()));
    }
    for (const auto& name : outputNames.value()) {
        auto tensor = session.CreateTensor(TensorIoType::OUTPUT, name);
        if (!tensor) return tensor.error();
        outputs.push_back(std::move(tensor.value()));
    }
    auto ran = session.Run(inputs, outputs);
    return ran ? std::error_code{} : ran.error();
}

}  // namespace

class UnitTestInferenceEngine : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        // Without the configured compute unit, test the CPU engine: tests that need a model cache skip themselves
        auto computeUnit = GetTestConfig().computeUnit;
        if (!HasComputeUnit(computeUnit)) {
            MLVC_LOG_WARN("No %s on this device, testing the CPU engine", ComputeUnitToString(computeUnit));
            computeUnit = ComputeUnit::CPU;
        }
#if defined(MLVC_PLATFORM_APPLE)
        // Core ML's model cache doesn't depend on the compute unit, and a fresh compile for the Neural Engine takes ~30 s
        computeUnit = ComputeUnit::CPU;
#endif
        auto engine = MakeEngine(computeUnit);
        ASSERT_TRUE(engine) << engine.error().message();
        auto model = LoadTestModel(engine.value()->GetInfo());
        ASSERT_TRUE(model) << model.error().message();
        s_engine = std::move(engine.value());
        s_model = std::move(model.value());
    }

    static void TearDownTestSuite()
    {
        s_engine.reset();
        s_model = {};
    }

    void SetUp() override
    {
        ASSERT_TRUE(s_engine) << "Creating the inference engine failed";
        const auto cacheDir = std::filesystem::temp_directory_path()
                              / ("mlvc_inference_engine_test_" + std::to_string(std::random_device{}()));
        ASSERT_TRUE(std::filesystem::create_directory(cacheDir));
        m_cacheDir = cacheDir;
        // The model cache root is process-wide: replace the one set by earlier tests
        ModelCache::Instance().SetRootCacheDir({});
        ModelCache::Instance().SetRootCacheDir(m_cacheDir);
    }

    void TearDown() override
    {
        if (!m_cacheDir.empty()) {
            ModelCache::Instance().SetRootCacheDir({});
            std::error_code ec;
            std::filesystem::remove_all(m_cacheDir, ec);
            EXPECT_FALSE(ec) << ec.message();
        }
    }

    expected<bool> PrepareModelCache()
    {
        return s_engine->PrepareModelCache(m_model.m_name, m_model.m_modelData, m_model.m_weightsData, kCacheKey);
    }

    expected<InferenceSessionPtr> CreateSession(std::span<const std::byte> modelData,
                                                const std::map<std::string, std::span<const std::byte>>& weightsData = {})
    {
        return s_engine->CreateSession(m_model.m_name, modelData, weightsData, kCacheKey, m_model.m_functionName);
    }

    // Write times as native ticks: gtest can't print file_time_type portably (std::format needs macOS 13.3+ for it)
    using CachedFileTimes = std::map<std::filesystem::path, int64_t>;

    CachedFileTimes CachedFiles() const
    {
        CachedFileTimes files;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(m_cacheDir)) {
            if (entry.is_regular_file() && entry.path().filename() != kAccessTimeFilename) {
                files[entry.path()] = static_cast<int64_t>(entry.last_write_time().time_since_epoch().count());
            }
        }
        return files;
    }

    // Removes the compiled models but keeps their cache entries; no session may be using them
    void RemoveCompiledModels() const
    {
        for (const auto& entry : std::filesystem::directory_iterator(m_cacheDir)) {
            for (const auto& item : std::filesystem::directory_iterator(entry.path())) {
                if (item.path().filename() != kAccessTimeFilename) std::filesystem::remove_all(item.path());
            }
        }
    }

    // Overwrites the compiled models with invalid data; no session may be using them
    void CorruptCompiledModels() const
    {
        for (const auto& path : CachedFiles() | std::views::keys) {
            auto file = std::ofstream{ path, std::ios::binary | std::ios::trunc };
            ASSERT_TRUE(file) << path;
            file << "corrupt";
            ASSERT_TRUE(file) << path;
        }
    }

    static inline std::unique_ptr<IInferenceEngine> s_engine;
    static inline TestModel s_model;

    const TestModel& m_model = s_model;
    std::filesystem::path m_cacheDir;
};

TEST_F(UnitTestInferenceEngine, RejectsInvalidArguments)
{
    {
        SCOPED_TRACE("CreateSession without model data or cache key");
        auto session = s_engine->CreateSession(m_model.m_name, {}, {}, "", m_model.m_functionName);
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::invalid_argument));
    }
    {
        SCOPED_TRACE("PrepareModelCache without model data");
        auto cached = s_engine->PrepareModelCache(m_model.m_name, {}, {}, kCacheKey);
        ASSERT_FALSE(cached);
        EXPECT_EQ(cached.error(), make_error_code(Error::invalid_argument));
    }
    {
        SCOPED_TRACE("PrepareModelCache without a cache key");
        auto cached = s_engine->PrepareModelCache(m_model.m_name, m_model.m_modelData, m_model.m_weightsData, "");
        ASSERT_FALSE(cached);
        EXPECT_EQ(cached.error(), make_error_code(Error::invalid_argument));
    }
    if (s_engine->GetInfo().inferenceBackend == InferenceBackend::COREML) {
        SCOPED_TRACE("External weights");
        const auto weightsData =
            std::map<std::string, std::span<const std::byte>>{ { "weights.bin", m_model.m_modelData } };
        auto cached = s_engine->PrepareModelCache(m_model.m_name, m_model.m_modelData, weightsData, kCacheKey);
        ASSERT_FALSE(cached);
        EXPECT_EQ(cached.error(), make_error_code(Error::invalid_argument));
        auto session = CreateSession(m_model.m_modelData, weightsData);
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::invalid_argument));
    }
    if (s_engine->GetInfo().inferenceBackend == InferenceBackend::COREML) {
        SCOPED_TRACE("Unknown function name: the freshly compiled model is deleted");
        auto session = s_engine->CreateSession(m_model.m_name, m_model.m_modelData, {}, kCacheKey, "missing");
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_init_error));
    } else {
        SCOPED_TRACE("Function name");
        auto session =
            s_engine->CreateSession(m_model.m_name, m_model.m_modelData, m_model.m_weightsData, kCacheKey, "main");
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::invalid_argument));
    }
    EXPECT_TRUE(std::filesystem::is_empty(m_cacheDir));
}

// The Windows CPU EP has no model cache and ignores the cache key; Core ML uses no cache without a cache key.
TEST_F(UnitTestInferenceEngine, CreateSessionWithoutModelCache)
{
    auto engine = MakeEngine(ComputeUnit::CPU);
    ASSERT_TRUE(engine) << engine.error().message();
    const bool engineCaches = SupportsModelCache(engine.value()->GetInfo());
    const std::string_view cacheKey = engineCaches ? "" : kCacheKey;
    const auto createSession = [&](std::span<const std::byte> modelData,
                                   const std::map<std::string, std::span<const std::byte>>& weightsData = {}) {
        return engine.value()->CreateSession(m_model.m_name, modelData, weightsData, cacheKey, m_model.m_functionName);
    };

    if (!engineCaches) {
        SCOPED_TRACE("PrepareModelCache");
        auto cached =
            engine.value()->PrepareModelCache(m_model.m_name, m_model.m_modelData, m_model.m_weightsData, kCacheKey);
        ASSERT_TRUE(cached) << cached.error().message();
        EXPECT_FALSE(cached.value());
    }
    {
        SCOPED_TRACE("Model data");
        auto session = createSession(m_model.m_modelData, m_model.m_weightsData);
        ASSERT_TRUE(session) << session.error().message();
        EXPECT_EQ(RunSession(*session.value()), std::error_code{});
        EXPECT_FALSE(session.value()->CreateTensor(TensorIoType::INPUT, "missing"));
        EXPECT_FALSE(session.value()->Run({}, {}));
    }
    {
        SCOPED_TRACE("No model data");
        auto session = createSession({});
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::invalid_argument));
    }
    {
        SCOPED_TRACE("Invalid model data");
        auto session = createSession(kInvalidModel);
        EXPECT_FALSE(session);
    }
    EXPECT_TRUE(std::filesystem::is_empty(m_cacheDir));
}

TEST_F(UnitTestInferenceEngine, ModelCacheLifecycle)
{
    if (!SupportsModelCache(s_engine->GetInfo())) GTEST_SKIP() << "The inference engine has no model cache";

    CachedFileTimes compiledFiles;
    {
        SCOPED_TRACE("Not cached: CreateSession() compiles the model into the cache");
        auto session = CreateSession(m_model.m_modelData, m_model.m_weightsData);
        ASSERT_TRUE(session) << session.error().message();
        EXPECT_EQ(RunSession(*session.value()), std::error_code{});
        compiledFiles = CachedFiles();
        ASSERT_FALSE(compiledFiles.empty());
    }

    {
        SCOPED_TRACE("Cached: loaded with or without model data, and nothing is rewritten");
        auto cached = PrepareModelCache();
        ASSERT_TRUE(cached) << cached.error().message();
        EXPECT_TRUE(cached.value());
        auto session = CreateSession({});
        ASSERT_TRUE(session) << session.error().message();
        EXPECT_EQ(RunSession(*session.value()), std::error_code{});
        session = CreateSession(m_model.m_modelData, m_model.m_weightsData);
        ASSERT_TRUE(session) << session.error().message();
        EXPECT_EQ(CachedFiles(), compiledFiles);
    }

    {
        SCOPED_TRACE("Cached: sessions created concurrently, as a call's encoder and decoder are");
        std::vector<std::error_code> errors(4);
        std::vector<std::thread> threads;
        for (auto& error : errors) {
            threads.emplace_back([this, &error] {
                auto session = CreateSession({});
                error = session ? RunSession(*session.value()) : session.error();
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        EXPECT_EQ(errors, std::vector<std::error_code>(errors.size()));
    }

    // All sessions are released here: OpenVINO only compiles a model again when none of its sessions is alive
    RemoveCompiledModels();
    ASSERT_TRUE(CachedFiles().empty());
    {
        SCOPED_TRACE("Compiled model missing, no model data");
        auto session = CreateSession({});
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_init_error));
    }
#if defined(MLVC_PLATFORM_WINCLASSIC)
    const auto cachedModel =
        std::ranges::find_if(compiledFiles, [](const auto& file) { return file.first.filename() == "model.onnx"; });
    ASSERT_NE(cachedModel, compiledFiles.end());
    const auto modelPath = cachedModel->first;
    {
        SCOPED_TRACE("Cache directory cannot be deleted");
        // Windows streams hold the file without delete sharing.
        const auto lockedFile = std::ofstream{ modelPath.parent_path() / "locked.bin" };
        ASSERT_TRUE(lockedFile);
        auto failedCache = PrepareModelCache();
        ASSERT_FALSE(failedCache);
        EXPECT_EQ(failedCache.error(), make_error_code(Error::io_error));
        auto failedSession = CreateSession(m_model.m_modelData, m_model.m_weightsData);
        ASSERT_FALSE(failedSession);
        EXPECT_EQ(failedSession.error(), make_error_code(Error::io_error));
    }
    // A directory in place of the compiled model doesn't count as cached
    ASSERT_TRUE(std::filesystem::create_directory(modelPath));
#endif
    {
        SCOPED_TRACE("Compiled model missing: PrepareModelCache() compiles it again");
        auto cached = PrepareModelCache();
        ASSERT_TRUE(cached) << cached.error().message();
        EXPECT_TRUE(cached.value());
        EXPECT_FALSE(CachedFiles().empty());
#if defined(MLVC_PLATFORM_WINCLASSIC)
        EXPECT_TRUE(std::filesystem::is_regular_file(modelPath));
#endif
        auto session = CreateSession({});
        ASSERT_TRUE(session) << session.error().message();
        EXPECT_EQ(RunSession(*session.value()), std::error_code{});
    }

    {
        SCOPED_TRACE("Corrupt model: counts as cached, kept without model data, recompiled with it");
        ASSERT_NO_FATAL_FAILURE(CorruptCompiledModels());
        auto cached = PrepareModelCache();
        ASSERT_TRUE(cached) << cached.error().message();
        EXPECT_TRUE(cached.value());
        const auto corruptFiles = CachedFiles();
        auto session = CreateSession({});
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_init_error));
        EXPECT_EQ(CachedFiles(), corruptFiles);
        session = CreateSession(m_model.m_modelData, m_model.m_weightsData);
        ASSERT_TRUE(session) << session.error().message();
        EXPECT_EQ(RunSession(*session.value()), std::error_code{});
        session = CreateSession({});
        ASSERT_TRUE(session) << session.error().message();
    }

    {
        SCOPED_TRACE("Corrupt model that can't be recompiled: deleted");
        ASSERT_NO_FATAL_FAILURE(CorruptCompiledModels());
        auto session = CreateSession(kInvalidModel);
        EXPECT_FALSE(session);
        EXPECT_TRUE(std::filesystem::is_empty(m_cacheDir));
    }
}

TEST_F(UnitTestInferenceEngine, ModelCacheFailuresReturnErrors)
{
    if (!SupportsModelCache(s_engine->GetInfo())) GTEST_SKIP() << "The inference engine has no model cache";

    {
        SCOPED_TRACE("Not cached, no model data");
        auto session = CreateSession({});
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_init_error));
        EXPECT_TRUE(std::filesystem::is_empty(m_cacheDir));
    }

    {
        SCOPED_TRACE("Invalid model data");
        auto cached = s_engine->PrepareModelCache(m_model.m_name, kInvalidModel, {}, kCacheKey);
        EXPECT_FALSE(cached);
        auto session = CreateSession(kInvalidModel);
        EXPECT_FALSE(session);
        EXPECT_TRUE(std::filesystem::is_empty(m_cacheDir));
    }

    {
        SCOPED_TRACE("Unwritable model cache");
        const auto blockingFile = m_cacheDir / "file";
        {
            const auto file = std::ofstream{ blockingFile };
            ASSERT_TRUE(file);
        }
        ModelCache::Instance().SetRootCacheDir({});
        ModelCache::Instance().SetRootCacheDir(blockingFile / "models");

        auto cached = PrepareModelCache();
        ASSERT_FALSE(cached);
        EXPECT_EQ(cached.error(), make_error_code(Error::io_error));

        auto session = CreateSession(m_model.m_modelData, m_model.m_weightsData);
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::io_error));
    }
}
