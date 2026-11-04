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
#include <chrono>
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
constexpr auto kManifestFilename = "mlvc_manifest.json";
constexpr auto kAccessTimeFilename = "last_access.time";
const auto kInvalidModel = std::vector<std::byte>(1024, std::byte{ 0x5a });

struct TestModel {
    std::vector<std::byte> m_bundle;
    std::string m_name;
    std::span<const std::byte> m_modelData{};
    std::map<std::string, std::span<const std::byte>> m_weightsData;
    std::optional<std::string> m_functionName;
};

expected<TestModel> LoadTestModel(const InferenceEngineInfo& info)
{
    // Use the smallest compatible decoder to keep real-model tests inexpensive.
    const auto versions = std::array{ GetTestConfig().mlvcVersion };
    auto bundles = ReadBundleBlobs(GetDefaultModelBundlesDir(), versions, info);
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

expected<std::unique_ptr<IInferenceEngine>> MakeEngine(ComputeUnit computeUnit, std::shared_ptr<ModelCache> modelCache)
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
    params.modelCache = std::move(modelCache);
    return MakeWinMlEngine(params);
#elif defined(MLVC_PLATFORM_APPLE)
    return MakeCoreMlEngine({ .computeUnit = computeUnit, .modelCache = std::move(modelCache) });
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
        auto computeUnit = GetTestConfig().computeUnit;
        if (!HasComputeUnit(computeUnit)) {
            MLVC_LOG_WARN("No %s on this device, testing the CPU engine", ComputeUnitToString(computeUnit));
            computeUnit = ComputeUnit::CPU;
        }
#if defined(MLVC_PLATFORM_APPLE)
        // Core ML's cache is compute-unit independent; avoid Neural Engine compilation overhead.
        computeUnit = ComputeUnit::CPU;
#endif
        s_cacheDir = std::filesystem::temp_directory_path()
                     / ("mlvc_inference_engine_test_" + std::to_string(std::random_device{}()));
        auto cache = ModelCache::Create(s_cacheDir / "models");
        ASSERT_TRUE(cache) << cache.error().message();
        s_cache = std::move(cache.value());
        auto engine = MakeEngine(computeUnit, s_cache);
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
        s_cache.reset();
    }

    void SetUp() override
    {
        ASSERT_TRUE(s_engine) << "Creating the inference engine failed";
        std::error_code ec;
        std::filesystem::remove_all(s_cacheDir, ec);
        ASSERT_TRUE(std::filesystem::create_directory(s_cacheDir, ec)) << ec.message();
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove_all(s_cacheDir, ec);
        EXPECT_FALSE(ec) << ec.message();
    }

    size_t CountCacheFiles() const
    {
        // Include staging and trash so failed operations cannot hide leftover files.
        return static_cast<size_t>(std::ranges::count_if(std::filesystem::recursive_directory_iterator(s_cacheDir),
                                                         [](const auto& entry) { return entry.is_regular_file(); }));
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
        for (const auto& entry : std::filesystem::recursive_directory_iterator(s_cacheDir)) {
            const auto name = entry.path().filename();
            if (entry.is_regular_file() && name != kAccessTimeFilename && name != kManifestFilename) {
                files[entry.path()] = static_cast<int64_t>(entry.last_write_time().time_since_epoch().count());
            }
        }
        return files;
    }

    void CorruptCompiledModels() const
    {
        // Keep sizes and metadata valid so only the backend detects the corruption.
        for (const auto& path : CachedFiles() | std::views::keys) {
            const auto size = std::filesystem::file_size(path);
            auto file = std::ofstream{ path, std::ios::binary | std::ios::trunc };
            ASSERT_TRUE(file) << path;
            file << std::string(static_cast<size_t>(size), 'x');
            file.close();
            ASSERT_TRUE(file) << path;
        }
    }

    static inline std::filesystem::path s_cacheDir;
    static inline std::shared_ptr<ModelCache> s_cache;
    static inline std::unique_ptr<IInferenceEngine> s_engine;
    static inline TestModel s_model;

    const TestModel& m_model = s_model;
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
    if (SupportsModelCache(s_engine->GetInfo())) {
        for (const auto* cacheKey : { "", "invalid.key" }) {
            SCOPED_TRACE("ValidateModelCache with an invalid key");
            SCOPED_TRACE(cacheKey);
            const auto validated = s_engine->ValidateModelCache(cacheKey);
            ASSERT_FALSE(validated);
            EXPECT_EQ(validated.error(), make_error_code(Error::invalid_argument));
        }
        SCOPED_TRACE("Invalid nonempty cache key");
        const auto cached =
            s_engine->PrepareModelCache(m_model.m_name, m_model.m_modelData, m_model.m_weightsData, "invalid.key");
        ASSERT_FALSE(cached);
        EXPECT_EQ(cached.error(), make_error_code(Error::invalid_argument));
        for (const auto modelData : { std::span<const std::byte>{}, m_model.m_modelData }) {
            const auto session = s_engine->CreateSession(m_model.m_name, modelData, m_model.m_weightsData,
                                                         "invalid.key", m_model.m_functionName);
            ASSERT_FALSE(session);
            EXPECT_EQ(session.error(), make_error_code(Error::invalid_argument));
        }
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
    EXPECT_EQ(CountCacheFiles(), 0u);
    if (s_engine->GetInfo().inferenceBackend == InferenceBackend::COREML) {
        SCOPED_TRACE("Unknown function name: the compiled model stays in the cache");
        auto session = s_engine->CreateSession(m_model.m_name, m_model.m_modelData, {}, kCacheKey, "missing");
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_cache_load_error));
        EXPECT_FALSE(CachedFiles().empty());
    } else {
        SCOPED_TRACE("Function name");
        auto session =
            s_engine->CreateSession(m_model.m_name, m_model.m_modelData, m_model.m_weightsData, kCacheKey, "main");
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::invalid_argument));
        EXPECT_EQ(CountCacheFiles(), 0u);
    }
}

TEST_F(UnitTestInferenceEngine, CreateSessionWithoutModelCache)
{
    auto engine = MakeEngine(ComputeUnit::CPU, nullptr);
    ASSERT_TRUE(engine) << engine.error().message();
    const auto createSession = [&](std::span<const std::byte> modelData,
                                   const std::map<std::string, std::span<const std::byte>>& weightsData = {}) {
        return engine.value()->CreateSession(m_model.m_name, modelData, weightsData, kCacheKey, m_model.m_functionName);
    };

    {
        SCOPED_TRACE("PrepareModelCache");
        auto cached =
            engine.value()->PrepareModelCache(m_model.m_name, m_model.m_modelData, m_model.m_weightsData, kCacheKey);
        ASSERT_TRUE(cached) << cached.error().message();
        EXPECT_FALSE(cached.value());
    }
    {
        SCOPED_TRACE("ValidateModelCache");
        const auto validated = engine.value()->ValidateModelCache(kCacheKey);
        ASSERT_TRUE(validated) << validated.error().message();
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
        ASSERT_FALSE(session);
        const auto error = engine.value()->GetInfo().inferenceBackend == InferenceBackend::COREML
                               ? Error::general_failure
                               : Error::model_init_error;
        EXPECT_EQ(session.error(), make_error_code(error));
    }
}

TEST_F(UnitTestInferenceEngine, ModelCacheLifecycle)
{
    // Backend load failures must not trigger repair; validation failures are tested separately.
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
        const auto validated = s_engine->ValidateModelCache(kCacheKey);
        ASSERT_TRUE(validated) << validated.error().message();
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

    // All sessions are released here, so the compiled models can change
    {
        SCOPED_TRACE("Corrupt model with valid sizes: counts as cached, but fails to load");
        ASSERT_NO_FATAL_FAILURE(CorruptCompiledModels());
        const auto corruptFiles = CachedFiles();
        const auto validated = s_engine->ValidateModelCache(kCacheKey);
        ASSERT_TRUE(validated) << validated.error().message();
        auto cached = PrepareModelCache();
        ASSERT_TRUE(cached) << cached.error().message();
        EXPECT_TRUE(cached.value());
        auto session = CreateSession({});
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_cache_load_error));
        session = CreateSession(m_model.m_modelData, m_model.m_weightsData);
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_cache_load_error));
        EXPECT_EQ(CachedFiles(), corruptFiles);
    }

    {
        SCOPED_TRACE("Removed by Cleanup(): compiled again");
        s_cache->Cleanup(std::chrono::seconds(0));
        EXPECT_EQ(CountCacheFiles(), 0u);
        const auto evicted = s_engine->ValidateModelCache(kCacheKey);
        ASSERT_FALSE(evicted);
        EXPECT_EQ(evicted.error(), make_error_code(Error::model_cache_error));
        EXPECT_EQ(CountCacheFiles(), 0u);
        auto cached = PrepareModelCache();
        ASSERT_TRUE(cached) << cached.error().message();
        EXPECT_TRUE(cached.value());
        EXPECT_FALSE(CachedFiles().empty());
        auto session = CreateSession({});
        ASSERT_TRUE(session) << session.error().message();
        EXPECT_EQ(RunSession(*session.value()), std::error_code{});
    }
}

TEST_F(UnitTestInferenceEngine, ModelCacheFailuresReturnErrors)
{
    if (!SupportsModelCache(s_engine->GetInfo())) GTEST_SKIP() << "The inference engine has no model cache";

    {
        SCOPED_TRACE("Not cached, no model data");
        auto session = CreateSession({});
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_cache_error));
        EXPECT_EQ(CountCacheFiles(), 0u);
    }

    {
        SCOPED_TRACE("Invalid model data");
        auto cached = s_engine->PrepareModelCache(m_model.m_name, kInvalidModel, {}, kCacheKey);
        ASSERT_FALSE(cached);
        EXPECT_EQ(cached.error(), make_error_code(Error::model_compile_error));
        auto session = CreateSession(kInvalidModel);
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_compile_error));
        EXPECT_EQ(CountCacheFiles(), 0u);
    }

    {
        SCOPED_TRACE("Unwritable model cache");
        ASSERT_GT(std::filesystem::remove_all(s_cacheDir), 0u);
        {
            const auto file = std::ofstream{ s_cacheDir };
            ASSERT_TRUE(file);
        }

        auto cached = PrepareModelCache();
        ASSERT_FALSE(cached);
        EXPECT_EQ(cached.error(), make_error_code(Error::model_cache_error));

        auto session = CreateSession(m_model.m_modelData, m_model.m_weightsData);
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::model_cache_error));
    }
}

TEST_F(UnitTestInferenceEngine, ModelCacheValidationFailuresAreClassified)
{
    if (!SupportsModelCache(s_engine->GetInfo())) {
        GTEST_SKIP() << "The inference engine has no model cache";
    }
    const auto prepared = PrepareModelCache();
    ASSERT_TRUE(prepared) << prepared.error().message();
    ASSERT_TRUE(prepared.value());
    const auto compiledFiles = CachedFiles();
    auto damagedEntries = size_t{};
    for (const auto& entry : std::filesystem::directory_iterator(s_cacheDir / "models")) {
        if (!entry.is_directory()) {
            continue;
        }
        auto manifest = std::ofstream{ entry.path() / kManifestFilename, std::ios::binary | std::ios::trunc };
        ASSERT_TRUE(manifest);
        manifest << "{invalid";
        manifest.close();
        ASSERT_TRUE(manifest);
        ++damagedEntries;
    }
    ASSERT_EQ(damagedEntries, 1u);
    const auto validated = s_engine->ValidateModelCache(kCacheKey);
    ASSERT_FALSE(validated);
    EXPECT_EQ(validated.error(), make_error_code(Error::model_cache_error));
    const auto preparedAgain = PrepareModelCache();
    ASSERT_FALSE(preparedAgain);
    EXPECT_EQ(preparedAgain.error(), make_error_code(Error::model_cache_error));
    const auto cachedOnly = CreateSession({});
    ASSERT_FALSE(cachedOnly);
    EXPECT_EQ(cachedOnly.error(), make_error_code(Error::model_cache_error));
    const auto withInputs = CreateSession(m_model.m_modelData, m_model.m_weightsData);
    ASSERT_FALSE(withInputs);
    EXPECT_EQ(withInputs.error(), make_error_code(Error::model_cache_error));
    EXPECT_EQ(CachedFiles(), compiledFiles);
}
