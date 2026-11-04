// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/manager.hpp"
#include "libmlvc/common/macros.hpp"
#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

using namespace libmlvc;

static_assert(!std::is_copy_constructible_v<MlvcEncoder>);
static_assert(!std::is_copy_assignable_v<MlvcEncoder>);
static_assert(std::is_nothrow_move_constructible_v<MlvcEncoder>);
static_assert(std::is_nothrow_move_assignable_v<MlvcEncoder>);
static_assert(!std::is_copy_constructible_v<MlvcDecoder>);
static_assert(!std::is_copy_assignable_v<MlvcDecoder>);
static_assert(std::is_nothrow_move_constructible_v<MlvcDecoder>);
static_assert(std::is_nothrow_move_assignable_v<MlvcDecoder>);

namespace {

class ScopedTempDir {
public:
    ScopedTempDir() = default;
    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;
    ScopedTempDir(ScopedTempDir&&) = delete;
    ScopedTempDir& operator=(ScopedTempDir&&) = delete;
    ~ScopedTempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(m_path, ec);
    }

    const std::filesystem::path m_path =
        std::filesystem::temp_directory_path() / ("mlvc_manager_test_" + std::to_string(std::random_device{}()));
};

testing::AssertionResult WriteCacheFile(const std::filesystem::path& path, std::string_view contents)
{
    auto file = std::ofstream{ path, std::ios::binary | std::ios::trunc };
    if (!file) {
        return testing::AssertionFailure() << "Failed to open " << path;
    }
    file << contents;
    file.close();
    if (!file) {
        return testing::AssertionFailure() << "Failed to write or close " << path;
    }
    return testing::AssertionSuccess();
}

testing::AssertionResult WriteAccessTime(const std::filesystem::path& dir, std::chrono::system_clock::duration age)
{
    // Little-endian int64 seconds since 1970, the byte order of every supported platform.
    const auto time = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>((std::chrono::system_clock::now() - age).time_since_epoch()).count());
    return WriteCacheFile(dir / "last_access.time", std::string_view{ reinterpret_cast<const char*>(&time), sizeof(time) });
}

testing::AssertionResult WriteCacheEntry(const std::filesystem::path& dir, std::chrono::system_clock::duration age)
{
    constexpr auto model = std::string_view{ "compiled model" };
    std::filesystem::create_directories(dir);
    if (auto written = WriteCacheFile(dir / "model.bin", model); !written) {
        return written;
    }
    const auto manifest = R"({"version":1,"files":[{"path":"model.bin","size":)" + std::to_string(model.size()) + "}]}";
    if (auto written = WriteCacheFile(dir / "mlvc_manifest.json", manifest); !written) {
        return written;
    }
    return WriteAccessTime(dir, age);
}

bool SupportsModelCache(const ManagerInfo& info)
{
    if (info.inferenceBackend == InferenceBackend::COREML) {
        return true;
    }
    return info.computeUnit == ComputeUnit::NPU
           && (info.onnxExecutionProvider == OnnxExecutionProvider::OPENVINO
               || info.onnxExecutionProvider == OnnxExecutionProvider::QNN);
}

expected<std::vector<std::byte>> ReadCompatibleBundleFixture(const ManagerInfo& info, const MlvcVersion version)
{
    auto bundleMarker = std::string_view{ "-onnx-" };
    if (info.inferenceBackend == InferenceBackend::COREML) {
        bundleMarker = "-coreml-apple-";
    } else if (info.onnxExecutionProvider == OnnxExecutionProvider::QNN) {
        bundleMarker = "-onnx-qualcomm-";
    } else if (info.onnxExecutionProvider == OnnxExecutionProvider::OPENVINO) {
        bundleMarker = "-onnx-intel-";
    }

    const auto versionSuffix = "-" + version.ToString() + ".tar";
    auto bundlePaths = std::vector<std::filesystem::path>{};
    for (const auto& entry : std::filesystem::directory_iterator(GetDefaultModelBundlesDir())) {
        const auto filename = entry.path().filename().string();
        if (!entry.is_regular_file() || entry.path().extension() != ".tar"
            || filename.find(bundleMarker) == std::string::npos || !filename.ends_with(versionSuffix)) {
            continue;
        }
        bundlePaths.push_back(entry.path());
    }
    std::ranges::sort(bundlePaths);
    if (bundlePaths.empty()) {
        return make_error_code(Error::invalid_argument);
    }

    std::ifstream file(bundlePaths.front(), std::ios::binary | std::ios::ate);
    if (!file) {
        return make_error_code(Error::io_error);
    }
    const auto fileSize = file.tellg();
    if (fileSize <= 0) {
        return make_error_code(Error::io_error);
    }
    auto bundleBytes = std::vector<std::byte>(static_cast<std::size_t>(fileSize));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bundleBytes.data()), fileSize)) {
        return make_error_code(Error::io_error);
    }
    return bundleBytes;
}

}  // namespace

class ManagerTests : public ::testing::Test {
protected:
    ManagerTests()
    {
        m_managerParams.computeUnit = GetTestConfig().computeUnit;
        m_managerParams.sessionPreloadMode = SessionPreloadMode::NONE;
        m_managerParams.winmlInitMode = GetTestConfig().winmlInitMode;
    }

    MlvcVersion m_mlvcVersion = GetTestConfig().mlvcVersion;
    ManagerParams m_managerParams;
};

// -----------------------------------------------------------------------------
// Creation tests
// -----------------------------------------------------------------------------

TEST_F(ManagerTests, CreateFromDirectory)
{
    auto managerCopy = std::optional<MlvcManager>{};
    {
        auto manager = MlvcManager::CreateFromDirectory(m_managerParams, {}, std::array{ m_mlvcVersion });
        ASSERT_TRUE(manager);
        ASSERT_EQ(manager.value().GetAvailableVersions(), std::vector{ m_mlvcVersion });
        managerCopy.emplace(manager.value());
    }

    ASSERT_EQ(managerCopy->GetAvailableVersions(), std::vector{ m_mlvcVersion });
}

#if defined(MLVC_PLATFORM_WINCLASSIC)
TEST_F(ManagerTests, PropagatesModelCacheCreationErrors)
{
    const auto tempDir = ScopedTempDir{};
    auto params = m_managerParams;
    params.cacheDir = std::filesystem::path{ std::string(1, '\0') };
    params.computeUnit = ComputeUnit::CPU;
    auto error = std::error_code{};
    ASSERT_TRUE(std::filesystem::absolute(params.cacheDir / "models", error).empty());
    ASSERT_TRUE(error);

    for (const bool fromBlobs : { false, true }) {
        SCOPED_TRACE(fromBlobs ? "From blobs" : "From directory");
        auto backendReady = false;
        const auto onProgress = [&](const InitializeProgressEvent& event) {
            if (std::holds_alternative<InferenceBackendReady>(event)) {
                backendReady = true;
            }
        };
        const auto manager = [&] {
            if (fromBlobs) {
                const auto blob = std::array<std::byte, 1024>{};
                return MlvcManager::CreateFromBlobs(params, std::array{ std::span<const std::byte>{ blob } }, {},
                                                    onProgress);
            }
            return MlvcManager::CreateFromDirectory(params, tempDir.m_path / "missing", std::array{ m_mlvcVersion }, {},
                                                    onProgress);
        }();
        ASSERT_FALSE(manager);
        EXPECT_EQ(manager.error(), make_error_code(Error::model_cache_error));
        EXPECT_FALSE(backendReady);
    }
}
#endif

TEST_F(ManagerTests, CleansModelCacheOnInitialization)
{
    const auto tempDir = ScopedTempDir{};
    const auto& cacheDir = tempDir.m_path;
    const auto expired = cacheDir / "models" / "v2_expired";
    const auto fresh = cacheDir / "models" / "v2_fresh";
    const auto untimed = cacheDir / "models" / "v2_untimed";
    const auto afterInitialization = cacheDir / "models" / "v2_after_initialization";
    const auto abandoned = cacheDir / "models.staging" / "abandoned";
    const auto trashed = cacheDir / "models.trash" / "leftover";
    auto params = m_managerParams;
    params.computeUnit = ComputeUnit::CPU;
    params.cacheDir = cacheDir;
    for (const bool enabled : { false, true }) {
        SCOPED_TRACE(enabled ? "Model cache enabled" : "Model cache disabled");
        ASSERT_TRUE(WriteCacheEntry(expired, std::chrono::days(30)));
        ASSERT_TRUE(WriteCacheEntry(fresh, {}));
        ASSERT_TRUE(WriteCacheEntry(untimed, {}));
        std::filesystem::remove(untimed / "last_access.time");
        ASSERT_TRUE(WriteCacheEntry(abandoned, std::chrono::hours(2)));
        ASSERT_TRUE(WriteCacheEntry(trashed, {}));
        params.enableModelCache = enabled;
        const auto manager = std::make_shared<MlvcManagerImpl>(params);
        auto cancelToken = CancelToken{};
        bool backendReady = false;
        const auto checkLeftoversRemoved = [&](const InitializeProgressEvent& event) {
            if (const auto* status = std::get_if<StatusMessage>(&event);
                status && status->message == "Initialization complete") {
                EXPECT_EQ(std::filesystem::exists(expired), !enabled);
                EXPECT_TRUE(WriteCacheEntry(afterInitialization, std::chrono::days(30)));
                return;
            }
            if (!std::holds_alternative<InferenceBackendReady>(event)) {
                return;
            }
            backendReady = true;
            EXPECT_TRUE(std::filesystem::exists(expired));
            EXPECT_EQ(std::filesystem::exists(untimed), !enabled);
            EXPECT_EQ(std::filesystem::exists(abandoned), !enabled);
            EXPECT_EQ(std::filesystem::exists(trashed), !enabled);
            // Avoid compilation here; non-caching backends still exercise the final retention cleanup.
            if (enabled && SupportsModelCache(manager->GetInfo())) {
                cancelToken.Cancel();
            }
        };
        const auto initialized = manager->Initialize({}, std::array{ m_mlvcVersion }, cancelToken, checkLeftoversRemoved);
        if (cancelToken.IsCancelled()) {
            ASSERT_FALSE(initialized);
            EXPECT_EQ(initialized.error(), make_error_code(Error::operation_cancelled)) << initialized.error().message();
        } else {
            ASSERT_TRUE(initialized) << initialized.error().message();
        }
        EXPECT_TRUE(backendReady);
        const bool cleanedUp = enabled && initialized;
        EXPECT_EQ(std::filesystem::exists(expired), !cleanedUp);
        EXPECT_TRUE(std::filesystem::exists(fresh));
        if (initialized) {
            EXPECT_TRUE(std::filesystem::exists(afterInitialization)) << "Cleanup ran after initialization completed";
        }
    }
}

TEST_F(ManagerTests, CleansModelCacheOnBundleLoadFailure)
{
    for (const bool fromBlobs : { false, true }) {
        SCOPED_TRACE(fromBlobs ? "Missing manifest in blob" : "Missing bundle directory");
        for (const bool enabled : { false, true }) {
            SCOPED_TRACE(enabled ? "Model cache enabled" : "Model cache disabled");
            const auto tempDir = ScopedTempDir{};
            const auto expired = tempDir.m_path / "models" / "v2_expired";
            const auto fresh = tempDir.m_path / "models" / "v2_fresh";
            ASSERT_TRUE(WriteCacheEntry(expired, std::chrono::days(30)));
            ASSERT_TRUE(WriteCacheEntry(fresh, {}));
            auto params = m_managerParams;
            params.cacheDir = tempDir.m_path;
            params.enableModelCache = enabled;
            params.computeUnit = ComputeUnit::CPU;

            const auto manager = [&] {
                if (fromBlobs) {
                    const auto blob = std::array<std::byte, 1024>{};
                    return MlvcManager::CreateFromBlobs(params, std::array{ std::span<const std::byte>{ blob } });
                }
                return MlvcManager::CreateFromDirectory(params, tempDir.m_path / "missing", std::array{ m_mlvcVersion });
            }();
            ASSERT_FALSE(manager);
            EXPECT_EQ(manager.error(), make_error_code(fromBlobs ? Error::io_error : Error::invalid_argument));
            const bool cleanedUp = enabled && fromBlobs;
            EXPECT_EQ(std::filesystem::exists(expired), !cleanedUp);
            EXPECT_TRUE(std::filesystem::exists(fresh));
        }
    }
}

TEST_F(ManagerTests, CleansModelCacheOnCancelledBundleLoad)
{
    const auto tempDir = ScopedTempDir{};
    const auto expired = tempDir.m_path / "models" / "v2_expired";
    ASSERT_TRUE(WriteCacheEntry(expired, std::chrono::days(30)));
    auto params = m_managerParams;
    params.cacheDir = tempDir.m_path;
    params.computeUnit = ComputeUnit::CPU;
    auto cancelToken = CancelToken{};
    auto bundleLoadingStarted = false;
    const auto blob = std::array<std::byte, 1024>{};
    const auto onProgress = [&](const InitializeProgressEvent& event) {
        const auto* status = std::get_if<StatusMessage>(&event);
        if (status && status->message == "Extracting bundle files") {
            bundleLoadingStarted = true;
            cancelToken.Cancel();
        }
    };
    const auto manager =
        MlvcManager::CreateFromBlobs(params, std::array{ std::span<const std::byte>{ blob } }, cancelToken, onProgress);
    EXPECT_TRUE(bundleLoadingStarted);
    ASSERT_FALSE(manager);
    EXPECT_EQ(manager.error(), make_error_code(Error::operation_cancelled));
    EXPECT_FALSE(std::filesystem::exists(expired));
}

TEST_F(ManagerTests, SharesModelCacheWithInferenceEngine)
{
    const auto tempDir = ScopedTempDir{};
    const auto modelsDir = tempDir.m_path / "models";
    // Manifests are written only on compilation. Use native clock ticks for portable gtest diagnostics.
    const auto compileTimes = [&] {
        auto times = std::map<std::string, int64_t>{};
        for (const auto& entry : std::filesystem::directory_iterator(modelsDir)) {
            const auto time = std::filesystem::last_write_time(entry.path() / "mlvc_manifest.json");
            times[entry.path().filename().string()] = static_cast<int64_t>(time.time_since_epoch().count());
        }
        return times;
    };
    auto params = m_managerParams;
    params.cacheDir = tempDir.m_path;
    // One compiled model is enough to verify sharing; avoid preparing the whole bundle.
    const auto createManager = [&] {
        auto cancelToken = CancelToken{};
        auto prepared = 0;
        return MlvcManager::CreateFromDirectory(
            params, {}, std::array{ m_mlvcVersion }, cancelToken, [&](const InitializeProgressEvent& event) {
                const auto* status = std::get_if<StatusMessage>(&event);
                if (status && status->message.starts_with("Preparing model cache for") && ++prepared == 2) {
                    cancelToken.Cancel();
                }
            });
    };
    auto manager = createManager();
    if (!manager) {
        ASSERT_EQ(manager.error(), make_error_code(Error::operation_cancelled)) << manager.error().message();
    } else if (!SupportsModelCache(manager.value().GetInfo())) {
        GTEST_SKIP() << "The inference engine has no model cache";
    }
    const auto compiled = compileTimes();
    ASSERT_FALSE(compiled.empty());
    for (const auto& entry : compiled) {
        EXPECT_TRUE(entry.first.starts_with("v2_")) << entry.first;
    }

    {
        SCOPED_TRACE("Unused for 15 days");
        for (const auto& entry : compiled) {
            ASSERT_TRUE(WriteAccessTime(modelsDir / entry.first, std::chrono::days(15)));
        }
        auto again = createManager();
        if (!again) {
            EXPECT_EQ(again.error(), make_error_code(Error::operation_cancelled)) << again.error().message();
        }
        EXPECT_EQ(compileTimes(), compiled);
    }

    const auto invalid = modelsDir / compiled.begin()->first;
    ASSERT_TRUE(std::filesystem::remove(invalid / "mlvc_manifest.json"));
    for (const auto age : { std::chrono::days{ 0 }, std::chrono::days{ 30 } }) {
        SCOPED_TRACE("Invalid entry unused for " + std::to_string(age.count()) + " days");
        ASSERT_TRUE(WriteAccessTime(invalid, age));
        const auto failed = createManager();
        ASSERT_FALSE(failed);
        EXPECT_EQ(failed.error(), make_error_code(Error::model_cache_error));
        EXPECT_EQ(std::filesystem::exists(invalid), age == std::chrono::days{ 0 });
    }
}

TEST_F(ManagerTests, ReportsInferenceBackendReadyBeforeLoadingBundles)
{
    std::vector<std::string> events;
    auto manager = MlvcManager::CreateFromDirectory(
        m_managerParams, {}, std::array{ m_mlvcVersion }, {}, [&](const InitializeProgressEvent& event) {
            if (std::holds_alternative<InferenceBackendReady>(event)) {
                events.emplace_back("(backend ready)");
            } else if (const auto* status = std::get_if<StatusMessage>(&event)) {
                events.push_back(status->message);
            }
        });
    ASSERT_TRUE(manager) << manager.error().message();
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.front(), "Setting up inference engine");
    ASSERT_EQ(std::ranges::count(events, "(backend ready)"), 1);
    const auto ready = std::ranges::find(events, "(backend ready)");
    ASSERT_NE(std::next(ready), events.end());
    EXPECT_EQ(*std::next(ready), "Reading model bundles");
}

TEST_F(ManagerTests, CodecsOutliveManagerHandles)
{
    auto encoder = std::optional<MlvcEncoder>{};
    auto decoder = std::optional<MlvcDecoder>{};
    {
        auto manager = MlvcManager::CreateFromDirectory(m_managerParams, {}, std::array{ m_mlvcVersion });
        ASSERT_TRUE(manager);

        auto config = manager.value().GetDefaultEncoderConfig(m_mlvcVersion);
        ASSERT_TRUE(config);
        config.value().SetSize(320, 180);

        auto result = manager.value().CreateEncoder(config.value());
        ASSERT_TRUE(result);
        encoder = std::move(result.value());

        auto decoderResult = manager.value().CreateDecoder();
        ASSERT_TRUE(decoderResult);
        decoder = std::move(decoderResult.value());
    }

    auto frameBytes = std::vector<std::byte>(320 * 180 * 3 / 2, std::byte{ 128 });
    auto encoded = encoder->Encode(Nv12FrameView{ 320, 180, frameBytes });
    ASSERT_TRUE(encoded) << encoded.error().message();

    auto decoded = decoder->Decode(encoded.value().bitStream);
    ASSERT_TRUE(decoded) << decoded.error().message();
}

TEST_F(ManagerTests, CreateFromBlobsRejectsEmptyInput)
{
    auto manager = MlvcManager::CreateFromBlobs(m_managerParams, {});
    ASSERT_FALSE(manager);
    ASSERT_EQ(manager.error(), make_error_code(Error::invalid_argument));
}

TEST_F(ManagerTests, ReusesOnDemandSessions)
{
    auto params = m_managerParams;
    params.sessionPreloadMode = SessionPreloadMode::NONE;
    const auto manager = std::make_shared<MlvcManagerImpl>(params);
    ASSERT_TRUE(manager->Initialize({}, std::array{ m_mlvcVersion }));
    const auto modelId = manager->GetOptimalModelId(m_mlvcVersion, 320, 180);
    ASSERT_TRUE(modelId) << modelId.error().message();

    auto sessions = std::vector<InferenceSessionPtr>(4);
    {
        auto threads = std::vector<std::thread>{};
        for (auto& session : sessions) {
            threads.emplace_back([&] {
                auto ret = manager->GetInferenceSession(m_mlvcVersion, modelId.value(), ModelPartId::DECODER);
                if (ret) session = ret.value();
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    ASSERT_NE(sessions.front(), nullptr);
    for (const auto& session : sessions) {
        EXPECT_EQ(session, sessions.front());
    }

    const auto again = manager->GetInferenceSession(m_mlvcVersion, modelId.value(), ModelPartId::DECODER);
    ASSERT_TRUE(again) << again.error().message();
    EXPECT_EQ(again.value(), sessions.front());

    const auto encoderSession = manager->GetInferenceSession(m_mlvcVersion, modelId.value(), ModelPartId::ENCODER);
    ASSERT_TRUE(encoderSession) << encoderSession.error().message();
    EXPECT_NE(encoderSession.value(), sessions.front());
}

TEST_F(ManagerTests, CreatesSessionsInBackground)
{
    auto params = m_managerParams;
    params.sessionPreloadMode = SessionPreloadMode::ON_FIRST_USE;
    const auto manager = std::make_shared<MlvcManagerImpl>(params);
    ASSERT_TRUE(manager->Initialize({}, std::array{ m_mlvcVersion }));
    EXPECT_EQ(manager->GetInfo().sessionPreloadMode, SessionPreloadMode::ON_FIRST_USE);

    for (const auto& [width, height] : { std::pair{ 320, 180 }, std::pair{ 640, 360 } }) {
        const auto modelId = manager->GetOptimalModelId(m_mlvcVersion, width, height);
        ASSERT_TRUE(modelId) << modelId.error().message();
        const auto session = manager->GetInferenceSession(m_mlvcVersion, modelId.value(), ModelPartId::DECODER);
        ASSERT_TRUE(session) << session.error().message();
        const auto again = manager->GetInferenceSession(m_mlvcVersion, modelId.value(), ModelPartId::DECODER);
        ASSERT_TRUE(again) << again.error().message();
        EXPECT_EQ(again.value(), session.value());
    }
}

TEST_F(ManagerTests, CreateFromBlobs)
{
    auto info = ManagerInfo{};
    {
        auto directoryManager = MlvcManager::CreateFromDirectory(m_managerParams, {}, std::array{ m_mlvcVersion });
        ASSERT_TRUE(directoryManager);
        info = directoryManager.value().GetInfo();
    }

    auto bundleBytes = ReadCompatibleBundleFixture(info, m_mlvcVersion);
    ASSERT_TRUE(bundleBytes) << bundleBytes.error().message();

    auto manager =
        MlvcManager::CreateFromBlobs(m_managerParams, std::array{ std::span<const std::byte>{ bundleBytes.value() } });
    ASSERT_TRUE(manager) << manager.error().message();
    ASSERT_EQ(manager.value().GetAvailableVersions(), std::vector{ m_mlvcVersion });

    // The manager doesn't need the blobs after it's created
    std::ranges::fill(bundleBytes.value(), std::byte{ 0 });
    auto config = manager.value().GetDefaultEncoderConfig(m_mlvcVersion);
    ASSERT_TRUE(config) << config.error().message();
    config.value().SetSize(320, 180);
    auto encoder = manager.value().CreateEncoder(config.value());
    ASSERT_TRUE(encoder) << encoder.error().message();
    const auto frame = std::vector<std::byte>(320 * 180 * 3 / 2, std::byte{ 128 });
    auto encoded = encoder.value().Encode(Nv12FrameView{ 320, 180, frame });
    EXPECT_TRUE(encoded) << encoded.error().message();
}

TEST_F(ManagerTests, CancelCreateFromDirectory)
{
    auto runCancellationScenario = [&](bool cancelInCallback) {
        const auto tempDir = ScopedTempDir{};
        const auto expired = tempDir.m_path / "models" / "v2_expired";
        ASSERT_TRUE(WriteCacheEntry(expired, std::chrono::days(30)));
        CancelToken cancelToken;

        ManagerParams params = m_managerParams;
        params.cacheDir = tempDir.m_path;
        params.sessionPreloadMode = SessionPreloadMode::AT_INIT;  // Keep initialization on the cancellable path.

        std::mutex progressMutex;
        std::condition_variable progressCv;
        bool firstCallbackReached = false;
        bool waitingForExternalCancel = false;
        bool releaseCallback = false;
        bool initFinished = false;
        std::atomic<int> progressCount{ 0 };
        std::optional<expected<MlvcManager>> createResult;

        auto progressCallback = [&](const libmlvc::InitializeProgressEvent& event) {
            (void)event;

            const int callbackCount = progressCount.fetch_add(1) + 1;
            if (callbackCount != 1) {
                return;
            }

            std::unique_lock<std::mutex> lock(progressMutex);
            firstCallbackReached = true;

            if (cancelInCallback) {
                lock.unlock();
                cancelToken.Cancel();
                progressCv.notify_all();
                return;
            }

            waitingForExternalCancel = true;
            progressCv.notify_all();
            progressCv.wait(lock, [&]() { return releaseCallback; });
        };

        std::thread initThread([&]() {
            createResult = MlvcManager::CreateFromDirectory(params, {}, {}, cancelToken, progressCallback);
            // Wake the waiter even if initialization fails before the first progress callback
            {
                std::lock_guard<std::mutex> lock(progressMutex);
                initFinished = true;
            }
            progressCv.notify_all();
        });

        {
            std::unique_lock<std::mutex> lock(progressMutex);
            progressCv.wait(lock, [&]() { return firstCallbackReached || initFinished; });
            if (firstCallbackReached && !cancelInCallback) {
                progressCv.wait(lock, [&]() { return waitingForExternalCancel; });
                cancelToken.Cancel();
                releaseCallback = true;
            }
        }
        progressCv.notify_all();

        initThread.join();

        ASSERT_GE(progressCount.load(), 1);
        ASSERT_TRUE(createResult);
        ASSERT_FALSE(*createResult);
        ASSERT_EQ(createResult->error(), make_error_code(Error::operation_cancelled));
        EXPECT_TRUE(std::filesystem::exists(expired));
    };

    runCancellationScenario(true);
    runCancellationScenario(false);
}

TEST_F(ManagerTests, CancelDuringBundleInitialization)
{
    for (const std::string_view step :
         { "Reading model bundles", "Preparing model cache for", "Preloading session for" }) {
        SCOPED_TRACE(step);
        CancelToken cancelToken;
        bool stepReached = false;
        auto progressCallback = [&](const libmlvc::InitializeProgressEvent& event) {
            const auto* status = std::get_if<libmlvc::StatusMessage>(&event);
            if (status && status->message.starts_with(step)) {
                stepReached = true;
                cancelToken.Cancel();
            }
        };
        auto params = m_managerParams;
        params.sessionPreloadMode = SessionPreloadMode::AT_INIT;
        auto manager = MlvcManager::CreateFromDirectory(params, {}, {}, cancelToken, progressCallback);
        EXPECT_TRUE(stepReached);
        ASSERT_FALSE(manager);
        EXPECT_EQ(manager.error(), make_error_code(Error::operation_cancelled));
    }
}

// -----------------------------------------------------------------------------
// Version/Capability API tests
// -----------------------------------------------------------------------------

TEST_F(ManagerTests, GetAvailableVersions)
{
    CancelToken cancelToken;

    auto progressCallback = [&](const libmlvc::InitializeProgressEvent& event) {
        if (const auto* status = std::get_if<libmlvc::StatusMessage>(&event)) {
            std::cerr << "Init progress status: " << status->message << '\n';
        } else if (const auto* appRuntimeVersion = std::get_if<libmlvc::WindowsAppRuntimeVersionAvailable>(&event)) {
            std::cerr << "Init progress apprtversion: Windows App Runtime v" << appRuntimeVersion->version << '\n';
        } else if (const auto* epInfo = std::get_if<libmlvc::WindowsAppRuntimeEPInfoAvailable>(&event)) {
            std::cerr << "Init progress apprtepversion: Windows Execution Provider "
                      << OnnxExecutionProviderToString(epInfo->provider) << " v" << epInfo->epPackageVersion << '\n';
        }
    };

    auto manager =
        MlvcManager::CreateFromDirectory(m_managerParams, {}, std::array{ m_mlvcVersion }, cancelToken, progressCallback);
    ASSERT_TRUE(manager);
    auto versions = manager.value().GetAvailableVersions();
    ASSERT_EQ(versions.size(), 1);
    ASSERT_EQ(versions[0], m_mlvcVersion);
}

TEST_F(ManagerTests, GetCapabilities)
{
    auto manager = MlvcManager::CreateFromDirectory(m_managerParams, {}, std::array{ m_mlvcVersion });
    ASSERT_TRUE(manager);

    auto capabilities = manager.value().GetCapabilities(m_mlvcVersion);
    ASSERT_TRUE(capabilities);
    ASSERT_GE(capabilities.value().maxWidth, 640);
    ASSERT_GE(capabilities.value().maxHeight, 360);
    ASSERT_GE(capabilities.value().maxFps, 30.0f);
    ASSERT_GE(capabilities.value().maxNumOfTemporalLayers, 2);
    ASSERT_GE(capabilities.value().maxNumOfLtrFrames, 8);
}

TEST_F(ManagerTests, GetDefaultEncoderConfig)
{
    auto manager = MlvcManager::CreateFromDirectory(m_managerParams, {}, std::array{ m_mlvcVersion });
    ASSERT_TRUE(manager);

    auto config = manager.value().GetDefaultEncoderConfig(m_mlvcVersion);
    ASSERT_TRUE(config) << "Failed to create encoder config: " << config.error().message();

    ASSERT_EQ(config.value().mlvcVersion, m_mlvcVersion);
    ASSERT_EQ(config.value().width, 640);
    ASSERT_EQ(config.value().height, 360);
    ASSERT_GT(config.value().iframePeriod, 0);
    ASSERT_EQ(config.value().numTemporalLayers, 1);
    ASSERT_EQ(config.value().ltrMode, LtrMode::INTERNAL);
    ASSERT_EQ(config.value().ltrNumSlots, 4);
}
