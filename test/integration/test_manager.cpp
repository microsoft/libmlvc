// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <type_traits>
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
        // To speed up tests, disable session caching
        m_managerParams.computeUnit = GetTestConfig().computeUnit;
        m_managerParams.enableSessionCaching = false;
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
}

TEST_F(ManagerTests, CancelCreateFromDirectory)
{
    auto runCancellationScenario = [&](bool cancelInCallback) {
        CancelToken cancelToken;

        ManagerParams params = m_managerParams;
        params.enableSessionCaching = true;  // Keep initialization on the cancellable path.

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
        ASSERT_FALSE(*createResult);  // Creation should fail due to cancellation.
        ASSERT_EQ(createResult->error(), make_error_code(Error::operation_cancelled));
    };

    runCancellationScenario(true);
    runCancellationScenario(false);
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
            std::cerr << "Init progress apprtepversion: Windows Execution Provider v" << epInfo->version << '\n';
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
