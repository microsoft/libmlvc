// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/inference/interface.hpp"
#include "libmlvc/inference/winml/bootstrap.hpp"
#include "libmlvc/inference/winml/factory.hpp"
#include "libmlvc/inference/winml/ort_helpers.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>

namespace libmlvc_winml {

using namespace libmlvc;

class WinMlInferenceEngine : public IInferenceEngine {
public:
    static expected<std::unique_ptr<WinMlInferenceEngine>>
    Create(const WinMlEngineParams& params, CancelToken cancelToken = {},
           const InitializeProgressCallback& progressCallback = nullptr);
    ~WinMlInferenceEngine();
    expected<InferenceSessionPtr> CreateSession(std::string_view name, std::span<const std::byte> modelData,
                                                const std::map<std::string, std::span<const std::byte>>& weightsData,
                                                std::string_view cacheKey,
                                                const std::optional<std::string>& functionName = std::nullopt) override;
    const InferenceEngineInfo& GetInfo() const override { return m_info; }

private:
    WinMlInferenceEngine(const WinMlEngineParams& params);
    expected<void> Initialize(CancelToken cancelToken, const InitializeProgressCallback& progressCallback);

    // Common helpers for session creation
    expected<OrtSessionOptionsPtr> CreateConfiguredSessionOptions();
    expected<InferenceSessionPtr> CreateSessionFromArray(std::string_view name, std::span<const std::byte> modelData,
                                                         const std::map<std::string, std::span<const std::byte>>& weightsData);

    // EP context caching helpers
    bool SupportsEpContextCaching() const;
    expected<InferenceSessionPtr> CreateSessionFromEpContext(std::string_view name, std::span<const std::byte> modelData,
                                                             const std::map<std::string, std::span<const std::byte>>& weightsData,
                                                             std::string_view cacheKey);
    expected<void> CompileModelToEpContext(std::string_view name, std::span<const std::byte> modelData,
                                           const std::map<std::string, std::span<const std::byte>>& weightsData,
                                           const std::filesystem::path& modelCacheDir);
    expected<InferenceSessionPtr> LoadSessionFromCache(std::string_view name, const std::filesystem::path& modelCacheDir);

    const WinMlEngineParams m_params;
    InferenceEngineInfo m_info;
    std::string m_envHash;
    // m_ortEnv must be released before m_hOnnxRuntime
    HModulePtr m_hOnnxRuntime;
    WinMlApi m_winMlApi;
    const OrtApi* m_ortApi;
    OrtEnvPtr m_ortEnv;
    const OrtEpDevice* m_epDevice = nullptr;
    OrtHardwareDeviceType m_hwDeviceType = OrtHardwareDeviceType_CPU;
    const OrtMemoryInfo* m_epMemoryInfo = nullptr;  // EP memory info for zero-copy allocator
    OrtMemoryInfoPtr m_epMemoryInfoStorage;         // Owns m_epMemoryInfo when created by name
    std::wstring m_providerLibraryPath;
    mutable std::mutex m_mutex;
};

}  // namespace libmlvc_winml
