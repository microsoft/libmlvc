// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/codec/bundle.hpp"
#include "libmlvc/inference/interface.hpp"
#include "libmlvc/schema/bundle.hpp"

#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace libmlvc {

class MlvcEncoderImpl;
class MlvcDecoderImpl;

class MlvcManagerImpl : public std::enable_shared_from_this<MlvcManagerImpl> {
public:
    static std::filesystem::path GetDefaultModelBundlesDir();
    static MlvcVersion GetDefaultModelVersion();

    MlvcManagerImpl(const ManagerParams& params);
    MlvcManagerImpl(const MlvcManagerImpl&) = delete;
    MlvcManagerImpl& operator=(const MlvcManagerImpl&) = delete;
    MlvcManagerImpl(MlvcManagerImpl&&) = delete;
    MlvcManagerImpl& operator=(MlvcManagerImpl&&) = delete;

    // Initialization (not thread safe, should be called once before using other APIs)
    expected<void> Initialize(const std::filesystem::path& bundlesDir = {}, std::span<const MlvcVersion> versions = {},
                              CancelToken cancelToken = {}, InitializeProgressCallback progressCallback = nullptr);
    expected<void> Initialize(std::span<const std::span<const std::byte>> bundleBlobs, CancelToken cancelToken = {},
                              InitializeProgressCallback progressCallback = nullptr);

    // Available versions, capabilities and default configs
    std::vector<MlvcVersion> GetAvailableVersions() const;
    expected<Capabilities> GetCapabilities(const MlvcVersion mlvcVersion) const;
    expected<EncoderConfig> GetDefaultEncoderConfig(const MlvcVersion mlvcVersion) const;

    // Create encoder/decoder
    expected<std::unique_ptr<MlvcEncoderImpl>> CreateEncoder(const EncoderConfig& config) const;
    expected<std::unique_ptr<MlvcDecoderImpl>> CreateDecoder() const;

    // Info
    const ManagerInfo& GetInfo() const { return m_info; }

    // Model queries (forwarded to ModelBundle class)
    bool IsVersionAvailable(const MlvcVersion mlvcVersion) const;
    expected<std::string> GetOptimalModelId(const MlvcVersion mlvcVersion, const int width, const int height) const;
    expected<ModelManifest> GetModelManifest(const MlvcVersion mlvcVersion, const std::string& modelId) const;
    expected<std::shared_ptr<const ModelMetadata>> GetModelMetadata(const MlvcVersion mlvcVersion,
                                                                    const std::string& modelId) const;
    expected<std::shared_ptr<const ScaleDecoderData>> GetScaleDecoderData(const MlvcVersion mlvcVersion,
                                                                          const std::string& modelId) const;
    expected<std::shared_ptr<const GaussianCoderPmf>> GetGaussianPmf(const MlvcVersion mlvcVersion,
                                                                     const std::string& modelId) const;
    expected<std::shared_ptr<const BitEstimatorPmf>> GetBitEstimatorPmf(const MlvcVersion mlvcVersion,
                                                                        const std::string& modelId) const;
    expected<InferenceSessionPtr> GetInferenceSession(const MlvcVersion mlvcVersion, const std::string& modelId,
                                                      const ModelPartId partId) const;

private:
    ManagerParams m_params{};
    ManagerInfo m_info{};
    std::unique_ptr<IInferenceEngine> m_inferenceEngine;
    std::map<MlvcVersion, ModelBundle> m_bundles;
    std::atomic<bool> m_initialized{ false };
    mutable std::atomic<int> m_encoderCounter{ 0 };
    mutable std::atomic<int> m_decoderCounter{ 0 };

    expected<void> InitInferenceEngine(CancelToken cancelToken, const InitializeProgressCallback& progressCallback);
    expected<void> LoadBundles(std::vector<std::vector<std::byte>>&& bundleBlobs, CancelToken cancelToken,
                               const InitializeProgressCallback& progressCallback);
    expected<void> LoadBundles(std::span<const std::span<const std::byte>> bundleBlobs, CancelToken cancelToken,
                               const InitializeProgressCallback& progressCallback);
    expected<void> CheckInitialized(MlvcVersion mlvcVersion) const;
};

}  // namespace libmlvc
