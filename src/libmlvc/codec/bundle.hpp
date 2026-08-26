// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/inference/interface.hpp"
#include "libmlvc/schema/bundle.hpp"
#include "libmlvc/schema/pmf.hpp"

#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace libmlvc {

struct ScaleDecoderData;

expected<std::vector<std::vector<std::byte>>> LoadModelBundles(const std::filesystem::path& bundlesDir,
                                                               std::span<const MlvcVersion> versions,
                                                               const InferenceEngineInfo& engineInfo);

class ModelBundle {
public:
    ModelBundle(std::span<const std::byte> bundleBlob, IInferenceEngine& engine, bool enableModelCache,
                bool enableSessionCaching);
    ModelBundle(const ModelBundle&) = delete;
    ModelBundle& operator=(const ModelBundle&) = delete;
    ModelBundle(ModelBundle&&) = default;
    ModelBundle& operator=(ModelBundle&&) = delete;
    expected<void> Initialize(CancelToken cancelToken, const InitializeProgressCallback& progressCallback);

    // Manifest, capabilities and encoder config
    const BundleManifest& GetManifest() const { return m_manifest; }
    Capabilities GetCapabilities() const;
    expected<EncoderConfig> GetDefaultEncoderConfig() const;

    // Accessors for metadata and model sessions corresponding to a given model ID
    expected<std::string> GetOptimalModelId(const int width, const int height) const;
    expected<ModelManifest> GetModelManifest(const std::string& modelId) const;
    expected<std::shared_ptr<const ModelMetadata>> GetModelMetadata(const std::string& modelId) const;
    expected<std::shared_ptr<const ScaleDecoderData>> GetScaleDecoderData(const std::string& modelId) const;
    expected<std::shared_ptr<const GaussianCoderPmf>> GetGaussianPmf(const std::string& modelId) const;
    expected<std::shared_ptr<const BitEstimatorPmf>> GetBitEstimatorPmf(const std::string& modelId) const;
    expected<InferenceSessionPtr> GetInferenceSession(const std::string& modelId, ModelPartId partId) const;

private:
    std::vector<std::byte> m_bundleBlob;
    IInferenceEngine& m_engine;
    bool m_enableModelCache{};
    bool m_enableSessionCaching{};
    BundleManifest m_manifest{};
    std::unordered_map<std::string, std::span<const std::byte>> m_bundleFiles;
    std::unordered_map<std::string, std::shared_ptr<const GaussianCoderPmf>> m_gaussianPmfs;
    std::unordered_map<std::string, std::shared_ptr<const BitEstimatorPmf>> m_bitEstimatorPmfs;
    std::unordered_map<std::string, std::shared_ptr<const ScaleDecoderData>> m_scaleDecoderData;
    std::map<std::pair<std::string, ModelPartId>, InferenceSessionPtr> m_inferenceSessions;

    expected<void> LoadScaleDecoderData();
    expected<void> LoadPmfs();
    expected<void> CacheSessions(CancelToken cancelToken, const InitializeProgressCallback& progressCallback);
    expected<InferenceSessionPtr> CreateInferenceSession(const std::string& modelId, ModelPartId partId) const;
};

}  // namespace libmlvc
