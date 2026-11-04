// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/thread_pool.hpp"
#include "libmlvc/inference/interface.hpp"
#include "libmlvc/schema/bundle.hpp"
#include "libmlvc/schema/pmf.hpp"

#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace libmlvc {

struct ScaleDecoderData;

expected<std::vector<std::vector<std::byte>>> ReadBundleBlobs(const std::filesystem::path& bundlesDir,
                                                              std::span<const MlvcVersion> versions,
                                                              const InferenceEngineInfo& engineInfo);

class ModelBundle {
public:
    static expected<std::unique_ptr<ModelBundle>> Create(IInferenceEngine& engine, ThreadPool* sessionCreationPool,
                                                         std::span<const std::byte> bundleBlob, bool enableModelCache,
                                                         SessionPreloadMode sessionPreloadMode, CancelToken cancelToken,
                                                         const InitializeProgressCallback& progressCallback);
    ModelBundle(const ModelBundle&) = delete;
    ModelBundle& operator=(const ModelBundle&) = delete;
    ModelBundle(ModelBundle&&) = delete;
    ModelBundle& operator=(ModelBundle&&) = delete;

    expected<void> ValidateModelCache() const;

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

    // Queues the creation of every session not queued yet; failures aren't retried
    void PreloadSessionsInBackground() const;

private:
    using BundleFiles = std::unordered_map<std::string, std::span<const std::byte>>;

    struct SessionSlot {
        std::mutex mutex;  // Guards `session`
        InferenceSessionPtr session;
        std::shared_ptr<ThreadPool::Task> task;  // ON_FIRST_USE: the pool task that creates `session`
    };

    struct ModelSource {
        std::span<const std::byte> modelData;
        std::map<std::string, std::span<const std::byte>> weightsData;
        std::string cacheKey;
    };

    std::vector<std::byte> m_bundleBlob;
    IInferenceEngine& m_engine;
    ThreadPool* m_sessionCreationPool{};
    bool m_enableModelCache{};
    SessionPreloadMode m_sessionPreloadMode{};
    BundleManifest m_manifest{};
    BundleFiles m_bundleFiles;
    std::unordered_map<std::string, std::shared_ptr<const GaussianCoderPmf>> m_gaussianPmfs;
    std::unordered_map<std::string, std::shared_ptr<const BitEstimatorPmf>> m_bitEstimatorPmfs;
    std::unordered_map<std::string, std::shared_ptr<const ScaleDecoderData>> m_scaleDecoderData;
    mutable std::map<std::pair<std::string, ModelPartId>, SessionSlot> m_inferenceSessions;
    mutable std::mutex m_inferenceSessionsMutex;  // Guards the map and the slots' tasks

    ModelBundle(IInferenceEngine& engine, ThreadPool* sessionCreationPool, bool enableModelCache,
                SessionPreloadMode sessionPreloadMode);
    expected<void> Initialize(std::span<const std::byte> bundleBlob, CancelToken cancelToken,
                              const InitializeProgressCallback& progressCallback);
    expected<void> LoadScaleDecoderData(const BundleFiles& files);
    expected<void> LoadPmfs(const BundleFiles& files);
    expected<bool> PrepareModelCache(const BundleFiles& files, CancelToken cancelToken,
                                     const InitializeProgressCallback& progressCallback);
    expected<void> PreloadSessions(const BundleFiles& files, CancelToken cancelToken,
                                   const InitializeProgressCallback& progressCallback);
    void QueueSessionCreations(const std::optional<std::pair<std::string, ModelPartId>>& requested) const;
    expected<ModelSource> GetModelSource(const std::string& modelId, const ModelPartManifest& part,
                                         const BundleFiles& files) const;
    expected<InferenceSessionPtr> CreateInferenceSession(const std::string& modelId, ModelPartId partId,
                                                         const BundleFiles& files) const;
};

}  // namespace libmlvc
