// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/bundle.hpp"
#include "libmlvc/codec/core/scale_decoder.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/tar.hpp"

#include <libmlvc/error_codes.hpp>

#include <fstream>
#include <limits>
#include <ranges>

namespace libmlvc {

namespace {

expected<std::vector<std::byte>> ReadFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (file.fail()) {
        MLVC_LOG_ERROR("Failed to open file: %s", path.filename().string().c_str());
        return make_error_code(Error::io_error);
    }

    file.seekg(0, std::ios::end);
    const std::size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> data(fileSize);
    const std::size_t bytesRead = file.read(reinterpret_cast<char*>(data.data()), fileSize).gcount();
    if (bytesRead != fileSize) {
        MLVC_LOG_ERROR("Failed to read data, got %zu bytes instead of %zu", bytesRead, fileSize);
        return make_error_code(Error::io_error);
    }
    return data;
}

expected<BundleManifest> PeekManifest(const std::filesystem::path& bundlePath)
{
    auto manifestData = ReadTarEntry(bundlePath, "bundle_manifest.json");
    if (!manifestData) {
        MLVC_LOG_ERROR("Failed to read bundle_manifest.json for peeking manifest: %s",
                       manifestData.error().message().c_str());
        return manifestData.error();
    }

    auto res = BundleManifest::FromJsonBuffer(manifestData.value());
    if (!res) {
        MLVC_LOG_ERROR("Failed to parse bundle_manifest.json for peeking manifest: %s", res.error().message().c_str());
        return res.error();
    }
    return res.value();
}

bool IsBundleCompatible(const BundleManifest& manifest, const InferenceEngineInfo& engineInfo)
{
    if (engineInfo.inferenceBackend == InferenceBackend::COREML) {
        return manifest.modelType == ModelType::COREML;
    } else if (engineInfo.inferenceBackend == InferenceBackend::WINDOWSML) {
        if (manifest.modelType != ModelType::ONNX) {
            // For WindowsML, currently only support ONNX models
            return false;
        }
        if (engineInfo.onnxExecutionProvider == OnnxExecutionProvider::OPENVINO) {
            // OpenVINO requires specialized bundle targeting Intel
            return manifest.targetDevice == TargetDevice::INTEL;
        }
        if (engineInfo.onnxExecutionProvider == OnnxExecutionProvider::QNN) {
            // QNN requires specialized bundle targeting Qualcomm
            return manifest.targetDevice == TargetDevice::QUALCOMM;
        }
    }
    return true;  // For other cases, assume compatible
}

int GetBundlePriority(const TargetDevice targetDevice)
{
    switch (targetDevice) {
    case TargetDevice::GENERIC:
        return 3;
    case TargetDevice::QUALCOMM:
        return 2;
    case TargetDevice::INTEL:
        return 1;
    case TargetDevice::APPLE:
        return 0;
    }
    return 0;
}

expected<void> WarmUpSession(const InferenceSessionPtr& session)
{
    auto inputNames = session->GetInputNames();
    if (!inputNames) return inputNames.error();

    auto outputNames = session->GetOutputNames();
    if (!outputNames) return outputNames.error();

    std::vector<InferenceTensorPtr> inputs;
    for (const auto& inputName : inputNames.value()) {
        auto tensor = session->CreateTensor(TensorIoType::INPUT, inputName);
        if (!tensor) return tensor.error();
        inputs.push_back(std::move(tensor.value()));
    }

    std::vector<InferenceTensorPtr> outputs;
    for (const auto& outputName : outputNames.value()) {
        auto tensor = session->CreateTensor(TensorIoType::OUTPUT, outputName);
        if (!tensor) return tensor.error();
        outputs.push_back(std::move(tensor.value()));
    }

    if (auto ret = session->Run(inputs, outputs); !ret) {
        return ret.error();
    }
    return {};
}

class BundleBlob {
public:
    static expected<BundleBlob> FromPath(const std::filesystem::path& path)
    {
        auto manifest = PeekManifest(path);
        if (!manifest) {
            MLVC_LOG_ERROR("Failed to peek manifest from bundle file: %s, error: %s", path.filename().string().c_str(),
                           manifest.error().message().c_str());
            return manifest.error();
        }

        return BundleBlob{ std::move(manifest.value()), path };
    }

    BundleBlob(BundleManifest manifest, std::filesystem::path path)
        : m_manifest(std::move(manifest)), m_path(std::move(path))
    {
    }
    const BundleManifest& GetManifest() const { return m_manifest; }
    expected<std::vector<std::byte>> ReadData() const { return ReadFile(m_path); }

private:
    BundleManifest m_manifest;
    std::filesystem::path m_path;
};

}  // namespace

// ----------------------------------------------------------------
// Functions
// ----------------------------------------------------------------

expected<std::vector<std::vector<std::byte>>> LoadModelBundles(const std::filesystem::path& bundlesDir,
                                                               std::span<const MlvcVersion> versions,
                                                               const InferenceEngineInfo& engineInfo)
{
    MLVC_LOG_INFO("Discovering model bundles...");

    std::error_code ec;
    if (!std::filesystem::is_directory(bundlesDir, ec)) {
        MLVC_LOG_ERROR("Model bundles directory does not exist or is not a directory: %s",
                       ec ? ec.message().c_str() : "not a directory");
        return make_error_code(Error::invalid_argument);
    }

    std::map<MlvcVersion, BundleBlob> bundlesMap;
    for (const auto& entry : std::filesystem::directory_iterator(bundlesDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".tar") {
            MLVC_LOG_DEBUG("Skipping non-regular file or non-tar file: %s", entry.path().filename().string().c_str());
            continue;
        }
        auto bundle = BundleBlob::FromPath(entry.path());
        if (!bundle) {
            MLVC_LOG_WARN("Failed to read bundle blob from %s, skipping", entry.path().filename().string().c_str());
            continue;
        }

        // Version filtering: Only keep selected versions
        const auto& manifest = bundle->GetManifest();
        if (std::ranges::find(versions, manifest.mlvcVersion) == versions.end()) {
            MLVC_LOG_DEBUG("Bundle %s version %s not in requested versions list, skipping",
                           entry.path().filename().string().c_str(), manifest.mlvcVersion.ToString().c_str());
            continue;
        }

        // Filter out bundles that are not compatible with the current platform
        if (!IsBundleCompatible(manifest, engineInfo)) {
            MLVC_LOG_DEBUG("Bundle %s is not compatible with this inference engine, skipping",
                           entry.path().filename().string().c_str());
            continue;
        }

        // If there are multiple bundles with the same version, keep the one with the largest priority
        if (bundlesMap.contains(manifest.mlvcVersion)) {
            const auto& existingBundle = bundlesMap.at(manifest.mlvcVersion);
            if (GetBundlePriority(existingBundle.GetManifest().targetDevice) >= GetBundlePriority(manifest.targetDevice)) {
                MLVC_LOG_DEBUG("Bundle %s has same version as previously found bundle but smaller priority, skipping",
                               entry.path().filename().string().c_str());
                continue;
            }
        }

        // Accept the bundle
        MLVC_LOG_INFO("Found model bundle: %s", entry.path().filename().string().c_str());
        bundlesMap.insert_or_assign(manifest.mlvcVersion, std::move(bundle.value()));
    }

    // Check that all requested versions are found
    for (const auto& version : versions) {
        if (!bundlesMap.contains(version)) {
            MLVC_LOG_ERROR("Requested bundle version %s not found", version.ToString().c_str());
            return make_error_code(Error::general_failure);
        }
    }

    std::vector<std::vector<std::byte>> res;
    res.reserve(bundlesMap.size());
    for (const auto& bundleBlob : bundlesMap | std::views::values) {
        auto data = bundleBlob.ReadData();
        if (!data) return data.error();
        res.push_back(std::move(data.value()));
    }
    return res;
}

// ----------------------------------------------------------------
// ModelBundle
// ----------------------------------------------------------------

ModelBundle::ModelBundle(IInferenceEngine& engine, bool enableModelCache, bool enableSessionPreloading)
    : m_engine(engine), m_enableModelCache(enableModelCache), m_enableSessionPreloading(enableSessionPreloading)
{
}

expected<void> ModelBundle::Initialize(std::span<const std::byte> bundleBlob, CancelToken cancelToken,
                                       const InitializeProgressCallback& progressCallback)
{
    auto checkpoint = [&cancelToken, &progressCallback](const std::string& status) -> expected<void> {
        if (progressCallback) progressCallback(StatusMessage{ status });
        if (cancelToken.IsCancelled()) {
            MLVC_LOG_INFO("Bundle initialization cancelled");
            return make_error_code(Error::operation_cancelled);
        }
        return {};
    };

    // Extract bundle files
    BundleFiles bundleFiles;
    {
        if (auto ret = checkpoint("Extracting bundle files"); !ret) return ret.error();
        auto entries = ExtractTar(bundleBlob);
        if (!entries) {
            MLVC_LOG_ERROR("Failed to extract tar: %s", entries.error().message().c_str());
            return entries.error();
        }
        bundleFiles = std::move(entries.value());
    }

    // Parse manifest
    {
        if (auto ret = checkpoint("Parsing model manifest"); !ret) return ret.error();
        const auto manifestIt = bundleFiles.find("bundle_manifest.json");
        if (manifestIt == bundleFiles.end()) {
            MLVC_LOG_ERROR("Manifest file not found in model bundle");
            return make_error_code(Error::io_error);
        }
        auto manifest = BundleManifest::FromJsonBuffer(manifestIt->second);
        if (!manifest) {
            MLVC_LOG_ERROR("Failed to parse bundle manifest: %s", manifest.error().message().c_str());
            return manifest.error();
        }
        m_manifest = std::move(manifest.value());
    }

    // Check compatibility
    if (!IsBundleCompatible(m_manifest, m_engine.GetInfo())) {
        MLVC_LOG_ERROR("Model bundle is not compatible with the inference engine: bundle(%s, %s, %s), engine(%s, %s)",
                       m_manifest.bundleName.c_str(), ModelTypeToString(m_manifest.modelType),
                       TargetDeviceToString(m_manifest.targetDevice),
                       InferenceBackendToString(m_engine.GetInfo().inferenceBackend),
                       OnnxExecutionProviderToString(m_engine.GetInfo().onnxExecutionProvider));
        return make_error_code(Error::invalid_argument);
    }

    // Load scale decoder data
    if (auto ret = checkpoint("Loading scale decoder data"); !ret) return ret.error();
    if (auto ret = LoadScaleDecoderData(bundleFiles); !ret) {
        MLVC_LOG_ERROR("Failed to load scale decoder data for bundle %s %s: %s", m_manifest.bundleName.c_str(),
                       m_manifest.mlvcVersion.ToString().c_str(), ret.error().message().c_str());
        return ret.error();
    }

    // Load PMFs
    if (auto ret = checkpoint("Loading PMFs"); !ret) return ret.error();
    if (auto ret = LoadPmfs(bundleFiles); !ret) {
        MLVC_LOG_ERROR("Failed to load PMFs for bundle %s %s: %s", m_manifest.bundleName.c_str(),
                       m_manifest.mlvcVersion.ToString().c_str(), ret.error().message().c_str());
        return ret.error();
    }

    // Compile the missing models into the model cache
    bool modelsCached = false;
    if (m_enableModelCache) {
        if (auto ret = checkpoint("Preparing model cache"); !ret) return ret.error();
        auto cached = PrepareModelCache(bundleFiles, cancelToken, progressCallback);
        if (!cached) {
            MLVC_LOG_ERROR("Failed to prepare model cache for bundle %s %s: %s", m_manifest.bundleName.c_str(),
                           m_manifest.mlvcVersion.ToString().c_str(), cached.error().message().c_str());
            return cached.error();
        }
        modelsCached = cached.value();
    }

    // Preload sessions if enabled
    if (m_enableSessionPreloading) {
        if (auto ret = checkpoint("Preloading inference sessions"); !ret) return ret.error();
        if (auto ret = PreloadSessions(bundleFiles, cancelToken, progressCallback); !ret) {
            MLVC_LOG_ERROR("Failed to preload sessions for bundle %s %s: %s", m_manifest.bundleName.c_str(),
                           m_manifest.mlvcVersion.ToString().c_str(), ret.error().message().c_str());
            return ret.error();
        }
    }

    // Keep a copy of the bundle data if sessions are created on demand from uncached models
    if (!m_enableSessionPreloading && !modelsCached) {
        MLVC_LOG_INFO("Keeping a copy of the bundle data (%.1lf MB)",
                      static_cast<double>(bundleBlob.size()) / (1024.0 * 1024.0));
        m_bundleBlob.assign(bundleBlob.begin(), bundleBlob.end());
        for (const auto& [path, data] : bundleFiles) {
            const auto offset = static_cast<size_t>(data.data() - bundleBlob.data());
            m_bundleFiles[path] = std::span<const std::byte>(m_bundleBlob).subspan(offset, data.size());
        }
    }

    MLVC_LOG_INFO("Loaded bundle: %s, %s, %s, %s, ts: %s, registry: %zu, models: %zu",
                  m_manifest.mlvcVersion.ToString().c_str(), m_manifest.bundleName.c_str(),
                  ModelTypeToString(m_manifest.modelType), TargetDeviceToString(m_manifest.targetDevice),
                  m_manifest.timestamp.c_str(), m_manifest.modelRegistry.size(), m_manifest.modelManifests.size());

    if (progressCallback) {
        progressCallback(StatusMessage{ "Bundle loaded successfully" });
    }
    return {};
}

Capabilities ModelBundle::GetCapabilities() const
{
    int maxWidth = 0;
    int maxHeight = 0;
    for (const auto& modelMetadata : m_manifest.modelMetadata | std::views::values) {
        if (std::tie(maxWidth, maxHeight) < std::tie(modelMetadata->modelWidth, modelMetadata->modelHeight)) {
            maxWidth = modelMetadata->modelWidth;
            maxHeight = modelMetadata->modelHeight;
        }
    }

    return Capabilities{ .maxWidth = maxWidth,
                         .maxHeight = maxHeight,
                         .maxFps = 120.0f,
                         .maxNumOfTemporalLayers = MAX_TEMPORAL_LAYERS,
                         .maxNumOfLtrFrames = MAX_LTR_SLOTS };
}

expected<EncoderConfig> ModelBundle::GetDefaultEncoderConfig() const
{
    static constexpr int defaultWidth = 640;
    static constexpr int defaultHeight = 360;

    const auto modelId = GetOptimalModelId(defaultWidth, defaultHeight);
    if (!modelId) {
        return modelId.error();
    }
    const auto modelMetadata = GetModelMetadata(modelId.value());
    if (!modelMetadata) {
        return modelMetadata.error();
    }

    const int iframePeriod = modelMetadata.value()->iframePeriod.value_or(1024);
    const int numTemporalLayers = 1;
    const LtrMode ltrMode = LtrMode::INTERNAL;
    const int ltrStartIdx = modelMetadata.value()->ltrStartIdx.value_or(0);
    int ltrPeriod = modelMetadata.value()->ltrPeriod.value_or(0);
    const int ltrNumSlots = 4;
    int ltrRecoveryPeriod = ltrPeriod > 0 ? 1 : 0;

    static constexpr int targetLtrPeriod = 16;
    if (ltrPeriod > targetLtrPeriod && ltrRecoveryPeriod > 0 && ltrNumSlots >= 2) {
        ltrRecoveryPeriod = ltrPeriod / targetLtrPeriod;
        ltrPeriod = targetLtrPeriod;
    }

    return EncoderConfig{ .mlvcVersion = m_manifest.mlvcVersion,
                          .width = defaultWidth,
                          .height = defaultHeight,
                          .iframePeriod = iframePeriod,
                          .numTemporalLayers = numTemporalLayers,
                          .ltrMode = ltrMode,
                          .ltrStartIdx = ltrStartIdx,
                          .ltrPeriod = ltrPeriod,
                          .ltrNumSlots = ltrNumSlots,
                          .ltrRecoveryPeriod = ltrRecoveryPeriod };
}

expected<std::string> ModelBundle::GetOptimalModelId(const int width, const int height) const
{
    std::string bestModelId;
    float bestComputeOverhead = std::numeric_limits<float>::max();
    for (const auto& [modelId, modelMetadata] : m_manifest.modelMetadata) {
        const int modelWidth = modelMetadata->modelWidth;
        const int modelHeight = modelMetadata->modelHeight;
        if (modelId != std::to_string(modelWidth) + "x" + std::to_string(modelHeight)) {
            MLVC_LOG_ERROR("Model ID %s does not match dimensions %dx%d", modelId.c_str(), modelWidth, modelHeight);
            return make_error_code(Error::invalid_argument);
        }

        const bool fitsNormally = width <= modelWidth && height <= modelHeight;
        const bool fitsRotated = height <= modelWidth && width <= modelHeight;
        if (!(fitsNormally || fitsRotated)) {
            continue;
        }

        float computeOverhead = static_cast<float>(modelWidth * modelHeight) / (width * height);
        if (!fitsNormally) {
            computeOverhead += 0.1f;
        }

        if (computeOverhead < bestComputeOverhead) {
            bestModelId = modelId;
            bestComputeOverhead = computeOverhead;
        }
    }

    if (bestModelId.empty()) {
        MLVC_LOG_ERROR("No suitable model found for input size: %dx%d", width, height);
        return make_error_code(Error::invalid_argument);
    }
    return bestModelId;
}

expected<ModelManifest> ModelBundle::GetModelManifest(const std::string& modelId) const
{
    const auto it = m_manifest.modelManifests.find(modelId);
    if (it == m_manifest.modelManifests.end()) {
        MLVC_LOG_ERROR("Model %s manifest not found in the bundle manifest", modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }
    return it->second;
}

expected<std::shared_ptr<const ModelMetadata>> ModelBundle::GetModelMetadata(const std::string& modelId) const
{
    const auto it = m_manifest.modelMetadata.find(modelId);
    if (it == m_manifest.modelMetadata.end()) {
        MLVC_LOG_ERROR("Model %s metadata not found in the bundle manifest", modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }
    return it->second;
}

expected<std::shared_ptr<const ScaleDecoderData>> ModelBundle::GetScaleDecoderData(const std::string& modelId) const
{
    const auto it = m_scaleDecoderData.find(modelId);
    if (it == m_scaleDecoderData.end()) {
        MLVC_LOG_ERROR("Scale decoder data for model %s not found", modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }
    return it->second;
}

expected<std::shared_ptr<const GaussianCoderPmf>> ModelBundle::GetGaussianPmf(const std::string& modelId) const
{
    const auto modelIt = m_manifest.modelManifests.find(modelId);
    if (modelIt == m_manifest.modelManifests.end()) {
        MLVC_LOG_ERROR("Model %s not found in manifest", modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }

    const auto pmfIt = m_gaussianPmfs.find(modelIt->second.gaussianPmfPath);
    if (pmfIt == m_gaussianPmfs.end()) {
        MLVC_LOG_ERROR("Gaussian PMF for model %s not found", modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }
    return pmfIt->second;
}

expected<std::shared_ptr<const BitEstimatorPmf>> ModelBundle::GetBitEstimatorPmf(const std::string& modelId) const
{
    const auto modelIt = m_manifest.modelManifests.find(modelId);
    if (modelIt == m_manifest.modelManifests.end()) {
        MLVC_LOG_ERROR("Model %s not found in manifest", modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }

    const auto pmfIt = m_bitEstimatorPmfs.find(modelIt->second.bitEstimatorPmfPath);
    if (pmfIt == m_bitEstimatorPmfs.end()) {
        MLVC_LOG_ERROR("Bit estimator PMF for model %s not found", modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }
    return pmfIt->second;
}

expected<InferenceSessionPtr> ModelBundle::GetInferenceSession(const std::string& modelId, const ModelPartId partId) const
{
    MLVC_LOG_DEBUG("GetInferenceSession: version=%s, modelId=%s, partId=%s", m_manifest.mlvcVersion.ToString().c_str(),
                   modelId.c_str(), ModelPartIdToString(partId));

    auto sessionIt = m_inferenceSessions.find({ modelId, partId });
    if (sessionIt != m_inferenceSessions.end()) {
        MLVC_LOG_DEBUG("Returning previously created inference session for model %s", modelId.c_str());
        return sessionIt->second;
    }

    return CreateInferenceSession(modelId, partId, m_bundleFiles);
}

expected<void> ModelBundle::LoadScaleDecoderData(const BundleFiles& files)
{
    for (const auto& [modelId, modelManifest] : m_manifest.modelManifests) {
        auto data = std::make_shared<ScaleDecoderData>();
        m_scaleDecoderData[modelId] = std::move(data);
    }
    return {};
}

expected<void> ModelBundle::LoadPmfs(const BundleFiles& files)
{
    for (const auto& [modelId, modelManifest] : m_manifest.modelManifests) {
        // Gaussian PMFs
        {
            const auto& pmfPath = modelManifest.gaussianPmfPath;
            if (m_gaussianPmfs.find(pmfPath) == m_gaussianPmfs.end()) {
                const auto it = files.find(pmfPath);
                if (it == files.end()) {
                    MLVC_LOG_ERROR("Gaussian PMF file %s not found in bundle for model %s", pmfPath.c_str(),
                                   modelId.c_str());
                    return make_error_code(Error::io_error);
                }

                auto pmf = GaussianCoderPmf::FromJsonBuffer(it->second);
                if (!pmf) {
                    MLVC_LOG_ERROR("Failed to parse Gaussian PMF from file %s: %s", pmfPath.c_str(),
                                   pmf.error().message().c_str());
                    return pmf.error();
                }
                if (!pmf.value().indexSpace) {
                    MLVC_LOG_ERROR("Gaussian PMF %s is not index-space; only index-space scales are supported",
                                   pmfPath.c_str());
                    return make_error_code(Error::invalid_argument);
                }
                m_gaussianPmfs[pmfPath] = std::make_shared<GaussianCoderPmf>(std::move(pmf.value()));
            }
        }
        // Bit estimator PMFs
        {
            const auto& pmfPath = modelManifest.bitEstimatorPmfPath;
            if (m_bitEstimatorPmfs.find(pmfPath) == m_bitEstimatorPmfs.end()) {
                const auto it = files.find(pmfPath);
                if (it == files.end()) {
                    MLVC_LOG_ERROR("Bit Estimator PMF file %s not found in bundle for model %s", pmfPath.c_str(),
                                   modelId.c_str());
                    return make_error_code(Error::io_error);
                }

                auto pmf = BitEstimatorPmf::FromJsonBuffer(it->second);
                if (!pmf) {
                    MLVC_LOG_ERROR("Failed to parse Bit Estimator PMF from file %s: %s", pmfPath.c_str(),
                                   pmf.error().message().c_str());
                    return pmf.error();
                }
                m_bitEstimatorPmfs[pmfPath] = std::make_shared<BitEstimatorPmf>(std::move(pmf.value()));
            }
        }
    }
    return {};
}

expected<bool> ModelBundle::PrepareModelCache(const BundleFiles& files, CancelToken cancelToken,
                                              const InitializeProgressCallback& progressCallback)
{
    MLVC_LOG_INFO("Preparing model cache for bundle %s...", m_manifest.mlvcVersion.ToString().c_str());
    for (const auto& [modelId, modelManifest] : m_manifest.modelManifests) {
        for (const auto& [modelPartId, bundlePart] : modelManifest.modelParts) {
            if (progressCallback) {
                progressCallback(
                    StatusMessage{ "Preparing model cache for " + modelId + " " + ModelPartIdToString(modelPartId) });
            }
            // Cancellation should be checked after progress callback, as user might trigger cancellation from the callback
            if (cancelToken.IsCancelled()) {
                MLVC_LOG_INFO("Model cache preparation cancelled");
                return make_error_code(Error::operation_cancelled);
            }
            auto source = GetModelSource(modelId, bundlePart, files);
            if (!source) return source.error();
            auto cached = m_engine.PrepareModelCache(bundlePart.registryId, source->modelData, source->weightsData,
                                                     source->cacheKey);
            if (!cached) {
                MLVC_LOG_ERROR("Failed to prepare model cache for model %s part %s: %s", modelId.c_str(),
                               ModelPartIdToString(modelPartId), cached.error().message().c_str());
                return cached.error();
            }
            // The inference engine has no model cache
            if (!cached.value()) return false;
        }
    }
    return true;
}

expected<void> ModelBundle::PreloadSessions(const BundleFiles& files, CancelToken cancelToken,
                                            const InitializeProgressCallback& progressCallback)
{
    MLVC_LOG_INFO("Preloading inference sessions for bundle %s...", m_manifest.mlvcVersion.ToString().c_str());
    for (const auto& [modelId, modelManifest] : m_manifest.modelManifests) {
        for (const auto& modelPartId : modelManifest.modelParts | std::views::keys) {
            if (progressCallback) {
                progressCallback(
                    StatusMessage{ "Preloading session for " + modelId + " " + ModelPartIdToString(modelPartId) });
            }
            // Cancellation should be checked after progress callback, as user might trigger cancellation from the callback
            if (cancelToken.IsCancelled()) {
                MLVC_LOG_INFO("Session preloading cancelled");
                return make_error_code(Error::operation_cancelled);
            }
            auto session = CreateInferenceSession(modelId, modelPartId, files);
            if (!session) {
                MLVC_LOG_ERROR("Failed to create inference session for model %s part %s: %s", modelId.c_str(),
                               ModelPartIdToString(modelPartId), session.error().message().c_str());
                return session.error();
            }
            m_inferenceSessions[{ modelId, modelPartId }] = std::move(session.value());
        }
    }
    MLVC_LOG_INFO("Preloaded %zu inference sessions", m_inferenceSessions.size());
    return {};
}

expected<ModelBundle::ModelSource> ModelBundle::GetModelSource(const std::string& modelId, const ModelPartManifest& part,
                                                               const BundleFiles& files) const
{
    const auto registryIt = m_manifest.modelRegistry.find(part.registryId);
    if (registryIt == m_manifest.modelRegistry.end()) {
        MLVC_LOG_ERROR("Registry entry %s not found in manifest for model %s", part.registryId.c_str(), modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }
    const auto& registryEntry = registryIt->second;

    ModelSource source;
    source.cacheKey = m_enableModelCache ? registryEntry.sha256.substr(0, 16) : "";
    std::replace(source.cacheKey.begin(), source.cacheKey.end(), '.', '_');
    if (files.empty()) {
        return source;
    }

    const auto modelIt = files.find(registryEntry.modelPath);
    if (modelIt == files.end()) {
        MLVC_LOG_ERROR("Model file %s not found in bundle", registryEntry.modelPath.c_str());
        return make_error_code(Error::invalid_argument);
    }
    source.modelData = modelIt->second;

    if (const auto& weightsPath = registryEntry.weightsPath) {
        if (const auto weightsIt = files.find(*weightsPath); weightsIt != files.end()) {
            source.weightsData[*weightsPath] = weightsIt->second;
        }
    }
    return source;
}

expected<InferenceSessionPtr> ModelBundle::CreateInferenceSession(const std::string& modelId, const ModelPartId partId,
                                                                  const BundleFiles& files) const
{
    const auto modelIt = m_manifest.modelManifests.find(modelId);
    if (modelIt == m_manifest.modelManifests.end()) {
        MLVC_LOG_ERROR("Model %s not found in manifest", modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }

    const auto& parts = modelIt->second.modelParts;
    const auto partIt = parts.find(partId);
    if (partIt == parts.end()) {
        MLVC_LOG_ERROR("Model part %d not found in model %s", static_cast<int>(partId), modelId.c_str());
        return make_error_code(Error::invalid_argument);
    }
    const ModelPartManifest& bundlePart = partIt->second;
    MLVC_LOG_INFO("Creating inference session for model %s, part=%s, registry_id=%s func=%s", modelId.c_str(),
                  ModelPartIdToString(partId), bundlePart.registryId.c_str(),
                  bundlePart.functionName.value_or("N/A").c_str());

    auto source = GetModelSource(modelId, bundlePart, files);
    if (!source) return source.error();

    std::string sessionName = modelId + "/" + ModelPartIdToString(partId);
    if (bundlePart.functionName) {
        sessionName += "/" + bundlePart.functionName.value();
    }
    auto session = m_engine.CreateSession(sessionName, source->modelData, source->weightsData, source->cacheKey,
                                          bundlePart.functionName);
    if (!session) {
        MLVC_LOG_ERROR("Failed to create inference session for model %s part %s: %s", modelId.c_str(),
                       ModelPartIdToString(partId), session.error().message().c_str());
        return session.error();
    }

    // Models compiled for the NPU don't benefit from a warm-up
    if (m_engine.GetInfo().computeUnit != ComputeUnit::NPU) {
        if (auto ret = WarmUpSession(session.value()); !ret) {
            MLVC_LOG_ERROR("Failed to warm up inference session for model %s part %s: %s", modelId.c_str(),
                           ModelPartIdToString(partId), ret.error().message().c_str());
            return ret.error();
        }
    }

    return session.value();
}

}  // namespace libmlvc
