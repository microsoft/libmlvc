// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/manager.hpp"
#include "libmlvc/codec/bundle.hpp"
#include "libmlvc/codec/codec.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/platform_names.hpp"
#include "libmlvc/common/utils.hpp"
#include "libmlvc/inference/model_cache.hpp"

#include <libmlvc/platform_info.hpp>

#if defined(MLVC_PLATFORM_APPLE)
    #include "libmlvc/inference/coreml/factory.hpp"
#endif
#if defined(MLVC_PLATFORM_WINCLASSIC)
    #include "libmlvc/inference/winml/factory.hpp"
    #include <shlobj.h>
    #pragma comment(lib, "shell32.lib")
#endif

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <ranges>
#include <sstream>
#include <string_view>

namespace libmlvc {
namespace {

constexpr const char* DEFAULT_CACHE_DIR = "mlvc_cache";

std::string MlvcVersions2Str(std::span<const MlvcVersion> versions)
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < versions.size(); ++i) {
        stream << versions[i].ToString();
        if (i + 1 < versions.size()) stream << ", ";
    }
    return stream.str();
}

std::string ToUpper(std::string_view value)
{
    std::string result{ value };
    for (auto& c : result) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return result;
}

expected<std::filesystem::path> GetAppDataDir()
{
#if defined(MLVC_PLATFORM_WINCLASSIC)
    PWSTR appDataPath = nullptr;
    const HRESULT result = SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &appDataPath);
    if (FAILED(result) || appDataPath == nullptr) {
        MLVC_LOG_ERROR("Failed to get local app data directory");
        if (appDataPath) CoTaskMemFree(appDataPath);
        return make_error_code(Error::io_error);
    }
    std::filesystem::path path(appDataPath);
    CoTaskMemFree(appDataPath);
    return path;
#elif defined(MLVC_PLATFORM_IOS)
    if (const auto home = std::getenv("HOME")) return std::filesystem::path(home) / "Documents";
    MLVC_LOG_ERROR("Failed to get app data directory: HOME not set");
    return make_error_code(Error::io_error);
#elif defined(MLVC_PLATFORM_MACOSX)
    if (const auto home = std::getenv("HOME")) {
        return std::filesystem::path(home) / "Library" / "Application Support";
    }
    MLVC_LOG_ERROR("Failed to get app data directory: HOME not set");
    return make_error_code(Error::io_error);
#else
    #error "Unsupported platform for GetAppDataDir"
#endif
}

// Minimum NPU driver versions required for correct MLVC operation, keyed by CPU vendor/series.
struct MinNpuDriver {
    const char* cpuVendor;
    const char* cpuSeries;
    const char* minDriverVersion;
};

constexpr MinNpuDriver MIN_NPU_DRIVERS[] = {
    // Lunar Lake NPU drivers below 32.0.100.4778 render 540p incorrectly.
    { VENDOR_INTEL, CPU_SERIES_LUNAR_LAKE, "32.0.100.4778" },
    { VENDOR_QUALCOMM, CPU_SERIES_SNAPDRAGON_X1, "30.0.140.1000" },
};

// Fails if the current platform's NPU driver is below the minimum required version.
expected<void> CheckMinNpuDriverVersion()
{
    const auto& platform = GetPlatformInfo();
    for (const auto& entry : MIN_NPU_DRIVERS) {
        if (platform.cpuVendor != entry.cpuVendor || platform.cpuSeries != entry.cpuSeries) {
            continue;
        }

        const AcceleratorInfo* npuAccel = nullptr;
        for (const auto& acc : platform.accelerators) {
            if (acc.type == AcceleratorType::NPU) {
                npuAccel = &acc;
                break;
            }
        }

        const std::string actual = npuAccel ? npuAccel->driverVersion : std::string{};
        if (actual.empty() || !VersionAtLeast(actual, entry.minDriverVersion)) {
            MLVC_LOG_ERROR(
                "NPU driver version %s is below the required minimum %s for %s %s. "
                "Please update the NPU driver to initialize MLVC on the NPU.",
                actual.empty() ? "unknown" : actual.c_str(), entry.minDriverVersion, platform.cpuVendor.c_str(),
                platform.cpuSeries.c_str());
            return make_error_code(Error::incompatible_driver_version_error);
        }
        break;
    }
    return {};
}

#if defined(MLVC_PLATFORM_WINCLASSIC)
OnnxExecutionProvider ResolveOnnxExecutionProvider(ComputeUnit computeUnit)
{
    const auto& platform = GetPlatformInfo();

    switch (computeUnit) {
    case ComputeUnit::CPU:
        return OnnxExecutionProvider::CPU;
    case ComputeUnit::GPU:
        if (platform.cpuVendor == VENDOR_QUALCOMM) {
            return OnnxExecutionProvider::QNN;
        }
        return OnnxExecutionProvider::DIRECTML;
    case ComputeUnit::NPU: {
        if (platform.cpuVendor == VENDOR_QUALCOMM) {
            return OnnxExecutionProvider::QNN;
        } else if (platform.cpuVendor == VENDOR_INTEL) {
            return OnnxExecutionProvider::OPENVINO;
        }
        MLVC_LOG_ERROR("NPU not supported for vendor %s on this architecture", platform.cpuVendor.c_str());
        return OnnxExecutionProvider::CPU;
    }
    default:
        MLVC_LOG_ERROR("Unexpected compute unit %d in ResolveOnnxExecutionProvider", static_cast<int>(computeUnit));
        return OnnxExecutionProvider::CPU;
    }
}
#endif

}  // anonymous namespace

// ----------------------------------------------------------------
// MlvcManagerImpl
// ----------------------------------------------------------------

std::filesystem::path MlvcManagerImpl::GetDefaultModelBundlesDir()
{
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif
    if (const auto modelBundlesDir = std::getenv("LIBMLVC_MODEL_BUNDLES_DIR");
        modelBundlesDir && modelBundlesDir[0] != '\0') {
        return std::filesystem::path{ modelBundlesDir };
    }
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif
    return std::filesystem::path{ "./data/model_bundles" };
}

MlvcManagerImpl::MlvcManagerImpl(const ManagerParams& params)
    : m_params(params)
    , m_info{
        .enableModelCache = params.enableModelCache,
        .enableSessionCaching = params.enableSessionCaching,
    }
{
    if (m_params.cacheDir.empty()) {
        auto appDataDir = GetAppDataDir();
        if (appDataDir) {
            m_params.cacheDir = *appDataDir / DEFAULT_CACHE_DIR;
        } else {
            MLVC_LOG_WARN("Failed to resolve app data directory, falling back to ./%s", DEFAULT_CACHE_DIR);
            m_params.cacheDir = DEFAULT_CACHE_DIR;
        }
    }
}

expected<void> MlvcManagerImpl::Initialize(const std::filesystem::path& bundlesDir, std::span<const MlvcVersion> versions,
                                           CancelToken cancelToken, InitializeProgressCallback progressCallback)
{
    if (m_initialized) {
        MLVC_LOG_WARN("MlvcManager already initialized");
        return {};
    }

    const auto defaultVersions = { GetDefaultModelVersion() };
    const std::span<const MlvcVersion> selectedVersions = versions.empty() ? defaultVersions : versions;

    MLVC_LOG_INFO(
        "Initializing MlvcManager from bundles dir: compute_unit=%s, enable_model_cache=%d, enable_session_caching=%d, "
        "versions=[%s]",
        ComputeUnitToString(m_params.computeUnit), m_params.enableModelCache, m_params.enableSessionCaching,
        MlvcVersions2Str(selectedVersions).c_str());

    if (!IsPlatformSupported()) {
        MLVC_LOG_ERROR("Platform is not supported for MLVC");
        return make_error_code(Error::unsupported_platform_error);
    }

    if (auto ret = InitInferenceEngine(cancelToken, progressCallback); !ret) {
        MLVC_LOG_ERROR("Failed to init inference engine: %s", ret.error().message().c_str());
        return ret.error();
    }

    auto bundleBlobs = LoadModelBundles(bundlesDir.empty() ? GetDefaultModelBundlesDir() : bundlesDir, selectedVersions,
                                        m_inferenceEngine->GetInfo());
    if (!bundleBlobs) {
        return bundleBlobs.error();
    }

    return LoadBundles(std::move(bundleBlobs.value()), std::move(cancelToken), std::move(progressCallback));
}

expected<void> MlvcManagerImpl::Initialize(std::span<const std::span<const std::byte>> bundleBlobs,
                                           CancelToken cancelToken, InitializeProgressCallback progressCallback)
{
    if (m_initialized) {
        MLVC_LOG_WARN("MlvcManager already initialized");
        return {};
    }
    if (bundleBlobs.empty()) {
        MLVC_LOG_ERROR("No model bundle blobs provided");
        return make_error_code(Error::invalid_argument);
    }
    MLVC_LOG_INFO(
        "Initializing MlvcManager from blobs: compute_unit=%s, enable_model_cache=%d, enable_session_caching=%d",
        ComputeUnitToString(m_params.computeUnit), m_params.enableModelCache, m_params.enableSessionCaching);

    if (!IsPlatformSupported()) {
        MLVC_LOG_ERROR("Platform is not supported for MLVC");
        return make_error_code(Error::unsupported_platform_error);
    }

    if (auto ret = InitInferenceEngine(cancelToken, progressCallback); !ret) {
        MLVC_LOG_ERROR("Failed to init inference engine: %s", ret.error().message().c_str());
        return ret.error();
    }

    return LoadBundles(bundleBlobs, std::move(cancelToken), std::move(progressCallback));
}

std::vector<MlvcVersion> MlvcManagerImpl::GetAvailableVersions() const
{
    if (!m_initialized.load(std::memory_order_acquire)) {
        return {};
    }
    std::vector<MlvcVersion> versions;
    versions.reserve(m_bundles.size());
    for (const auto version : m_bundles | std::views::keys) {
        versions.push_back(version);
    }
    return versions;
}

expected<Capabilities> MlvcManagerImpl::GetCapabilities(const MlvcVersion mlvcVersion) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetCapabilities();
}

expected<EncoderConfig> MlvcManagerImpl::GetDefaultEncoderConfig(const MlvcVersion mlvcVersion) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetDefaultEncoderConfig();
}

expected<std::unique_ptr<MlvcEncoderImpl>> MlvcManagerImpl::CreateEncoder(const EncoderConfig& config) const
{
    if (auto ret = CheckInitialized(config.mlvcVersion); !ret) return ret.error();

    const auto encoderTag = "Enc#" + std::to_string(m_encoderCounter++);
    auto encoder = std::unique_ptr<MlvcEncoderImpl>(new MlvcEncoderImpl(encoderTag, shared_from_this()));
    if (auto ret = encoder->Initialize(config); !ret) {
        MLVC_LOG_ERROR("Failed to initialize encoder: %s", ret.error().message().c_str());
        return ret.error();
    }
    return encoder;
}

expected<std::unique_ptr<MlvcDecoderImpl>> MlvcManagerImpl::CreateDecoder() const
{
    if (!m_initialized.load(std::memory_order_acquire)) {
        MLVC_LOG_ERROR("MlvcManager not initialized");
        return make_error_code(Error::not_initialized);
    }

    const auto decoderTag = "Dec#" + std::to_string(m_decoderCounter++);
    auto decoder = std::unique_ptr<MlvcDecoderImpl>(new MlvcDecoderImpl(decoderTag, shared_from_this()));
    if (auto ret = decoder->Initialize(); !ret) {
        MLVC_LOG_ERROR("Failed to initialize decoder: %s", ret.error().message().c_str());
        return ret.error();
    }
    return decoder;
}

bool MlvcManagerImpl::IsVersionAvailable(const MlvcVersion mlvcVersion) const
{
    if (!m_initialized.load(std::memory_order_acquire)) {
        return false;
    }
    return m_bundles.contains(mlvcVersion);
}

expected<std::string> MlvcManagerImpl::GetOptimalModelId(const MlvcVersion mlvcVersion, const int width, const int height) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetOptimalModelId(width, height);
}

expected<ModelManifest> MlvcManagerImpl::GetModelManifest(const MlvcVersion mlvcVersion, const std::string& modelId) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetModelManifest(modelId);
}

expected<std::shared_ptr<const ModelMetadata>> MlvcManagerImpl::GetModelMetadata(const MlvcVersion mlvcVersion,
                                                                                 const std::string& modelId) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetModelMetadata(modelId);
}

expected<std::shared_ptr<const ScaleDecoderData>> MlvcManagerImpl::GetScaleDecoderData(const MlvcVersion mlvcVersion,
                                                                                       const std::string& modelId) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetScaleDecoderData(modelId);
}

expected<std::shared_ptr<const GaussianCoderPmf>> MlvcManagerImpl::GetGaussianPmf(const MlvcVersion mlvcVersion,
                                                                                  const std::string& modelId) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetGaussianPmf(modelId);
}

expected<std::shared_ptr<const BitEstimatorPmf>> MlvcManagerImpl::GetBitEstimatorPmf(const MlvcVersion mlvcVersion,
                                                                                     const std::string& modelId) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetBitEstimatorPmf(modelId);
}

expected<InferenceSessionPtr> MlvcManagerImpl::GetInferenceSession(const MlvcVersion mlvcVersion,
                                                                   const std::string& modelId, const ModelPartId partId) const
{
    if (auto ret = CheckInitialized(mlvcVersion); !ret) return ret.error();
    return m_bundles.at(mlvcVersion).GetInferenceSession(modelId, partId);
}

expected<void> MlvcManagerImpl::InitInferenceEngine(CancelToken cancelToken,
                                                    const InitializeProgressCallback& progressCallback)
{
    ModelCache::Instance().SetRootCacheDir(m_params.cacheDir / "models");

    ComputeUnit computeUnit = m_params.computeUnit;
    if (computeUnit == ComputeUnit::AUTO) {
        computeUnit = HasSupportedNpu() ? ComputeUnit::NPU : ComputeUnit::GPU;
        MLVC_LOG_DEBUG("Resolved compute unit: %s -> %s", ComputeUnitToString(m_params.computeUnit),
                       ComputeUnitToString(computeUnit));
    }

    if (computeUnit == ComputeUnit::NPU) {
        if (auto ret = CheckMinNpuDriverVersion(); !ret) {
            return ret.error();
        }

        if (!HasSupportedNpu()) {
            const auto& platform = GetPlatformInfo();
            if (platform.cpuVendor != VENDOR_APPLE && platform.cpuVendor != VENDOR_INTEL
                && platform.cpuVendor != VENDOR_QUALCOMM) {
                MLVC_LOG_ERROR("NPU compute unit requested but vendor %s does not have supported NPUs.",
                               platform.cpuVendor.c_str());
                return make_error_code(Error::invalid_argument);
            } else {
                MLVC_LOG_WARN(
                    "NPU compute unit requested but current platform (vendor=%s, series=%s) is not in the "
                    "list of tested/supported NPUs. Inference may fail or produce unexpected results.",
                    platform.cpuVendor.c_str(), platform.cpuSeries.c_str());
            }
        }
    }

    if (computeUnit != ComputeUnit::NPU) {
        MLVC_LOG_WARN(
            "MLVC is primarily designed for NPU inference. Running on %s may result in degraded "
            "performance or unexpected behavior.",
            ToUpper(ComputeUnitToString(computeUnit)).c_str());
    }

#if defined(MLVC_PLATFORM_WINCLASSIC)
    auto modelType = ModelType::ONNX;
    WinMlEngineParams engineParams{};
    engineParams.computeUnit = computeUnit;
    engineParams.onnxExecutionProvider = ResolveOnnxExecutionProvider(computeUnit);
    engineParams.initMode = m_params.winmlInitMode;
    auto engine = MakeWinMlEngine(engineParams, cancelToken, progressCallback);
    if (!engine) {
        MLVC_LOG_ERROR("Failed to create WinML inference engine: %s", engine.error().message().c_str());
        return engine.error();
    }
    m_inferenceEngine = std::move(engine.value());

#elif defined(MLVC_PLATFORM_APPLE)
    auto modelType = ModelType::COREML;
    CoreMlEngineParams engineParams{};
    engineParams.computeUnit = computeUnit;
    auto engine = MakeCoreMlEngine(engineParams);
    if (!engine) {
        MLVC_LOG_ERROR("Failed to create CoreML inference engine: %s", engine.error().message().c_str());
        return engine.error();
    }
    m_inferenceEngine = std::move(engine.value());
#else
    #error "Inference engine not implemented for this platform"
#endif

    const auto& engineInfo = m_inferenceEngine->GetInfo();
    m_info.inferenceBackend = engineInfo.inferenceBackend;
    m_info.modelType = modelType;
    m_info.computeUnit = engineInfo.computeUnit;
    m_info.onnxExecutionProvider = engineInfo.onnxExecutionProvider;
    m_info.windowsAppRuntimeVersion = engineInfo.windowsAppRuntimeVersion;
    m_info.windowsAppRuntimeEpVersion = engineInfo.windowsAppRuntimeEpVersion;
    m_info.onnxRuntimeVersion = engineInfo.onnxRuntimeVersion;
    m_info.onnxRuntimeEpVersion = engineInfo.onnxRuntimeEpVersion;
    m_info.deviceLuid = engineInfo.deviceLuid;
    m_info.driverVersion = engineInfo.driverVersion;
    m_info.driverDate = engineInfo.driverDate;
    return {};
}

expected<void> MlvcManagerImpl::LoadBundles(std::vector<std::vector<std::byte>>&& bundleBlobs, CancelToken cancelToken,
                                            const InitializeProgressCallback& progressCallback)
{
    std::vector<std::span<const std::byte>> bundleBlobsSpan;
    bundleBlobsSpan.reserve(bundleBlobs.size());
    for (const auto& bundle : bundleBlobs) {
        bundleBlobsSpan.emplace_back(bundle);
    }
    return LoadBundles(bundleBlobsSpan, std::move(cancelToken), progressCallback);
}

expected<void> MlvcManagerImpl::LoadBundles(std::span<const std::span<const std::byte>> bundleBlobs,
                                            CancelToken cancelToken, const InitializeProgressCallback& progressCallback)
{
    std::map<MlvcVersion, ModelBundle> bundles;
    for (const auto& bundleBlob : bundleBlobs) {
        const auto index = bundles.size() + 1;
        MLVC_LOG_INFO("Loading model bundle %zu/%zu (%.1lf MB)...", index, bundleBlobs.size(),
                      static_cast<double>(bundleBlob.size()) / std::pow(1024.0, 2));

        if (cancelToken.IsCancelled()) {
            return make_error_code(Error::operation_cancelled);
        }

        ModelBundle bundle(bundleBlob, *m_inferenceEngine, m_params.enableModelCache, m_params.enableSessionCaching);
        if (auto ret = bundle.Initialize(cancelToken, progressCallback); !ret) {
            MLVC_LOG_ERROR("Failed to initialize bundle: %s", ret.error().message().c_str());
            return ret.error();
        }

        const auto& manifest = bundle.GetManifest();
        if (bundles.contains(manifest.mlvcVersion)) {
            MLVC_LOG_ERROR("Duplicate MLVC version %s, bundle already loaded", manifest.mlvcVersion.ToString().c_str());
            return make_error_code(Error::invalid_argument);
        }

        if (manifest.modelType != m_info.modelType) {
            MLVC_LOG_ERROR("Bundle %s has model type %s but platform expects %s", manifest.mlvcVersion.ToString().c_str(),
                           ModelTypeToString(manifest.modelType), ModelTypeToString(m_info.modelType));
            return make_error_code(Error::invalid_argument);
        }
        bundles.emplace(manifest.mlvcVersion, std::move(bundle));
        MLVC_LOG_INFO("Loading model bundle %zu/%zu completed", index, bundleBlobs.size());
    }

    if (bundles.empty()) {
        MLVC_LOG_ERROR("No compatible model bundles loaded");
        return make_error_code(Error::invalid_argument);
    }

    m_bundles = std::move(bundles);
    m_initialized.store(true, std::memory_order_release);
    MLVC_LOG_INFO("MLVC Manager initialized with %zu bundle(s)", m_bundles.size());
    if (progressCallback) {
        progressCallback(StatusMessage{ "Initialization complete" });
    }
    return {};
}

expected<void> MlvcManagerImpl::CheckInitialized(MlvcVersion mlvcVersion) const
{
    if (!m_initialized.load(std::memory_order_acquire)) {
        MLVC_LOG_ERROR("MlvcManager not initialized");
        return make_error_code(Error::not_initialized);
    }
    if (!m_bundles.contains(mlvcVersion)) {
        MLVC_LOG_ERROR("Requested MLVC version %s not available", mlvcVersion.ToString().c_str());
        return make_error_code(Error::invalid_argument);
    }
    return {};
}

}  // namespace libmlvc
