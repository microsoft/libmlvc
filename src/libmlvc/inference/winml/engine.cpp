// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/inference/winml/engine.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/utils.hpp"
#include "libmlvc/inference/model_cache.hpp"
#include "libmlvc/inference/winml/bootstrap.hpp"
#include "libmlvc/inference/winml/ort_helpers.hpp"
#include "libmlvc/inference/winml/session.hpp"

#include <libmlvc/platform_info.hpp>

#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

using namespace libmlvc;

expected<std::unique_ptr<IInferenceEngine>> libmlvc::MakeWinMlEngine(const WinMlEngineParams& params,
                                                                     CancelToken cancelToken,
                                                                     const InitializeProgressCallback& progressCallback)
{
    auto engineResult = libmlvc_winml::WinMlInferenceEngine::Create(params, cancelToken, progressCallback);
    if (!engineResult) {
        MLVC_LOG_ERROR("Failed to create WinML engine: %s", engineResult.error().message().c_str());
        return engineResult.error();
    }
    return std::move(engineResult.value());
}

namespace libmlvc_winml {

static const char* HardwareDeviceTypeToString(OrtHardwareDeviceType type)
{
    switch (type) {
    case OrtHardwareDeviceType_CPU:
        return "CPU";
    case OrtHardwareDeviceType_GPU:
        return "GPU";
    case OrtHardwareDeviceType_NPU:
        return "NPU";
    default:
        return "unknown";
    }
}

static OrtHardwareDeviceType ComputeUnitToHardwareDeviceType(ComputeUnit computeUnit)
{
    switch (computeUnit) {
    case ComputeUnit::CPU:
        return OrtHardwareDeviceType_CPU;
    case ComputeUnit::GPU:
        return OrtHardwareDeviceType_GPU;
    case ComputeUnit::NPU:
    default:
        return OrtHardwareDeviceType_NPU;
    }
}

static expected<void> CheckMinEpVersion(OnnxExecutionProvider onnxExecutionProvider, const std::string& epPackageVersion)
{
    if (onnxExecutionProvider == OnnxExecutionProvider::CPU || onnxExecutionProvider == OnnxExecutionProvider::DIRECTML) {
        return {};
    }

    if (epPackageVersion.empty()) {
        MLVC_LOG_WARN("Unable to determine %s execution provider version",
                      OnnxExecutionProviderToString(onnxExecutionProvider));
        return {};
    }

    constexpr auto minOpenvinoEpVersion = "1.8.84.0";
    if (onnxExecutionProvider == OnnxExecutionProvider::OPENVINO && !VersionAtLeast(epPackageVersion, minOpenvinoEpVersion)) {
        MLVC_LOG_ERROR("OpenVINO execution provider version %s is below the required minimum %s",
                       epPackageVersion.c_str(), minOpenvinoEpVersion);
        return make_error_code(Error::incompatible_ep_version_error);
    }

    constexpr auto minQnnEpVersion = "1.8.30.0";
    if (onnxExecutionProvider == OnnxExecutionProvider::QNN && !VersionAtLeast(epPackageVersion, minQnnEpVersion)) {
        MLVC_LOG_ERROR("QNN execution provider version %s is below the required minimum %s", epPackageVersion.c_str(),
                       minQnnEpVersion);
        return make_error_code(Error::incompatible_ep_version_error);
    }
    return {};
}

// Convert a uint32 to a hex string (e.g., 0x8086 -> "8086").
static std::string Uint32ToHexTag(uint32_t value)
{
    if (value == 0) return std::string();
    char hex[9];
    snprintf(hex, sizeof(hex), "%X", value);
    return hex;
}

// Look up a value by key in ORT key-value pairs. Returns nullptr if not found.
static const char* FindOrtKvpValue(const OrtApi* ortApi, const OrtKeyValuePairs* kvps, const char* key)
{
    if (!kvps) return nullptr;
    const char* const* keys = nullptr;
    const char* const* values = nullptr;
    size_t n = 0;
    ortApi->GetKeyValuePairs(kvps, &keys, &values, &n);
    for (size_t j = 0; j < n; j++) {
        if (keys[j] && std::strcmp(keys[j], key) == 0) return values[j];
    }
    return nullptr;
}

static std::string Fnv1a32(std::string_view s)
{
    uint32_t h = 0x811c9dc5;
    for (auto c : s) {
        h = (h ^ static_cast<uint8_t>(c)) * 0x01000193;
    }
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08x", h);
    return buf;
}

// Compute a short hash for EP context cache keys from the backend, compute unit,
// EP, EP package version, and (except for OpenVINO) driver version.
// OpenVINO ships its NPU compiler with the EP, so driver updates do not invalidate its cache.
static std::string ComputeEngineEnvHash(const InferenceEngineInfo& config)
{
    std::string fingerprint;
    fingerprint += InferenceBackendToString(config.inferenceBackend);
    fingerprint += '|';
    fingerprint += ComputeUnitToString(config.computeUnit);
    fingerprint += '|';
    fingerprint += OnnxExecutionProviderToString(config.onnxExecutionProvider);
    fingerprint += '|';
    fingerprint += config.windowsAppRuntimeEpVersion;
    if (config.onnxExecutionProvider != OnnxExecutionProvider::OPENVINO) {
        fingerprint += '|';
        fingerprint += config.driverVersion;
    }
    return Fnv1a32(fingerprint);
}

// ORT session creation is not thread-safe with custom execution providers.
// Use a global mutex to serialize session creation across all engine instances.
static std::mutex g_ortCreateSessionMutex;

static expected<void> AddExternalWeightsToSessionOptions(const OrtApi* ortApi, OrtSessionOptions* sessionOptions,
                                                         const std::map<std::string, std::span<const std::byte>>& weightsData)
{
    if (weightsData.empty()) {
        return {};
    }

    MLVC_LOG_DEBUG("Adding %zu external initializers from files in memory", weightsData.size());

    // ORT copies names and lengths by value, and stores raw buffer pointers (char*) into session options.
    // The buffer pointers reference the caller's weightsData spans, which must stay alive through session creation.
    std::vector<std::wstring> weightNamesWide;
    std::vector<char*> weightBuffers;
    std::vector<size_t> weightLengths;

    for (const auto& [name, data] : weightsData) {
        weightNamesWide.push_back(std::wstring(name.begin(), name.end()));
        weightBuffers.push_back(const_cast<char*>(reinterpret_cast<const char*>(data.data())));
        weightLengths.push_back(data.size());
    }

    auto weightNamePtrs = ToRawWideStrings(weightNamesWide);

    if (auto ret = CheckOrtStatus(ortApi, ortApi->AddExternalInitializersFromFilesInMemory(
                                              sessionOptions, weightNamePtrs.data(), weightBuffers.data(),
                                              weightLengths.data(), weightsData.size()));
        !ret) {
        MLVC_LOG_ERROR("Failed to add external initializers from files in memory: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    MLVC_LOG_DEBUG("Successfully added %zu external initializers from files in memory", weightsData.size());
    return {};
}

// --------------------------------------------------------------------------------------------
// WinMlInferenceEngine Implementation
// --------------------------------------------------------------------------------------------

WinMlInferenceEngine::WinMlInferenceEngine(const WinMlEngineParams& params) : m_params(params), m_ortApi(nullptr) {}

expected<std::unique_ptr<WinMlInferenceEngine>>
WinMlInferenceEngine::Create(const WinMlEngineParams& params, CancelToken cancelToken,
                             const InitializeProgressCallback& progressCallback)
{
    auto engine = std::unique_ptr<WinMlInferenceEngine>(new WinMlInferenceEngine(params));
    if (auto ret = engine->Initialize(cancelToken, progressCallback); !ret) {
        return ret.error();
    }
    return engine;
}

expected<void> WinMlInferenceEngine::Initialize(CancelToken cancelToken, const InitializeProgressCallback& progressCallback)
{
    MLVC_LOG_INFO("Initializing WindowsML: compute_unit=%s, onnx_execution_provider=%s, init_mode=%s",
                  ComputeUnitToString(m_params.computeUnit),
                  OnnxExecutionProviderToString(m_params.onnxExecutionProvider), WinMlInitModeToString(m_params.initMode));

    auto modulesResult = LoadWinMlModules(m_params.initMode);
    if (!modulesResult) {
        MLVC_LOG_ERROR("Failed to load WinML / ONNX Runtime modules");
        return modulesResult.error();
    }
    m_hOnnxRuntime = std::move(modulesResult->hOnnxRuntime);
    m_winMlApi = std::move(modulesResult->winMlApi);

    auto ortApiBaseResult = GetOrtApiBaseFromModule(m_hOnnxRuntime.get());
    if (!ortApiBaseResult) {
        MLVC_LOG_ERROR("Failed to get ORT API base from loaded module");
        m_hOnnxRuntime.reset();
        return ortApiBaseResult.error();
    }
    const OrtApiBase* ortApiBase = *ortApiBaseResult;
    const char* ortVersion = ortApiBase->GetVersionString();
    m_ortApi = ortApiBase->GetApi(ORT_API_VERSION);
    if (!m_ortApi) {
        MLVC_LOG_ERROR("Failed to get OrtApi for version %u", ORT_API_VERSION);
        m_hOnnxRuntime.reset();
        return make_error_code(Error::model_init_error);
    }
    const auto winAppRuntimeVersion = GetWindowsAppRuntimeVersion(m_params.initMode);
    MLVC_LOG_DEBUG("Windows App Runtime v%s, ONNX Runtime v%s",
                   winAppRuntimeVersion.empty() ? "N/A" : winAppRuntimeVersion.c_str(), ortVersion ? ortVersion : "N/A");

    // Emit Windows App Runtime version discovery event
    if (progressCallback) {
        progressCallback(
            WindowsAppRuntimeVersionAvailable{ winAppRuntimeVersion.empty() ? "unavailable" : winAppRuntimeVersion });
        if (cancelToken.IsCancelled()) {
            MLVC_LOG_INFO("Initialization cancelled by caller after Windows App Runtime version discovery");
            return make_error_code(Error::operation_cancelled);
        }
    }

    // Gets or creates OrtEnv. OrtEnv is a global ref-counted singleton
    OrtEnv* env = nullptr;
    if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "libmlvc", &env)); !ret) {
        MLVC_LOG_ERROR("Failed to create ONNX Runtime environment: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }
    m_ortEnv = MakeOrtEnv(m_ortApi, env);

    std::string epCatalogPackageVersion;
    if (m_params.onnxExecutionProvider == OnnxExecutionProvider::TENSOR_RT
        || m_params.onnxExecutionProvider == OnnxExecutionProvider::OPENVINO
        || m_params.onnxExecutionProvider == OnnxExecutionProvider::QNN) {

        // Ensure execution provider is downloaded and registered
        if (progressCallback) {
            progressCallback(StatusMessage{ "Ensuring execution provider is ready" });
        }
        auto ret = EnsureExecutionProviderReady(
            m_winMlApi, m_ortApi, m_ortEnv.get(), m_params.onnxExecutionProvider, cancelToken,
            [&progressCallback](double progress) {
                if (progressCallback) {
                    progressCallback(StatusMessage{
                        "Downloading execution provider: " + std::to_string(static_cast<int>(progress)) + "%" });
                }
            });
        if (!ret) {
            MLVC_LOG_ERROR("Failed to ensure %s execution provider is ready: %s",
                           OnnxExecutionProviderToString(m_params.onnxExecutionProvider), ret.error().message().c_str());
            return ret.error();
        }
        epCatalogPackageVersion = ret->epPackageVersion;

        // Ensure minimum required EP version
        if (auto epCheck = CheckMinEpVersion(m_params.onnxExecutionProvider, epCatalogPackageVersion); !epCheck) {
            return epCheck.error();
        }

        // Emit EP info once readiness is established.
        if (progressCallback) {
            progressCallback(WindowsAppRuntimeEPInfoAvailable{
                .version = epCatalogPackageVersion.empty() ? "unavailable" : epCatalogPackageVersion,
                .downloadTimeMs = ret->downloadTimeMs });
            if (cancelToken.IsCancelled()) {
                MLVC_LOG_INFO("Initialization cancelled by caller after execution provider version discovery");
                return make_error_code(Error::operation_cancelled);
            }
        }
    }

    // Validate provider name early
    const char* ortProviderName = OnnxExecutionProviderToOrtName(m_params.onnxExecutionProvider);
    if (!ortProviderName) {
        MLVC_LOG_ERROR("No ORT provider name for %s", OnnxExecutionProviderToString(m_params.onnxExecutionProvider));
        return make_error_code(Error::model_init_error);
    }

    // Enumerate EP devices
    const OrtEpDevice* const* epDevices = nullptr;
    size_t numEpDevices = 0;
    if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->GetEpDevices(m_ortEnv.get(), &epDevices, &numEpDevices)); !ret) {
        MLVC_LOG_ERROR("Failed to enumerate EP devices: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    // Log available EP devices
    MLVC_LOG_INFO("Available EP devices (%zu):", numEpDevices);
    for (size_t i = 0; i < numEpDevices; i++) {
        const char* epName = m_ortApi->EpDevice_EpName(epDevices[i]);
        const auto* hwDevice = m_ortApi->EpDevice_Device(epDevices[i]);
        const char* deviceTypeStr =
            hwDevice ? HardwareDeviceTypeToString(m_ortApi->HardwareDevice_Type(hwDevice)) : "unknown";
        const char* version = FindOrtKvpValue(m_ortApi, m_ortApi->EpDevice_EpMetadata(epDevices[i]), "version");

        std::string hwId;
        const char* luidStr = nullptr;
        const char* highPerformanceIndexStr = nullptr;
        if (hwDevice) {
            auto vendor = Uint32ToHexTag(m_ortApi->HardwareDevice_VendorId(hwDevice));
            if (!vendor.empty() && vendor[0] != '\0') {
                hwId = vendor + ":" + Uint32ToHexTag(m_ortApi->HardwareDevice_DeviceId(hwDevice));
            }
            const auto* metadata = m_ortApi->HardwareDevice_Metadata(hwDevice);
            luidStr = FindOrtKvpValue(m_ortApi, metadata, "LUID");
            highPerformanceIndexStr = FindOrtKvpValue(m_ortApi, metadata, "DxgiHighPerformanceIndex");
        }
        MLVC_LOG_INFO("  [%zu] %s (%s, %s, v%s, luid=%s, dxgi_high_performance_index=%s)", i, epName ? epName : "(null)",
                      deviceTypeStr, hwId.empty() ? "N/A" : hwId.c_str(), version ? version : "N/A",
                      luidStr ? luidStr : "N/A", highPerformanceIndexStr ? highPerformanceIndexStr : "N/A");
    }

    // Resolve compute unit (AUTO -> NPU) before device selection
    const auto resolvedComputeUnit = (m_params.computeUnit == ComputeUnit::AUTO) ? ComputeUnit::NPU : m_params.computeUnit;

    // Select EP device matching provider name and preferred hardware device type
    const auto preferredHwType = ComputeUnitToHardwareDeviceType(resolvedComputeUnit);
    const auto getPerformanceIndex = [&](const OrtHardwareDevice* hwDevice) {
        if (preferredHwType != OrtHardwareDeviceType_GPU) return 0;

        int value = std::numeric_limits<int>::max();
        const char* str =
            FindOrtKvpValue(m_ortApi, m_ortApi->HardwareDevice_Metadata(hwDevice), "DxgiHighPerformanceIndex");
        if (str) {
            const char* end = str + std::strlen(str);
            const auto [ptr, ec] = std::from_chars(str, end, value);
            if (ec != std::errc{} || ptr != end || value < 0) value = std::numeric_limits<int>::max();
        }
        return value;
    };

    int epDeviceIndex = -1;
    int bestPerformanceIndex = std::numeric_limits<int>::max();
    for (size_t i = 0; i < numEpDevices; i++) {
        const char* epName = m_ortApi->EpDevice_EpName(epDevices[i]);
        if (epName && std::strcmp(epName, ortProviderName) == 0) {
            const auto* hwDevice = m_ortApi->EpDevice_Device(epDevices[i]);
            const auto hwType = hwDevice ? m_ortApi->HardwareDevice_Type(hwDevice) : OrtHardwareDeviceType_CPU;
            if (hwType == preferredHwType) {
                const auto performanceIndex = getPerformanceIndex(hwDevice);
                if (epDeviceIndex < 0 || performanceIndex < bestPerformanceIndex) {
                    epDeviceIndex = static_cast<int>(i);
                    bestPerformanceIndex = performanceIndex;
                }
            }
        }
    }

    if (epDeviceIndex < 0) {
        MLVC_LOG_ERROR("%s device with hardware type %s not found among %zu registered EP devices", ortProviderName,
                       HardwareDeviceTypeToString(preferredHwType), numEpDevices);
        return make_error_code(Error::model_init_error);
    }
    m_epDevice = epDevices[epDeviceIndex];
    const auto* hwDev = m_ortApi->EpDevice_Device(epDevices[epDeviceIndex]);
    m_hwDeviceType = hwDev ? m_ortApi->HardwareDevice_Type(hwDev) : OrtHardwareDeviceType_CPU;

    // Resolve EP-specific memory info for zero-copy allocator.
    // V2 API returns null for QNN, fall back to well-known allocator name.
    m_epMemoryInfo = m_ortApi->EpDevice_MemoryInfo(m_epDevice, OrtDeviceMemoryType_HOST_ACCESSIBLE);
    if (!m_epMemoryInfo && m_params.onnxExecutionProvider == OnnxExecutionProvider::QNN) {
        OrtMemoryInfo* mi = nullptr;
        if (CheckOrtStatus(m_ortApi,
                           m_ortApi->CreateMemoryInfo("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeDefault, &mi))) {
            m_epMemoryInfoStorage = MakeOrtMemoryInfo(m_ortApi, mi);
            m_epMemoryInfo = m_epMemoryInfoStorage.get();
            MLVC_LOG_INFO("Resolved EP memory info via named allocator 'QnnHtpShared'");
        }
    }

    // Read EP version from selected device metadata
    const char* epVersion = FindOrtKvpValue(m_ortApi, m_ortApi->EpDevice_EpMetadata(m_epDevice), "version");

    // Extract LUID from hardware device metadata
    int64_t deviceLuid = 0;
    if (hwDev) {
        const char* luidStr = FindOrtKvpValue(m_ortApi, m_ortApi->HardwareDevice_Metadata(hwDev), "LUID");
        if (luidStr) {
            deviceLuid = std::strtoll(luidStr, nullptr, 10);
        }
    }

    // Populate info fields
    m_info.inferenceBackend = InferenceBackend::WINDOWSML;
    m_info.computeUnit = resolvedComputeUnit;
    m_info.onnxExecutionProvider = m_params.onnxExecutionProvider;
    m_info.windowsAppRuntimeVersion = winAppRuntimeVersion;
    m_info.windowsAppRuntimeEpVersion = epCatalogPackageVersion;
    m_info.onnxRuntimeVersion = ortVersion ? ortVersion : "";
    m_info.onnxRuntimeEpVersion = epVersion ? epVersion : "";
    m_info.deviceLuid = deviceLuid;

    // Look up driver info by matching device LUID with PlatformInfo accelerators
    if (deviceLuid != 0) {
        const auto& platformInfo = GetPlatformInfo();
        for (const auto& accel : platformInfo.accelerators) {
            if (accel.luid == deviceLuid) {
                m_info.driverVersion = accel.driverVersion;
                m_info.driverDate = accel.driverDate;
                break;
            }
        }
    }

    // Compute engine environment hash for EP context cache key prefix
    m_envHash = ComputeEngineEnvHash(m_info);

    MLVC_LOG_INFO("WindowsML initialized: runtime=%s, ort=%s, ep=%s-%s[%d], ep_ver=%s(%s), driver=%s(%s), env_hash=%s",
                  m_info.windowsAppRuntimeVersion.empty() ? "N/A" : m_info.windowsAppRuntimeVersion.c_str(),
                  m_info.onnxRuntimeVersion.empty() ? "N/A" : m_info.onnxRuntimeVersion.c_str(),
                  OnnxExecutionProviderToString(m_info.onnxExecutionProvider), ComputeUnitToString(m_info.computeUnit),
                  epDeviceIndex,
                  m_info.windowsAppRuntimeEpVersion.empty() ? "N/A" : m_info.windowsAppRuntimeEpVersion.c_str(),
                  m_info.onnxRuntimeEpVersion.empty() ? "N/A" : m_info.onnxRuntimeEpVersion.c_str(),
                  m_info.driverVersion.empty() ? "N/A" : m_info.driverVersion.c_str(),
                  m_info.driverDate.empty() ? "N/A" : m_info.driverDate.c_str(), m_envHash.c_str());
    return {};
}

WinMlInferenceEngine::~WinMlInferenceEngine() {}

expected<OrtSessionOptionsPtr> WinMlInferenceEngine::CreateConfiguredSessionOptions()
{
    const char* ortProviderName = OnnxExecutionProviderToOrtName(m_params.onnxExecutionProvider);

    // Create session options
    OrtSessionOptions* sessionOptions = nullptr;
    if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->CreateSessionOptions(&sessionOptions)); !ret) {
        MLVC_LOG_ERROR("Failed to create session options: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }
    auto sessionOptionsPtr = MakeOrtSessionOptions(m_ortApi, sessionOptions);

    // For non-CPU devices: disable graph optimizations, CPU memory arena, and memory pattern.
    if (m_hwDeviceType != OrtHardwareDeviceType_CPU) {
        if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->SetSessionGraphOptimizationLevel(sessionOptions, ORT_DISABLE_ALL));
            !ret) {
            MLVC_LOG_ERROR("Failed to set graph optimization level: %s", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
        if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->DisableCpuMemArena(sessionOptions)); !ret) {
            MLVC_LOG_ERROR("Failed to disable CPU memory arena: %s", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
        if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->DisableMemPattern(sessionOptions)); !ret) {
            MLVC_LOG_ERROR("Failed to disable memory pattern: %s", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
    }

    // Set session config entries
    if (m_hwDeviceType == OrtHardwareDeviceType_NPU) {
        // Disable CPU EP fallback so failures are surfaced instead of silently falling back
        if (auto ret = CheckOrtStatus(
                m_ortApi, m_ortApi->AddSessionConfigEntry(sessionOptions, "session.disable_cpu_ep_fallback", "1"));
            !ret) {
            MLVC_LOG_ERROR("Failed to set session config: %s", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
    }

    // Append EP with options using V2 API
    if (m_params.onnxExecutionProvider != OnnxExecutionProvider::CPU) {
        std::vector<const char*> optionKeys;
        std::vector<const char*> optionValues;
        if (m_params.onnxExecutionProvider == OnnxExecutionProvider::QNN && m_hwDeviceType == OrtHardwareDeviceType_NPU) {
            optionKeys = { "htp_performance_mode", "htp_graph_finalization_optimization_mode",
                           "enable_htp_shared_memory_allocator" };
            optionValues = { "burst", "3", "1" };
        }

        if (optionKeys.size() != optionValues.size()) {
            MLVC_LOG_ERROR("EP option keys/values size mismatch: %zu keys vs %zu values", optionKeys.size(),
                           optionValues.size());
            return make_error_code(Error::model_init_error);
        }

        if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->SessionOptionsAppendExecutionProvider_V2(
                                                    sessionOptions, m_ortEnv.get(), &m_epDevice, 1, optionKeys.data(),
                                                    optionValues.data(), optionKeys.size()));
            !ret) {
            MLVC_LOG_ERROR("Failed to append %s execution provider: %s", ortProviderName, ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
        MLVC_LOG_DEBUG("Successfully added %s execution provider using V2 API", ortProviderName);
    }
    return sessionOptionsPtr;
}

expected<InferenceSessionPtr>
WinMlInferenceEngine::CreateSession(std::string_view name, std::span<const std::byte> modelData,
                                    const std::map<std::string, std::span<const std::byte>>& weightsData,
                                    std::string_view cacheKey, const std::optional<std::string>& functionName)
{
    (void)functionName;  // ONNX Runtime doesn't support function selection

    std::lock_guard lock(m_mutex);

    // Creating sessions with non-CPU EPs is not thread-safe
    std::unique_lock globalLock{ g_ortCreateSessionMutex, std::defer_lock };
    if (m_params.onnxExecutionProvider != OnnxExecutionProvider::CPU) {
        globalLock.lock();
    }

    // Check if we should use EP context caching
    const bool useEpContextCache = !cacheKey.empty() && SupportsEpContextCaching();

    if (useEpContextCache) {
        const auto fullCacheKey = m_envHash + "_" + std::string(cacheKey);
        MLVC_LOG_DEBUG("EP context caching enabled for model '%.*s' with key '%s'", static_cast<int>(name.size()),
                       name.data(), fullCacheKey.c_str());
        return CreateSessionFromEpContext(name, modelData, weightsData, fullCacheKey);
    }

    return CreateSessionFromArray(name, modelData, weightsData);
}

expected<InferenceSessionPtr>
WinMlInferenceEngine::CreateSessionFromArray(std::string_view name, std::span<const std::byte> modelData,
                                             const std::map<std::string, std::span<const std::byte>>& weightsData)
{
    auto sessionOptionsResult = CreateConfiguredSessionOptions();
    if (!sessionOptionsResult) {
        return sessionOptionsResult.error();
    }
    auto sessionOptionsPtr = std::move(sessionOptionsResult.value());

    // Add external weights to session options if provided
    if (auto ret = AddExternalWeightsToSessionOptions(m_ortApi, sessionOptionsPtr.get(), weightsData); !ret) {
        return ret.error();
    }

    // Create session directly from memory
    MLVC_LOG_INFO("[%.*s] Creating ORT session from %zu bytes", static_cast<int>(name.size()), name.data(),
                  modelData.size());
    OrtSession* session = nullptr;
    if (auto ret =
            CheckOrtStatus(m_ortApi, m_ortApi->CreateSessionFromArray(m_ortEnv.get(), modelData.data(), modelData.size(),
                                                                      sessionOptionsPtr.get(), &session));
        !ret) {
        MLVC_LOG_ERROR("[%.*s] Session creation failed: %s", static_cast<int>(name.size()), name.data(),
                       ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    auto sessionPtr = MakeOrtSession(m_ortApi, session);
    MLVC_LOG_INFO("[%.*s] ORT session created successfully", static_cast<int>(name.size()), name.data());

    return WinMlInferenceSession::Create(name, std::move(sessionPtr), m_ortApi, m_epMemoryInfo);
}

// --------------------------------------------------------------------------------------------
// EP Context Caching Implementation
// --------------------------------------------------------------------------------------------

static constexpr auto kCachedModelFilename = "model.onnx";

bool WinMlInferenceEngine::SupportsEpContextCaching() const
{
    if (m_hwDeviceType != OrtHardwareDeviceType_NPU) {
        return false;
    }
    return m_params.onnxExecutionProvider == OnnxExecutionProvider::OPENVINO
           || m_params.onnxExecutionProvider == OnnxExecutionProvider::QNN;
}

expected<InferenceSessionPtr>
WinMlInferenceEngine::CreateSessionFromEpContext(std::string_view name, std::span<const std::byte> modelData,
                                                 const std::map<std::string, std::span<const std::byte>>& weightsData,
                                                 std::string_view cacheKey)
{
    auto& cache = libmlvc::ModelCache::Instance();

    // 1. Try to load from existing cache
    if (auto cacheDirResult = cache.LocateCacheDir(cacheKey)) {
        MLVC_LOG_INFO("[%.*s] Found cached EP context at '%s'", static_cast<int>(name.size()), name.data(),
                      std::string(cacheKey).c_str());

        if (auto sessionResult = LoadSessionFromCache(name, cacheDirResult.value())) {
            cache.CleanupCacheDirs();  // Trigger cleanup after successful load
            MLVC_LOG_INFO("[%.*s] Successfully loaded session from EP context cache", static_cast<int>(name.size()),
                          name.data());
            return sessionResult;
        }

        // Cache exists but load failed - delete and recompile
        MLVC_LOG_WARN("[%.*s] Failed to load from cache, recompiling model", static_cast<int>(name.size()), name.data());
        cache.DeleteCacheDir(cacheKey);
    }

    // 2. Create cache directory and compile model to EP context
    auto cacheDirResult = cache.CreateCacheDir(cacheKey);
    if (!cacheDirResult) {
        MLVC_LOG_WARN("[%.*s] Failed to create cache directory, falling back to non-cached session",
                      static_cast<int>(name.size()), name.data());
        return CreateSessionFromArray(name, modelData, weightsData);
    }

    if (auto ret = CompileModelToEpContext(name, modelData, weightsData, cacheDirResult.value()); !ret) {
        MLVC_LOG_WARN("[%.*s] Failed to compile model to EP context: %s", static_cast<int>(name.size()), name.data(),
                      ret.error().message().c_str());
        cache.DeleteCacheDir(cacheKey);
        return CreateSessionFromArray(name, modelData, weightsData);
    }
    MLVC_LOG_DEBUG("[%.*s] Successfully compiled EP context with cache key '%s'", static_cast<int>(name.size()),
                   name.data(), std::string(cacheKey).c_str());

    // 3. Load from freshly created cache (Intel recommendation)
    auto sessionResult = LoadSessionFromCache(name, cacheDirResult.value());
    if (!sessionResult) {
        MLVC_LOG_ERROR("[%.*s] Failed to load from freshly compiled cache", static_cast<int>(name.size()), name.data());
        cache.DeleteCacheDir(cacheKey);
        return sessionResult.error();
    }

    // 4. Trigger cleanup
    cache.CleanupCacheDirs();

    MLVC_LOG_INFO("[%.*s] Successfully created session from EP context, cache key '%s'", static_cast<int>(name.size()),
                  name.data(), std::string(cacheKey).c_str());
    return sessionResult;
}

expected<void> WinMlInferenceEngine::CompileModelToEpContext(std::string_view name, std::span<const std::byte> modelData,
                                                             const std::map<std::string, std::span<const std::byte>>& weightsData,
                                                             const std::filesystem::path& modelCacheDir)
{
    auto sessionOptionsResult = CreateConfiguredSessionOptions();
    if (!sessionOptionsResult) {
        return sessionOptionsResult.error();
    }
    auto sessionOptionsPtr = std::move(sessionOptionsResult.value());

    // Configure EP context options
    auto contextFilePath = (modelCacheDir / kCachedModelFilename).string();

    // Enable EP context generation
    if (auto ret =
            CheckOrtStatus(m_ortApi, m_ortApi->AddSessionConfigEntry(sessionOptionsPtr.get(), "ep.context_enable", "1"));
        !ret) {
        MLVC_LOG_ERROR("Failed to enable EP context: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    // Set embed mode to 0 (external file, not embedded in ONNX)
    if (auto ret = CheckOrtStatus(
            m_ortApi, m_ortApi->AddSessionConfigEntry(sessionOptionsPtr.get(), "ep.context_embed_mode", "0"));
        !ret) {
        MLVC_LOG_ERROR("Failed to set EP context embed mode: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    // Set the output file path for the EP context
    if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->AddSessionConfigEntry(
                                                sessionOptionsPtr.get(), "ep.context_file_path", contextFilePath.c_str()));
        !ret) {
        MLVC_LOG_ERROR("Failed to set EP context file path: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    MLVC_LOG_DEBUG("EP context cache dir: %s", modelCacheDir.filename().string().c_str());

    // Add external weights if provided
    if (auto ret = AddExternalWeightsToSessionOptions(m_ortApi, sessionOptionsPtr.get(), weightsData); !ret) {
        return ret.error();
    }

    // Create session - this will compile and save the EP context
    MLVC_LOG_INFO("[%.*s] Compiling model to EP context...", static_cast<int>(name.size()), name.data());
    OrtSession* session = nullptr;
    if (auto ret =
            CheckOrtStatus(m_ortApi, m_ortApi->CreateSessionFromArray(m_ortEnv.get(), modelData.data(), modelData.size(),
                                                                      sessionOptionsPtr.get(), &session));
        !ret) {
        MLVC_LOG_ERROR("[%.*s] EP context compilation failed: %s", static_cast<int>(name.size()), name.data(),
                       ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    // Release the compilation session - we'll load from cache (Intel recommendation)
    m_ortApi->ReleaseSession(session);

    // Verify the cache file was created
    std::error_code ec;
    if (!std::filesystem::exists(modelCacheDir / kCachedModelFilename, ec)) {
        MLVC_LOG_ERROR("EP context file was not created at expected path");
        return make_error_code(Error::model_init_error);
    }

    // Log all files created in the cache directory for debugging
    MLVC_LOG_DEBUG("Cache directory contents after EP context compilation:");
    for (const auto& entry : std::filesystem::directory_iterator(modelCacheDir, ec)) {
        if (!ec) {
            auto fileSize = std::filesystem::file_size(entry.path(), ec);
            MLVC_LOG_DEBUG("  - %s (%zu bytes)", entry.path().filename().string().c_str(),
                           ec ? 0 : static_cast<size_t>(fileSize));
        }
    }

    MLVC_LOG_DEBUG("[%.*s] EP context compiled successfully", static_cast<int>(name.size()), name.data());
    return {};
}

expected<InferenceSessionPtr> WinMlInferenceEngine::LoadSessionFromCache(std::string_view name,
                                                                         const std::filesystem::path& modelCacheDir)
{
    auto cachedModelPath = modelCacheDir / kCachedModelFilename;

    // Log cache directory contents for debugging
    std::error_code ec;
    MLVC_LOG_DEBUG("Cache directory contents:");
    for (const auto& entry : std::filesystem::directory_iterator(modelCacheDir, ec)) {
        if (!ec) {
            auto fileSize = std::filesystem::file_size(entry.path(), ec);
            MLVC_LOG_DEBUG("  - %s (%zu bytes)", entry.path().filename().string().c_str(),
                           ec ? 0 : static_cast<size_t>(fileSize));
        }
    }

    auto sessionOptionsResult = CreateConfiguredSessionOptions();
    if (!sessionOptionsResult) {
        return sessionOptionsResult.error();
    }
    auto sessionOptionsPtr = std::move(sessionOptionsResult.value());

    // IMPORTANT: Load from FILE PATH, not from memory!
    // EP context creates auxiliary files (e.g., .blob for OpenVINO) alongside the ONNX file,
    // and they are referenced by relative path. Loading from memory breaks this.
    std::wstring cachedModelPathW = cachedModelPath.wstring();
    MLVC_LOG_DEBUG("[%.*s] Loading cached EP context from cache dir: %s", static_cast<int>(name.size()), name.data(),
                   modelCacheDir.filename().string().c_str());

    OrtSession* session = nullptr;
    if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->CreateSession(m_ortEnv.get(), cachedModelPathW.c_str(),
                                                                    sessionOptionsPtr.get(), &session));
        !ret) {
        MLVC_LOG_ERROR("[%.*s] Failed to load session from cache: %s", static_cast<int>(name.size()), name.data(),
                       ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    auto sessionPtr = MakeOrtSession(m_ortApi, session);
    return WinMlInferenceSession::Create(name, std::move(sessionPtr), m_ortApi, m_epMemoryInfo);
}

}  // namespace libmlvc_winml
