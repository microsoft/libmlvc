// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/inference/winml/bootstrap.hpp"
#include "libmlvc/common/logging.hpp"

#include <libmlvc/error_codes.hpp>

#include <MddBootstrap.h>
#include <Windows.h>
#include <appmodel.h>
#include <winml/onnxruntime_c_api.h>
#include <winrt/Microsoft.Windows.ApplicationModel.WindowsAppRuntime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

// WINDOWSAPPSDK_VERSION_MAJOR and WINDOWSAPPSDK_VERSION_MINOR are provided as
// compile definitions by the WindowsAppSdk CMake target, derived from the
// Foundation NuGet package version.
#if !defined(WINDOWSAPPSDK_VERSION_MAJOR) || !defined(WINDOWSAPPSDK_VERSION_MINOR)
    #error "WINDOWSAPPSDK_VERSION_MAJOR and WINDOWSAPPSDK_VERSION_MINOR must be defined by the build system"
#endif

#if WINDOWSAPPSDK_VERSION_MAJOR < 1 || (WINDOWSAPPSDK_VERSION_MAJOR == 1 && WINDOWSAPPSDK_VERSION_MINOR < 8)
    #error "Windows App SDK version must be at least 1.8"
#endif

#define WINDOWSAPPSDK_RELEASE_MAJORMINOR \
    static_cast<uint32_t>((WINDOWSAPPSDK_VERSION_MAJOR << 16) | WINDOWSAPPSDK_VERSION_MINOR)
#define WINDOWSAPPSDK_RELEASE_VERSION_TAG_W L""

using namespace libmlvc;

namespace libmlvc_winml {

// Configuration constants
constexpr std::chrono::milliseconds ASYNC_POLL_INTERVAL{ 100 };  // Polling interval for async download operations
constexpr wchar_t WINDOWSAPPSDK_PUBLISHER_ID[] = L"8wekyb3d8bbwe";
constexpr wchar_t kOnnxRuntimeDllName[] = L"onnxruntime.dll";
constexpr wchar_t kWinMLDllName[] = L"Microsoft.Windows.AI.MachineLearning.dll";

static PACKAGE_VERSION GetWindowsAppSdkMinVersion()
{
    PACKAGE_VERSION minVersion{};

#if WINDOWSAPPSDK_VERSION_MAJOR == 1 && WINDOWSAPPSDK_VERSION_MINOR == 8
    // Windows App SDK 1.8.1 (1.8.250916003) is the first release with Windows ML support.
    // Its runtime package version is 8000.625.330.0.
    minVersion.Major = 8000;
    minVersion.Minor = 625;
    minVersion.Build = 330;
    minVersion.Revision = 0;
#endif
    return minVersion;
}

static expected<std::wstring> GetModulePathW(HMODULE hModule);

static const char* ReadyStateToString(WinMLEpReadyState state)
{
    switch (state) {
    case WinMLEpReadyState_Ready:
        return "Ready";
    case WinMLEpReadyState_NotReady:
        return "NotReady";
    case WinMLEpReadyState_NotPresent:
        return "NotPresent";
    default:
        return "Unknown";
    }
}

static const char* CertificationToString(WinMLEpCertification certification)
{
    switch (certification) {
    case WinMLEpCertification_Unknown:
        return "Unknown";
    case WinMLEpCertification_Certified:
        return "Certified";
    case WinMLEpCertification_Uncertified:
        return "Uncertified";
    default:
        return "Unrecognized";
    }
}

expected<const OrtApiBase*> GetOrtApiBaseFromModule(HMODULE hOnnxRuntime)
{
    if (!hOnnxRuntime) {
        MLVC_LOG_ERROR("GetOrtApiBaseFromModule: onnxruntime.dll not loaded");
        return make_unexpected(make_error_code(Error::invalid_argument));
    }

    typedef const OrtApiBase* (*OrtGetApiBaseFunc)(void);
    auto getApiBase = reinterpret_cast<OrtGetApiBaseFunc>(GetProcAddress(hOnnxRuntime, "OrtGetApiBase"));

    if (!getApiBase) {
        MLVC_LOG_ERROR("Failed to get OrtGetApiBase from onnxruntime.dll");
        return make_unexpected(make_error_code(Error::general_failure));
    }

    const OrtApiBase* apiBase = getApiBase();
    if (!apiBase) {
        MLVC_LOG_ERROR("OrtGetApiBase returned null");
        return make_unexpected(make_error_code(Error::general_failure));
    }

    return apiBase;
}

static bool IsPackagedApp()
{
    UINT32 length = 0;
    return GetCurrentPackageFullName(&length, nullptr) != APPMODEL_ERROR_NO_PACKAGE;
}

static expected<std::wstring> GetPackagePathFromFullName(PCWSTR packageFullName)
{
    UINT32 pathLength = 0;
    LONG rc = GetPackagePathByFullName(packageFullName, &pathLength, nullptr);
    if (rc != ERROR_INSUFFICIENT_BUFFER || pathLength == 0) {
        MLVC_LOG_ERROR("GetPackagePathByFullName size query failed for '%ls' (Error: 0x%08lX)", packageFullName,
                       static_cast<unsigned long>(rc));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    std::wstring packagePath(pathLength, L'\0');
    rc = GetPackagePathByFullName(packageFullName, &pathLength, packagePath.data());
    if (rc != ERROR_SUCCESS) {
        MLVC_LOG_ERROR("GetPackagePathByFullName failed for '%ls' (Error: 0x%08lX)", packageFullName,
                       static_cast<unsigned long>(rc));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    // pathLength includes terminating null; trim it for std::wstring semantics.
    if (!packagePath.empty() && packagePath.back() == L'\0') {
        packagePath.pop_back();
    }

    return packagePath;
}

static expected<std::wstring> FindStaticWindowsAppRuntimeDependency(const std::wstring& packageFamilyName,
                                                                    const PACKAGE_VERSION& minVersion)
{
    // PACKAGE_FILTER_DIRECT enumerates the current package's direct framework deps.
    // PACKAGE_FILTER_OPTIONAL must be OR'd in to include dependencies marked with
    // uap6:Optional="true" in the appxmanifest.
    constexpr UINT32 kPackageFilter = PACKAGE_FILTER_DIRECT | PACKAGE_FILTER_OPTIONAL;

    UINT32 bufferLength = 0;
    UINT32 count = 0;
    LONG rc = GetCurrentPackageInfo(kPackageFilter, &bufferLength, nullptr, &count);
    if (rc != ERROR_INSUFFICIENT_BUFFER) {
        // ERROR_SUCCESS with bufferLength==0 means no direct deps; anything else is a real error.
        if (rc != ERROR_SUCCESS) {
            MLVC_LOG_WARN("GetCurrentPackageInfo size query failed (Error: 0x%08lX)", static_cast<unsigned long>(rc));
        }
        return make_unexpected(make_error_code(Error::general_failure));
    }

    std::vector<BYTE> buffer(bufferLength);
    rc = GetCurrentPackageInfo(kPackageFilter, &bufferLength, buffer.data(), &count);
    if (rc != ERROR_SUCCESS) {
        MLVC_LOG_WARN("GetCurrentPackageInfo failed (Error: 0x%08lX)", static_cast<unsigned long>(rc));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    MLVC_LOG_INFO("Current package has %u direct dependencies", count);

    const PACKAGE_INFO* packages = reinterpret_cast<const PACKAGE_INFO*>(buffer.data());
    for (UINT32 i = 0; i < count; ++i) {
        const PACKAGE_INFO& info = packages[i];
        MLVC_LOG_INFO(
            "Found dependency: familyName='%ls', fullName='%ls', version=%u.%u.%u.%u, path='%ls' compare with target "
            "familyName='%ls' minVersion=%u.%u.%u.%u",
            info.packageFamilyName ? info.packageFamilyName : L"<null>",
            info.packageFullName ? info.packageFullName : L"<null>", info.packageId.version.Major,
            info.packageId.version.Minor, info.packageId.version.Build, info.packageId.version.Revision,
            info.path ? info.path : L"<null>", packageFamilyName.c_str(), minVersion.Major, minVersion.Minor,
            minVersion.Build, minVersion.Revision);
        if (!info.packageFamilyName || packageFamilyName != info.packageFamilyName) {
            continue;
        }
        const PACKAGE_VERSION& v = info.packageId.version;
        // PACKAGE_VERSION layout is (Revision, Build, Minor, Major) low-to-high and Windows runs on LE only - so compare via Version.
        if (v.Version < minVersion.Version) {
            MLVC_LOG_WARN(
                "Static Windows App Runtime dependency '%ls' present but version %u.%u.%u.%u "
                "is below required minVersion %u.%u.%u.%u",
                packageFamilyName.c_str(), v.Major, v.Minor, v.Build, v.Revision, minVersion.Major, minVersion.Minor,
                minVersion.Build, minVersion.Revision);
            continue;
        }
        MLVC_LOG_INFO("Found static Windows App Runtime dependency '%ls' v%u.%u.%u.%u at: %ls", packageFamilyName.c_str(),
                      v.Major, v.Minor, v.Build, v.Revision, info.path ? info.path : L"<unknown>");
        if (!info.path) {
            return make_unexpected(make_error_code(Error::general_failure));
        }
        return std::wstring(info.path);
    }

    return make_unexpected(make_error_code(Error::general_failure));
}

// For packaged apps: resolves the Windows App Runtime framework directory. First checks if the
// appxmanifest already declares a matching <PackageDependency>. Otherwise falls back to
// TryCreatePackageDependency + AddPackageDependency to dynamically
// attach the framework. Returns the runtime directory on success.
static expected<std::wstring> ResolveWindowsAppRuntimePathForPackagedApp()
{
    using TryCreatePackageDependencyFn = HRESULT(WINAPI*)(
        PSID user, PCWSTR packageFamilyName, PACKAGE_VERSION minVersion, UINT32 packageDependencyProcessorArchitectures,
        UINT32 lifetimeKind, PCWSTR lifetimeArtifact, UINT32 options, PWSTR * packageDependencyId);
    using AddPackageDependencyFn = HRESULT(WINAPI*)(PCWSTR packageDependencyId, INT32 rank, UINT32 options,
                                                    void** packageDependencyContext, PWSTR* packageFullName);
    auto heapFreeDeleter = [](wchar_t* p) {
        if (p) {
            HeapFree(GetProcessHeap(), 0, p);
        }
    };
    using UniqueHeapString = std::unique_ptr<wchar_t, decltype(heapFreeDeleter)>;

    static std::mutex s_dependencyMutex;
    static bool s_dependencyAdded{};
    static void* s_dependencyContext{};
    static std::wstring s_runtimeDirectory;

    std::lock_guard<std::mutex> lock(s_dependencyMutex);
    if (s_dependencyAdded) {
        return s_runtimeDirectory;
    }

    PACKAGE_VERSION minVersion = GetWindowsAppSdkMinVersion();
    std::wstring packageFamilyName = L"Microsoft.WindowsAppRuntime." + std::to_wstring(WINDOWSAPPSDK_VERSION_MAJOR);
    if constexpr (WINDOWSAPPSDK_VERSION_MAJOR == 1) {
        packageFamilyName += L"." + std::to_wstring(WINDOWSAPPSDK_VERSION_MINOR);
    }
    packageFamilyName += L"_";
    packageFamilyName += WINDOWSAPPSDK_PUBLISHER_ID;

    // If the appxmanifest already declares a static <PackageDependency> on the
    // matching Windows App Runtime framework, no dynamic AddPackageDependency call is needed
    if (auto staticPath = FindStaticWindowsAppRuntimeDependency(packageFamilyName, minVersion); staticPath) {
        s_runtimeDirectory = *staticPath;
        s_dependencyAdded = true;
        return s_runtimeDirectory;
    }

    HMODULE kernelBase = GetModuleHandleW(L"kernelbase.dll");
    if (!kernelBase) {
        MLVC_LOG_ERROR("GetModuleHandleW(kernelbase.dll) failed (Error: 0x%08lX)",
                       static_cast<unsigned long>(GetLastError()));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    // otherwise add package dependency here via addPackageDependency in order to be able to use
    // winrt::Microsoft::Windows::AI::MachineLearning apis
    auto tryCreatePackageDependency =
        reinterpret_cast<TryCreatePackageDependencyFn>(GetProcAddress(kernelBase, "TryCreatePackageDependency"));
    auto addPackageDependency =
        reinterpret_cast<AddPackageDependencyFn>(GetProcAddress(kernelBase, "AddPackageDependency"));

    if (!tryCreatePackageDependency || !addPackageDependency) {
        MLVC_LOG_WARN(
            "OS dynamic dependency APIs are not available on this Windows version. "
            "TryCreatePackageDependency=%p AddPackageDependency=%p",
            static_cast<void*>(tryCreatePackageDependency), static_cast<void*>(addPackageDependency));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    PWSTR packageDependencyIdRaw = nullptr;
    HRESULT hr = tryCreatePackageDependency(nullptr, packageFamilyName.c_str(), minVersion,
                                            0,  // PackageDependencyProcessorArchitectures::None
                                            0,  // PackageDependencyLifetimeKind::Process
                                            nullptr,
                                            0,  // CreatePackageDependencyOptions_None
                                            &packageDependencyIdRaw);
    UniqueHeapString packageDependencyId{ packageDependencyIdRaw, heapFreeDeleter };
    if (FAILED(hr)) {
        MLVC_LOG_ERROR(
            "TryCreatePackageDependency failed for package family '%ls' and minVersion %u.%u.%u.%u "
            "(HRESULT: 0x%08lX)",
            packageFamilyName.c_str(), minVersion.Major, minVersion.Minor, minVersion.Build, minVersion.Revision,
            static_cast<unsigned long>(hr));
        return make_unexpected(make_error_code(Error::windows_app_runtime_unavailable_error));
    }

    PWSTR packageFullNameRaw = nullptr;
    hr = addPackageDependency(packageDependencyId.get(),
                              0,  // PACKAGE_DEPENDENCY_RANK_DEFAULT
                              0,  // AddPackageDependencyOptions_None
                              &s_dependencyContext, &packageFullNameRaw);
    UniqueHeapString packageFullName{ packageFullNameRaw, heapFreeDeleter };
    if (FAILED(hr)) {
        MLVC_LOG_ERROR("AddPackageDependency failed (HRESULT: 0x%08lX)", static_cast<unsigned long>(hr));
        return make_unexpected(make_error_code(Error::windows_app_runtime_unavailable_error));
    }

    MLVC_LOG_INFO("Added packaged Windows App Runtime dependency: %ls",
                  packageFullName ? packageFullName.get() : L"<unknown>");

    if (!packageFullName) {
        MLVC_LOG_ERROR("AddPackageDependency succeeded but returned null package full name");
        return make_unexpected(make_error_code(Error::general_failure));
    }

    auto runtimeDirectory = GetPackagePathFromFullName(packageFullName.get());
    if (!runtimeDirectory) {
        return make_unexpected(runtimeDirectory.error());
    }

    s_runtimeDirectory = *runtimeDirectory;
    s_dependencyAdded = true;
    return s_runtimeDirectory;
}

static expected<void> TryInitializeAppSDKForUnpackagedApp()
{
    static std::mutex s_initMutex;
    static bool s_initialized{};

    std::lock_guard<std::mutex> lock(s_initMutex);
    if (s_initialized) {
        return {};
    }

    PACKAGE_VERSION minVersion = GetWindowsAppSdkMinVersion();

    const MddBootstrapInitializeOptions options =
        static_cast<MddBootstrapInitializeOptions>(MddBootstrapInitializeOptions_OnNoMatch_ShowUI);

    HModulePtr hBootstrap(LoadLibraryW(L"Microsoft.WindowsAppRuntime.Bootstrap.dll"), [](HMODULE h) {
        if (h) {
            FreeLibrary(h);
        }
    });
    if (!hBootstrap) {
        auto lastError = GetLastError();
        MLVC_LOG_ERROR("Failed to load Microsoft.WindowsAppRuntime.Bootstrap.dll (Error: 0x%08lX)",
                       static_cast<unsigned long>(lastError));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    auto initialize2 = reinterpret_cast<decltype(&MddBootstrapInitialize2)>(
        GetProcAddress(hBootstrap.get(), "MddBootstrapInitialize2"));
    if (!initialize2) {
        auto lastError = GetLastError();
        MLVC_LOG_ERROR("Failed to get MddBootstrapInitialize2 from bootstrap DLL (Error: 0x%08lX)",
                       static_cast<unsigned long>(lastError));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    // It's ok to call this multiple times as long as the version paramters are compatible.
    // So if someone else in Teams also uses windowsappsdk, they can call this too.
    MLVC_LOG_INFO("MddBootstrapInitialize2: SDK=%d.%d, minVersion=%u.%u.%u.%u, options=0x%04X",
                  WINDOWSAPPSDK_VERSION_MAJOR, WINDOWSAPPSDK_VERSION_MINOR, minVersion.Major, minVersion.Minor,
                  minVersion.Build, minVersion.Revision, static_cast<unsigned int>(options));
    HRESULT hr = initialize2(WINDOWSAPPSDK_RELEASE_MAJORMINOR, WINDOWSAPPSDK_RELEASE_VERSION_TAG_W, minVersion, options);

    if (FAILED(hr)) {
        MLVC_LOG_ERROR("Failed to initialize Windows App SDK bootstrapper (HRESULT: 0x%08lX)",
                       static_cast<unsigned long>(hr));
        return make_unexpected(make_error_code(Error::windows_app_runtime_unavailable_error));
    }

    s_initialized = true;
    return {};
}

static expected<std::wstring> GetUnpackagedWindowsAppSdkRuntimeDir()
{
    constexpr wchar_t kWindowsAppRuntimeDllName[] = L"Microsoft.WindowsAppRuntime.dll";

    HModulePtr hWinAppRuntime(LoadLibraryW(kWindowsAppRuntimeDllName), [](HMODULE h) {
        if (h) {
            FreeLibrary(h);
        }
    });
    if (!hWinAppRuntime) {
        auto lastError = GetLastError();
        MLVC_LOG_ERROR("Failed to load %ls (Error: 0x%08lX)", kWindowsAppRuntimeDllName,
                       static_cast<unsigned long>(lastError));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    auto modulePath = GetModulePathW(hWinAppRuntime.get());
    if (!modulePath) return make_unexpected(modulePath.error());

    auto lastSeparator = modulePath->find_last_of(L"\\/");
    if (lastSeparator == std::wstring::npos) {
        MLVC_LOG_ERROR("Invalid module path format for %ls: %ls", kWindowsAppRuntimeDllName, modulePath->c_str());
        return make_unexpected(make_error_code(Error::general_failure));
    }

    std::wstring runtimeDir = modulePath->substr(0, lastSeparator);
    MLVC_LOG_INFO("Windows App SDK runtime directory: %ls", runtimeDir.c_str());
    return runtimeDir;
}

static expected<std::wstring> GetModulePathW(HMODULE hModule)
{
    // Windows \\?\ extended paths are capped at 32767 wchars (UNICODE_STRING.Length is a USHORT,
    // so max bytes / sizeof(WCHAR) = 32767). Size for that worst case + null terminator.
    constexpr DWORD kExtendedPathBufferChars = 32768;
    std::wstring path(kExtendedPathBufferChars, L'\0');
    DWORD len = GetModuleFileNameW(hModule, path.data(), kExtendedPathBufferChars);
    if (len == 0 || len == kExtendedPathBufferChars) {
        // 0: API failure. len == buffer size: path was truncated at the OS limit.
        MLVC_LOG_ERROR("GetModuleFileNameW failed (Error: 0x%08lX, len=%lu)",
                       static_cast<unsigned long>(GetLastError()), static_cast<unsigned long>(len));
        return make_unexpected(make_error_code(Error::general_failure));
    }
    path.resize(len);
    return path;
}

// Used by self-contained mode: DLLs are deployed next to this DLL/EXE.
static expected<std::wstring> GetCurrentModuleDir()
{
    HMODULE currentModule = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&GetCurrentModuleDir), &currentModule)) {
        auto lastError = GetLastError();
        MLVC_LOG_ERROR("GetModuleHandleExW failed (Error: 0x%08lX)", static_cast<unsigned long>(lastError));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    auto modulePath = GetModulePathW(currentModule);
    if (!modulePath) return make_unexpected(modulePath.error());

    auto lastSeparator = modulePath->find_last_of(L"\\/");
    if (lastSeparator == std::wstring::npos) {
        MLVC_LOG_ERROR("Invalid module path: %ls", modulePath->c_str());
        return make_unexpected(make_error_code(Error::general_failure));
    }
    return modulePath->substr(0, lastSeparator);
}

static expected<HModulePtr> LoadDllFromDir(const std::wstring& dir, const wchar_t* dllName)
{
    const std::wstring dllPath = (std::filesystem::path(dir) / dllName).wstring();

    auto hDll = HModulePtr(LoadLibraryExW(dllPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH), [](HMODULE h) {
        if (h) {
            FreeLibrary(h);
        }
    });
    if (!hDll) {
        auto lastError = GetLastError();
        MLVC_LOG_ERROR("Failed to load %ls from '%ls' (Error: 0x%08lX)", dllName, dllPath.c_str(),
                       static_cast<unsigned long>(lastError));
        return make_unexpected(make_error_code(Error::general_failure));
    }

    if (auto loadedPath = GetModulePathW(hDll.get()); loadedPath) {
        MLVC_LOG_INFO("Loaded %ls from: %ls", dllName, loadedPath->c_str());
        if (_wcsicmp(loadedPath->c_str(), dllPath.c_str()) != 0) {
            MLVC_LOG_ERROR(
                "%ls path mismatch: expected '%ls', actually loaded from '%ls'. "
                "Another copy was already loaded in this process.",
                dllName, dllPath.c_str(), loadedPath->c_str());
            return make_unexpected(make_error_code(Error::general_failure));
        }
    }

    return hDll;
}

// Resolves the directory from which onnxruntime.dll and Microsoft.Windows.AI.MachineLearning.dll
// should be loaded for the given init mode.
static expected<std::wstring> ResolveDllDir(WinMlInitMode mode)
{
    if (mode == WinMlInitMode::SelfContained) {
        auto dir = GetCurrentModuleDir();
        return dir;
    }

    // AppSdk mode: bootstrap WindowsAppRuntime and discover its framework directory.
    bool isPackaged = IsPackagedApp();
    expected<std::wstring> appSdkPath = make_unexpected(make_error_code(Error::general_failure));
    if (isPackaged) {
        appSdkPath = ResolveWindowsAppRuntimePathForPackagedApp();
    } else {
        if (auto initResult = TryInitializeAppSDKForUnpackagedApp(); !initResult) {
            return make_unexpected(initResult.error());
        }
        appSdkPath = GetUnpackagedWindowsAppSdkRuntimeDir();
    }
    if (!appSdkPath) {
        MLVC_LOG_ERROR("Failed to initialize Windows App SDK: %s", appSdkPath.error().message().c_str());
        return make_unexpected(appSdkPath.error());
    }
    MLVC_LOG_INFO("%s app detected - Windows App SDK path: %ls", isPackaged ? "Packaged" : "Unpackaged",
                  appSdkPath->c_str());
    return appSdkPath;
}

// Resolves all WinMLEp*/WinMLAsync* function pointers from a loaded Microsoft.Windows.AI.MachineLearning.dll.
// Fills out everything except hWinML (which the caller owns).
static expected<void> ResolveWinMlApiFunctions(HMODULE hWinML, WinMlApi& outApi)
{
#define MLVC_RESOLVE_WINML_FN(NAME)                                                   \
    outApi.NAME = reinterpret_cast<decltype(&::NAME)>(GetProcAddress(hWinML, #NAME)); \
    if (!outApi.NAME) {                                                               \
        MLVC_LOG_ERROR("Failed to resolve %s from %ls", #NAME, kWinMLDllName);        \
        return make_unexpected(make_error_code(Error::general_failure));              \
    }

    MLVC_RESOLVE_WINML_FN(WinMLEpCatalogCreate)
    MLVC_RESOLVE_WINML_FN(WinMLEpCatalogRelease)
    MLVC_RESOLVE_WINML_FN(WinMLEpCatalogEnumProviders)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetNameSize)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetName)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetVersionSize)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetVersion)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetPackageFamilyNameSize)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetPackageFamilyName)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetLibraryPathSize)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetLibraryPath)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetReadyState)
    MLVC_RESOLVE_WINML_FN(WinMLEpGetCertification)
    MLVC_RESOLVE_WINML_FN(WinMLEpEnsureReady)
    MLVC_RESOLVE_WINML_FN(WinMLEpEnsureReadyAsync)
    MLVC_RESOLVE_WINML_FN(WinMLAsyncGetStatus)
    MLVC_RESOLVE_WINML_FN(WinMLAsyncCancel)
    MLVC_RESOLVE_WINML_FN(WinMLAsyncClose)

#undef MLVC_RESOLVE_WINML_FN
    return {};
}

expected<LoadedWinMlModules> LoadWinMlModules(WinMlInitMode mode)
{
    auto dirResult = ResolveDllDir(mode);
    if (!dirResult) {
        return make_unexpected(dirResult.error());
    }
    const std::wstring& dir = *dirResult;

    auto hOnnxRuntime = LoadDllFromDir(dir, kOnnxRuntimeDllName);
    if (!hOnnxRuntime) {
        return make_unexpected(hOnnxRuntime.error());
    }

    auto hWinML = LoadDllFromDir(dir, kWinMLDllName);
    if (!hWinML) {
        return make_unexpected(hWinML.error());
    }

    LoadedWinMlModules result;
    result.hOnnxRuntime = std::move(*hOnnxRuntime);
    result.winMlApi.hWinML = std::move(*hWinML);
    if (auto ret = ResolveWinMlApiFunctions(result.winMlApi.hWinML.get(), result.winMlApi); !ret) {
        return make_unexpected(ret.error());
    }
    return result;
}

std::string GetWindowsAppRuntimeVersion(WinMlInitMode mode)
{
    // cant call winrt apis in self-contained mode since the Windows App Runtime may not be present at all,
    // so just return empty string
    if (mode == WinMlInitMode::SelfContained) {
        return {};
    }
    try {
        return winrt::to_string(winrt::Microsoft::Windows::ApplicationModel::WindowsAppRuntime::RuntimeInfo::AsString());
    } catch (const winrt::hresult_error& ex) {
        MLVC_LOG_WARN("Failed to get Windows App Runtime version: %s", winrt::to_string(ex.message()).c_str());
        return {};
    }
}

namespace {

// Read a variable-length string out of an ep handle using the (GetSize, Get) C-API pair.
template <typename GetSizeFn, typename GetFn>
static std::string ReadWinMlEpString(WinMLEpHandle ep, GetSizeFn getSize, GetFn get)
{
    size_t size = 0;
    if (FAILED(getSize(ep, &size)) || size == 0) {
        return {};
    }
    std::string buf(size, '\0');
    size_t used = 0;
    if (FAILED(get(ep, size, buf.data(), &used))) {
        return {};
    }
    // `used` is the byte count written, including the null terminator. Resize to drop it.
    buf.resize(used > 0 ? used - 1 : 0);
    return buf;
}

static std::wstring Utf8ToWide(std::string_view utf8)
{
    if (utf8.empty()) return {};
    const int byteLen = static_cast<int>(utf8.size());
    const int wideLen = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), byteLen, nullptr, 0);
    if (wideLen <= 0) return {};
    std::wstring wide(static_cast<size_t>(wideLen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), byteLen, wide.data(), wideLen);
    return wide;
}

static BOOL CALLBACK LogProviderCallback([[maybe_unused]] WinMLEpHandle ep, const WinMLEpInfo* info, void* context)
{
    auto* indexPtr = static_cast<int*>(context);
    int index = indexPtr ? (*indexPtr)++ : -1;
    MLVC_LOG_INFO("  [%d] %s (version=%s, ReadyState=%s(%d), Certification=%s(%d), packageFamily=%s, libraryPath='%s')",
                  index, info && info->name ? info->name : "(null)", info && info->version ? info->version : "(null)",
                  ReadyStateToString(info ? info->readyState : WinMLEpReadyState_NotPresent),
                  static_cast<int>(info ? info->readyState : WinMLEpReadyState_NotPresent),
                  CertificationToString(info ? info->certification : WinMLEpCertification_Unknown),
                  static_cast<int>(info ? info->certification : WinMLEpCertification_Unknown),
                  info && info->packageFamilyName ? info->packageFamilyName : "(null)",
                  info && info->libraryPath ? info->libraryPath : "(null)");
    return TRUE;  // keep enumerating
}

struct FindProviderContext {
    const char* providerName = nullptr;
    bool readyOnly = false;
    int bestScore = -1;
    WinMLEpHandle best = nullptr;
};

static BOOL CALLBACK FindProviderCallback(WinMLEpHandle ep, const WinMLEpInfo* info, void* context)
{
    auto* ctx = static_cast<FindProviderContext*>(context);
    if (!info || !info->name || std::string_view{ info->name } != ctx->providerName) return TRUE;
    if (ctx->readyOnly && info->readyState != WinMLEpReadyState_Ready) return TRUE;
    const bool installed = info->readyState != WinMLEpReadyState_NotPresent;
    const bool ready = info->readyState == WinMLEpReadyState_Ready;
    // Higher is better: Ready > NotReady > NotPresent. Catalog order breaks ties.
    const int score = installed + ready;
    if (score > ctx->bestScore) {
        ctx->bestScore = score;
        ctx->best = ep;
    }
    return TRUE;  // keep enumerating
}

// Avoid WinMLEpCatalogFindProvider, which may return different matches across calls.
static HRESULT FindProvider(const WinMlApi& winMl, WinMLEpCatalogHandle catalog, const char* providerName,
                            bool readyOnly, WinMLEpHandle* ep)
{
    FindProviderContext ctx{ .providerName = providerName, .readyOnly = readyOnly };
    const HRESULT hr = winMl.WinMLEpCatalogEnumProviders(catalog, FindProviderCallback, &ctx);
    if (FAILED(hr)) return hr;
    if (!ctx.best) return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
    *ep = ctx.best;
    return S_OK;
}

struct EnsureReadyProgressContext {
    DownloadProgressCallback userCallback;
};

static void CALLBACK EnsureReadyProgress(WinMLAsyncBlock* async, double progress)
{
    if (!async || !async->context) return;
    auto* ctx = static_cast<EnsureReadyProgressContext*>(async->context);
    if (ctx->userCallback) {
        ctx->userCallback(progress);
    }
}

// Drives WinMLEpEnsureReadyAsync to completion, supporting progress reporting and cooperative
// cancellation. Returns success when the EP reaches Ready, or an error otherwise.
static expected<void> EnsureProviderReadyAsync(const WinMlApi& winMl, WinMLEpHandle ep, const char* providerName,
                                               WinMLEpReadyState startingState, CancelToken cancelToken,
                                               DownloadProgressCallback progressCallback)
{
    if (progressCallback) progressCallback(0.0);

    if (startingState == WinMLEpReadyState_NotPresent) {
        MLVC_LOG_INFO("Downloading execution provider '%s'...", providerName);
    } else {
        MLVC_LOG_DEBUG("Attaching execution provider '%s' to process...", providerName);
    }

    EnsureReadyProgressContext ctx{ progressCallback };
    WinMLAsyncBlock async{};
    async.context = &ctx;
    async.progress = progressCallback ? EnsureReadyProgress : nullptr;
    async.callback = nullptr;

    HRESULT hr = winMl.WinMLEpEnsureReadyAsync(ep, &async);
    if (FAILED(hr)) {
        MLVC_LOG_ERROR("WinMLEpEnsureReadyAsync failed for '%s' (HRESULT: 0x%08lX)", providerName,
                       static_cast<unsigned long>(hr));
        winMl.WinMLAsyncClose(&async);
        return make_unexpected(make_error_code(Error::ep_download_error));
    }

    // Poll for completion so we can respond to cancellation.
    while (true) {
        HRESULT status = winMl.WinMLAsyncGetStatus(&async, FALSE);
        if (status != E_PENDING) {
            hr = status;
            break;
        }
        if (cancelToken.IsCancelled()) {
            MLVC_LOG_INFO("Cancelling download for provider '%s'", providerName);
            winMl.WinMLAsyncCancel(&async);
            // Drain to completion so close is safe.
            winMl.WinMLAsyncGetStatus(&async, TRUE);
            winMl.WinMLAsyncClose(&async);
            return make_unexpected(make_error_code(Error::operation_cancelled));
        }
        std::this_thread::sleep_for(ASYNC_POLL_INTERVAL);
    }

    winMl.WinMLAsyncClose(&async);

    if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED) || hr == E_ABORT) {
        MLVC_LOG_INFO("Download was cancelled for provider '%s'", providerName);
        return make_unexpected(make_error_code(Error::operation_cancelled));
    }
    if (FAILED(hr)) {
        MLVC_LOG_ERROR("EnsureReadyAsync failed for provider '%s' (HRESULT: 0x%08lX)", providerName,
                       static_cast<unsigned long>(hr));
        return make_unexpected(make_error_code(Error::ep_download_error));
    }

    if (startingState == WinMLEpReadyState_NotPresent) {
        MLVC_LOG_INFO("Successfully downloaded execution provider '%s'", providerName);
    } else {
        MLVC_LOG_DEBUG("Successfully attached execution provider '%s'", providerName);
    }
    if (progressCallback) progressCallback(100.0);
    return {};
}

}  // anonymous namespace

expected<EnsureExecutionProviderReadyResult> EnsureExecutionProviderReady(const WinMlApi& winMl, const OrtApi* ortApi,
                                                                          OrtEnv* ortEnv, OnnxExecutionProvider provider,
                                                                          CancelToken cancelToken,
                                                                          DownloadProgressCallback progressCallback)
{
    const char* providerName = OnnxExecutionProviderToOrtName(provider);
    const char* providerLogName = providerName ? providerName : OnnxExecutionProviderToString(provider);
    if (!providerName) {
        MLVC_LOG_ERROR("Provider %s does not support EnsureExecutionProviderReady", OnnxExecutionProviderToString(provider));
        return make_unexpected(make_error_code(Error::invalid_argument));
    }
    if (!winMl.WinMLEpCatalogCreate || !ortApi || !ortEnv) {
        MLVC_LOG_ERROR("EnsureExecutionProviderReady called with uninitialized WinML/ORT state");
        return make_unexpected(make_error_code(Error::invalid_argument));
    }

    // Create WindowsML EP catalog
    WinMLEpCatalogHandle catalogRaw = nullptr;
    HRESULT hr = winMl.WinMLEpCatalogCreate(&catalogRaw);
    if (FAILED(hr) || !catalogRaw) {
        MLVC_LOG_ERROR("WinMLEpCatalogCreate failed (HRESULT: 0x%08lX)", static_cast<unsigned long>(hr));
        return make_unexpected(make_error_code(Error::ep_register_error));
    }
    using CatalogPtr = std::unique_ptr<std::remove_pointer_t<WinMLEpCatalogHandle>, decltype(&::WinMLEpCatalogRelease)>;
    CatalogPtr catalog(catalogRaw, winMl.WinMLEpCatalogRelease);

    // Log every provider currently visible in the catalog, for diagnostics.
    {
        MLVC_LOG_INFO("WindowsML execution providers catalog:");
        int enumIndex = 0;
        winMl.WinMLEpCatalogEnumProviders(catalog.get(), LogProviderCallback, &enumIndex);
    }

    // Find the requested provider, download, and ensure it's ready for use
    std::optional<uint32_t> downloadTimeMs;
    {
        WinMLEpHandle epRaw = nullptr;
        hr = FindProvider(winMl, catalog.get(), providerName, /*readyOnly=*/false, &epRaw);
        if (FAILED(hr)) {
            MLVC_LOG_ERROR("FindProvider failed for '%s' (HRESULT: 0x%08lX)", providerLogName,
                           static_cast<unsigned long>(hr));
            return make_unexpected(make_error_code(Error::ep_download_error));
        }

        WinMLEpReadyState readyState = WinMLEpReadyState_NotPresent;
        if (FAILED(winMl.WinMLEpGetReadyState(epRaw, &readyState))) {
            MLVC_LOG_ERROR("WinMLEpGetReadyState failed for '%s'", providerLogName);
            return make_unexpected(make_error_code(Error::ep_download_error));
        }

        if (cancelToken.IsCancelled()) {
            return make_unexpected(make_error_code(Error::operation_cancelled));
        }

        if (readyState != WinMLEpReadyState_Ready) {
            MLVC_LOG_INFO("Ensuring execution provider '%s' is ready (current ReadyState=%s)", providerLogName,
                          ReadyStateToString(readyState));
            const bool wasDownload = (readyState == WinMLEpReadyState_NotPresent);
            const auto startTime = std::chrono::steady_clock::now();
            if (auto ret =
                    EnsureProviderReadyAsync(winMl, epRaw, providerLogName, readyState, cancelToken, progressCallback);
                !ret) {
                return make_unexpected(ret.error());
            }
            if (wasDownload) {
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
                downloadTimeMs = static_cast<uint32_t>(std::min<int64_t>(elapsed, std::numeric_limits<uint32_t>::max()));
            }
        } else {
            MLVC_LOG_DEBUG("Execution provider '%s' is already ready", providerLogName);
            if (progressCallback) progressCallback(100.0);
        }
    }

    // Register the provider with onnxruntime via OrtApi::RegisterExecutionProviderLibrary
    std::string epPackageVersion;
    {
        WinMLEpHandle epRaw = nullptr;
        hr = FindProvider(winMl, catalog.get(), providerName, /*readyOnly=*/true, &epRaw);
        if (FAILED(hr)) {
            MLVC_LOG_ERROR("FindProvider failed for '%s' (HRESULT: 0x%08lX)", providerLogName,
                           static_cast<unsigned long>(hr));
            return make_unexpected(make_error_code(Error::ep_register_error));
        }

        epPackageVersion = ReadWinMlEpString(epRaw, winMl.WinMLEpGetVersionSize, winMl.WinMLEpGetVersion);
        std::string packageFamily =
            ReadWinMlEpString(epRaw, winMl.WinMLEpGetPackageFamilyNameSize, winMl.WinMLEpGetPackageFamilyName);
        std::string libraryPathUtf8 =
            ReadWinMlEpString(epRaw, winMl.WinMLEpGetLibraryPathSize, winMl.WinMLEpGetLibraryPath);
        if (libraryPathUtf8.empty()) {
            MLVC_LOG_ERROR("WinMLEpGetLibraryPath returned empty path for '%s'", providerLogName);
            return make_unexpected(make_error_code(Error::ep_register_error));
        }
        std::wstring libraryPathWide = Utf8ToWide(libraryPathUtf8);
        if (libraryPathWide.empty()) {
            MLVC_LOG_ERROR("Failed to convert library path to wide string for '%s'", providerLogName);
            return make_unexpected(make_error_code(Error::ep_register_error));
        }

        MLVC_LOG_INFO(
            "Registering execution provider '%s' (version: '%s', package_family: '%s') from '%s' via "
            "OrtApi::RegisterExecutionProviderLibrary",
            providerName, epPackageVersion.c_str(), packageFamily.c_str(), libraryPathUtf8.c_str());

        OrtStatus* status = ortApi->RegisterExecutionProviderLibrary(ortEnv, providerName, libraryPathWide.c_str());
        if (status) {
            const char* msg = ortApi->GetErrorMessage(status);
            MLVC_LOG_ERROR("RegisterExecutionProviderLibrary failed for '%s': %s", providerName, msg ? msg : "(unknown)");
            ortApi->ReleaseStatus(status);
            return make_unexpected(make_error_code(Error::ep_register_error));
        }
    }

    return EnsureExecutionProviderReadyResult{
        .epPackageVersion = std::move(epPackageVersion),
        .downloadTimeMs = downloadTimeMs,
    };
}

const char* OnnxExecutionProviderToOrtName(const OnnxExecutionProvider onnxExecutionProvider)
{
    switch (onnxExecutionProvider) {
    case OnnxExecutionProvider::CPU:
        return "CPUExecutionProvider";
    case OnnxExecutionProvider::DIRECTML:
        return "DmlExecutionProvider";
    case OnnxExecutionProvider::QNN:
        return "QNNExecutionProvider";
    case OnnxExecutionProvider::OPENVINO:
        return "OpenVINOExecutionProvider";
    case OnnxExecutionProvider::TENSOR_RT:
        return "NvTensorRTRTXExecutionProvider";
    default:
        return nullptr;
    }
}

}  // namespace libmlvc_winml
