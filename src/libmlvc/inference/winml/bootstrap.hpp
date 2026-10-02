// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "libmlvc/inference/interface.hpp"

#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <WinMLEpCatalog.h>
#include <Windows.h>

struct OrtApiBase;
struct OrtApi;
struct OrtEnv;

namespace libmlvc_winml {

using namespace libmlvc;

using HModulePtr = std::unique_ptr<std::remove_pointer_t<HMODULE>, std::function<void(HMODULE)>>;

struct EnsureExecutionProviderReadyResult {
    std::string epPackageVersion;
    std::optional<uint32_t> downloadTimeMs;
};

using DownloadProgressCallback = std::function<void(double progressPercentage)>;

// Function pointers resolved from Microsoft.Windows.AI.MachineLearning.dll. Loading the DLL
// dynamically (instead of linking against an import lib) lets us pick the load path at runtime --
// either from the WindowsAppRuntime framework package (AppSdk mode) or from the module directory
// next to the executable (SelfContained mode).
struct WinMlApi {
    HModulePtr hWinML;

    decltype(&::WinMLEpCatalogCreate) WinMLEpCatalogCreate = nullptr;
    decltype(&::WinMLEpCatalogRelease) WinMLEpCatalogRelease = nullptr;
    decltype(&::WinMLEpCatalogEnumProviders) WinMLEpCatalogEnumProviders = nullptr;
    decltype(&::WinMLEpGetNameSize) WinMLEpGetNameSize = nullptr;
    decltype(&::WinMLEpGetName) WinMLEpGetName = nullptr;
    decltype(&::WinMLEpGetVersionSize) WinMLEpGetVersionSize = nullptr;
    decltype(&::WinMLEpGetVersion) WinMLEpGetVersion = nullptr;
    decltype(&::WinMLEpGetPackageFamilyNameSize) WinMLEpGetPackageFamilyNameSize = nullptr;
    decltype(&::WinMLEpGetPackageFamilyName) WinMLEpGetPackageFamilyName = nullptr;
    decltype(&::WinMLEpGetLibraryPathSize) WinMLEpGetLibraryPathSize = nullptr;
    decltype(&::WinMLEpGetLibraryPath) WinMLEpGetLibraryPath = nullptr;
    decltype(&::WinMLEpGetReadyState) WinMLEpGetReadyState = nullptr;
    decltype(&::WinMLEpGetCertification) WinMLEpGetCertification = nullptr;
    decltype(&::WinMLEpEnsureReady) WinMLEpEnsureReady = nullptr;
    decltype(&::WinMLEpEnsureReadyAsync) WinMLEpEnsureReadyAsync = nullptr;
    decltype(&::WinMLAsyncGetStatus) WinMLAsyncGetStatus = nullptr;
    decltype(&::WinMLAsyncCancel) WinMLAsyncCancel = nullptr;
    decltype(&::WinMLAsyncClose) WinMLAsyncClose = nullptr;
};

struct LoadedWinMlModules {
    HModulePtr hOnnxRuntime;
    WinMlApi winMlApi;
};

// Loads onnxruntime.dll and Microsoft.Windows.AI.MachineLearning.dll.
//   AppSdk:        bootstrap WindowsAppRuntime (MddBootstrap / dynamic package dependency) and
//                  load both DLLs from the resolved framework package directory.
//   SelfContained: load both DLLs from the calling module's directory, without bootstrapping
//                  WindowsAppRuntime. The host is responsible for deploying both files alongside
//                  the binary.
// Vendor execution providers (OpenVINO, QNN, TensorRT) work in both modes -- the flat-C
// WinMLEpCatalog API exposed by the WinML DLL handles their discovery and download regardless
// of whether the WindowsAppRuntime has been bootstrapped.
expected<LoadedWinMlModules> LoadWinMlModules(WinMlInitMode mode);

// Returns the deployed WindowsAppRuntime framework version (e.g. "1.8.260222000") in AppSdk mode,
// or an empty string in SelfContained mode (no framework package present).
std::string GetWindowsAppRuntimeVersion(WinMlInitMode mode);
expected<const OrtApiBase*> GetOrtApiBaseFromModule(HMODULE hOnnxRuntime);

// Downloads (if needed), verifies readiness, and registers the execution provider library
// with ONNX Runtime via WinML's TryRegister. The EP is registered into the process-wide
// singleton OrtEnv.
// Returns the EP package version and, when the package had to be downloaded, the measured
// download duration.
expected<EnsureExecutionProviderReadyResult> EnsureExecutionProviderReady(const WinMlApi& winMl, const OrtApi* ortApi,
                                                                          OrtEnv* ortEnv, OnnxExecutionProvider provider,
                                                                          CancelToken cancelToken,
                                                                          DownloadProgressCallback progressCallback);

const char* OnnxExecutionProviderToOrtName(const OnnxExecutionProvider onnxExecutionProvider);

}  // namespace libmlvc_winml
