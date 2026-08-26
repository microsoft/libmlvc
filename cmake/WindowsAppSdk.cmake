# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# WindowsAppSdk setup for standalone mlvc builds
# Find Windows App SDK packages and create IMPORTED target
#
# This module defines:
#   WindowsAppSdk::WindowsAppSdk - INTERFACE IMPORTED target bundling everything
#   WINDOWSAPPSDK_ML_DIR         - Path to ML package (for advanced use)
#   WINDOWSAPPSDK_FOUNDATION_DIR - Path to Foundation package (for advanced use)
#   WINDOWSAPPSDK_BOOTSTRAP_DLL  - Path to Bootstrap DLL for runtime deployment
#
# The target also sets INTERFACE_COMPILE_DEFINITIONS:
#   WINDOWSAPPSDK_VERSION_MAJOR  - Major version extracted from Foundation package
#   WINDOWSAPPSDK_VERSION_MINOR  - Minor version extracted from Foundation package
#
# Standalone builds use fixed Windows App SDK versions.
set(WINDOWSAPPSDK_ML_VERSION "1.8.2141")
set(WINDOWSAPPSDK_FOUNDATION_VERSION "1.8.260222000")

include(${CMAKE_CURRENT_LIST_DIR}/DownloadNuGet.cmake)
set(SDK_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/sdk")

# Create SDK directory
file(MAKE_DIRECTORY "${SDK_LOCATION}")

# Download the ML package (contains ONNX Runtime)
download_nuget_package(Microsoft.WindowsAppSDK.ML ${WINDOWSAPPSDK_ML_VERSION} "${SDK_LOCATION}")

# Download the Foundation package (contains bootstrapper)
download_nuget_package(Microsoft.WindowsAppSDK.Foundation ${WINDOWSAPPSDK_FOUNDATION_VERSION} "${SDK_LOCATION}")

message(STATUS "Windows App SDK packages ready in ${SDK_LOCATION}")

# Find cppwinrt.exe from Windows SDK (ensures ABI compatibility)
# Auto-detect SDK version since CMAKE_SYSTEM_VERSION may not match installed SDK
set(_progfiles_x86 "ProgramFiles(x86)")
set(WIN_SDK_BIN_BASE "$ENV{${_progfiles_x86}}/Windows Kits/10/bin")
file(GLOB SDK_VERSION_DIRS "${WIN_SDK_BIN_BASE}/10.*")
if(SDK_VERSION_DIRS)
    list(SORT SDK_VERSION_DIRS)
    list(GET SDK_VERSION_DIRS -1 LATEST_SDK_DIR)
    get_filename_component(WIN_SDK_VERSION "${LATEST_SDK_DIR}" NAME)
else()
    message(FATAL_ERROR "No Windows SDK versions found in ${WIN_SDK_BIN_BASE}")
endif()
set(WIN_SDK_BIN "${WIN_SDK_BIN_BASE}/${WIN_SDK_VERSION}/x64")
find_program(CPPWINRT_EXE cppwinrt.exe PATHS "${WIN_SDK_BIN}" NO_DEFAULT_PATH)
if(CPPWINRT_EXE)
    message(
        STATUS
        "Using cppwinrt.exe from Windows SDK ${WIN_SDK_VERSION}: ${CPPWINRT_EXE}"
    )
else()
    message(
        FATAL_ERROR
        "cppwinrt.exe not found in Windows SDK at: ${WIN_SDK_BIN}"
    )
endif()

# Set package directories directly (paths are deterministic after download)
set(WINDOWSAPPSDK_ML_DIR
    "${SDK_LOCATION}/Microsoft.WindowsAppSDK.ML.${WINDOWSAPPSDK_ML_VERSION}"
)
set(WINDOWSAPPSDK_FOUNDATION_DIR
    "${SDK_LOCATION}/Microsoft.WindowsAppSDK.Foundation.${WINDOWSAPPSDK_FOUNDATION_VERSION}"
)

# Locate Windows SDK UnionMetadata for type resolution (e.g. Windows.Foundation.Uri)
set(WIN_SDK_UNION_METADATA
    "$ENV{${_progfiles_x86}}/Windows Kits/10/UnionMetadata/${WIN_SDK_VERSION}"
)
if(NOT EXISTS "${WIN_SDK_UNION_METADATA}")
    message(
        FATAL_ERROR
        "Windows SDK UnionMetadata not found at: ${WIN_SDK_UNION_METADATA}"
    )
endif()

# Validate packages were downloaded
if(NOT EXISTS "${WINDOWSAPPSDK_ML_DIR}/include/winml/onnxruntime_c_api.h")
    message(
        FATAL_ERROR
        "Could NOT find Microsoft.WindowsAppSDK.ML at ${WINDOWSAPPSDK_ML_DIR}"
    )
endif()
if(NOT EXISTS "${WINDOWSAPPSDK_FOUNDATION_DIR}/include/MddBootstrap.h")
    message(
        FATAL_ERROR
        "Could NOT find Microsoft.WindowsAppSDK.Foundation at ${WINDOWSAPPSDK_FOUNDATION_DIR}"
    )
endif()

message(STATUS "Windows App SDK ready:")
message(STATUS "  ML: ${WINDOWSAPPSDK_ML_DIR}")
message(STATUS "  Foundation: ${WINDOWSAPPSDK_FOUNDATION_DIR}")

# Determine architecture
set(WINDOWSAPPSDK_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_VS_PLATFORM_NAME)
    set(WINDOWSAPPSDK_TARGET_PROCESSOR "${CMAKE_VS_PLATFORM_NAME}")
endif()
if(WINDOWSAPPSDK_TARGET_PROCESSOR MATCHES "AMD64|amd64|x64|x86_64")
    set(WINDOWSAPPSDK_ARCH "x64")
    set(WINDOWSAPPSDK_RUNTIME_ARCH "win-x64")
elseif(
    WINDOWSAPPSDK_TARGET_PROCESSOR
        MATCHES
        "ARM64|arm64|aarch64|armv8A64|armv8a64"
)
    set(WINDOWSAPPSDK_ARCH "arm64")
    set(WINDOWSAPPSDK_RUNTIME_ARCH "win-arm64")
else()
    message(
        FATAL_ERROR
        "Unsupported target architecture: ${WINDOWSAPPSDK_TARGET_PROCESSOR}"
    )
endif()

set(WINDOWSAPPSDK_BOOTSTRAP_DLL
    "${WINDOWSAPPSDK_FOUNDATION_DIR}/runtimes/${WINDOWSAPPSDK_RUNTIME_ARCH}/native/Microsoft.WindowsAppRuntime.Bootstrap.dll"
)

# Extract major.minor from Foundation package version (e.g. "1.8.250906002" -> 1, 8)
# and expose as preprocessor defines via the INTERFACE target.
string(
    REGEX MATCH "^([0-9]+)\\.([0-9]+)"
    _ver_match
    "${WINDOWSAPPSDK_FOUNDATION_VERSION}"
)
set(WINDOWSAPPSDK_VERSION_MAJOR ${CMAKE_MATCH_1})
set(WINDOWSAPPSDK_VERSION_MINOR ${CMAKE_MATCH_2})
message(
    STATUS
    "Windows App SDK version: ${WINDOWSAPPSDK_VERSION_MAJOR}.${WINDOWSAPPSDK_VERSION_MINOR}"
)

# C++/WinRT generation output directory
set(WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR "${CMAKE_BINARY_DIR}/generated/cppwinrt")
set(WINMD_ML_PATH
    "${WINDOWSAPPSDK_ML_DIR}/metadata/Microsoft.Windows.AI.MachineLearning.winmd"
)
set(WINMD_FOUNDATION_APPRUNTIME_PATH
    "${WINDOWSAPPSDK_FOUNDATION_DIR}/metadata/Microsoft.Windows.ApplicationModel.WindowsAppRuntime.winmd"
)

# Create output directory at configure time so CMake doesn't complain about non-existent include paths
file(MAKE_DIRECTORY "${WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR}")

# Create custom command to generate WinRT headers
# Generate all headers (Windows SDK + custom metadata) in a single command
# This makes the build completely independent of system-installed cppwinrt versions
add_custom_command(
    OUTPUT
        "${WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR}/winrt/Microsoft.Windows.AI.MachineLearning.h"
        "${WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR}/winrt/Microsoft.Windows.ApplicationModel.WindowsAppRuntime.h"
    COMMAND
        "${CPPWINRT_EXE}" -input "${WINMD_ML_PATH}" -input
        "${WINMD_FOUNDATION_APPRUNTIME_PATH}" -ref "${WIN_SDK_UNION_METADATA}"
        -output "${WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR}" -verbose
    DEPENDS "${WINMD_ML_PATH}" "${WINMD_FOUNDATION_APPRUNTIME_PATH}"
    COMMENT
        "Generating C++/WinRT headers from Windows SDK and Windows App SDK metadata"
    VERBATIM
)

# Create a target for the generated headers
add_custom_target(
    libmlvc_windowsappsdk_cppwinrt_headers
    DEPENDS
        "${WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR}/winrt/Microsoft.Windows.AI.MachineLearning.h"
        "${WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR}/winrt/Microsoft.Windows.ApplicationModel.WindowsAppRuntime.h"
)

message(
    STATUS
    "C++/WinRT headers will be generated to: ${WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR}"
)

# Create the INTERFACE IMPORTED target
# Using IMPORTED so it doesn't need to be exported
add_library(WindowsAppSdk INTERFACE IMPORTED GLOBAL)
add_library(WindowsAppSdk::WindowsAppSdk ALIAS WindowsAppSdk)

# Include directories: generated C++/WinRT first, then SDK headers
set_target_properties(
    WindowsAppSdk
    PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "${WINDOWSAPPSDK_CPPWINRT_OUTPUT_DIR};${WINDOWSAPPSDK_ML_DIR}/include;${WINDOWSAPPSDK_FOUNDATION_DIR}/include"
        INTERFACE_LINK_LIBRARIES "oleaut32;RuntimeObject"
        INTERFACE_COMPILE_DEFINITIONS
            "WINDOWSAPPSDK_VERSION_MAJOR=${WINDOWSAPPSDK_VERSION_MAJOR};WINDOWSAPPSDK_VERSION_MINOR=${WINDOWSAPPSDK_VERSION_MINOR}"
)

# Make the target depend on C++/WinRT header generation
add_dependencies(WindowsAppSdk libmlvc_windowsappsdk_cppwinrt_headers)
