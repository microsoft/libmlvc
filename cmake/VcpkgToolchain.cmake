# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

option(
    LIBMLVC_ALLOW_VCPKG_DOWNLOADS
    "Allow downloading or bootstrapping missing vcpkg tooling"
    ON
)

if(NOT "$ENV{VCPKG_ROOT}" STREQUAL "")
    set(_vcpkg_root "$ENV{VCPKG_ROOT}")
elseif(NOT "$ENV{VCPKG_INSTALLATION_ROOT}" STREQUAL "")
    set(_vcpkg_root "$ENV{VCPKG_INSTALLATION_ROOT}")
else()
    set(_vcpkg_root "${CMAKE_CURRENT_LIST_DIR}/../.vcpkg")
    if(NOT EXISTS "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
        if(NOT LIBMLVC_ALLOW_VCPKG_DOWNLOADS)
            message(
                FATAL_ERROR
                "vcpkg was not found at ${_vcpkg_root}; downloads are disabled."
            )
        endif()
        find_package(Git REQUIRED)

        file(
            READ "${CMAKE_CURRENT_LIST_DIR}/../vcpkg-configuration.json"
            _vcpkg_cfg
        )
        string(
            JSON _vcpkg_repo
            GET "${_vcpkg_cfg}"
            "default-registry"
            "repository"
        )
        string(
            JSON _vcpkg_baseline
            GET "${_vcpkg_cfg}"
            "default-registry"
            "baseline"
        )

        message(
            STATUS
            "VcpkgToolchain.cmake: no vcpkg root environment variable set; cloning ${_vcpkg_repo} @ "
            "${_vcpkg_baseline} into ${_vcpkg_root}"
        )
        file(REMOVE_RECURSE "${_vcpkg_root}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" init --quiet "${_vcpkg_root}"
            RESULT_VARIABLE _vcpkg_rc
        )
        if(_vcpkg_rc EQUAL 0)
            execute_process(
                COMMAND
                    "${GIT_EXECUTABLE}" -C "${_vcpkg_root}" fetch --depth 1
                    "${_vcpkg_repo}" "${_vcpkg_baseline}"
                RESULT_VARIABLE _vcpkg_rc
            )
        endif()
        if(_vcpkg_rc EQUAL 0)
            execute_process(
                COMMAND
                    "${GIT_EXECUTABLE}" -C "${_vcpkg_root}" checkout --quiet
                    --detach FETCH_HEAD
                RESULT_VARIABLE _vcpkg_rc
            )
        endif()
        if(NOT _vcpkg_rc EQUAL 0)
            file(REMOVE_RECURSE "${_vcpkg_root}")
            message(
                FATAL_ERROR
                "VcpkgToolchain.cmake: failed to clone vcpkg (${_vcpkg_repo} @ ${_vcpkg_baseline}). "
                "Check network access, or set VCPKG_ROOT or VCPKG_INSTALLATION_ROOT "
                "to a bootstrapped vcpkg."
            )
        endif()
    endif()
endif()

if(NOT EXISTS "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
    message(
        FATAL_ERROR
        "VcpkgToolchain.cmake: vcpkg toolchain not found at '${_vcpkg_root}'. "
        "Set VCPKG_ROOT or VCPKG_INSTALLATION_ROOT to a valid vcpkg checkout."
    )
endif()

file(REAL_PATH "${_vcpkg_root}" _vcpkg_selected_root)
if(DEFINED CACHE{_LIBMLVC_VCPKG_ROOT})
    file(REAL_PATH "${_LIBMLVC_VCPKG_ROOT}" _vcpkg_cached_root)
    if(CMAKE_HOST_WIN32)
        string(TOLOWER "${_vcpkg_selected_root}" _vcpkg_selected_root)
        string(TOLOWER "${_vcpkg_cached_root}" _vcpkg_cached_root)
    endif()
    if(NOT _vcpkg_selected_root STREQUAL _vcpkg_cached_root)
        message(
            FATAL_ERROR
            "VcpkgToolchain.cmake: vcpkg root changed from '${_LIBMLVC_VCPKG_ROOT}' to '${_vcpkg_root}'. "
            "Reconfigure with cmake --fresh to discard the cached vcpkg root."
        )
    endif()
else()
    set(_LIBMLVC_VCPKG_ROOT
        "${_vcpkg_selected_root}"
        CACHE INTERNAL
        "Vcpkg root selected for this build tree"
    )
endif()
list(
    APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    LIBMLVC_ALLOW_VCPKG_DOWNLOADS
    _LIBMLVC_VCPKG_ROOT
)

if(CMAKE_HOST_WIN32)
    set(_vcpkg_executable "${_vcpkg_root}/vcpkg.exe")
else()
    set(_vcpkg_executable "${_vcpkg_root}/vcpkg")
endif()
if(NOT LIBMLVC_ALLOW_VCPKG_DOWNLOADS AND NOT EXISTS "${_vcpkg_executable}")
    message(
        FATAL_ERROR
        "vcpkg executable was not found at ${_vcpkg_executable}; downloads are disabled."
    )
endif()
include("${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
