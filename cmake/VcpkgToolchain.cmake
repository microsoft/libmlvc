# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(DEFINED ENV{VCPKG_ROOT})
    set(_vcpkg_root "$ENV{VCPKG_ROOT}")
else()
    set(_vcpkg_root "${CMAKE_CURRENT_LIST_DIR}/../.vcpkg")
    if(NOT EXISTS "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
        find_program(_vcpkg_git git)
        if(NOT _vcpkg_git)
            message(
                FATAL_ERROR
                "VcpkgToolchain.cmake: VCPKG_ROOT is not set and 'git' was not found to "
                "auto-clone vcpkg. Install git, or set VCPKG_ROOT to a bootstrapped vcpkg."
            )
        endif()

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
            "VcpkgToolchain.cmake: VCPKG_ROOT not set; cloning ${_vcpkg_repo} @ "
            "${_vcpkg_baseline} into ${_vcpkg_root}"
        )
        file(REMOVE_RECURSE "${_vcpkg_root}")
        execute_process(
            COMMAND "${_vcpkg_git}" init --quiet "${_vcpkg_root}"
            RESULT_VARIABLE _vcpkg_rc
        )
        if(_vcpkg_rc EQUAL 0)
            execute_process(
                COMMAND
                    "${_vcpkg_git}" -C "${_vcpkg_root}" remote add origin
                    "${_vcpkg_repo}"
                RESULT_VARIABLE _vcpkg_rc
            )
        endif()
        if(_vcpkg_rc EQUAL 0)
            execute_process(
                COMMAND
                    "${_vcpkg_git}" -C "${_vcpkg_root}" fetch --depth 1 origin
                    "${_vcpkg_baseline}"
                RESULT_VARIABLE _vcpkg_rc
            )
        endif()
        if(_vcpkg_rc EQUAL 0)
            execute_process(
                COMMAND
                    "${_vcpkg_git}" -C "${_vcpkg_root}" checkout --quiet
                    --detach FETCH_HEAD
                RESULT_VARIABLE _vcpkg_rc
            )
        endif()
        if(NOT _vcpkg_rc EQUAL 0)
            file(REMOVE_RECURSE "${_vcpkg_root}")
            message(
                FATAL_ERROR
                "VcpkgToolchain.cmake: failed to clone vcpkg (${_vcpkg_repo} @ ${_vcpkg_baseline}). "
                "Check network access, or set VCPKG_ROOT to a bootstrapped vcpkg."
            )
        endif()
    endif()
endif()

if(NOT EXISTS "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
    message(
        FATAL_ERROR
        "VcpkgToolchain.cmake: vcpkg toolchain not found at '${_vcpkg_root}'. "
        "Set VCPKG_ROOT to a valid vcpkg checkout."
    )
endif()
include("${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
