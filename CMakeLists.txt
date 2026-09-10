# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.30)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------

option(LIBMLVC_BUILD_TOOLS "Build the mlvc command-line tool" OFF)
option(LIBMLVC_BUILD_TESTS "Build libmlvc tests" OFF)

project(libmlvc VERSION 0.1.0 LANGUAGES CXX)
include(cmake/GitInfo.cmake)

# ---------------------------------------------------------------------------
# Git hooks (pre-commit)
# ---------------------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL AND IS_DIRECTORY "${PROJECT_SOURCE_DIR}/.git")
    find_program(PRE_COMMIT_EXECUTABLE pre-commit)
    if(PRE_COMMIT_EXECUTABLE)
        execute_process(
            COMMAND "${PRE_COMMIT_EXECUTABLE}" install
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE _pre_commit_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(_pre_commit_result EQUAL 0)
            message(STATUS "Installed pre-commit git hooks")
        else()
            message(
                WARNING
                "'pre-commit install' failed; git hooks not installed"
            )
        endif()
    else()
        message(
            WARNING
            "pre-commit not found in PATH; git hooks not installed. "
            "Install it with: uv tool install pre-commit"
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

if(NOT TARGET Boost::json)
    find_package(Boost CONFIG REQUIRED COMPONENTS json)
endif()

if((LIBMLVC_BUILD_TOOLS OR LIBMLVC_BUILD_TESTS) AND NOT TARGET ZLIB::ZLIB)
    find_package(ZLIB REQUIRED)
endif()

if(LIBMLVC_BUILD_TESTS)
    if(NOT TARGET GTest::gtest)
        find_package(GTest CONFIG REQUIRED)
    endif()
    if(NOT TARGET benchmark::benchmark)
        find_package(benchmark CONFIG REQUIRED)
    endif()
endif()

if(WIN32 AND NOT TARGET WindowsAppSdk::WindowsAppSdk)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    include(WindowsAppSdk)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------

add_subdirectory(src)

# ---------------------------------------------------------------------------
# Tools and tests
# ---------------------------------------------------------------------------

if(LIBMLVC_BUILD_TOOLS OR LIBMLVC_BUILD_TESTS)
    add_subdirectory(tools)
endif()

if(LIBMLVC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
