# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.30)
project(libmlvc_example LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(libmlvc CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

add_executable(libmlvc_example main.cpp)
target_link_libraries(libmlvc_example PRIVATE libmlvc::libmlvc)

if(WIN32)
    # Copy runtime DLLs next to the exe so it can run directly from the build dir
    add_custom_command(
        TARGET libmlvc_example
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_RUNTIME_DLLS:libmlvc_example>"
            "$<TARGET_PROPERTY:libmlvc_example,LIBMLVC_RUNTIME_FILES>"
            "$<TARGET_FILE_DIR:libmlvc_example>"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLLs next to libmlvc_example"
        VERBATIM
    )
endif()
