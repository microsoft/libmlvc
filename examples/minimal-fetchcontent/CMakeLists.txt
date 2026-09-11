# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.30)
project(libmlvc_example LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

include(FetchContent)

set(BOOST_INCLUDE_LIBRARIES json)
FetchContent_Declare(
    Boost
    URL
        https://github.com/boostorg/boost/releases/download/boost-1.91.0-1/boost-1.91.0-1-cmake.tar.xz
    URL_HASH
        SHA256=cc5dc5006ecbdf0051f90979be31b4eee5987d9ae14ae9fb9c03cfa43fa3cdad
    EXCLUDE_FROM_ALL
)
FetchContent_Declare(
    libmlvc
    GIT_REPOSITORY https://github.com/microsoft/libmlvc.git
    GIT_TAG main # Replace main with a tag or commit to pin the dependency
)
FetchContent_MakeAvailable(Boost libmlvc)

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
