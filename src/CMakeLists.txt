# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

include(CMakePackageConfigHelpers)

# ---------------------------------------------------------------------------
# Sources, headers, and dependencies
# ---------------------------------------------------------------------------

set(LIBMLVC_GENERATED_PRIVATE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/private)
file(MAKE_DIRECTORY ${LIBMLVC_GENERATED_PRIVATE_INCLUDE_DIR}/libmlvc)
configure_file(
    ../cmake/libmlvc_version.hpp.in
    ${LIBMLVC_GENERATED_PRIVATE_INCLUDE_DIR}/libmlvc/version.hpp
    @ONLY
)

set(LIBMLVC_PUBLIC_HEADERS
    ../include/libmlvc/libmlvc.hpp
    ../include/libmlvc/error_codes.hpp
    ../include/libmlvc/expected.hpp
    ../include/libmlvc/build_info.hpp
    ../include/libmlvc/platform_info.hpp
    ../include/libmlvc/types.hpp
    ../include/libmlvc/unexpected.hpp
    ../include/libmlvc/export.hpp
)

set(LIBMLVC_SOURCES
    libmlvc/bitstream/coder.hpp
    libmlvc/common/logging.hpp
    libmlvc/bitstream/bit_io.hpp
    libmlvc/common/macros.hpp
    libmlvc/common/fp16.hpp
    libmlvc/common/platform_names.hpp
    libmlvc/common/tar.hpp
    libmlvc/codec/codec.hpp
    libmlvc/entropy/rans_coder.hpp
    libmlvc/schema/pmf.hpp
    libmlvc/codec/bundle.hpp
    libmlvc/codec/manager.hpp
    libmlvc/common/tensor.hpp
    libmlvc/codec/core/core.hpp
    libmlvc/codec/core/inference_model.hpp
    libmlvc/codec/core/gop_manager.hpp
    libmlvc/codec/core/reference_manager.hpp
    libmlvc/codec/core/reference_data.hpp
    libmlvc/codec/core/transforms.hpp
    libmlvc/codec/core/frame_processing.hpp
    libmlvc/entropy/rans.hpp
    libmlvc/common/utils.hpp
    libmlvc/entropy/latent_coder.hpp
    libmlvc/codec/core/scale_decoder.hpp
    libmlvc/schema/bundle.hpp
    libmlvc/inference/interface.hpp
    libmlvc/inference/model_cache.hpp
    ${LIBMLVC_GENERATED_PRIVATE_INCLUDE_DIR}/libmlvc/version.hpp
    libmlvc/common/logging.cpp
    libmlvc/entropy/rans_coder.cpp
    libmlvc/schema/pmf.cpp
    libmlvc/common/utils.cpp
    libmlvc/platform_info.cpp
    libmlvc/common/tar.cpp
    libmlvc/entropy/latent_coder.cpp
    libmlvc/error_codes.cpp
    libmlvc/codec/core/inference_model.cpp
    libmlvc/codec/core/gop_manager.cpp
    libmlvc/codec/core/reference_manager.cpp
    libmlvc/codec/core/reference_data.cpp
    libmlvc/codec/core/transforms.cpp
    libmlvc/build_info.cpp
    libmlvc/bitstream/coder.cpp
    libmlvc/bitstream/bit_io.cpp
    libmlvc/codec/core/core.cpp
    libmlvc/codec/codec.cpp
    libmlvc/schema/bundle.cpp
    libmlvc/codec/core/scale_decoder.cpp
    libmlvc/codec/bundle.cpp
    libmlvc/codec/manager.cpp
    libmlvc/inference/model_cache.cpp
    libmlvc/libmlvc.cpp
)
set(LIBMLVC_PRIVATE_LINK_LIBRARIES Boost::json)
if(WIN32)
    list(
        APPEND LIBMLVC_SOURCES
        libmlvc/inference/winml/factory.hpp
        libmlvc/inference/winml/ort_helpers.hpp
        libmlvc/inference/winml/session.hpp
        libmlvc/inference/winml/engine.hpp
        libmlvc/inference/winml/bootstrap.hpp
        libmlvc/inference/winml/session.cpp
        libmlvc/inference/winml/engine.cpp
        libmlvc/inference/winml/bootstrap.cpp
    )
    list(
        APPEND LIBMLVC_PRIVATE_LINK_LIBRARIES
        $<BUILD_INTERFACE:WindowsAppSdk::WindowsAppSdk>
        $<INSTALL_INTERFACE:oleaut32>
        $<INSTALL_INTERFACE:RuntimeObject>
        advapi32
        setupapi
        ole32
    )
elseif(APPLE)
    list(
        APPEND LIBMLVC_SOURCES
        libmlvc/inference/coreml/factory.hpp
        libmlvc/inference/coreml/engine.mm
    )
    list(
        APPEND LIBMLVC_PRIVATE_LINK_LIBRARIES
        "-framework CoreML"
        "-framework CoreVideo"
        "-framework Foundation"
    )
endif()

# ---------------------------------------------------------------------------
# IDE source tree
# ---------------------------------------------------------------------------

set(LIBMLVC_SOURCE_TREE_FILES ${LIBMLVC_SOURCES})
list(
    REMOVE_ITEM LIBMLVC_SOURCE_TREE_FILES
    ${LIBMLVC_GENERATED_PRIVATE_INCLUDE_DIR}/libmlvc/version.hpp
)
source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}/libmlvc
    PREFIX "Source"
    FILES ${LIBMLVC_SOURCE_TREE_FILES}
)
source_group(
    TREE ${PROJECT_SOURCE_DIR}/include/libmlvc
    PREFIX "Public API"
    FILES ${LIBMLVC_PUBLIC_HEADERS}
)
source_group(
    "Generated"
    FILES ${LIBMLVC_GENERATED_PRIVATE_INCLUDE_DIR}/libmlvc/version.hpp
)

# ---------------------------------------------------------------------------
# Compile flags
# ---------------------------------------------------------------------------

set(LIBMLVC_SIMD_COMPILE_OPTIONS "")
set(LIBMLVC_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_VS_PLATFORM_NAME)
    set(LIBMLVC_TARGET_PROCESSOR "${CMAKE_VS_PLATFORM_NAME}")
endif()
if(MSVC)
    list(APPEND LIBMLVC_SIMD_COMPILE_OPTIONS $<$<NOT:$<CONFIG:Debug>>:/O2>)
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND LIBMLVC_SIMD_COMPILE_OPTIONS -O2)
endif()

if(LIBMLVC_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64|x86")
    if(MSVC)
        list(APPEND LIBMLVC_SIMD_COMPILE_OPTIONS /arch:AVX2)
    else()
        list(APPEND LIBMLVC_SIMD_COMPILE_OPTIONS -mavx2 -mfma)
    endif()
endif()

if(
    NOT MSVC
    AND NOT APPLE
    AND LIBMLVC_TARGET_PROCESSOR MATCHES "[Aa][Rr][Mm]64|[Aa][Aa][Rr][Cc][Hh]64"
)
    list(APPEND LIBMLVC_SIMD_COMPILE_OPTIONS -march=armv8.2-a+fp16+fp16fml)
endif()

if(LIBMLVC_SIMD_COMPILE_OPTIONS)
    set_source_files_properties(
        libmlvc/codec/core/transforms.cpp
        libmlvc/codec/core/scale_decoder.cpp
        PROPERTIES COMPILE_OPTIONS "${LIBMLVC_SIMD_COMPILE_OPTIONS}"
    )
endif()

# ---------------------------------------------------------------------------
# Main library
# ---------------------------------------------------------------------------

add_library(libmlvc)
add_library(libmlvc::libmlvc ALIAS libmlvc)
set_target_properties(
    libmlvc
    PROPERTIES
        OUTPUT_NAME mlvc
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        TRANSITIVE_LINK_PROPERTIES LIBMLVC_RUNTIME_FILES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
)
target_compile_features(libmlvc PUBLIC cxx_std_20)
target_include_directories(
    libmlvc
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBMLVC_GENERATED_PRIVATE_INCLUDE_DIR}
)
target_sources(
    libmlvc
    PUBLIC
        FILE_SET public_headers
            TYPE HEADERS
            BASE_DIRS ../include
            FILES ${LIBMLVC_PUBLIC_HEADERS}
    PRIVATE ${LIBMLVC_SOURCES}
)
target_link_libraries(libmlvc PRIVATE ${LIBMLVC_PRIVATE_LINK_LIBRARIES})

if(WIN32)
    get_target_property(
        _windowsappsdk_runtime_files
        WindowsAppSdk::WindowsAppSdk
        WINDOWSAPPSDK_RUNTIME_FILES
    )
    if(NOT _windowsappsdk_runtime_files AND WINDOWSAPPSDK_BOOTSTRAP_DLL)
        set(_windowsappsdk_runtime_files "${WINDOWSAPPSDK_BOOTSTRAP_DLL}")
    endif()
    if(_windowsappsdk_runtime_files)
        set(_libmlvc_install_runtime_files "")
        foreach(_runtime_file IN LISTS _windowsappsdk_runtime_files)
            get_filename_component(_runtime_name "${_runtime_file}" NAME)
            list(
                APPEND _libmlvc_install_runtime_files
                "$<INSTALL_PREFIX>/bin/${_runtime_name}"
            )
        endforeach()
        set_target_properties(
            libmlvc
            PROPERTIES
                LIBMLVC_RUNTIME_FILES "${_windowsappsdk_runtime_files}"
                INTERFACE_LIBMLVC_RUNTIME_FILES
                    "$<BUILD_INTERFACE:${_windowsappsdk_runtime_files}>$<INSTALL_INTERFACE:${_libmlvc_install_runtime_files}>"
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
    configure_package_config_file(
        ../cmake/libmlvcConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/libmlvcConfig.cmake
        INSTALL_DESTINATION share/cmake/libmlvc
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/libmlvcConfigVersion.cmake
        COMPATIBILITY SameMinorVersion
    )
    if(WIN32)
        install(
            FILES "$<TARGET_PROPERTY:libmlvc,LIBMLVC_RUNTIME_FILES>"
            DESTINATION bin
        )
    endif()
    install(TARGETS libmlvc EXPORT libmlvcTargets FILE_SET public_headers)
    install(
        EXPORT libmlvcTargets
        DESTINATION share/cmake/libmlvc
        NAMESPACE libmlvc::
    )
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/libmlvcConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/libmlvcConfigVersion.cmake
        DESTINATION share/cmake/libmlvc
    )
endif()
