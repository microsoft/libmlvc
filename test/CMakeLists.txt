# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ---------------------------------------------------------------------------
# Test sources and dependencies
# ---------------------------------------------------------------------------

set(LIBMLVC_TEST_SOURCES
    support/test_config.cpp
    support/test_config.hpp
    unit/test_mlvc_io.cpp
    unit/test_video_io.cpp
    integration/test_manager.cpp
    integration/test_validation.cpp
    integration/test_interop.cpp
    test_main.cpp
)

if(NOT BUILD_SHARED_LIBS)
    list(
        APPEND LIBMLVC_TEST_SOURCES
        support/recovery_controller.hpp
        support/rans_test_pmf.cpp
        support/rans_test_pmf.hpp
        unit/test_bit_io.cpp
        unit/test_bitstream_spec_v0.cpp
        unit/test_gop_manager.cpp
        unit/test_model_cache.cpp
        unit/test_rans_coder.cpp
        unit/test_ref_manager.cpp
        unit/test_scale_decoder.cpp
        unit/test_transforms.cpp
        unit/test_utils.cpp
        integration/test_bitexact.cpp
        integration/test_codec_features.cpp
        benchmark/bench_entropy.cpp
        benchmark/bench_tensor.cpp
        benchmark/bench_transforms.cpp
    )
endif()

source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    PREFIX "Tests"
    FILES ${LIBMLVC_TEST_SOURCES}
)

set(LIBMLVC_TEST_LIBRARIES
    libmlvc
    libmlvc_support
    benchmark::benchmark
    Boost::json
)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------

if(NOT TARGET gtest_libmlvc) # Alternate build integrations may define this target.
    add_executable(gtest_libmlvc)
    target_compile_features(gtest_libmlvc PRIVATE cxx_std_20)
    target_sources(gtest_libmlvc PRIVATE ${LIBMLVC_TEST_SOURCES})
    target_link_libraries(
        gtest_libmlvc
        PRIVATE ${LIBMLVC_TEST_LIBRARIES} GTest::gtest
    )
    target_compile_definitions(
        gtest_libmlvc
        PRIVATE LIBMLVC_PROVIDE_TESTMAIN_ENTRY
    )

    include(GoogleTest)
    gtest_discover_tests(gtest_libmlvc DISCOVERY_MODE PRE_TEST)
endif()

target_include_directories(gtest_libmlvc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT BUILD_SHARED_LIBS)
    target_include_directories(gtest_libmlvc PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()

if(WIN32)
    # Copy runtime DLLs next to the exe so it can run directly from the build dir
    add_custom_command(
        TARGET gtest_libmlvc
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_PROPERTY:libmlvc,LIBMLVC_RUNTIME_FILES>"
            "$<TARGET_RUNTIME_DLLS:gtest_libmlvc>"
            "$<TARGET_FILE_DIR:gtest_libmlvc>"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLLs next to gtest_libmlvc"
        VERBATIM
    )
endif()
