# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_library(libmlvc_support STATIC)
set_target_properties(libmlvc_support PROPERTIES OUTPUT_NAME mlvc_support)
target_compile_features(libmlvc_support PUBLIC cxx_std_20)
target_sources(
    libmlvc_support
    PUBLIC
        FILE_SET headers
            TYPE HEADERS
            BASE_DIRS ..
            FILES
                compress.hpp
                dataset.hpp
                encoder_overrides.hpp
                hash.hpp
                interop.hpp
                metrics.hpp
                mlvc_io.hpp
                serialization.hpp
                test_data.hpp
                validate.hpp
                video_io.hpp
    PRIVATE
        compress.cpp
        dataset.cpp
        hash.cpp
        interop.cpp
        metrics.cpp
        mkv_writer.cpp
        mkv_writer.hpp
        mlvc_io.cpp
        serialization.cpp
        test_data.cpp
        validate.cpp
        video_io.cpp
)
target_link_libraries(
    libmlvc_support
    PUBLIC libmlvc
    PRIVATE Boost::json ZLIB::ZLIB $<$<PLATFORM_ID:Windows>:bcrypt>
)
