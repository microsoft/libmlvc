# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_executable(mlvc)
target_sources(
    mlvc
    PRIVATE
        arguments.cpp
        arguments.hpp
        benchmark_runner.cpp
        benchmark_runner.hpp
        main.cpp
        reporting.cpp
        reporting.hpp
)
target_link_libraries(mlvc PRIVATE libmlvc libmlvc_support Boost::json)

if(WIN32)
    # Copy runtime DLLs next to the exe so it can run directly from the build dir
    add_custom_command(
        TARGET mlvc
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_PROPERTY:libmlvc,LIBMLVC_RUNTIME_FILES>"
            "$<TARGET_RUNTIME_DLLS:mlvc>" "$<TARGET_FILE_DIR:mlvc>"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLLs next to mlvc"
        VERBATIM
    )
endif()

if(PROJECT_IS_TOP_LEVEL)
    install(TARGETS mlvc)
endif()
