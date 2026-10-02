# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ---------------------------------------------------------------------------
# Tool executable
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Build-tree runtime support
# ---------------------------------------------------------------------------

set(MLVC_TOOL_RUNTIME_FILES)
if(WIN32)
    list(
        APPEND MLVC_TOOL_RUNTIME_FILES
        "$<TARGET_RUNTIME_DLLS:mlvc>"
        "$<TARGET_PROPERTY:mlvc,LIBMLVC_RUNTIME_FILES>"
    )
endif()

if(MLVC_TOOL_RUNTIME_FILES)
    add_custom_command(
        TARGET mlvc
        POST_BUILD
        COMMAND
            "$<$<BOOL:$<JOIN:${MLVC_TOOL_RUNTIME_FILES},>>:${CMAKE_COMMAND};-E;copy_if_different;${MLVC_TOOL_RUNTIME_FILES};$<TARGET_FILE_DIR:mlvc>>"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime libraries next to mlvc"
        VERBATIM
    )
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
    install(TARGETS mlvc)
endif()
