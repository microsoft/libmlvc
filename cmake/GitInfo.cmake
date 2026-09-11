# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Detect git commit hash and branch at configure time.
# Sets LIBMLVC_GIT_HASH, LIBMLVC_GIT_SHORT_HASH, LIBMLVC_GIT_BRANCH.
# Falls back to "unknown" when git is unavailable or the source is not a git repo.

find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE LIBMLVC_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_hash_result
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE LIBMLVC_GIT_SHORT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_hash_short_result
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE LIBMLVC_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _git_branch_result
    )

    if(NOT _git_hash_result EQUAL 0)
        set(LIBMLVC_GIT_HASH "unknown")
    endif()
    if(NOT _git_hash_short_result EQUAL 0)
        set(LIBMLVC_GIT_SHORT_HASH "unknown")
    endif()
    if(NOT _git_branch_result EQUAL 0)
        set(LIBMLVC_GIT_BRANCH "unknown")
    endif()
else()
    set(LIBMLVC_GIT_HASH "unknown")
    set(LIBMLVC_GIT_SHORT_HASH "unknown")
    set(LIBMLVC_GIT_BRANCH "unknown")
endif()

message(STATUS "Git: ${LIBMLVC_GIT_SHORT_HASH} (${LIBMLVC_GIT_BRANCH})")
