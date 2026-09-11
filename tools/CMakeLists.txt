# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_subdirectory(libmlvc_support)

if(LIBMLVC_BUILD_TOOLS)
    add_subdirectory(mlvc)
endif()
