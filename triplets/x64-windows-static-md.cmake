# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

# Keep in sync with the msvc-hardening preset
set(VCPKG_C_FLAGS "/W3 /guard:cf /Qspectre")
set(VCPKG_CXX_FLAGS "/W3 /guard:cf /Qspectre")
