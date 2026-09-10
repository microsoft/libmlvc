# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(DEFINED ENV{LIBMLVC_SOURCE_PATH})
    set(SOURCE_PATH "$ENV{LIBMLVC_SOURCE_PATH}")
else()
    vcpkg_from_git(
        OUT_SOURCE_PATH SOURCE_PATH
        URL https://github.com/microsoft/libmlvc.git
        # TODO: Update to the first public source commit.
        REF 00e3fb60f2d5b240bd701a6492a2b88408ad47a8
        FETCH_REF main
        HEAD_REF main
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME libmlvc CONFIG_PATH share/cmake/libmlvc)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    set(VCPKG_POLICY_DLLS_IN_STATIC_LIBRARY enabled)
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
