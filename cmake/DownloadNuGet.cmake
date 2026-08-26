# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NuGet package helper for standalone builds.
# Usage: download_nuget_package(PACKAGE_NAME VERSION OUTPUT_DIR)

function(download_nuget_package PACKAGE_NAME VERSION OUTPUT_DIR)
    set(PACKAGE_URL
        "https://www.nuget.org/api/v2/package/${PACKAGE_NAME}/${VERSION}"
    )
    set(OUTPUT_PATH "${OUTPUT_DIR}/${PACKAGE_NAME}.${VERSION}")

    if(NOT EXISTS "${OUTPUT_PATH}")
        message(STATUS "Downloading ${PACKAGE_NAME} ${VERSION}...")

        # Download the NuGet package
        file(
            DOWNLOAD "${PACKAGE_URL}"
            "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}.nupkg"
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
        )

        list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
        if(NOT STATUS_CODE EQUAL 0)
            list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
            message(
                FATAL_ERROR
                "Failed to download ${PACKAGE_NAME}: ${ERROR_MESSAGE}"
            )
        endif()

        # Extract the package (NuGet packages are zip files)
        file(
            ARCHIVE_EXTRACT
            INPUT "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}.nupkg"
            DESTINATION "${OUTPUT_PATH}"
        )

        # Clean up downloaded package
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}.nupkg")

        message(STATUS "Extracted ${PACKAGE_NAME} to ${OUTPUT_PATH}")
    else()
        message(STATUS "Using cached ${PACKAGE_NAME} ${VERSION}")
    endif()
endfunction()
