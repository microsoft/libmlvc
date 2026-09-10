// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/// @file
/// Build version and Git revision API.

#pragma once
#include <libmlvc/export.hpp>

#include <string>

namespace libmlvc {

/// Library version and Git revision compiled into libmlvc.
struct BuildInfo {
    std::string libmlvcVersion{};  ///< Semantic libmlvc version.
    std::string gitHash{};         ///< Full commit hash, or `unknown`.
    std::string gitShortHash{};    ///< Abbreviated commit hash, or `unknown`.
    std::string gitBranch{};       ///< Branch name, or `unknown`.
};

/// Returns version and Git revision information for the loaded libmlvc library.
/// The returned reference remains valid until the process exits.
LIBMLVC_EXPORT const BuildInfo& GetBuildInfo();

}  // namespace libmlvc
