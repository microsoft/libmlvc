// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/export.hpp>

#include <string>

namespace libmlvc {

struct BuildInfo {
    std::string libmlvcVersion{};
    std::string gitHash{};
    std::string gitShortHash{};
    std::string gitBranch{};
};

LIBMLVC_EXPORT const BuildInfo& GetBuildInfo();

}  // namespace libmlvc
