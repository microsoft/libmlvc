// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/version.hpp"

#include <libmlvc/build_info.hpp>

namespace libmlvc {

const BuildInfo& GetBuildInfo()
{
    static const auto info = []() {
        return BuildInfo{
            .libmlvcVersion = LIBMLVC_VERSION_STRING,
            .gitHash = LIBMLVC_GIT_HASH,
            .gitShortHash = LIBMLVC_GIT_SHORT_HASH,
            .gitBranch = LIBMLVC_GIT_BRANCH,
        };
    }();
    return info;
}

}  // namespace libmlvc
