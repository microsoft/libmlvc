// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/utils.hpp"

#include <cstdint>

namespace libmlvc {

bool VersionAtLeast(std::string_view actual, std::string_view required)
{
    auto nextComponent = [](std::string_view& s) -> uint64_t {
        if (s.empty()) return 0;
        const size_t dot = s.find('.');
        const std::string_view tok = s.substr(0, dot);
        s = (dot == std::string_view::npos) ? std::string_view{} : s.substr(dot + 1);
        uint64_t v = 0;
        for (const char c : tok) {
            if (c < '0' || c > '9') break;
            v = v * 10 + static_cast<uint64_t>(c - '0');
        }
        return v;
    };

    while (!actual.empty() || !required.empty()) {
        const uint64_t a = nextComponent(actual);
        const uint64_t r = nextComponent(required);
        if (a != r) return a > r;
    }
    return true;  // all components equal
}

}  // namespace libmlvc
