// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/test_data.hpp"

#include <cstdlib>

namespace libmlvc {

std::filesystem::path GetTestDataDir()
{
    static const std::filesystem::path dir = [] {
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif
        const char* testDataDir = std::getenv("LIBMLVC_TEST_DATA_DIR");
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif
        return testDataDir && testDataDir[0] != '\0' ? std::filesystem::path{ testDataDir }
                                                     : std::filesystem::path{ "./data/test_data" };
    }();
    return dir;
}

}  // namespace libmlvc
