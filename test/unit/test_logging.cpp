// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/logging.hpp"

#include <gtest/gtest.h>

using namespace libmlvc;

TEST(UnitTestLogging, ToLogString)
{
    EXPECT_EQ(ToLogString({}), "");
    EXPECT_EQ(ToLogString(u8"\u65e5\U0001F600"), "\xe6\x97\xa5\xf0\x9f\x98\x80");
#if defined(_WIN32)
    EXPECT_EQ(ToLogString(L"\xD800"), "\xef\xbf\xbd");
#endif
}
