// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/utils.hpp"

#include <gtest/gtest.h>

using namespace libmlvc;

//------------------------------------------------------------------------------
// VersionAtLeast
//------------------------------------------------------------------------------

TEST(UnitTestUtils, VersionAtLeast_Comparison)
{
    EXPECT_TRUE(VersionAtLeast("32.0.100.4778", "32.0.100.4778"));   // equal
    EXPECT_TRUE(VersionAtLeast("32.0.100.4779", "32.0.100.4778"));   // higher
    EXPECT_FALSE(VersionAtLeast("32.0.100.4512", "32.0.100.4778"));  // lower
    // Numeric, not lexicographic: "4512" must rank below "4778".
    EXPECT_TRUE(VersionAtLeast("32.0.10.0", "32.0.9.0"));
}

TEST(UnitTestUtils, VersionAtLeast_EdgeCases)
{
    EXPECT_TRUE(VersionAtLeast("32.0", "32.0.0.0"));                   // missing components -> 0
    EXPECT_FALSE(VersionAtLeast("", "32.0.100.4778"));                 // empty actual
    EXPECT_TRUE(VersionAtLeast("32.0.100.4778abc", "32.0.100.4778"));  // parses up to non-digit
}

TEST(UnitTestUtils, VersionAtLeast_EpThresholds)
{
    EXPECT_TRUE(VersionAtLeast("1.8.79.0", "1.8.79.0"));    // OpenVINO at minimum
    EXPECT_FALSE(VersionAtLeast("1.8.78.0", "1.8.79.0"));   // OpenVINO below minimum
    EXPECT_TRUE(VersionAtLeast("1.8.30.0", "1.8.30.0"));    // QNN at minimum
    EXPECT_FALSE(VersionAtLeast("1.8.29.99", "1.8.30.0"));  // QNN below minimum
}
