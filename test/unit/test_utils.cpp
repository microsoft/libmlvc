// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/utils.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <type_traits>

using namespace libmlvc;

//------------------------------------------------------------------------------
// ScopeExit
//------------------------------------------------------------------------------

TEST(UnitTestUtils, ScopeExit_RunsOnceOnScopeExit)
{
    auto calls = 0;
    const auto action = [&calls]() noexcept { ++calls; };
    {
        auto onExit = ScopeExit{ action };
        using Guard = decltype(onExit);
        static_assert(!std::is_copy_constructible_v<Guard>);
        static_assert(!std::is_copy_assignable_v<Guard>);
        static_assert(!std::is_move_constructible_v<Guard>);
        static_assert(!std::is_move_assignable_v<Guard>);
        static_assert(std::is_nothrow_destructible_v<Guard>);
        EXPECT_EQ(calls, 0);
    }
    EXPECT_EQ(calls, 1);
}

TEST(UnitTestUtils, ScopeExit_RunsOnEarlyReturn)
{
    auto calls = 0;
    const auto run = [&] {
        const auto onExit = ScopeExit{ [&calls]() noexcept { ++calls; } };
        return 42;
    };
    EXPECT_EQ(run(), 42);
    EXPECT_EQ(calls, 1);
}

TEST(UnitTestUtils, ScopeExit_ReleaseSkipsCallback)
{
    auto calls = 0;
    {
        auto onExit = ScopeExit{ [&calls]() noexcept { ++calls; } };
        static_assert(noexcept(onExit.Release()));
        onExit.Release();
        onExit.Release();
        EXPECT_EQ(calls, 0);
    }
    EXPECT_EQ(calls, 0);
}

TEST(UnitTestUtils, ScopeExit_OwnsMoveOnlyCallback)
{
    auto result = 0;
    {
        const auto onExit = ScopeExit{ [value = std::make_unique<int>(42), &result]() mutable noexcept {
            result = *value;
            value.reset();
        } };
        EXPECT_EQ(result, 0);
    }
    EXPECT_EQ(result, 42);
}

#if GTEST_HAS_EXCEPTIONS
TEST(UnitTestUtils, ScopeExit_RunsDuringUnwinding)
{
    auto calls = 0;
    const auto run = [&] {
        const auto onExit = ScopeExit{ [&calls]() noexcept { ++calls; } };
        throw std::runtime_error{ "test" };
    };
    EXPECT_THROW(run(), std::runtime_error);
    EXPECT_EQ(calls, 1);
}
#endif

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
