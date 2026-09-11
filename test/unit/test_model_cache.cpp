// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/inference/model_cache.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>

using namespace libmlvc;

constexpr auto kRetention = std::chrono::hours(60 * 24);

class UnitTestModelCache : public ::testing::Test {
protected:
    void SetUp() override
    {
        m_testCacheDir = std::filesystem::temp_directory_path() / "mlvc_model_cache_test";
        std::error_code ec;
        std::filesystem::remove_all(m_testCacheDir, ec);
        std::filesystem::create_directories(m_testCacheDir, ec);
        ASSERT_FALSE(ec) << "create_directories: " << ec.message();
        ModelCache::Instance().SetRootCacheDir(m_testCacheDir);
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove_all(m_testCacheDir, ec);
        ModelCache::Instance().SetRootCacheDir({});
    }

    void WriteAccessTime(const std::filesystem::path& cacheDir, int64_t secondsSinceEpoch)
    {
        std::ofstream file(cacheDir / "last_access.time", std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(file.is_open());
        file.write(reinterpret_cast<const char*>(&secondsSinceEpoch), sizeof(secondsSinceEpoch));
    }

    int64_t ReadAccessTime(const std::filesystem::path& cacheDir)
    {
        std::ifstream file(cacheDir / "last_access.time", std::ios::binary);
        EXPECT_TRUE(file.is_open());
        int64_t seconds = 0;
        file.read(reinterpret_cast<char*>(&seconds), sizeof(seconds));
        return seconds;
    }

    std::filesystem::path m_testCacheDir;
};

TEST_F(UnitTestModelCache, CreateLocateDelete)
{
    auto& cache = ModelCache::Instance();
    const std::string key = "test_key";
    std::error_code ec;

    // Locate non-existent should fail
    ASSERT_FALSE(cache.LocateCacheDir(key));

    // Create
    auto createResult = cache.CreateCacheDir(key);
    ASSERT_TRUE(createResult);
    ASSERT_TRUE(std::filesystem::exists(createResult.value(), ec)) << ec.message();
    ASSERT_TRUE(std::filesystem::exists(createResult.value() / "last_access.time", ec)) << ec.message();

    // Locate existing
    auto locateResult = cache.LocateCacheDir(key);
    ASSERT_TRUE(locateResult);
    ASSERT_EQ(locateResult.value(), createResult.value());

    // Delete
    ASSERT_TRUE(cache.DeleteCacheDir(key));
    ASSERT_FALSE(std::filesystem::exists(createResult.value(), ec)) << ec.message();

    // Delete non-existent should succeed
    ASSERT_TRUE(cache.DeleteCacheDir("nonexistent"));
}

TEST_F(UnitTestModelCache, CreateDuplicateKeyOverwritesTimestamp)
{
    auto& cache = ModelCache::Instance();
    const std::string key = "dup_key";

    // Create initial cache
    auto result = cache.CreateCacheDir(key);
    ASSERT_TRUE(result);

    // Backdate the timestamp
    WriteAccessTime(result.value(), 1000);

    // Re-create with same key — should overwrite timestamp
    auto result2 = cache.CreateCacheDir(key);
    ASSERT_TRUE(result2);
    ASSERT_EQ(result2.value(), result.value());

    // Read back timestamp and verify it was refreshed
    ASSERT_GT(ReadAccessTime(result.value()), 1000);
}

TEST_F(UnitTestModelCache, CleanupExpiredAndStaleCaches)
{
    auto& cache = ModelCache::Instance();

    // Create fresh cache
    auto freshResult = cache.CreateCacheDir("fresh");
    ASSERT_TRUE(freshResult);

    // Create expired cache with old timestamp
    auto expiredResult = cache.CreateCacheDir("expired");
    ASSERT_TRUE(expiredResult);
    auto pastTime = std::chrono::duration_cast<std::chrono::seconds>(
                        (std::chrono::system_clock::now() - std::chrono::hours(24 * 90)).time_since_epoch())
                        .count();
    WriteAccessTime(expiredResult.value(), pastTime);

    // Create stale cache without timestamp
    auto stalePath = m_testCacheDir / "stale";
    std::error_code ec;
    std::filesystem::create_directories(stalePath, ec);
    ASSERT_FALSE(ec) << ec.message();

    // Run cleanup with 60-day retention
    ASSERT_TRUE(cache.CleanupCacheDirs(kRetention, true));

    // Fresh should remain, expired and stale should be deleted
    ASSERT_TRUE(std::filesystem::exists(freshResult.value(), ec)) << ec.message();
    ASSERT_FALSE(std::filesystem::exists(expiredResult.value(), ec)) << ec.message();
    ASSERT_FALSE(std::filesystem::exists(stalePath, ec)) << ec.message();
}

TEST_F(UnitTestModelCache, CleanupThrottleSkipsWhenNotForced)
{
    auto& cache = ModelCache::Instance();
    std::error_code ec;

    // Prime the throttle by running forced cleanup
    ASSERT_TRUE(cache.CleanupCacheDirs(kRetention, true));

    // Create an expired cache
    auto expiredResult = cache.CreateCacheDir("expired");
    ASSERT_TRUE(expiredResult);
    WriteAccessTime(expiredResult.value(), 1000);

    // Non-forced cleanup should be throttled — expired cache survives
    ASSERT_TRUE(cache.CleanupCacheDirs(kRetention, false));
    ASSERT_TRUE(std::filesystem::exists(expiredResult.value(), ec)) << ec.message();

    // Forced cleanup should bypass throttle — expired cache deleted
    ASSERT_TRUE(cache.CleanupCacheDirs(kRetention, true));
    ASSERT_FALSE(std::filesystem::exists(expiredResult.value(), ec)) << ec.message();
}

TEST_F(UnitTestModelCache, OperationsFailWithoutRootDir)
{
    auto& cache = ModelCache::Instance();

    // Clear root directory
    cache.SetRootCacheDir({});

    // Create, Locate, Delete should fail
    ASSERT_FALSE(cache.CreateCacheDir("key"));
    ASSERT_FALSE(cache.LocateCacheDir("key"));
    ASSERT_FALSE(cache.DeleteCacheDir("key"));

    // CleanupCacheDirs returns success when root is empty
    ASSERT_TRUE(cache.CleanupCacheDirs(kRetention, true));

    // Restore for TearDown
    cache.SetRootCacheDir(m_testCacheDir);
}
