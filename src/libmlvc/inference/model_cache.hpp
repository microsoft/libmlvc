// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <chrono>
#include <filesystem>
#include <mutex>
#include <string_view>

namespace libmlvc {

// Singleton cache manager for compiled model caching (EP context, CoreML, etc.).
// Manages cache directories for compiled models, with timestamp-based expiration.
// Thread-safe for concurrent session creation.
// Platform-agnostic: works on Windows, macOS, and other platforms.
class ModelCache {
public:
    static ModelCache& Instance();

    // Set the root cache directory. Must be set before any cache operations.
    // Once set, changing to a different non-empty path is ignored (with warning).
    // Setting to empty resets.
    void SetRootCacheDir(const std::filesystem::path& rootDir);

    // Create a new cache directory for the given key.
    // Returns path to: {cacheRoot}/{cacheKey}/
    expected<std::filesystem::path> CreateCacheDir(std::string_view cacheKey) noexcept;

    // Locate existing cache directory and update access timestamp.
    // Returns path if directory exists, error otherwise.
    expected<std::filesystem::path> LocateCacheDir(std::string_view cacheKey) noexcept;

    // Delete cache directory for given key.
    expected<void> DeleteCacheDir(std::string_view cacheKey) noexcept;

    // Cleanup old cache dirs. Default retention: 60 days.
    // Runs at most once per hour unless force=true.
    expected<void> CleanupCacheDirs(std::chrono::seconds retentionTime = std::chrono::hours(60 * 24),
                                    bool force = false) noexcept;

private:
    ModelCache() = default;
    ~ModelCache() = default;
    ModelCache(const ModelCache&) = delete;
    ModelCache& operator=(const ModelCache&) = delete;

    expected<void> UpdateAccessTime(const std::filesystem::path& cacheDir) noexcept;
    expected<std::chrono::system_clock::time_point> ReadAccessTime(const std::filesystem::path& cacheDir) noexcept;

    std::mutex m_mutex;
    std::filesystem::path m_rootDir;
    std::chrono::system_clock::time_point m_lastCleanupTime{};
};

}  // namespace libmlvc
