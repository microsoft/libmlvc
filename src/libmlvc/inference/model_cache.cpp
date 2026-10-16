// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/inference/model_cache.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"

#include <libmlvc/error_codes.hpp>

#include <cstdint>
#include <fstream>

// clang-format off
#if defined(MLVC_PLATFORM_WINCLASSIC)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif
// clang-format on

namespace libmlvc {

static constexpr auto kAccessTimeFilename = "last_access.time";
static constexpr auto kCleanupThrottleInterval = std::chrono::hours(1);

// Converts a path to UTF-8 for logging without throwing: on Windows, path::string() throws for characters outside
// the ANSI code page and path::u8string() for unpaired surrogates, which become U+FFFD here.
static std::string ToLogString(const std::filesystem::path& path) noexcept
{
#if defined(MLVC_PLATFORM_WINCLASSIC)
    const std::wstring& wide = path.native();
    if (wide.empty()) return {};
    const int wideLen = static_cast<int>(wide.size());
    const int len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wideLen, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return "<unprintable path>";
    std::string result(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), wideLen, result.data(), len, nullptr, nullptr);
    return result;
#else
    return path.native();
#endif
}

ModelCache& ModelCache::Instance()
{
    static ModelCache instance;
    return instance;
}

void ModelCache::SetRootCacheDir(const std::filesystem::path& rootDir)
{
    std::lock_guard lock(m_mutex);

    if (rootDir.empty()) {
        // Reset is always allowed
        m_rootDir.clear();
        return;
    }

    if (!m_rootDir.empty() && m_rootDir != rootDir) {
        MLVC_LOG_WARN("Model cache root directory already set, ignoring new value");
        return;
    }

    m_rootDir = rootDir;
}

expected<std::filesystem::path> ModelCache::CreateCacheDir(std::string_view cacheKey) noexcept
{
    std::lock_guard lock(m_mutex);

    if (m_rootDir.empty()) {
        MLVC_LOG_ERROR("Cache directory not configured");
        return make_error_code(Error::invalid_argument);
    }

    auto cacheDir = m_rootDir / std::string(cacheKey);

    // Start from an empty directory, without files left by an earlier compile
    std::error_code ec;
    std::filesystem::remove_all(cacheDir, ec);
    if (ec) {
        MLVC_LOG_ERROR("Failed to delete existing directory for cache key '%s': %s", std::string(cacheKey).c_str(),
                       ec.message().c_str());
        return make_error_code(Error::io_error);
    }
    std::filesystem::create_directories(cacheDir, ec);
    if (ec) {
        MLVC_LOG_ERROR("Failed to create directory for cache key '%s': %s", std::string(cacheKey).c_str(),
                       ec.message().c_str());
        return make_error_code(Error::io_error);
    }

    // Write initial access time
    if (auto ret = UpdateAccessTime(cacheDir); !ret) {
        MLVC_LOG_ERROR("Failed to write initial access time for cache key '%s': %s", std::string(cacheKey).c_str(),
                       ret.error().message().c_str());
        return ret.error();
    }

    MLVC_LOG_DEBUG("Created cache directory for key '%s'", std::string(cacheKey).c_str());
    return cacheDir;
}

expected<std::filesystem::path> ModelCache::LocateCacheDir(std::string_view cacheKey) noexcept
{
    std::lock_guard lock(m_mutex);

    if (m_rootDir.empty()) {
        return make_error_code(Error::invalid_argument);
    }

    auto cacheDir = m_rootDir / std::string(cacheKey);

    // Check if directory exists
    std::error_code ec;
    if (!std::filesystem::exists(cacheDir, ec) || ec) {
        return make_error_code(Error::io_error);
    }

    // Update access time
    if (auto ret = UpdateAccessTime(cacheDir); !ret) {
        MLVC_LOG_WARN("Failed to update access time for cache key '%s': %s", std::string(cacheKey).c_str(),
                      ret.error().message().c_str());
        // Continue anyway - cache is still usable
    }

    MLVC_LOG_DEBUG("Located cache directory for key '%s'", std::string(cacheKey).c_str());
    return cacheDir;
}

expected<void> ModelCache::DeleteCacheDir(std::string_view cacheKey) noexcept
{
    std::lock_guard lock(m_mutex);

    if (m_rootDir.empty()) {
        return make_error_code(Error::invalid_argument);
    }

    auto cacheDir = m_rootDir / std::string(cacheKey);

    std::error_code ec;
    if (std::filesystem::exists(cacheDir, ec) && !ec) {
        std::filesystem::remove_all(cacheDir, ec);
        if (ec) {
            MLVC_LOG_WARN("Failed to delete cache directory %s: %s", std::string(cacheKey).c_str(), ec.message().c_str());
            return make_error_code(Error::io_error);
        }
        MLVC_LOG_DEBUG("Deleted cache directory: %s", std::string(cacheKey).c_str());
    }

    return {};
}

expected<void> ModelCache::CleanupCacheDirs(std::chrono::seconds retentionTime, bool force) noexcept
{
    std::lock_guard lock(m_mutex);

    // Throttle cleanup to once per hour unless forced
    auto now = std::chrono::system_clock::now();
    if (!force && (now - m_lastCleanupTime) < kCleanupThrottleInterval) {
        return {};
    }
    m_lastCleanupTime = now;

    std::error_code ec;
    if (!std::filesystem::exists(m_rootDir, ec) || ec) {
        return {};  // Cache root doesn't exist, nothing to clean
    }

    MLVC_LOG_INFO("Starting cache cleanup with %lld second retention", static_cast<long long>(retentionTime.count()));

    int deletedCount = 0;
    std::error_code iterEc;
    std::filesystem::directory_iterator dirIter(m_rootDir, iterEc);
    if (iterEc) {
        MLVC_LOG_ERROR("Failed to iterate over cache root directory: %s", iterEc.message().c_str());
        return make_error_code(Error::io_error);
    }

    // Use error codes, not exceptions: an exception thrown in this noexcept function would terminate the process
    for (; !iterEc && dirIter != std::filesystem::directory_iterator(); dirIter.increment(iterEc)) {
        const auto& entry = *dirIter;
        if (!entry.is_directory(ec)) {
            continue;
        }

        auto accessTimeResult = ReadAccessTime(entry.path());
        if (!accessTimeResult) {
            // No access time file - delete stale cache
            std::filesystem::remove_all(entry.path(), ec);
            if (ec) {
                MLVC_LOG_ERROR("Failed to delete cache directory with missing access time %s: %s",
                               ToLogString(entry.path().filename()).c_str(), ec.message().c_str());
            } else {
                deletedCount++;
            }
            continue;
        }

        auto age = std::chrono::duration_cast<std::chrono::seconds>(now - accessTimeResult.value());
        if (age > retentionTime) {
            std::filesystem::remove_all(entry.path(), ec);
            if (ec) {
                MLVC_LOG_ERROR("Failed to delete cache directory %s: %s", ToLogString(entry.path().filename()).c_str(),
                               ec.message().c_str());
            } else {
                deletedCount++;
                MLVC_LOG_DEBUG("Deleted expired cache: %s (age: %lld days)", ToLogString(entry.path().filename()).c_str(),
                               static_cast<long long>(age.count() / (24 * 60 * 60)));
            }
        }
    }
    if (iterEc) {
        MLVC_LOG_ERROR("Failed to iterate over cache root directory: %s", iterEc.message().c_str());
    }

    MLVC_LOG_INFO("Cache cleanup completed: deleted %d expired entries", deletedCount);

    return {};
}

expected<void> ModelCache::UpdateAccessTime(const std::filesystem::path& cacheDir) noexcept
{
    auto accessTimePath = cacheDir / kAccessTimeFilename;

    std::ofstream file(accessTimePath, std::ios::binary | std::ios::trunc);
    if (!file) {
        MLVC_LOG_ERROR("Failed to open access time file for writing");
        return make_error_code(Error::io_error);
    }

    auto const now = std::chrono::system_clock::now();
    auto const nowSeconds =
        static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());
    file.write(reinterpret_cast<const char*>(&nowSeconds), sizeof(nowSeconds));
    if (!file) {
        MLVC_LOG_ERROR("Failed to write access time");
        return make_error_code(Error::io_error);
    }

    return {};
}

expected<std::chrono::system_clock::time_point> ModelCache::ReadAccessTime(const std::filesystem::path& cacheDir) noexcept
{
    auto accessTimePath = cacheDir / kAccessTimeFilename;

    std::ifstream file(accessTimePath, std::ios::binary);
    if (!file) {
        return make_error_code(Error::io_error);
    }

    int64_t accessTimeSeconds{};
    file.read(reinterpret_cast<char*>(&accessTimeSeconds), sizeof(accessTimeSeconds));
    if (!file) {
        return make_error_code(Error::io_error);
    }

    return std::chrono::system_clock::time_point(std::chrono::seconds(accessTimeSeconds));
}

}  // namespace libmlvc
