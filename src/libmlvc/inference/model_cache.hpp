// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace libmlvc {

// On-disk format and lifecycle: docs/model-cache-design.md.
class ModelCache {
public:
    using CompileFunction = std::function<expected<void>(const std::filesystem::path& dir)>;

    static constexpr std::chrono::seconds kRetention = std::chrono::days(14);
    static constexpr uintmax_t kSizeLimit = uintmax_t{ 512 } << 20;  // 512 MiB

    // Cache-operation failures return model_cache_error; invalid arguments and callback errors are unchanged.
    static expected<std::shared_ptr<ModelCache>> Create(const std::filesystem::path& entriesDir);
    expected<std::filesystem::path> GetOrCreate(std::string_view key, const CompileFunction& compile) const;
    expected<std::filesystem::path> Get(std::string_view key) const;
    void Cleanup(std::chrono::seconds retention = kRetention, uintmax_t sizeLimit = kSizeLimit) const noexcept;

private:
    std::filesystem::path m_entriesDir;
    std::filesystem::path m_stagingDir;
    std::filesystem::path m_trashDir;

    explicit ModelCache(std::filesystem::path entriesDir);
    expected<std::filesystem::path> Acquire(std::string_view key, const CompileFunction* compile) const;
    expected<void> CreateEntry(const std::filesystem::path& entryDir, const CompileFunction& compile) const;
    void CleanLeftovers() const noexcept;
    void ApplyRetention(std::chrono::seconds retention, uintmax_t sizeLimit) const noexcept;
    bool Remove(const std::filesystem::path& dir, std::optional<int64_t> chosenTime) const;
    static std::string UniqueName();
    static bool IsStagingAbandoned(const std::filesystem::path& stagingDir, std::optional<int64_t> time);
};

}  // namespace libmlvc
