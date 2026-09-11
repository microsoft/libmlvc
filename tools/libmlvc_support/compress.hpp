// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace libmlvc {

expected<std::vector<std::byte>> Compress(std::span<const std::byte> data) noexcept;
expected<std::vector<std::byte>> Uncompress(std::span<const std::byte> data) noexcept;
expected<std::vector<std::byte>> ReadGzipFile(const std::filesystem::path& filename);

}  // namespace libmlvc
