// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace libmlvc {

expected<std::unordered_map<std::string, std::span<const std::byte>>> ExtractTar(std::span<const std::byte> data);
expected<std::vector<std::byte>> ReadTarEntry(const std::filesystem::path& archivePath, std::string_view entryName);

}  // namespace libmlvc
