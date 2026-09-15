// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>

namespace libmlvc {

expected<std::unordered_map<std::string, std::span<const std::byte>>> ExtractTar(std::span<const std::byte> data);

}  // namespace libmlvc
