// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/bitstream/bit_io.hpp"

namespace libmlvc {

void AddEmulationPreventionBytes(std::span<const std::byte> input, std::vector<std::byte>& output)
{
    // Appends input bytes to output, adding emulation prevention bytes where needed.
    for (size_t i = 0; i < input.size(); i++) {
        if (i >= 2 && output.size() >= 2 && output[output.size() - 2] == std::byte{ 0x00 }
            && output[output.size() - 1] == std::byte{ 0x00 } && input[i] <= std::byte{ 0x03 }) {
            output.push_back(std::byte{ 0x03 });
        }
        output.push_back(input[i]);
    }
}

void RemoveEmulationPreventionBytes(std::span<const std::byte> input, std::vector<std::byte>& output)
{
    output.clear();
    for (size_t i = 0; i < input.size(); i++) {
        if (i >= 2 && input[i - 2] == std::byte{ 0x00 } && input[i - 1] == std::byte{ 0x00 }
            && input[i] == std::byte{ 0x03 }) {
            continue;
        }
        output.push_back(input[i]);
    }
}

}  // namespace libmlvc
