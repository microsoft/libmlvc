// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <libmlvc/expected.hpp>

#include <cstddef>
#include <span>
#include <vector>

namespace libmlvc {

struct BasePmf {
    std::vector<int> pmfLengths;
    std::vector<int> pmfOffsets;
    std::vector<int> pmfTable;
};

struct GaussianCoderPmf : public BasePmf {
    float scaleMin{};
    float scaleMax{};
    int scaleLevels{};
    bool indexSpace{};

    static expected<GaussianCoderPmf> FromJsonBuffer(std::span<const std::byte> buffer);
};

struct BitEstimatorPmf : public BasePmf {
    int qpNum{};
    int channels{};

    static expected<BitEstimatorPmf> FromJsonBuffer(std::span<const std::byte> buffer);
};

}  // namespace libmlvc
