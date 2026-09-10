// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <cstddef>
#include <vector>

namespace libmlvc {

struct FrameMetrics {
    double psnr{};
    double psnrY{};
    double psnrU{};
    double psnrV{};
    double bpp{};
    double kbps{};
};

struct AggregatedMetrics {
    double psnr{};
    double psnrY{};
    double psnrU{};
    double psnrV{};
    double bpp{};
    double kbps{};
    double psnrMin{};
    double psnrMax{};
    int count{};

    static AggregatedMetrics Aggregate(const std::vector<FrameMetrics>& metrics);
    static AggregatedMetrics Aggregate(const std::vector<AggregatedMetrics>& metrics);
};

FrameMetrics CalculateFrameMetrics(const Nv12FrameView& inputFrame, const Nv12FrameView& reconFrame,
                                   std::size_t encodedBytes, double fps);

expected<double> CalculateBdRate(const std::vector<double>& rate1, const std::vector<double>& metric1,
                                 const std::vector<double>& rate2, const std::vector<double>& metric2);

}  // namespace libmlvc
