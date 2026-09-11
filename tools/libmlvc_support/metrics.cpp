// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/metrics.hpp"

#include <libmlvc/error_codes.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ranges>
#include <sstream>

namespace libmlvc {

namespace {

using Poly3Coeffs = std::array<double, 4>;

template <std::size_t N>
expected<std::array<double, N>> SolveLinearSystem(std::array<std::array<double, N>, N> A, std::array<double, N> b)
{
    std::array<double, N> x;
    constexpr int n = static_cast<int>(N);

    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(A[j][i]) > std::abs(A[pivot][i])) {
                pivot = j;
            }
        }

        if (std::abs(A[pivot][i]) < 1e-12) {
            std::cerr << "Error: Matrix is singular or nearly singular during polynomial fitting\n";
            return make_error_code(Error::general_failure);
        }

        if (pivot != i) {
            std::swap(A[i], A[pivot]);
            std::swap(b[i], b[pivot]);
        }

        for (int j = i + 1; j < n; ++j) {
            const double f = A[j][i] / A[i][i];
            for (int k = i; k < n; ++k) {
                A[j][k] -= f * A[i][k];
            }
            b[j] -= f * b[i];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    return x;
}

expected<Poly3Coeffs> PolyFit3(const std::vector<double>& xVec, const std::vector<double>& yVec)
{
    if (xVec.size() < 4) {
        std::cerr << "Error: Not enough points for 3rd-order polynomial fitting (need at least 4, got " << xVec.size()
                  << ")\n";
        return make_error_code(Error::invalid_argument);
    }

    if (xVec.size() != yVec.size()) {
        std::cerr << "Error: x and y vectors have different sizes (" << xVec.size() << " vs " << yVec.size() << ")\n";
        return make_error_code(Error::invalid_argument);
    }

    std::array<std::array<double, 4>, 4> XtX{};
    Poly3Coeffs XtY{};
    for (size_t i = 0; i < xVec.size(); ++i) {
        const double x = xVec[i];
        const double y = yVec[i];
        const double x2 = x * x;
        const double x3 = x2 * x;
        const double x4 = x2 * x2;
        const double x5 = x3 * x2;
        const double x6 = x3 * x3;

        XtX[0][0] += x6;
        XtX[0][1] += x5;
        XtX[0][2] += x4;
        XtX[0][3] += x3;
        XtX[1][1] += x4;
        XtX[1][2] += x3;
        XtX[1][3] += x2;
        XtX[2][2] += x2;
        XtX[2][3] += x;
        XtX[3][3] += 1.0;

        XtY[0] += x3 * y;
        XtY[1] += x2 * y;
        XtY[2] += x * y;
        XtY[3] += y;
    }

    // Fill symmetric parts
    XtX[1][0] = XtX[0][1];
    XtX[2][0] = XtX[0][2];
    XtX[2][1] = XtX[1][2];
    XtX[3][0] = XtX[0][3];
    XtX[3][1] = XtX[1][3];
    XtX[3][2] = XtX[2][3];

    return SolveLinearSystem<4>(XtX, XtY);
}

double PolyIntegralVal(const Poly3Coeffs& coeffs, double x)
{
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x2 * x2;
    return (coeffs[0] / 4.0) * x4 + (coeffs[1] / 3.0) * x3 + (coeffs[2] / 2.0) * x2 + coeffs[3] * x;
}

}  // namespace

FrameMetrics CalculateFrameMetrics(const Nv12FrameView& inputFrame, const Nv12FrameView& reconFrame,
                                   std::size_t encodedBytes, double fps)
{
    const int width = inputFrame.Width();
    const int height = inputFrame.Height();
    assert(width == reconFrame.Width() && height == reconFrame.Height());
    assert(width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0);
    assert(inputFrame.Stride() >= width && reconFrame.Stride() >= width);

    const auto squaredError = [](const uint8_t input, const uint8_t recon) {
        const double normalizedInput = static_cast<double>(input) / 255.0;
        const double normalizedRecon = static_cast<double>(recon) / 255.0;
        return std::pow(normalizedInput - normalizedRecon, 2);
    };

    std::array<double, 3> mse{};
    const auto* inputY = reinterpret_cast<const uint8_t*>(inputFrame.YPlane().data());
    const auto* reconY = reinterpret_cast<const uint8_t*>(reconFrame.YPlane().data());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            mse[0] += squaredError(inputY[y * inputFrame.Stride() + x], reconY[y * reconFrame.Stride() + x]);
        }
    }
    mse[0] /= static_cast<double>(width * height);

    const auto* inputUv = reinterpret_cast<const uint8_t*>(inputFrame.UvPlane().data());
    const auto* reconUv = reinterpret_cast<const uint8_t*>(reconFrame.UvPlane().data());
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width / 2; ++x) {
            const int inputIndex = y * inputFrame.Stride() + 2 * x;
            const int reconIndex = y * reconFrame.Stride() + 2 * x;
            mse[1] += squaredError(inputUv[inputIndex], reconUv[reconIndex]);
            mse[2] += squaredError(inputUv[inputIndex + 1], reconUv[reconIndex + 1]);
        }
    }
    const double chromaSamples = static_cast<double>((width / 2) * (height / 2));
    mse[1] /= chromaSamples;
    mse[2] /= chromaSamples;

    auto mseToPsnr = [](double mse) {
        if (!std::isfinite(mse)) return -999.0;
        if (mse < 1e-10) return 999.0;
        return -10.0 * std::log10(mse);
    };

    return FrameMetrics{
        .psnr = (6.0 * mseToPsnr(mse[0]) + mseToPsnr(mse[1]) + mseToPsnr(mse[2])) / 8.0,
        .psnrY = mseToPsnr(mse[0]),
        .psnrU = mseToPsnr(mse[1]),
        .psnrV = mseToPsnr(mse[2]),
        .bpp = CHAR_BIT * static_cast<double>(encodedBytes) / static_cast<double>(inputFrame.Width() * inputFrame.Height()),
        .kbps = (8.0 * fps * static_cast<double>(encodedBytes)) / 1024.0,
    };
}

AggregatedMetrics AggregatedMetrics::Aggregate(const std::vector<FrameMetrics>& metrics)
{
    if (metrics.empty()) return {};

    AggregatedMetrics res;
    res.psnrMin = std::numeric_limits<double>::max();
    res.psnrMax = std::numeric_limits<double>::lowest();
    for (const auto& m : metrics) {
        res.psnr += m.psnr;
        res.psnrY += m.psnrY;
        res.psnrU += m.psnrU;
        res.psnrV += m.psnrV;
        res.bpp += m.bpp;
        res.kbps += m.kbps;
        res.psnrMin = std::min(res.psnrMin, m.psnr);
        res.psnrMax = std::max(res.psnrMax, m.psnr);
    }
    double n = static_cast<double>(metrics.size());
    res.psnr /= n;
    res.psnrY /= n;
    res.psnrU /= n;
    res.psnrV /= n;
    res.bpp /= n;
    res.kbps /= n;
    res.count = static_cast<int>(metrics.size());
    return res;
}

AggregatedMetrics AggregatedMetrics::Aggregate(const std::vector<AggregatedMetrics>& metrics)
{
    if (metrics.empty()) return {};

    AggregatedMetrics res;
    res.psnrMin = std::numeric_limits<double>::max();
    res.psnrMax = std::numeric_limits<double>::lowest();
    for (const auto& m : metrics) {
        res.psnr += m.psnr;
        res.psnrY += m.psnrY;
        res.psnrU += m.psnrU;
        res.psnrV += m.psnrV;
        res.bpp += m.bpp;
        res.kbps += m.kbps;
        res.psnrMin = std::min(res.psnrMin, m.psnrMin);
        res.psnrMax = std::max(res.psnrMax, m.psnrMax);
        res.count += m.count;
    }
    double n = static_cast<double>(metrics.size());
    res.psnr /= n;
    res.psnrY /= n;
    res.psnrU /= n;
    res.psnrV /= n;
    res.bpp /= n;
    res.kbps /= n;
    return res;
}

expected<double> CalculateBdRate(const std::vector<double>& rate1, const std::vector<double>& metric1,
                                 const std::vector<double>& rate2, const std::vector<double>& metric2)
{
    // Validate we have enough points for polynomial fitting
    if (metric1.size() < 4 || metric2.size() < 4) {
        std::cerr << "Error: Not enough QP points for BD-rate calculation (need at least 4, got " << metric1.size()
                  << " and " << metric2.size() << ")\n";
        return make_error_code(Error::invalid_argument);
    }

    if (metric1.size() != rate1.size()) {
        std::cerr << "Error: First metric and rate vectors have different sizes (" << metric1.size() << " vs "
                  << rate1.size() << ")\n";
        return make_error_code(Error::invalid_argument);
    }

    if (metric2.size() != rate2.size()) {
        std::cerr << "Error: Second metric and rate vectors have different sizes (" << metric2.size() << " vs "
                  << rate2.size() << ")\n";
        return make_error_code(Error::invalid_argument);
    }

    // Compute log(rate) for polynomial fitting
    auto toLog = [](const std::vector<double>& v) {
        std::vector<double> result;
        for (double x : v)
            result.push_back(std::log(x));
        return result;
    };
    const auto logRate1 = toLog(rate1);
    const auto logRate2 = toLog(rate2);

    // Find integration interval (overlapping metric range)
    const auto [minMetric1, maxMetric1] = std::ranges::minmax(metric1);
    const auto [minMetric2, maxMetric2] = std::ranges::minmax(metric2);
    const double minInt = std::max(minMetric1, minMetric2);
    const double maxInt = std::min(maxMetric1, maxMetric2);

    if (maxInt <= minInt) {
        std::ostringstream message;
        message << std::fixed << std::setprecision(2) << "No overlapping metric range ([" << minMetric1 << ", "
                << maxMetric1 << "] vs [" << minMetric2 << ", " << maxMetric2 << "])";
        std::cerr << "Error: " << message.str() << '\n';
        return make_error_code(Error::invalid_argument);
    }

    // Fit 3rd-order polynomials
    const auto poly1 = PolyFit3(metric1, logRate1);
    if (!poly1) {
        std::cerr << "Error: Failed to fit polynomial for first curve\n";
        return poly1.error();
    }
    const auto poly2 = PolyFit3(metric2, logRate2);
    if (!poly2) {
        std::cerr << "Error: Failed to fit polynomial for second curve\n";
        return poly2.error();
    }

    // Integrate both polynomials over [minInt, maxInt]
    const double int1 = PolyIntegralVal(poly1.value(), maxInt) - PolyIntegralVal(poly1.value(), minInt);
    const double int2 = PolyIntegralVal(poly2.value(), maxInt) - PolyIntegralVal(poly2.value(), minInt);

    // Calculate BD-rate
    const double avgExpDiff = (int2 - int1) / (maxInt - minInt);
    const double bdRate = (std::exp(avgExpDiff) - 1.0) * 100.0;
    return bdRate;
}

}  // namespace libmlvc
