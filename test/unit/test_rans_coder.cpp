// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/logging.hpp"
#include "libmlvc/entropy/rans_coder.hpp"
#include "support/rans_test_pmf.hpp"

#include <gtest/gtest.h>

#include <span>
#include <vector>

namespace {

using namespace libmlvc;

void RunShannonTest(const RansTestPmf& pmf, int symbolsPerSegment, int numSegments, int numFrames)
{
    // Set up encoder
    rans::EntropyEncoder encoder;
    auto ec = encoder.Initialize(rans::RansVariant::RansByte, pmf.PmfLengths(), pmf.PmfOffsets(), pmf.PmfTable(),
                                 pmf.SymbolBits(), pmf.BypassBits());
    ASSERT_FALSE(ec) << ec.message();

    rans::HeapResizableBuffer buffer;
    rans::RansEncoderStream encodeStream;
    ec = encodeStream.Initialize(rans::RansVariant::RansByte, buffer);
    ASSERT_FALSE(ec) << ec.message();

    // Encode all frames and accumulate total bytes
    std::vector<int32_t> indices;
    std::vector<int32_t> values;
    size_t totalBytes = 0;
    for (int i = 0; i < numFrames; i++) {
        for (int j = 0; j < numSegments; j++) {
            pmf.GenerateSymbols(indices, values, symbolsPerSegment);
            ec = encoder.Encode(encodeStream, indices, values);
            ASSERT_FALSE(ec) << "Frame " << i << " segment " << j << ": " << ec.message();
        }
        auto encoded = encodeStream.Flush();
        ASSERT_FALSE(encoded.empty()) << "Frame " << i;
        totalBytes += encoded.size();
    }

    // Verify actual bits/symbol is close to Shannon entropy
    const size_t totalSymbols = static_cast<size_t>(numFrames) * numSegments * symbolsPerSegment;
    const double actualBitsPerSymbol = static_cast<double>(totalBytes * 8) / static_cast<double>(totalSymbols);
    const double shannonBitsPerSymbol = pmf.CalcShannonEntropy();
    const double overheadPct = (actualBitsPerSymbol / shannonBitsPerSymbol - 1.0) * 100.0;
    MLVC_LOG_INFO("actual=%.6f bits/sym, shannon=%.6f bits/sym, overhead=%.2f%% (%d frames, %d segments)",
                  actualBitsPerSymbol, shannonBitsPerSymbol, overheadPct, numFrames, numSegments);

    // Allow up to 2% relative overhead + 0.005 bits/sym absolute tolerance
    const double maxOverhead = shannonBitsPerSymbol * 0.02 + 0.005;
    EXPECT_GE(actualBitsPerSymbol, shannonBitsPerSymbol * 0.99)
        << "Actual=" << actualBitsPerSymbol << " Shannon=" << shannonBitsPerSymbol;
    EXPECT_LE(actualBitsPerSymbol, shannonBitsPerSymbol + maxOverhead)
        << "Actual=" << actualBitsPerSymbol << " Shannon=" << shannonBitsPerSymbol
        << " MaxAllowed=" << (shannonBitsPerSymbol + maxOverhead);
}

void RunRoundTripTest(const RansTestPmf& pmf, int symbolsPerSegment, int numSegments, int numFrames)
{
    // Set up encoder and decoder
    rans::EntropyEncoder encoder;
    auto ec = encoder.Initialize(rans::RansVariant::RansByte, pmf.PmfLengths(), pmf.PmfOffsets(), pmf.PmfTable(),
                                 pmf.SymbolBits(), pmf.BypassBits());
    ASSERT_FALSE(ec) << "Encoder init failed: " << ec.message();

    rans::EntropyDecoder decoder;
    ec = decoder.Initialize(rans::RansVariant::RansByte, pmf.PmfLengths(), pmf.PmfOffsets(), pmf.PmfTable(),
                            pmf.SymbolBits(), pmf.BypassBits());
    ASSERT_FALSE(ec) << "Decoder init failed: " << ec.message();

    rans::HeapResizableBuffer encodeBuffer;
    rans::RansEncoderStream encodeStream;
    ec = encodeStream.Initialize(rans::RansVariant::RansByte, encodeBuffer);
    ASSERT_FALSE(ec) << "Encoder stream init failed: " << ec.message();

    rans::RansDecoderStream decodeStream;
    ec = decodeStream.Initialize(rans::RansVariant::RansByte);
    ASSERT_FALSE(ec) << "Decoder stream init failed: " << ec.message();

    // Per-segment storage for round-trip verification (must retain all segments since rANS decode is LIFO)
    struct SegmentData {
        std::vector<int32_t> indices;
        std::vector<int32_t> values;
    };
    std::vector<SegmentData> segments(numSegments);
    std::vector<int32_t> decoded;  // reusable decode buffer

    // Encode and decode each frame, verifying round-trip correctness
    for (int i = 0; i < numFrames; i++) {
        // Encode each segment
        for (int j = 0; j < numSegments; j++) {
            pmf.GenerateSymbols(segments[j].indices, segments[j].values, symbolsPerSegment);
            ec = encoder.Encode(encodeStream, segments[j].indices, segments[j].values);
            ASSERT_FALSE(ec) << "Encode frame " << i << " segment " << j << ": " << ec.message();
        }
        auto encoded = encodeStream.Flush();
        ASSERT_FALSE(encoded.empty()) << "Frame " << i;

        // Decode each segment separately (reverse order)
        decodeStream.Open(encoded);
        for (int j = numSegments - 1; j >= 0; j--) {
            decoded.resize(symbolsPerSegment);
            ec = decoder.Decode(std::span<int32_t>{ decoded }, segments[j].indices, decodeStream);
            ASSERT_FALSE(ec) << "Decode frame " << i << " segment " << j << ": " << ec.message();
            ASSERT_EQ(decoded, segments[j].values) << "Mismatch in frame " << i << " segment " << j;
        }
    }
}

}  // namespace

TEST(UnitTestRansCoder, ErrorCategoryIdentity)
{
    auto ec = rans::make_error_code(rans::error::invalid_params);

    EXPECT_TRUE(ec);
    EXPECT_STREQ(rans::error_category().name(), "libmlvc.rans");
    EXPECT_EQ(&ec.category(), &rans::error_category());
    EXPECT_EQ(ec.message(), "invalid parameter value");
}

TEST(UnitTestRansCoder, ShannonEntropy_Gaussian)
{
    auto pmf = RansTestPmf::MakeGaussianPmf();
    // 960x540 Gaussian latent: 24ch x 34h x 60w = 48960 symbols/segment, 2 segments
    RunShannonTest(pmf, 48960, 2, 150);
}

TEST(UnitTestRansCoder, ShannonEntropy_BitEstimator)
{
    auto pmf = RansTestPmf::MakeBitEstimatorPmf();
    // 960x540 BitEstimator latent: 48ch x 9h x 15w = 6480 symbols/segment, 1 segment
    RunShannonTest(pmf, 6480, 1, 150);
}

TEST(UnitTestRansCoder, RoundTrip_Gaussian)
{
    auto pmf = RansTestPmf::MakeGaussianPmf();
    // 960x540 Gaussian latent: 24ch x 34h x 60w = 48960 symbols/segment, 2 segments
    RunRoundTripTest(pmf, 48960, 2, 150);
}

TEST(UnitTestRansCoder, RoundTrip_BitEstimator)
{
    auto pmf = RansTestPmf::MakeBitEstimatorPmf();
    // 960x540 BitEstimator latent: 48ch x 9h x 15w = 6480 symbols/segment, 1 segment
    RunRoundTripTest(pmf, 6480, 1, 150);
}
