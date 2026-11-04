// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/logging.hpp"
#include "libmlvc/entropy/rans.hpp"
#include "libmlvc/entropy/rans_coder.hpp"
#include "support/rans_test_pmf.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
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

// One distribution over three values and the bypass tail mass
const std::vector<int32_t> kSmallPmfLengths{ 4 };
const std::vector<int32_t> kSmallPmfTable{ 8192, 40960, 8192, 8192 };

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

TEST(UnitTestRansCoder, RoundTrip_ExtremeValues)
{
    // The ends of the supported range and values around the bypass symbols
    const std::vector<int32_t> values{
        0, 1, -1, 2, -2, 3, -3, rans::kMaxValue, rans::kMaxValue - 1, rans::kMinValue, rans::kMinValue + 1
    };
    const std::vector<int32_t> indices(values.size(), 0);

    for (const auto variant : { rans::RansVariant::RansByte, rans::RansVariant::Rans64 }) {
        for (int32_t offset = 0; offset < kSmallPmfLengths[0] - 1; offset++) {
            for (int bypassBits = 2; bypassBits <= 5; bypassBits++) {
                SCOPED_TRACE(testing::Message() << "variant " << static_cast<int>(variant) << ", offset " << offset
                                                << ", bypass bits " << bypassBits);
                const std::vector<int32_t> pmfOffsets{ offset };
                rans::EntropyEncoder encoder;
                auto ec = encoder.Initialize(variant, kSmallPmfLengths, pmfOffsets, kSmallPmfTable, 16, bypassBits);
                ASSERT_FALSE(ec) << ec.message();
                rans::EntropyDecoder decoder;
                ec = decoder.Initialize(variant, kSmallPmfLengths, pmfOffsets, kSmallPmfTable, 16, bypassBits);
                ASSERT_FALSE(ec) << ec.message();

                rans::HeapResizableBuffer buffer;
                const auto encoded = encoder.Encode(buffer, indices, values);
                ASSERT_FALSE(encoded.empty());
                std::vector<int32_t> decoded(values.size());
                ec = decoder.Decode(std::span<int32_t>{ decoded }, indices, encoded);
                ASSERT_FALSE(ec) << ec.message();
                EXPECT_EQ(decoded, values);
            }
        }
    }
}

TEST(UnitTestRansCoder, RoundTrip_Empty)
{
    const std::vector<int32_t> pmfOffsets{ 1 };
    rans::EntropyEncoder encoder;
    auto ec = encoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, pmfOffsets, kSmallPmfTable);
    ASSERT_FALSE(ec) << ec.message();
    rans::EntropyDecoder decoder;
    ec = decoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, pmfOffsets, kSmallPmfTable);
    ASSERT_FALSE(ec) << ec.message();
    rans::HeapResizableBuffer buffer;
    const auto encoded = encoder.Encode(buffer, {}, {});
    ASSERT_FALSE(encoded.empty());
    ec = decoder.Decode({}, {}, encoded);
    EXPECT_FALSE(ec) << ec.message();
}

TEST(UnitTestRansCoder, Encode_InvalidInput)
{
    struct TestCase {
        std::vector<int32_t> indices;
        std::vector<int32_t> values;
        rans::error error;
    };
    const std::vector<TestCase> cases{
        { { 0 }, { rans::kMinValue - 1 }, rans::error::value_out_of_range },
        { { 0, 0, 0, 0 }, { 1, rans::kMinValue - 1, rans::kMinValue, 0 }, rans::error::value_out_of_range },
        { { 0, 0, 0, 0 }, { 0, 0, 0 }, rans::error::invalid_params },
    };

    for (int32_t offset = 0; offset < kSmallPmfLengths[0] - 1; offset++) {
        rans::EntropyEncoder encoder;
        auto ec = encoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, std::vector<int32_t>{ offset },
                                     kSmallPmfTable);
        ASSERT_FALSE(ec) << ec.message();
        for (size_t i = 0; i < cases.size(); i++) {
            SCOPED_TRACE(testing::Message() << "offset " << offset << ", case " << i);
            rans::HeapResizableBuffer buffer;
            EXPECT_TRUE(encoder.Encode(buffer, cases[i].indices, cases[i].values).empty());
            rans::RansEncoderStream stream;
            ec = stream.Initialize(rans::RansVariant::RansByte, buffer);
            ASSERT_FALSE(ec) << ec.message();
            EXPECT_EQ(encoder.Encode(stream, cases[i].indices, cases[i].values), rans::make_error_code(cases[i].error));
        }
    }
}

TEST(UnitTestRansCoder, Decode_ValuesOutsideRange)
{
    // An encoder offset that differs from the decoder's, like a corrupt stream, decodes just outside the range
    struct TestCase {
        int32_t encoderOffset;
        int32_t value;
        int32_t decoderOffset;
    };
    for (const auto& [encoderOffset, value, decoderOffset] :
         { TestCase{ 0, rans::kMinValue, 1 }, TestCase{ 0, rans::kMinValue, 2 }, TestCase{ 2, rans::kMaxValue, 1 },
           TestCase{ 2, rans::kMaxValue, 0 } }) {
        SCOPED_TRACE(testing::Message() << "encoder offset " << encoderOffset << ", value " << value
                                        << ", decoder offset " << decoderOffset);
        rans::EntropyEncoder encoder;
        auto ec = encoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths,
                                     std::vector<int32_t>{ encoderOffset }, kSmallPmfTable);
        ASSERT_FALSE(ec) << ec.message();
        rans::EntropyDecoder decoder;
        ec = decoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, std::vector<int32_t>{ decoderOffset },
                                kSmallPmfTable);
        ASSERT_FALSE(ec) << ec.message();

        const std::vector<int32_t> values{ value };
        const std::vector<int32_t> indices{ 0 };
        rans::HeapResizableBuffer buffer;
        const auto encoded = encoder.Encode(buffer, indices, values);
        ASSERT_FALSE(encoded.empty());
        std::vector<int32_t> decoded(values.size());
        EXPECT_EQ(decoder.Decode(std::span<int32_t>{ decoded }, indices, encoded),
                  rans::make_error_code(rans::error::invalid_stream));
    }
}

TEST(UnitTestRansCoder, Decode_TooManyBypassParts)
{
    // 32-bit bypass values take at most 16 two-bit parts
    rans::EntropyDecoder decoder;
    auto ec = decoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, std::vector<int32_t>{ 1 },
                                 kSmallPmfTable, 16, 2);
    ASSERT_FALSE(ec) << ec.message();
    for (const uint32_t partCount : { 16u, 17u }) {
        SCOPED_TRACE(partCount);
        // The value 3 as the encoder writes it, but with its bypass value 2 in partCount parts
        std::vector<std::byte> stream;
        auto sink = [&stream](uint8_t unit) { stream.push_back(static_cast<std::byte>(unit)); };
        rans::RansEncoder<uint32_t, uint8_t, decltype(sink)> encoder(std::move(sink));
        for (uint32_t part = partCount; part-- > 0;) {
            encoder.Put(part == 0 ? 2u : 0u, 1u, 2u);
        }
        encoder.Put(partCount % 3u, 1u, 2u);
        for (uint32_t i = 0; i < partCount / 3u; i++) {
            encoder.Put(3u, 1u, 2u);
        }
        const auto sentinelFreq = static_cast<uint32_t>(kSmallPmfTable.back());
        encoder.Put((1u << 16) - sentinelFreq, sentinelFreq, 16u);
        encoder.Flush();
        std::reverse(stream.begin(), stream.end());

        std::vector<int32_t> decoded(1);
        ec = decoder.Decode(std::span<int32_t>{ decoded }, std::vector<int32_t>{ 0 }, stream);
        if (partCount == 16) {
            ASSERT_FALSE(ec) << ec.message();
            EXPECT_EQ(decoded[0], 3);
        } else {
            EXPECT_EQ(ec, rans::make_error_code(rans::error::invalid_stream));
        }
    }
}

TEST(UnitTestRansCoder, Decode_IndicesPastLastPmf)
{
    // Indices past the last PMF decode with the last PMF
    const std::vector<int32_t> pmfLengths{ 4, 4 };
    const std::vector<int32_t> pmfOffsets{ 1, 2 };
    const std::vector<int32_t> pmfTable{ 8192, 40960, 8192, 8192, 16384, 16384, 16384, 16384 };
    const std::vector<int32_t> values{ 0, 1, -1, 2, -2, 5 };
    rans::EntropyEncoder encoder;
    auto ec = encoder.Initialize(rans::RansVariant::RansByte, pmfLengths, pmfOffsets, pmfTable);
    ASSERT_FALSE(ec) << ec.message();
    rans::EntropyDecoder decoder;
    ec = decoder.Initialize(rans::RansVariant::RansByte, pmfLengths, pmfOffsets, pmfTable);
    ASSERT_FALSE(ec) << ec.message();
    rans::HeapResizableBuffer buffer;
    const auto encoded = encoder.Encode(buffer, std::vector<int32_t>(values.size(), 1), values);
    ASSERT_FALSE(encoded.empty());

    const std::vector<int32_t> indices{
        1, 2, 3, 1000, std::numeric_limits<int32_t>::max() - 1, std::numeric_limits<int32_t>::max()
    };
    std::vector<int32_t> decoded(values.size());
    ec = decoder.Decode(std::span<int32_t>{ decoded }, indices, encoded);
    ASSERT_FALSE(ec) << ec.message();
    EXPECT_EQ(decoded, values);
}

TEST(UnitTestRansCoder, Initialize_InvalidPmf)
{
    const auto invalidPmf = rans::make_error_code(rans::error::invalid_pmf);
    rans::EntropyEncoder encoder;
    rans::EntropyDecoder decoder;
    EXPECT_EQ(encoder.Initialize(rans::RansVariant::RansByte, {}, {}, {}), invalidPmf);
    EXPECT_EQ(decoder.Initialize(rans::RansVariant::RansByte, {}, {}, {}), invalidPmf);

    // The value 0 must be a symbol before the bypass sentinel
    for (const int32_t offset : { std::numeric_limits<int32_t>::min(), -1, 3, 4, std::numeric_limits<int32_t>::max() }) {
        SCOPED_TRACE(offset);
        const std::vector<int32_t> pmfOffsets{ offset };
        EXPECT_EQ(encoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, pmfOffsets, kSmallPmfTable), invalidPmf);
        EXPECT_EQ(decoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, pmfOffsets, kSmallPmfTable), invalidPmf);
    }
}

TEST(UnitTestRansCoder, Initialize_InvalidBits)
{
    // At most the bits of the variant's normalization lower bound
    const std::vector<int32_t> pmfOffsets{ 1 };
    const std::vector<int32_t> values{ 0, 1, -2, rans::kMaxValue, rans::kMinValue };
    const std::vector<int32_t> indices(values.size(), 0);
    const auto invalidParams = rans::make_error_code(rans::error::invalid_params);
    for (const auto& [variant, maxBits] :
         { std::pair{ rans::RansVariant::RansByte, 23 }, std::pair{ rans::RansVariant::Rans64, 31 } }) {
        SCOPED_TRACE(static_cast<int>(variant));
        // The int32_t PMF checks reject all tables of 31 symbol bits
        const int maxSymbolBits = std::min(maxBits, 30);
        rans::EntropyEncoder encoder;
        rans::EntropyDecoder decoder;
        ASSERT_FALSE(encoder.Initialize(variant, kSmallPmfLengths, pmfOffsets, kSmallPmfTable, maxSymbolBits, maxBits));
        ASSERT_FALSE(decoder.Initialize(variant, kSmallPmfLengths, pmfOffsets, kSmallPmfTable, maxSymbolBits, maxBits));
        rans::HeapResizableBuffer buffer;
        const auto encoded = encoder.Encode(buffer, indices, values);
        ASSERT_FALSE(encoded.empty());
        std::vector<int32_t> decoded(values.size());
        EXPECT_FALSE(decoder.Decode(std::span<int32_t>{ decoded }, indices, encoded));
        EXPECT_EQ(decoded, values);
        for (const int bits : { 1, maxBits + 1 }) {
            SCOPED_TRACE(bits);
            EXPECT_EQ(encoder.Initialize(variant, kSmallPmfLengths, pmfOffsets, kSmallPmfTable, bits, 2), invalidParams);
            EXPECT_EQ(encoder.Initialize(variant, kSmallPmfLengths, pmfOffsets, kSmallPmfTable, 16, bits), invalidParams);
            EXPECT_EQ(decoder.Initialize(variant, kSmallPmfLengths, pmfOffsets, kSmallPmfTable, bits, 2), invalidParams);
            EXPECT_EQ(decoder.Initialize(variant, kSmallPmfLengths, pmfOffsets, kSmallPmfTable, 16, bits), invalidParams);
        }
    }
}

TEST(UnitTestRansCoder, Decode_StreamClosedByFailedOpen)
{
    const std::vector<int32_t> pmfOffsets{ 1 };
    const std::vector<int32_t> values{ 0, 1, -1, 5 };
    const std::vector<int32_t> indices(values.size(), 0);
    rans::EntropyEncoder encoder;
    auto ec = encoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, pmfOffsets, kSmallPmfTable);
    ASSERT_FALSE(ec) << ec.message();
    rans::EntropyDecoder decoder;
    ec = decoder.Initialize(rans::RansVariant::RansByte, kSmallPmfLengths, pmfOffsets, kSmallPmfTable);
    ASSERT_FALSE(ec) << ec.message();
    rans::HeapResizableBuffer buffer;
    const auto encoded = encoder.Encode(buffer, indices, values);
    ASSERT_FALSE(encoded.empty());

    rans::RansDecoderStream stream;
    ec = stream.Initialize(rans::RansVariant::RansByte);
    ASSERT_FALSE(ec) << ec.message();
    std::vector<int32_t> decoded(values.size());
    const auto invalidState = rans::make_error_code(rans::error::invalid_state);
    EXPECT_EQ(decoder.Decode(std::span<int32_t>{ decoded }, indices, stream), invalidState);

    // A failed Open closes the stream instead of keeping the previous data
    for (size_t size = 0; size < sizeof(uint32_t); size++) {
        SCOPED_TRACE(size);
        ec = stream.Open(encoded);
        ASSERT_FALSE(ec) << ec.message();
        EXPECT_EQ(stream.Open(encoded.first(size)), rans::make_error_code(rans::error::invalid_stream));
        EXPECT_EQ(decoder.Decode(std::span<int32_t>{ decoded }, indices, stream), invalidState);
    }

    ec = stream.Open(encoded);
    ASSERT_FALSE(ec) << ec.message();
    ec = decoder.Decode(std::span<int32_t>{ decoded }, indices, stream);
    ASSERT_FALSE(ec) << ec.message();
    EXPECT_EQ(decoded, values);
}
