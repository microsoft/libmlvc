// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/logging.hpp"
#include "libmlvc/entropy/rans_coder.hpp"
#include "support/rans_test_pmf.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using namespace libmlvc;

// ------------------------------------------------------------
// Entropy coder (rANS)
// ------------------------------------------------------------

static void BM_RansEncodeGaussian(benchmark::State& state)
{
    auto testPmf = RansTestPmf::MakeGaussianPmf();

    // 960x540 Gaussian latent: 24ch x 34h x 60w = 48960 symbols/segment, 2 segments
    constexpr int symbolsPerSegment = 48960;
    constexpr int numSegments = 2;
    constexpr int numSymbols = symbolsPerSegment * numSegments;
    std::vector<int32_t> indices;
    std::vector<int32_t> values;
    testPmf.GenerateSymbols(indices, values, numSymbols);

    // Initialize encoder
    rans::EntropyEncoder encoder;
    if (auto ec = encoder.Initialize(rans::RansVariant::RansByte, testPmf.PmfLengths(), testPmf.PmfOffsets(),
                                     testPmf.PmfTable(), testPmf.SymbolBits(), testPmf.BypassBits())) {
        MLVC_LOG_ABORT("Failed to initialize EntropyEncoder: %s", ec.message().c_str());
    }
    rans::HeapResizableBuffer encodeBuffer{};
    rans::RansEncoderStream encodeStream{};
    if (auto ec = encodeStream.Initialize(rans::RansVariant::RansByte, encodeBuffer)) {
        MLVC_LOG_ABORT("Failed to initialize RansEncoderStream: %s", ec.message().c_str());
    }

    for (auto _ : state) {
        for (int s = 0; s < numSegments; s++) {
            auto segIndices = std::span<const int32_t>(indices).subspan(s * symbolsPerSegment, symbolsPerSegment);
            auto segValues = std::span<const int32_t>(values).subspan(s * symbolsPerSegment, symbolsPerSegment);
            if (auto ec = encoder.Encode(encodeStream, segIndices, segValues)) {
                MLVC_LOG_ABORT("Failed to encode: %s", ec.message().c_str());
            }
        }
        [[maybe_unused]] auto bitstream = encodeStream.Flush();
    }
}
BENCHMARK(BM_RansEncodeGaussian);

static void BM_RansDecodeGaussian(benchmark::State& state)
{
    auto testPmf = RansTestPmf::MakeGaussianPmf();

    // 960x540 Gaussian latent: 24ch x 34h x 60w = 48960 symbols/segment, 2 segments
    constexpr int numSymbols = 48960 * 2;
    std::vector<int32_t> indices;
    std::vector<int32_t> values;
    testPmf.GenerateSymbols(indices, values, numSymbols);

    // Encode once to produce bitstream for decoding
    rans::EntropyEncoder encoder;
    if (auto ec = encoder.Initialize(rans::RansVariant::RansByte, testPmf.PmfLengths(), testPmf.PmfOffsets(),
                                     testPmf.PmfTable(), testPmf.SymbolBits(), testPmf.BypassBits())) {
        MLVC_LOG_ABORT("Failed to initialize EntropyEncoder: %s", ec.message().c_str());
    }
    rans::HeapResizableBuffer encodeBuffer{};
    rans::RansEncoderStream encodeStream{};
    if (auto ec = encodeStream.Initialize(rans::RansVariant::RansByte, encodeBuffer)) {
        MLVC_LOG_ABORT("Failed to initialize RansEncoderStream: %s", ec.message().c_str());
    }
    if (auto ec = encoder.Encode(encodeStream, indices, values)) {
        MLVC_LOG_ABORT("Failed to encode: %s", ec.message().c_str());
    }
    auto bitstream = encodeStream.Flush();
    // Copy bitstream to a persistent vector (Flush span is invalidated on next use)
    std::vector<std::byte> bitstreamCopy(bitstream.begin(), bitstream.end());

    // Initialize decoder
    rans::EntropyDecoder decoder;
    if (auto ec = decoder.Initialize(rans::RansVariant::RansByte, testPmf.PmfLengths(), testPmf.PmfOffsets(),
                                     testPmf.PmfTable(), testPmf.SymbolBits(), testPmf.BypassBits())) {
        MLVC_LOG_ABORT("Failed to initialize EntropyDecoder: %s", ec.message().c_str());
    }
    std::vector<int32_t> decoded(numSymbols);

    for (auto _ : state) {
        if (auto ec = decoder.Decode(decoded, indices, std::span<const std::byte>(bitstreamCopy))) {
            MLVC_LOG_ABORT("Failed to decode: %s", ec.message().c_str());
        }
    }
}
BENCHMARK(BM_RansDecodeGaussian);

static void BM_RansEncodeBitEst(benchmark::State& state)
{
    auto testPmf = RansTestPmf::MakeBitEstimatorPmf();

    // 960x540 BitEstimator latent: 48ch x 9h x 15w = 6480 symbols/segment, 1 segment
    const int numSymbols = 6480;
    std::vector<int32_t> indices;
    std::vector<int32_t> values;
    testPmf.GenerateSymbols(indices, values, numSymbols);

    // Initialize encoder
    rans::EntropyEncoder encoder;
    if (auto ec = encoder.Initialize(rans::RansVariant::RansByte, testPmf.PmfLengths(), testPmf.PmfOffsets(),
                                     testPmf.PmfTable(), testPmf.SymbolBits(), testPmf.BypassBits())) {
        MLVC_LOG_ABORT("Failed to initialize EntropyEncoder: %s", ec.message().c_str());
    }
    rans::HeapResizableBuffer encodeBuffer{};
    rans::RansEncoderStream encodeStream{};
    if (auto ec = encodeStream.Initialize(rans::RansVariant::RansByte, encodeBuffer)) {
        MLVC_LOG_ABORT("Failed to initialize RansEncoderStream: %s", ec.message().c_str());
    }

    for (auto _ : state) {
        if (auto ec = encoder.Encode(encodeStream, indices, values)) {
            MLVC_LOG_ABORT("Failed to encode: %s", ec.message().c_str());
        }
        [[maybe_unused]] auto bitstream = encodeStream.Flush();
    }
}
BENCHMARK(BM_RansEncodeBitEst);

static void BM_RansDecodeBitEst(benchmark::State& state)
{
    auto testPmf = RansTestPmf::MakeBitEstimatorPmf();

    // 960x540 BitEstimator latent: 48ch x 9h x 15w = 6480 symbols/segment, 1 segment
    const int numSymbols = 6480;
    std::vector<int32_t> indices;
    std::vector<int32_t> values;
    testPmf.GenerateSymbols(indices, values, numSymbols);

    // Encode once to produce bitstream for decoding
    rans::EntropyEncoder encoder;
    if (auto ec = encoder.Initialize(rans::RansVariant::RansByte, testPmf.PmfLengths(), testPmf.PmfOffsets(),
                                     testPmf.PmfTable(), testPmf.SymbolBits(), testPmf.BypassBits())) {
        MLVC_LOG_ABORT("Failed to initialize EntropyEncoder: %s", ec.message().c_str());
    }
    rans::HeapResizableBuffer encodeBuffer{};
    rans::RansEncoderStream encodeStream{};
    if (auto ec = encodeStream.Initialize(rans::RansVariant::RansByte, encodeBuffer)) {
        MLVC_LOG_ABORT("Failed to initialize RansEncoderStream: %s", ec.message().c_str());
    }
    if (auto ec = encoder.Encode(encodeStream, indices, values)) {
        MLVC_LOG_ABORT("Failed to encode: %s", ec.message().c_str());
    }
    auto bitstream = encodeStream.Flush();
    std::vector<std::byte> bitstreamCopy(bitstream.begin(), bitstream.end());

    // Initialize decoder
    rans::EntropyDecoder decoder;
    if (auto ec = decoder.Initialize(rans::RansVariant::RansByte, testPmf.PmfLengths(), testPmf.PmfOffsets(),
                                     testPmf.PmfTable(), testPmf.SymbolBits(), testPmf.BypassBits())) {
        MLVC_LOG_ABORT("Failed to initialize EntropyDecoder: %s", ec.message().c_str());
    }
    std::vector<int32_t> decoded(numSymbols);

    for (auto _ : state) {
        if (auto ec = decoder.Decode(decoded, indices, std::span<const std::byte>(bitstreamCopy))) {
            MLVC_LOG_ABORT("Failed to decode: %s", ec.message().c_str());
        }
    }
}
BENCHMARK(BM_RansDecodeBitEst);
