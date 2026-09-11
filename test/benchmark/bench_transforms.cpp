// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/frame_processing.hpp"
#include "libmlvc/codec/core/transforms.hpp"
#include "libmlvc/common/fp16.hpp"
#include "libmlvc/common/tensor.hpp"
#include "libmlvc/types.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace libmlvc;

// ------------------------------------------------------------
// Benchmarks for image format transforms
// ------------------------------------------------------------

static void BM_Nv12ToYuv444Fp16(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    const int modelWidth = 960;
    const int modelHeight = 544;
    std::vector<std::byte> bytes(height * width * 3 / 2, std::byte(128));
    Nv12FrameView frame{ width, height, bytes };
    Tensor<uint16_t, 3> output({ 3, modelHeight, modelWidth });
    output.SetValue(Fp32To16(0.5f));
    for (auto _ : state) {
        Nv12ToYuv444Fp16(frame, output);
    }
}
BENCHMARK(BM_Nv12ToYuv444Fp16);

static void BM_Yuv444Fp16ToNv12(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    Tensor<uint16_t, 3> yuv444({ 3, height, width });
    yuv444.SetValue(Fp32To16(0.5f));
    std::vector<std::byte> buffer;
    for (auto _ : state) {
        Yuv444Fp16ToNv12(yuv444, buffer);
    }
}
BENCHMARK(BM_Yuv444Fp16ToNv12);

// ------------------------------------------------------------
// Benchmarks for other transforms
// ------------------------------------------------------------

static void BM_TransposeNv12(benchmark::State& state)
{
    const int width = 540;
    const int height = 960;
    std::vector<std::byte> inputData(height * width * 3 / 2);
    std::vector<std::byte> outputData(height * width * 3 / 2);
    Nv12FrameView inputFrame{ width, height, inputData };
    for (auto _ : state) {
        TransposeNv12(inputFrame, outputData);
    }
}
BENCHMARK(BM_TransposeNv12);

// ------------------------------------------------------------
// Benchmarks for input and output transformers
// ------------------------------------------------------------

static void BM_InputTransformerLandscape(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    const int modelWidth = 960;
    const int modelHeight = 544;
    std::vector<std::byte> bytes(height * width * 3 / 2, std::byte(128));
    Nv12FrameView frame{ width, height, bytes };
    Tensor<uint16_t, 3> output({ 3, modelHeight, modelWidth });

    InputTransformer inputTransformer;
    for (auto _ : state) {
        inputTransformer.Transform(frame, modelWidth, modelHeight, output);
    }
}
BENCHMARK(BM_InputTransformerLandscape);

static void BM_InputTransformerPortrait(benchmark::State& state)
{
    const int width = 540;
    const int height = 960;
    const int modelWidth = 960;
    const int modelHeight = 544;
    std::vector<std::byte> bytes(height * width * 3 / 2, std::byte(128));
    Nv12FrameView frame{ width, height, bytes };
    Tensor<uint16_t, 3> output({ 3, modelHeight, modelWidth });

    InputTransformer inputTransformer;
    for (auto _ : state) {
        inputTransformer.Transform(frame, modelWidth, modelHeight, output);
    }
}
BENCHMARK(BM_InputTransformerPortrait);

static void BM_OutputTransformerLandscape(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    Tensor<uint16_t, 3> modelOutput({ 3, 544, 960 });

    OutputTransformer<mlvc_f16_t> outputTransformer;
    outputTransformer.Initialize();
    for (auto _ : state) {
        auto res = outputTransformer.Transform(modelOutput, width, height, { 0, 0, 0, 8 }, false);
        (void)res;
    }
}
BENCHMARK(BM_OutputTransformerLandscape);

static void BM_OutputTransformerPortrait(benchmark::State& state)
{
    const int width = 540;
    const int height = 960;
    Tensor<uint16_t, 3> modelOutput({ 3, 544, 960 });

    OutputTransformer<mlvc_f16_t> outputTransformer;
    outputTransformer.Initialize();
    for (auto _ : state) {
        auto res = outputTransformer.Transform(modelOutput, width, height, { 0, 0, 0, 8 }, true);
        (void)res;
    }
}
BENCHMARK(BM_OutputTransformerPortrait);

// ------------------------------------------------------------
// Float16 / Int32 conversions
// ------------------------------------------------------------

static void BM_Float16ToInt32(benchmark::State& state)
{
    Tensor<uint16_t, 3> input({ 384, 34, 60 });
    Tensor<int32_t, 3> output({ 384, 34, 60 });
    for (auto _ : state) {
        Float16ToInt32(input, output);
    }
}
BENCHMARK(BM_Float16ToInt32);

static void BM_Int32ToFloat16(benchmark::State& state)
{
    Tensor<int32_t, 3> input({ 384, 34, 60 });
    Tensor<uint16_t, 3> output({ 384, 34, 60 });
    for (auto _ : state) {
        Int32ToFloat16(input, output);
    }
}
BENCHMARK(BM_Int32ToFloat16);
