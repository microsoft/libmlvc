// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/tensor.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace libmlvc;

// ------------------------------------------------------------
// Benchmarks for memory operations
// ------------------------------------------------------------

static void BM_TensorMemset540p(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    std::vector<uint16_t> buffer(height * width * 3, 16789);
    for (auto _ : state) {
        std::memset(buffer.data(), 0, buffer.size() * sizeof(uint16_t));
        benchmark::DoNotOptimize(buffer.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_TensorMemset540p);

static void BM_TensorFill540p(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    std::vector<uint16_t> buffer(height * width * 3, 16789);
    for (auto _ : state) {
        std::fill(buffer.begin(), buffer.end(), 0);
        benchmark::DoNotOptimize(buffer.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_TensorFill540p);

static void BM_TensorFillLoop540p(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    std::vector<uint16_t> buffer(height * width * 3, 16789);
    auto data = buffer.data();
    for (auto _ : state) {
        for (int i = 0; i < height * width * 3; ++i) {
            data[i] = 123;
        }
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_TensorFillLoop540p);

static void BM_TensorFillLoops540p(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    std::vector<uint16_t> buffer(height * width * 3, 16789);
    auto data = buffer.data();
    const int strideY = width * 3;
    const int strideX = 3;
    for (auto _ : state) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int idx = y * strideY + x * strideX;
                data[idx + 0] = 123;
                data[idx + 1] = 456;
                data[idx + 2] = 789;
            }
        }
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_TensorFillLoops540p);

static void BM_TensorCopy540p(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    std::vector<uint16_t> srcbuffer(height * width * 3, 16789);
    std::vector<uint16_t> dstbuffer(srcbuffer.size(), 16789);
    for (auto _ : state) {
        std::copy(srcbuffer.begin(), srcbuffer.end(), dstbuffer.begin());
        benchmark::DoNotOptimize(dstbuffer.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_TensorCopy540p);

static void BM_TensorMemcpy540p(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    std::vector<uint16_t> srcbuffer(height * width * 3, 16789);
    std::vector<uint16_t> dstbuffer(srcbuffer.size(), 16789);
    for (auto _ : state) {
        std::memcpy(dstbuffer.data(), srcbuffer.data(), srcbuffer.size() * sizeof(uint16_t));
        benchmark::DoNotOptimize(dstbuffer.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_TensorMemcpy540p);

static void BM_TensorAllocCopy540p(benchmark::State& state)
{
    const int width = 960;
    const int height = 540;
    std::vector<uint16_t> srcbuffer(height * width * 3, 16789);
    for (auto _ : state) {
        std::vector<uint16_t> dstbuffer(srcbuffer.size(), 16789);
        std::copy(srcbuffer.begin(), srcbuffer.end(), dstbuffer.begin());
        benchmark::DoNotOptimize(dstbuffer.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_TensorAllocCopy540p);

static void BM_TensorSetZero(benchmark::State& state)
{
    Tensor<uint16_t, 3> tensor({ 32, 184, 320 });

    for (auto _ : state) {
        tensor.SetZero();
    }
}
BENCHMARK(BM_TensorSetZero);
