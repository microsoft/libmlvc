// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/transforms.hpp"
#include "libmlvc/common/tensor.hpp"

#include <libmlvc/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace libmlvc;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

std::vector<std::byte> MakeRandomNv12(int width, int height, std::mt19937& rng)
{
    std::vector<std::byte> buf(width * height * 3 / 2);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& b : buf) {
        b = static_cast<std::byte>(dist(rng));
    }
    return buf;
}

Tensor<uint16_t, 3> MakeRandomFp16Tensor(int channels, int height, int width, std::mt19937& rng)
{
    Tensor<uint16_t, 3> tensor({ channels, height, width });
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : tensor.Data()) {
        val = Fp32To16(dist(rng));
    }
    return tensor;
}

Tensor<int32_t, 3> MakeRandomInt32Tensor(int channels, int height, int width, std::mt19937& rng)
{
    Tensor<int32_t, 3> tensor({ channels, height, width });
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (auto& val : tensor.Data()) {
        val = dist(rng);
    }
    return tensor;
}

void AssertNv12Equal(const Nv12FrameView& a, const Nv12FrameView& b)
{
    ASSERT_EQ(a.Width(), b.Width());
    ASSERT_EQ(a.Height(), b.Height());
    ASSERT_EQ(a.YPlane().size(), b.YPlane().size());
    ASSERT_EQ(a.UvPlane().size(), b.UvPlane().size());
    EXPECT_TRUE(std::ranges::equal(a.YPlane(), b.YPlane())) << "Y plane mismatch";
    EXPECT_TRUE(std::ranges::equal(a.UvPlane(), b.UvPlane())) << "UV plane mismatch";
}

template <typename T>
void AssertTensorEqual(const Tensor<T, 3>& a, const Tensor<T, 3>& b)
{
    ASSERT_EQ(a.Shape(), b.Shape());
    ASSERT_EQ(a.Data().size(), b.Data().size());
    EXPECT_TRUE(std::ranges::equal(a.Data(), b.Data())) << "Tensor data mismatch";
}

// -----------------------------------------------------------------------------
// Scalar reference implementations
// -----------------------------------------------------------------------------

Nv12FrameView TransposeNv12Scalar(const Nv12FrameView& frame, std::vector<std::byte>& buffer)
{
    const int width = frame.Width();
    const int height = frame.Height();
    const int stride = frame.Stride();

    buffer.resize(static_cast<std::size_t>(width) * height * 3 / 2);

    // Luma
    const std::byte* inputDataLuma = frame.YPlane().data();
    std::byte* outputDataLuma = buffer.data();
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            outputDataLuma[x * height + y] = inputDataLuma[y * stride + x];
        }
    }

    // Chroma
    const std::byte* inputDataChroma = frame.UvPlane().data();
    std::byte* outputDataChroma = outputDataLuma + width * height;
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width / 2; x++) {
            outputDataChroma[x * height + 2 * y] = inputDataChroma[y * stride + 2 * x];
            outputDataChroma[x * height + 2 * y + 1] = inputDataChroma[y * stride + 2 * x + 1];
        }
    }

    return Nv12FrameView{ height, width, buffer };
}

void Nv12ToYuv444Fp16Scalar(const Nv12FrameView& frame, Tensor<uint16_t, 3>& output)
{
    const int width = frame.Width();
    const int height = frame.Height();
    const int stride = frame.Stride();

    MLVC_ASSERT(output.Shape()[0] == 3);
    MLVC_ASSERT(output.Shape()[1] >= height);
    MLVC_ASSERT(output.Shape()[2] >= width);

    const auto outputData = output.Data();
    const auto outputStrides = output.Strides();
    MLVC_ASSERT(outputStrides[2] == 1);

    static auto lut = []() {
        std::array<uint16_t, 256> table;
        for (int i = 0; i < 256; i++) {
            const float fvalue = static_cast<float>(i) / 255.0f;
            table[i] = Fp32To16(fvalue);
        }
        return table;
    }();

    auto int8ToFp16 = [](uint8_t value) -> uint16_t { return lut[value]; };

    // Y-channel
    const uint8_t* lumaData = reinterpret_cast<const uint8_t*>(frame.YPlane().data());
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const uint8_t value = lumaData[y * stride + x];
            outputData[y * outputStrides[1] + x] = int8ToFp16(value);
        }
    }

    // U, V channels
    const uint8_t* chromaData = reinterpret_cast<const uint8_t*>(frame.UvPlane().data());
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width / 2; x++) {
            const int index = y * stride + 2 * x;
            const uint16_t u = int8ToFp16(chromaData[index]);
            const uint16_t v = int8ToFp16(chromaData[index + 1]);
            const int uPlaneIndex = 1 * outputStrides[0];
            const int vPlaneIndex = 2 * outputStrides[0];

            {
                outputData[uPlaneIndex + (2 * y) * outputStrides[1] + (2 * x)] = u;
                outputData[uPlaneIndex + (2 * y) * outputStrides[1] + (2 * x + 1)] = u;
                outputData[uPlaneIndex + (2 * y + 1) * outputStrides[1] + (2 * x)] = u;
                outputData[uPlaneIndex + (2 * y + 1) * outputStrides[1] + (2 * x + 1)] = u;
            }

            {
                outputData[vPlaneIndex + (2 * y) * outputStrides[1] + (2 * x)] = v;
                outputData[vPlaneIndex + (2 * y) * outputStrides[1] + (2 * x + 1)] = v;
                outputData[vPlaneIndex + (2 * y + 1) * outputStrides[1] + (2 * x)] = v;
                outputData[vPlaneIndex + (2 * y + 1) * outputStrides[1] + (2 * x + 1)] = v;
            }
        }
    }
}

Nv12FrameView Yuv444Fp16ToNv12Scalar(const Tensor<uint16_t, 3>& yuv444, std::vector<std::byte>& buffer)
{
    const auto shape = yuv444.Shape();
    const auto strides = yuv444.Strides();
    auto inputData = yuv444.Data();

    auto fp16ToByte = [](const uint16_t value) -> std::byte {
        const float f = Fp16To32(value) * 255.0f;
        return static_cast<std::byte>(std::clamp(std::roundf(f), 0.0f, 255.0f));
    };

    const int height = shape[1];
    const int width = shape[2];
    buffer.resize(static_cast<std::size_t>(height * width) * 3 / 2);

    // Y-channel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int inputIndex = y * strides[1] + x;
            const int outputIndex = y * width + x;
            buffer[outputIndex] = fp16ToByte(inputData[inputIndex]);
        }
    }

    // UV-channels
    auto downsampleUv = [&inputData, &strides](const int ch, const int y, const int x) -> std::byte {
        const int i1 = ch * strides[0] + 2 * y * strides[1] + 2 * x;
        const int i2 = ch * strides[0] + 2 * y * strides[1] + 2 * x + 1;
        const int i3 = ch * strides[0] + (2 * y + 1) * strides[1] + 2 * x;
        const int i4 = ch * strides[0] + (2 * y + 1) * strides[1] + 2 * x + 1;
        const float f1 = Fp16To32(inputData[i1]);
        const float f2 = Fp16To32(inputData[i2]);
        const float f3 = Fp16To32(inputData[i3]);
        const float f4 = Fp16To32(inputData[i4]);
        const float avg = (f1 + f2 + f3 + f4) * 0.25f;
        return static_cast<std::byte>(std::clamp(std::roundf(255.0f * avg), 0.0f, 255.0f));
    };

    const int halfHeight = height / 2;
    const int halfWidth = width / 2;
    for (int y = 0; y < halfHeight; y++) {
        for (int x = 0; x < halfWidth; x++) {
            const std::byte u = downsampleUv(1, y, x);
            const std::byte v = downsampleUv(2, y, x);
            const int outputIndexU = height * width + 2 * y * halfWidth + 2 * x;
            const int outputIndexV = outputIndexU + 1;
            buffer[outputIndexU] = u;
            buffer[outputIndexV] = v;
        }
    }

    return Nv12FrameView{ width, height, buffer };
}

void Float16ToInt32Scalar(const Tensor<uint16_t, 3>& input, Tensor<int32_t, 3>& output)
{
    output.Create(input.Shape());

    const auto inputShape = input.Shape();
    const auto inputData = input.Data();
    auto outputData = output.Data();
    const auto inputStrides = input.Strides();
    const auto outputStrides = output.Strides();

    for (int ch = 0; ch < inputShape[0]; ch++) {
        for (int y = 0; y < inputShape[1]; y++) {
            for (int x = 0; x < inputShape[2]; x++) {
                const int inputIndex = ch * inputStrides[0] + y * inputStrides[1] + x;
                const int outputIndex = ch * outputStrides[0] + y * outputStrides[1] + x;
                const auto fValue = Fp16To32(inputData[inputIndex]);
                outputData[outputIndex] = static_cast<int32_t>(fValue);
            }
        }
    }
}

void Int32ToFloat16Scalar(const Tensor<int32_t, 3>& input, Tensor<uint16_t, 3>& output)
{
    output.Create(input.Shape());
    const auto inputShape = input.Shape();
    const auto inputData = input.Data();
    auto outputData = output.Data();
    const auto inputStrides = input.Strides();
    const auto outputStrides = output.Strides();

    for (int ch = 0; ch < inputShape[0]; ch++) {
        for (int y = 0; y < inputShape[1]; y++) {
            for (int x = 0; x < inputShape[2]; x++) {
                const int inputIndex = ch * inputStrides[0] + y * inputStrides[1] + x;
                const int outputIndex = ch * outputStrides[0] + y * outputStrides[1] + x;
                const auto fValue = static_cast<float>(inputData[inputIndex]);
                outputData[outputIndex] = Fp32To16(fValue);
            }
        }
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// UnitTestTransforms
// -----------------------------------------------------------------------------

TEST(UnitTestTransforms, TransposeNv12)
{
    std::mt19937 rng(42);
    for (auto [width, height] : std::initializer_list<std::pair<int, int>>{
             { 960, 540 },
             { 640, 368 },
             { 432, 240 },
             { 320, 192 },
             { 34, 22 },
             { 16, 16 },
         }) {
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
        auto bytes = MakeRandomNv12(width, height, rng);
        Nv12FrameView frame{ width, height, bytes };

        std::vector<std::byte> bufDispatch, bufScalar;
        auto resultDispatch = TransposeNv12(frame, bufDispatch);
        auto resultScalar = TransposeNv12Scalar(frame, bufScalar);
        ASSERT_NO_FATAL_FAILURE(AssertNv12Equal(resultDispatch, resultScalar));
    }
}

TEST(UnitTestTransforms, Nv12ToYuv444Fp16)
{
    std::mt19937 rng(42);
    for (auto [width, height] : std::initializer_list<std::pair<int, int>>{
             { 960, 540 },
             { 640, 368 },
             { 432, 240 },
             { 320, 192 },
             { 34, 22 },
             { 16, 16 },
         }) {
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
        auto bytes = MakeRandomNv12(width, height, rng);
        Nv12FrameView frame{ width, height, bytes };

        Tensor<uint16_t, 3> outDispatch({ 3, height, width });
        Tensor<uint16_t, 3> outScalar({ 3, height, width });
        Nv12ToYuv444Fp16(frame, outDispatch);
        Nv12ToYuv444Fp16Scalar(frame, outScalar);
        ASSERT_NO_FATAL_FAILURE(AssertTensorEqual(outDispatch, outScalar));
    }
}

TEST(UnitTestTransforms, Yuv444Fp16ToNv12)
{
    std::mt19937 rng(42);
    for (auto [width, height] : std::initializer_list<std::pair<int, int>>{
             { 960, 540 },
             { 640, 368 },
             { 432, 240 },
             { 320, 192 },
             { 34, 22 },
             { 16, 16 },
         }) {
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
        auto tensor = MakeRandomFp16Tensor(3, height, width, rng);

        std::vector<std::byte> bufDispatch, bufScalar;
        auto resultDispatch = Yuv444Fp16ToNv12(tensor, bufDispatch);
        auto resultScalar = Yuv444Fp16ToNv12Scalar(tensor, bufScalar);
        ASSERT_NO_FATAL_FAILURE(AssertNv12Equal(resultDispatch, resultScalar));
    }
}

TEST(UnitTestTransforms, Float16ToInt32)
{
    std::mt19937 rng(42);
    for (auto [ch, h, w] : std::initializer_list<std::tuple<int, int, int>>{
             { 384, 34, 60 },
             { 3, 16, 16 },
             { 3, 34, 33 },
             { 128, 10, 8 },
         }) {
        SCOPED_TRACE(std::to_string(ch) + "x" + std::to_string(h) + "x" + std::to_string(w));
        auto input = MakeRandomFp16Tensor(ch, h, w, rng);

        Tensor<int32_t, 3> outDispatch({ ch, h, w });
        Tensor<int32_t, 3> outScalar({ ch, h, w });
        Float16ToInt32(input, outDispatch);
        Float16ToInt32Scalar(input, outScalar);
        ASSERT_NO_FATAL_FAILURE(AssertTensorEqual(outDispatch, outScalar));
    }
}

TEST(UnitTestTransforms, Int32ToFloat16)
{
    std::mt19937 rng(42);
    for (auto [ch, h, w] : std::initializer_list<std::tuple<int, int, int>>{
             { 384, 34, 60 },
             { 3, 16, 16 },
             { 3, 34, 33 },
             { 128, 10, 8 },
         }) {
        SCOPED_TRACE(std::to_string(ch) + "x" + std::to_string(h) + "x" + std::to_string(w));
        auto input = MakeRandomInt32Tensor(ch, h, w, rng);

        Tensor<uint16_t, 3> outDispatch({ ch, h, w });
        Tensor<uint16_t, 3> outScalar({ ch, h, w });
        Int32ToFloat16(input, outDispatch);
        Int32ToFloat16Scalar(input, outScalar);
        ASSERT_NO_FATAL_FAILURE(AssertTensorEqual(outDispatch, outScalar));
    }
}
