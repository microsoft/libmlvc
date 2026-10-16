// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/transforms.hpp"
#include "libmlvc/common/tensor.hpp"
#include "libmlvc/schema/bundle.hpp"

#include <libmlvc/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace libmlvc;

// -----------------------------------------------------------------------------
// Test data helpers
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

// -----------------------------------------------------------------------------
// Comparison helpers
// -----------------------------------------------------------------------------

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

Nv12FrameView Yuv444Fp16ToNv12Scalar(const Tensor<uint16_t, 3>& yuv444, std::vector<std::byte>& buffer,
                                     TensorLayout layout = TensorLayout::NCHW)
{
    const auto shape = yuv444.Shape();
    const auto strides = yuv444.Strides();
    auto inputData = yuv444.Data();

    auto fp16ToByte = [](const uint16_t value) -> std::byte {
        const float f = Fp16To32(value) * 255.0f;
        return static_cast<std::byte>(std::clamp(std::roundf(f), 0.0f, 255.0f));
    };

    const int height = shape[layout == TensorLayout::NHWC ? 0 : 1];
    const int width = shape[layout == TensorLayout::NHWC ? 1 : 2];
    const int rowStride = strides[layout == TensorLayout::NHWC ? 0 : 1];
    const int columnStride = strides[layout == TensorLayout::NHWC ? 1 : 2];
    const int channelStride = strides[layout == TensorLayout::NHWC ? 2 : 0];
    buffer.resize(static_cast<std::size_t>(height * width) * 3 / 2);

    // Y-channel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int inputIndex = y * rowStride + x * columnStride;
            const int outputIndex = y * width + x;
            buffer[outputIndex] = fp16ToByte(inputData[inputIndex]);
        }
    }

    // UV-channels
    auto downsampleUv = [&inputData, rowStride, columnStride, channelStride](const int ch, const int y,
                                                                             const int x) -> std::byte {
        const int i1 = ch * channelStride + 2 * y * rowStride + 2 * x * columnStride;
        const int i2 = i1 + columnStride;
        const int i3 = i1 + rowStride;
        const int i4 = i3 + columnStride;
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
// NV12 transpose
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

// -----------------------------------------------------------------------------
// NV12 -> YUV444 FP16
// -----------------------------------------------------------------------------

TEST(UnitTestTransforms, Nv12ToYuv444Fp16AllByteValuesInEveryPosition)
{
    constexpr int height = 512;
    constexpr int modelHeight = height + 2;
    constexpr uint16_t sentinel = 0x3555;
    for (int width = 2; width <= 66; width += 2) {
        SCOPED_TRACE(width);
        for (const bool sourcePadded : { false, true }) {
            SCOPED_TRACE(sourcePadded);
            const int sourceStride = width + (sourcePadded ? 7 : 0);
            const int sourceOffset = sourcePadded ? 1 : 0;
            auto bytes = std::vector<std::byte>(sourceStride * height * 3 / 2 + 2, std::byte{ 0xa5 });
            std::byte* luma = bytes.data() + sourceOffset;
            std::byte* chroma = luma + sourceStride * height;
            for (int value = 0; value < 256; ++value) {
                for (int column = 0; column < width; ++column) {
                    luma[2 * value * sourceStride + column] = static_cast<std::byte>((value + column) & 0xff);
                    luma[(2 * value + 1) * sourceStride + column] =
                        static_cast<std::byte>((255 - value + 3 * column) & 0xff);
                }
                for (int column = 0; column < width; column += 2) {
                    chroma[value * sourceStride + column] = static_cast<std::byte>((value + column) & 0xff);
                    chroma[value * sourceStride + column + 1] = static_cast<std::byte>((255 - value + 3 * column) & 0xff);
                }
            }
            const Nv12FrameView frame{ width, height, sourceStride, luma, chroma };
            Tensor<uint16_t, 3> reference{ std::array{ 3, height, width } };
            Nv12ToYuv444Fp16Scalar(frame, reference);

            for (const auto layout : { TensorLayout::NCHW, TensorLayout::NHWC }) {
                SCOPED_TRACE(layout == TensorLayout::NCHW ? "NCHW" : "NHWC");
                for (const bool aligned : { false, true }) {
                    SCOPED_TRACE(aligned);
                    const auto shape = layout == TensorLayout::NHWC ? std::array{ modelHeight, width, 3 }
                                                                    : std::array{ 3, modelHeight, width };
                    const int rowElements = layout == TensorLayout::NHWC ? width * 3 : width;
                    const int rowStride = aligned ? ((rowElements + 15) & ~15) : rowElements + 5;
                    const auto strides = layout == TensorLayout::NHWC
                                             ? std::array{ rowStride, 3, 1 }
                                             : std::array{ rowStride * modelHeight, rowStride, 1 };
                    const auto dataSize = static_cast<std::size_t>(shape[0]) * strides[0];
                    auto storage = std::vector<uint16_t>(dataSize + 32, sentinel);
                    const auto alignedOffset =
                        ((32 - reinterpret_cast<std::uintptr_t>(storage.data()) % 32) % 32) / sizeof(uint16_t);
                    const auto offset = alignedOffset + (aligned ? 0 : 1);
                    Tensor<uint16_t, 3> output{ shape, strides, std::span{ storage }.subspan(offset, dataSize) };
                    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(output.Data().data()) % 32, aligned ? 0 : 2);
                    auto expectedStorage = storage;
                    for (int row = 0; row < height; ++row) {
                        for (int column = 0; column < width; ++column) {
                            for (int channel = 0; channel < 3; ++channel) {
                                const int index = layout == TensorLayout::NHWC
                                                      ? row * strides[0] + column * 3 + channel
                                                      : channel * strides[0] + row * strides[1] + column;
                                expectedStorage[offset + index] = reference(channel, row, column);
                            }
                        }
                    }
                    if (layout == TensorLayout::NHWC) {
                        Nv12ToYuv444Fp16Nhwc(frame, output);
                    } else {
                        Nv12ToYuv444Fp16Nchw(frame, output);
                    }
                    ASSERT_EQ(storage, expectedStorage);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// YUV444 FP16 -> NV12
// -----------------------------------------------------------------------------

TEST(UnitTestTransforms, Yuv444Fp16ToNv12)
{
    std::mt19937 rng(42);
    for (auto [width, height] : std::initializer_list<std::pair<int, int>>{
             { 960, 540 },
             { 34, 22 },
             { 14, 4 },
         }) {
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
        auto tensor = MakeRandomFp16Tensor(3, height, width, rng);
        std::uniform_int_distribution<uint32_t> values{ 0, 0xf7ff };
        for (auto& value : tensor.Data()) {
            const auto index = values(rng);
            value = static_cast<uint16_t>(index < 0x7c00 ? index : index + 0x400);
        }

        std::vector<std::byte> expectedBuffer{};
        const auto expected = Yuv444Fp16ToNv12Scalar(tensor, expectedBuffer);
        for (const auto layout : { TensorLayout::NCHW, TensorLayout::NHWC }) {
            SCOPED_TRACE(layout == TensorLayout::NCHW ? "NCHW" : "NHWC");
            for (const bool padded : { false, true }) {
                SCOPED_TRACE(padded);
                const auto shape =
                    layout == TensorLayout::NHWC ? std::array{ height, width, 3 } : std::array{ 3, height, width };
                auto strides = StridesFromShape<3>(shape);
                if (padded) {
                    strides = layout == TensorLayout::NHWC ? std::array{ width * 3 + 5, 3, 1 }
                                                           : std::array{ (width + 5) * height + 7, width + 5, 1 };
                }
                const auto dataSize = static_cast<std::size_t>(shape[0]) * strides[0];
                auto storage = std::vector<uint16_t>(dataSize + 2, 0x7e00);
                Tensor<uint16_t, 3> input{ shape, strides, std::span{ storage }.subspan(padded ? 1 : 0, dataSize) };
                for (int row = 0; row < height; ++row) {
                    for (int column = 0; column < width; ++column) {
                        for (int channel = 0; channel < 3; ++channel) {
                            const int index = layout == TensorLayout::NHWC
                                                  ? row * strides[0] + column * 3 + channel
                                                  : channel * strides[0] + row * strides[1] + column;
                            input.Data()[index] = tensor(channel, row, column);
                        }
                    }
                }
                std::vector<std::byte> actualBuffer{};
                const auto actual = layout == TensorLayout::NHWC ? Yuv444Fp16NhwcToNv12(input, actualBuffer)
                                                                 : Yuv444Fp16NchwToNv12(input, actualBuffer);
                ASSERT_NO_FATAL_FAILURE(AssertNv12Equal(expected, actual));
            }
        }
    }
}

TEST(UnitTestTransforms, Yuv444Fp16AllFiniteValuesInEveryPosition)
{
    constexpr int finiteValueCount = 0xf800;
    constexpr int batchSize = 512;
    constexpr int height = 2 * batchSize;
    for (int width = 2; width <= 66; width += 2) {
        SCOPED_TRACE(width);
        for (const auto layout : { TensorLayout::NCHW, TensorLayout::NHWC }) {
            SCOPED_TRACE(layout == TensorLayout::NCHW ? "NCHW" : "NHWC");
            for (const bool padded : { false, true }) {
                SCOPED_TRACE(padded);
                const auto shape =
                    layout == TensorLayout::NHWC ? std::array{ height, width, 3 } : std::array{ 3, height, width };
                auto strides = StridesFromShape<3>(shape);
                if (padded) {
                    strides = layout == TensorLayout::NHWC ? std::array{ width * 3 + 5, 3, 1 }
                                                           : std::array{ (width + 5) * height + 7, width + 5, 1 };
                }
                const auto dataSize = static_cast<std::size_t>(shape[0]) * strides[0];
                auto storage = std::vector<uint16_t>(dataSize + 32, 0x7e00);
                const auto alignedOffset =
                    ((32 - reinterpret_cast<std::uintptr_t>(storage.data()) % 32) % 32) / sizeof(uint16_t);
                const auto offset = alignedOffset + (padded ? 1 : 0);
                Tensor<uint16_t, 3> input{ shape, strides, std::span{ storage }.subspan(offset, dataSize) };
                const int rowStride = strides[layout == TensorLayout::NHWC ? 0 : 1];
                const int columnStride = strides[layout == TensorLayout::NHWC ? 1 : 2];
                const int channelStride = strides[layout == TensorLayout::NHWC ? 2 : 0];
                std::vector<std::byte> expectedBuffer{};
                std::vector<std::byte> actualBuffer{};
                for (int firstIndex = 0; firstIndex < finiteValueCount; firstIndex += batchSize) {
                    SCOPED_TRACE(firstIndex);
                    for (int sample = 0; sample < batchSize; ++sample) {
                        const int index = firstIndex + sample;
                        const auto value = static_cast<uint16_t>(index < 0x7c00 ? index : index + 0x400);
                        for (int row = 2 * sample; row < 2 * sample + 2; ++row) {
                            uint16_t* luma = input.Data().data() + row * rowStride;
                            uint16_t* chromaU = luma + channelStride;
                            uint16_t* chromaV = chromaU + channelStride;
                            for (int column = 0; column < width; ++column) {
                                luma[column * columnStride] = value;
                                chromaU[column * columnStride] = value;
                                chromaV[column * columnStride] = value ^ 0x8000;
                            }
                        }
                    }
                    Yuv444Fp16ToNv12Scalar(input, expectedBuffer, layout);
                    if (layout == TensorLayout::NHWC) {
                        Yuv444Fp16NhwcToNv12(input, actualBuffer);
                    } else {
                        Yuv444Fp16NchwToNv12(input, actualBuffer);
                    }
                    ASSERT_EQ(actualBuffer, expectedBuffer);
                }
            }
        }
    }
}

TEST(UnitTestTransforms, Yuv444Fp16ChromaRoundingBoundaries)
{
    constexpr int offsetsPerBoundary = 33;
    constexpr int ordersPerOffset = 24;
    constexpr int casesPerBoundary = offsetsPerBoundary * ordersPerOffset;
    constexpr double unitsPerOne = 16777216.0;
    std::vector<std::array<uint16_t, 4>> cases{ { 0x3814, 0x0000, 0x0283, 0x0000 },
                                                { 0x2000, 0x0000, 0x0202, 0x0000 },
                                                { 0x7bff, 0xfbff, 0x2010, 0x0000 },
                                                { 0x7bff, 0x2010, 0xfbff, 0x0000 },
                                                { 0xfbff, 0xfbff, 0xfbff, 0xfbff } };
    const auto regressionCaseCount = cases.size();
    cases.reserve(regressionCaseCount + 256 * casesPerBoundary);
    for (int boundary = 0; boundary < 256; ++boundary) {
        const int sumUnits = static_cast<int>(std::floor((boundary + 0.5) * (4.0 / 255.0) * unitsPerOne));
        bool foundBelow = false;
        bool foundAbove = false;
        for (int offset = -16; offset <= 16; ++offset) {
            float remaining = static_cast<float>((sumUnits + offset) / unitsPerOne);
            std::array<uint16_t, 4> values{};
            for (auto& value : values) {
                value = Fp32To16(remaining);
                remaining -= Fp16To32(value);
            }
            ASSERT_EQ(remaining, 0.0f);
            const float sum = Fp16To32(values[0]) + Fp16To32(values[1]) + Fp16To32(values[2]) + Fp16To32(values[3]);
            const float scaled = 255.0f * (sum * 0.25f);
            foundBelow |= scaled < boundary + 0.5f;
            foundAbove |= scaled > boundary + 0.5f;
            std::array<int, 4> order{ 0, 1, 2, 3 };
            do {
                cases.push_back({ values[order[0]], values[order[1]], values[order[2]], values[order[3]] });
            } while (std::next_permutation(order.begin(), order.end()));
        }
        ASSERT_TRUE(foundBelow && foundAbove) << boundary;
    }
    ASSERT_EQ(cases.size(), regressionCaseCount + 256 * casesPerBoundary);
    std::sort(cases.begin(), cases.end());
    cases.erase(std::unique(cases.begin(), cases.end()), cases.end());

    constexpr int batchSize = 512;
    constexpr int height = 2 * batchSize;
    for (int width = 2; width <= 66; width += 2) {
        SCOPED_TRACE(width);
        for (const auto layout : { TensorLayout::NCHW, TensorLayout::NHWC }) {
            SCOPED_TRACE(layout == TensorLayout::NCHW ? "NCHW" : "NHWC");
            const auto shape =
                layout == TensorLayout::NHWC ? std::array{ height, width, 3 } : std::array{ 3, height, width };
            const auto strides = layout == TensorLayout::NHWC ? std::array{ width * 3 + 5, 3, 1 }
                                                              : std::array{ (width + 5) * height + 7, width + 5, 1 };
            const auto dataSize = static_cast<std::size_t>(shape[0]) * strides[0];
            auto storage = std::vector<uint16_t>(dataSize + 2, 0x7e00);
            Tensor<uint16_t, 3> input{ shape, strides, std::span{ storage }.subspan(1, dataSize) };
            const int rowStride = strides[layout == TensorLayout::NHWC ? 0 : 1];
            const int columnStride = strides[layout == TensorLayout::NHWC ? 1 : 2];
            const int channelStride = strides[layout == TensorLayout::NHWC ? 2 : 0];
            std::vector<std::byte> expectedBuffer{};
            std::vector<std::byte> actualBuffer{};
            for (std::size_t firstCase = 0; firstCase < cases.size(); firstCase += batchSize) {
                SCOPED_TRACE(firstCase);
                for (int sample = 0; sample < batchSize; ++sample) {
                    const auto caseIndex = (firstCase + sample) % cases.size();
                    const auto& valuesU = cases[caseIndex];
                    const auto& valuesV = cases[(caseIndex + casesPerBoundary) % cases.size()];
                    for (int row = 2 * sample; row < 2 * sample + 2; ++row) {
                        uint16_t* luma = input.Data().data() + row * rowStride;
                        uint16_t* chromaU = luma + channelStride;
                        uint16_t* chromaV = chromaU + channelStride;
                        for (int column = 0; column < width; ++column) {
                            const int position = 2 * (row % 2) + column % 2;
                            luma[column * columnStride] = 0x3800;
                            chromaU[column * columnStride] = valuesU[position];
                            chromaV[column * columnStride] = valuesV[position];
                        }
                    }
                }
                Yuv444Fp16ToNv12Scalar(input, expectedBuffer, layout);
                if (layout == TensorLayout::NHWC) {
                    Yuv444Fp16NhwcToNv12(input, actualBuffer);
                } else {
                    Yuv444Fp16NchwToNv12(input, actualBuffer);
                }
                ASSERT_EQ(actualBuffer, expectedBuffer);
            }
        }
    }
}

TEST(UnitTestTransforms, NhwcConversionMatchesPlanarFp16Range)
{
    constexpr int finiteValueCount = 0xf800;
    for (const int width : { 6, 8, 14, 18, 256 }) {
        SCOPED_TRACE(width);
        const int height = 2 * ((finiteValueCount + 2 * width - 1) / (2 * width));
        Tensor<uint16_t, 3> nchw{ std::array{ 3, height, width } };
        Tensor<uint16_t, 3> nhwc{ std::array{ height, width, 3 } };
        nchw.SetValue(0);
        nhwc.SetValue(0);
        int pixel = 0;
        for (int bits = 0; bits <= 0xffff; ++bits) {
            if ((bits & 0x7c00) == 0x7c00) continue;
            const int row = pixel / width;
            const int column = pixel % width;
            for (int channel = 0; channel < 3; ++channel) {
                const auto value = static_cast<uint16_t>(channel == 1 ? bits ^ 0x8000 : bits);
                nchw(channel, row, column) = value;
                nhwc(row, column, channel) = value;
            }
            ++pixel;
        }
        ASSERT_EQ(pixel, finiteValueCount);
        std::vector<std::byte> planarOutput{};
        std::vector<std::byte> interleavedOutput{};
        const auto expected = Yuv444Fp16NchwToNv12(nchw, planarOutput);
        const auto actual = Yuv444Fp16NhwcToNv12(nhwc, interleavedOutput);
        ASSERT_NO_FATAL_FAILURE(AssertNv12Equal(expected, actual));
    }
}

// -----------------------------------------------------------------------------
// NV12 / YUV444 round trips and strides
// -----------------------------------------------------------------------------

TEST(UnitTestTransforms, ImageLayoutsPreserveRowStrides)
{
    auto random = std::mt19937{ 42 };
    constexpr uint16_t sentinel = 0x3555;
    for (const auto [width, height] :
         std::initializer_list<std::pair<int, int>>{ { 2, 2 }, { 16, 4 }, { 34, 6 }, { 66, 6 }, { 960, 540 } }) {
        SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
        const auto bytes = MakeRandomNv12(width, height, random);
        const int sourceStride = width + 6;
        auto padded = std::vector<std::byte>(sourceStride * height * 3 / 2 + 2, std::byte{ 0xa5 });
        for (int row = 0; row < height * 3 / 2; ++row) {
            std::copy_n(bytes.data() + row * width, width, padded.data() + 1 + row * sourceStride);
        }
        const Nv12FrameView input{ width, height, sourceStride, padded.data() + 1,
                                   padded.data() + 1 + height * sourceStride };
        const Nv12FrameView expectedFrame{ width, height, bytes };
        Tensor<uint16_t, 3> planarReference{ std::array{ 3, height, width } };
        Nv12ToYuv444Fp16Scalar(input, planarReference);

        for (const auto layout : { TensorLayout::NCHW, TensorLayout::NHWC }) {
            const auto shape =
                layout == TensorLayout::NHWC ? std::array{ height, width, 3 } : std::array{ 3, height, width };
            const auto strides = layout == TensorLayout::NHWC ? std::array{ width * 3 + 5, 3, 1 }
                                                              : std::array{ (width + 5) * height + 7, width + 5, 1 };
            const auto dataSize = static_cast<std::size_t>(shape[0]) * strides[0];
            auto storage = std::vector<uint16_t>(dataSize + 2, sentinel);
            auto expectedStorage = storage;
            Tensor<uint16_t, 3> tensor{ shape, strides, std::span{ storage }.subspan(1, dataSize) };
            if (layout == TensorLayout::NHWC) {
                Nv12ToYuv444Fp16Nhwc(input, tensor);
            } else {
                Nv12ToYuv444Fp16Nchw(input, tensor);
            }
            for (int row = 0; row < height; ++row) {
                for (int column = 0; column < width; ++column) {
                    for (int channel = 0; channel < 3; ++channel) {
                        const int offset = layout == TensorLayout::NHWC ? row * strides[0] + column * 3 + channel
                                                                        : channel * strides[0] + row * strides[1] + column;
                        expectedStorage[1 + offset] = planarReference(channel, row, column);
                    }
                }
            }
            EXPECT_EQ(storage, expectedStorage);
            std::vector<std::byte> reconstructed{};
            const auto actual = layout == TensorLayout::NHWC ? Yuv444Fp16NhwcToNv12(tensor, reconstructed)
                                                             : Yuv444Fp16NchwToNv12(tensor, reconstructed);
            ASSERT_NO_FATAL_FAILURE(AssertNv12Equal(expectedFrame, actual));
        }
    }
}

// -----------------------------------------------------------------------------
// Tensor numeric conversions
// -----------------------------------------------------------------------------

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
