// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/frame_processing.hpp"
#include "libmlvc/common/fp16.hpp"
#include "libmlvc/common/tensor.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace libmlvc;

namespace {

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

void AssertNv12Equal(const Nv12FrameView& a, const Nv12FrameView& b)
{
    ASSERT_EQ(a.Width(), b.Width());
    ASSERT_EQ(a.Height(), b.Height());
    ASSERT_EQ(a.YPlane().size(), b.YPlane().size());
    ASSERT_EQ(a.UvPlane().size(), b.UvPlane().size());
    EXPECT_TRUE(std::ranges::equal(a.YPlane(), b.YPlane())) << "Y plane mismatch";
    EXPECT_TRUE(std::ranges::equal(a.UvPlane(), b.UvPlane())) << "UV plane mismatch";
}

}  // namespace

TEST(UnitTestFrameProcessing, ImageLayoutsPreserveBoundBuffers)
{
    struct TestCase {
        int width;
        int height;
        int modelWidth;
        int modelHeight;
        bool transposed;
        int padRight;
        int padBottom;
    };
    // Expected rotation and right/bottom padding; 16x20 fits the model both ways and must not be rotated
    const TestCase testCases[] = {
        { 34, 22, 40, 24, false, 6, 2 }, { 22, 34, 40, 24, true, 6, 2 }, { 16, 20, 40, 24, false, 24, 4 },
        { 32, 16, 32, 16, false, 0, 0 }, { 2, 2, 4, 4, false, 2, 2 },
    };
    constexpr uint16_t unwritten = 0x7e00;  // NaN, never produced by the transformers
    const uint16_t gray = Fp32To16(0.5f);
    std::mt19937 random{ 42 };
    for (const auto& testCase : testCases) {
        const auto [width, height, modelWidth, modelHeight, transposed, padRight, padBottom] = testCase;
        for (const auto imageLayout : { TensorLayout::NCHW, TensorLayout::NHWC }) {
            const bool nhwc = imageLayout == TensorLayout::NHWC;
            for (const int rowPadding : { 0, 5 }) {
                SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + (nhwc ? " nhwc" : " nchw")
                             + (rowPadding ? " padded" : ""));
                const auto shape =
                    nhwc ? std::array{ modelHeight, modelWidth, 3 } : std::array{ 3, modelHeight, modelWidth };
                const int rowStride = (nhwc ? modelWidth * 3 : modelWidth) + rowPadding;
                const auto strides = nhwc ? std::array{ rowStride, 3, 1 }
                                          : std::array{ rowStride * modelHeight + rowPadding, rowStride, 1 };
                std::vector<uint16_t> storage(static_cast<std::size_t>(shape[0]) * strides[0], unwritten);
                Tensor<uint16_t, 3> output{ shape, strides, std::span{ storage } };
                Tensor<uint16_t, 3> expected{};
                InputTransformer transformer{ imageLayout };
                InputTransformer referenceTransformer{ TensorLayout::NCHW };
                OutputTransformer<uint16_t> outputTransformer{ imageLayout };
                for (int frameIndex = 0; frameIndex < 2; ++frameIndex) {
                    const auto bytes = MakeRandomNv12(width, height, random);
                    const Nv12FrameView frame{ width, height, bytes };
                    const auto result = transformer.Transform(frame, modelWidth, modelHeight, output);
                    referenceTransformer.Transform(frame, modelWidth, modelHeight, expected);
                    ASSERT_EQ(output.Data().data(), storage.data());
                    ASSERT_EQ(output.Shape(), shape);
                    ASSERT_EQ(output.Strides(), strides);
                    EXPECT_EQ(result.transposeFlag, transposed);
                    EXPECT_EQ(result.cropOffsets.left, 0);
                    EXPECT_EQ(result.cropOffsets.right, padRight);
                    EXPECT_EQ(result.cropOffsets.top, 0);
                    EXPECT_EQ(result.cropOffsets.bottom, padBottom);
                    for (int row = 0; row < modelHeight; ++row) {
                        for (int column = 0; column < modelWidth; ++column) {
                            const bool isPadding = row >= modelHeight - padBottom || column >= modelWidth - padRight;
                            for (int channel = 0; channel < 3; ++channel) {
                                EXPECT_EQ(nhwc ? output(row, column, channel) : output(channel, row, column),
                                          isPadding ? gray : expected(channel, row, column));
                            }
                        }
                    }
                    // Every tensor element is written, stride padding is not
                    EXPECT_EQ(std::ranges::count(storage, unwritten),
                              static_cast<std::ptrdiff_t>(storage.size()) - 3 * modelWidth * modelHeight);
                    const auto reconstructed =
                        outputTransformer.Transform(output, width, height, result.cropOffsets, result.transposeFlag);
                    ASSERT_TRUE(reconstructed);
                    ASSERT_NO_FATAL_FAILURE(AssertNv12Equal(frame, reconstructed.value()));
                }
            }
        }
    }
}

TEST(UnitTestFrameProcessing, ImageLayoutsReconstructSameNv12)
{
    std::mt19937 random{ 42 };
    auto nchw = MakeRandomFp16Tensor(3, 24, 40, random);
    nchw(0, 2, 2) = Fp32To16(-0.25f);
    nchw(0, 2, 3) = Fp32To16(1.25f);
    Tensor<uint16_t, 3> nhwc{ std::array{ 24, 40, 3 } };
    for (int row = 0; row < 24; ++row) {
        for (int column = 0; column < 40; ++column) {
            for (int channel = 0; channel < 3; ++channel) {
                nhwc(row, column, channel) = nchw(channel, row, column);
            }
        }
    }
    const CropOffsets crop{ 2, 4, 2, 4 };
    OutputTransformer<uint16_t> nchwTransformer{ TensorLayout::NCHW };
    OutputTransformer<uint16_t> nhwcTransformer{ TensorLayout::NHWC };
    for (const bool transposeFlag : { false, true }) {
        const int width = transposeFlag ? 18 : 34;
        const int height = transposeFlag ? 34 : 18;
        const auto expected = nchwTransformer.Transform(nchw, width, height, crop, transposeFlag);
        const auto actual = nhwcTransformer.Transform(nhwc, width, height, crop, transposeFlag);
        ASSERT_TRUE(expected);
        ASSERT_TRUE(actual);
        ASSERT_NO_FATAL_FAILURE(AssertNv12Equal(expected.value(), actual.value()));
    }
}

TEST(UnitTestFrameProcessing, ValidateImageTensor)
{
    constexpr int width = 40;
    constexpr int height = 24;
    // Rows of 40 FP16 values padded to a multiple of 64 bytes
    constexpr int paddedRow = 64;
    // Smallest NCHW plane stride: the last row of a plane ends right before the next plane
    constexpr int planeExtent = (height - 1) * paddedRow + width;
    struct TestCase {
        const char* name;
        TensorLayout layout;
        std::array<int, 3> shape;
        std::array<int, 3> strides;
        bool valid;
    };
    const TestCase testCases[] = {
        { "nhwc packed", TensorLayout::NHWC, { height, width, 3 }, { width * 3, 3, 1 }, true },
        { "nhwc padded rows", TensorLayout::NHWC, { height, width, 3 }, { width * 3 + 5, 3, 1 }, true },
        { "nchw packed", TensorLayout::NCHW, { 3, height, width }, { height * width, width, 1 }, true },
        { "nchw padded rows", TensorLayout::NCHW, { 3, height, width }, { height * paddedRow, paddedRow, 1 }, true },
        { "nchw tight planes", TensorLayout::NCHW, { 3, height, width }, { planeExtent, paddedRow, 1 }, true },
        { "nhwc padded pixels", TensorLayout::NHWC, { height, width, 3 }, { width * 32, 32, 1 }, false },
        { "nhwc partially padded pixels", TensorLayout::NHWC, { height, width, 3 }, { width * 8, 8, 1 }, false },
        { "nhwc channel stride", TensorLayout::NHWC, { height, width, 3 }, { width * 3, 3, 2 }, false },
        { "nhwc row overlap", TensorLayout::NHWC, { height, width, 3 }, { width * 3 - 1, 3, 1 }, false },
        { "nchw row overlap", TensorLayout::NCHW, { 3, height, width }, { height * width, width - 1, 1 }, false },
        { "nchw plane overlap", TensorLayout::NCHW, { 3, height, width }, { planeExtent - 1, paddedRow, 1 }, false },
        { "nchw column stride", TensorLayout::NCHW, { 3, height, width }, { height * width, width, 2 }, false },
        { "nhwc shape, nchw layout", TensorLayout::NCHW, { height, width, 3 }, { width * 3, 3, 1 }, false },
        { "nchw shape, nhwc layout", TensorLayout::NHWC, { 3, height, width }, { height * width, width, 1 }, false },
        { "wrong height", TensorLayout::NHWC, { height + 2, width, 3 }, { width * 3, 3, 1 }, false },
        { "wrong width", TensorLayout::NCHW, { 3, height, width - 2 }, { height * width, width, 1 }, false },
        { "four channels", TensorLayout::NHWC, { height, width, 4 }, { width * 4, 4, 1 }, false },
    };
    for (const auto& testCase : testCases) {
        SCOPED_TRACE(testCase.name);
        const auto result = ValidateImageTensor(testCase.layout, width, height, testCase.shape, testCase.strides);
        EXPECT_EQ(static_cast<bool>(result), testCase.valid);
        if (!result) {
            EXPECT_EQ(result.error(), make_error_code(Error::invalid_argument));
        }
    }
}
