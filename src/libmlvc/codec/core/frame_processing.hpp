// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/codec/core/transforms.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/tensor.hpp"
#include "libmlvc/schema/bundle.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace libmlvc {

inline expected<void> ValidateImageTensor(const TensorLayout layout, const int width, const int height,
                                          const std::array<int, 3>& shape, const std::array<int, 3>& strides)
{
    const bool nhwc = layout == TensorLayout::NHWC;
    const auto expectedShape = nhwc ? std::array{ height, width, 3 } : std::array{ 3, height, width };
    const bool validStrides =
        nhwc ? strides[2] == 1 && strides[1] == 3 && strides[0] >= int64_t{ 3 } * width
             : strides[2] == 1 && strides[1] >= width && strides[0] >= int64_t{ height - 1 } * strides[1] + width;
    if (shape != expectedShape || !validStrides) {
        MLVC_LOG_ERROR("Unsupported %s image tensor: shape [%d, %d, %d], strides [%d, %d, %d], model size %dx%d",
                       nhwc ? "nhwc" : "nchw", shape[0], shape[1], shape[2], strides[0], strides[1], strides[2], width,
                       height);
        return make_error_code(Error::invalid_argument);
    }
    return {};
}

class InputTransformer {
public:
    struct Result {
        bool transposeFlag{ false };
        CropOffsets cropOffsets{};
    };

    explicit InputTransformer(const TensorLayout modelInputLayout) : m_modelInputLayout{ modelInputLayout } {}

    template <typename T>
    Result Transform(const Nv12FrameView& frame, const int modelWidth, const int modelHeight, Tensor<T, 3>& output)
    {
        Result result{};
        const int width = frame.Width();
        const int height = frame.Height();
        const bool fitsNormally = width <= modelWidth && height <= modelHeight;
        const bool fitsRotated = height <= modelWidth && width <= modelHeight;
        MLVC_ASSERT(fitsNormally || fitsRotated);
        result.transposeFlag = !fitsNormally && fitsRotated;

        const int alignedWidth = result.transposeFlag ? height : width;
        const int alignedHeight = result.transposeFlag ? width : height;
        const int padWidth = modelWidth - alignedWidth;
        const int padHeight = modelHeight - alignedHeight;
        result.cropOffsets = { 0, padWidth, 0, padHeight };

        // Allocate output and fill with gray padding (once)
        output.Create(m_modelInputLayout == TensorLayout::NHWC ? std::array{ modelHeight, modelWidth, 3 }
                                                               : std::array{ 3, modelHeight, modelWidth });
        if (!m_outputInitialized) {
            output.SetValue(Fp32To16(0.5f));
            m_outputInitialized = true;
        }

        const auto alignedFrame = result.transposeFlag ? TransposeNv12(frame, m_transposedNv12) : frame;
        if (m_modelInputLayout == TensorLayout::NHWC) {
            Nv12ToYuv444Fp16Nhwc(alignedFrame, output);
        } else {
            Nv12ToYuv444Fp16Nchw(alignedFrame, output);
        }
        return result;
    }

protected:
    const TensorLayout m_modelInputLayout;
    std::vector<std::byte> m_transposedNv12;
    bool m_outputInitialized = false;
};

template <typename T>
class OutputTransformer {
public:
    explicit OutputTransformer(const TensorLayout modelOutputLayout) : m_modelOutputLayout{ modelOutputLayout } {}

    expected<Nv12FrameView> Transform(const Tensor<T, 3>& modelOutput, const int width, const int height,
                                      const CropOffsets& cropOffsets, const bool transposeFlag)
    {
        // Remove padding
        const Tensor<T, 3> cropped =
            CropImageView(modelOutput, cropOffsets.left, cropOffsets.right, cropOffsets.top, cropOffsets.bottom);

        // Convert YUV444 to NV12
        const auto frame = m_modelOutputLayout == TensorLayout::NHWC ? Yuv444Fp16NhwcToNv12(cropped, m_outputBuffer)
                                                                     : Yuv444Fp16NchwToNv12(cropped, m_outputBuffer);

        // Transpose if needed
        if (transposeFlag) {
            return TransposeNv12(frame, m_outputBufferTransposed);
        }
        return frame;
    }

protected:
    const TensorLayout m_modelOutputLayout;
    std::vector<std::byte> m_outputBuffer;
    std::vector<std::byte> m_outputBufferTransposed;

    Tensor<T, 3> CropImageView(const Tensor<T, 3>& input, const int left, const int right, const int top, const int bottom) const
    {
        const int heightAxis = m_modelOutputLayout == TensorLayout::NHWC ? 0 : 1;
        const int widthAxis = heightAxis + 1;
        MLVC_ASSERT(left >= 0 && right >= 0 && top >= 0 && bottom >= 0);
        MLVC_ASSERT(input.Shape()[heightAxis] > top + bottom);
        MLVC_ASSERT(input.Shape()[widthAxis] > left + right);

        auto shape = input.Shape();
        shape[heightAxis] -= (top + bottom);
        shape[widthAxis] -= (left + right);
        const int offset = top * input.Strides()[heightAxis] + left * input.Strides()[widthAxis];
        const std::span<T> croppedData = input.Data().subspan(offset);
        return Tensor<T, 3>{ shape, input.Strides(), croppedData };
    }
};

}  // namespace libmlvc
