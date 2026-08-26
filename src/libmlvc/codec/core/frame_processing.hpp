// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/codec/core/transforms.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/tensor.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace libmlvc {

class InputTransformer {
public:
    struct Result {
        bool transposeFlag{ false };
        CropOffsets cropOffsets{};
    };

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
        output.Create({ 3, modelHeight, modelWidth });
        if (!m_outputInitialized) {
            output.SetValue(Fp32To16(0.5f));
            m_outputInitialized = true;
        }

        const auto alignedFrame = result.transposeFlag ? TransposeNv12(frame, m_transposedNv12) : frame;
        Nv12ToYuv444Fp16(alignedFrame, output);
        return result;
    }

protected:
    std::vector<std::byte> m_transposedNv12;
    bool m_outputInitialized = false;
};

template <typename T>
class OutputTransformer {
public:
    expected<void> Initialize() { return {}; }

    expected<Nv12FrameView> Transform(const Tensor<T, 3>& modelOutput, const int width, const int height,
                                      const CropOffsets& cropOffsets, const bool transposeFlag)
    {
        // Remove padding
        const Tensor<T, 3> cropped =
            CropImageView(modelOutput, cropOffsets.left, cropOffsets.right, cropOffsets.top, cropOffsets.bottom);

        // Convert YUV444 to NV12
        const auto frame = Yuv444Fp16ToNv12(cropped, m_outputBuffer);

        // Transpose if needed
        if (transposeFlag) {
            return TransposeNv12(frame, m_outputBufferTransposed);
        }
        return frame;
    }

protected:
    std::vector<std::byte> m_outputBuffer;
    std::vector<std::byte> m_outputBufferTransposed;

    static Tensor<T, 3> CropImageView(const Tensor<T, 3>& input, const int left, const int right, const int top,
                                      const int bottom)
    {
        MLVC_ASSERT(left >= 0 && right >= 0 && top >= 0 && bottom >= 0);
        MLVC_ASSERT(input.Shape()[1] > top + bottom);
        MLVC_ASSERT(input.Shape()[2] > left + right);

        auto shape = input.Shape();
        shape[1] -= (top + bottom);
        shape[2] -= (left + right);
        const int offset = top * input.Strides()[1] + left * input.Strides()[2];
        const std::span<T> croppedData = input.Data().subspan(offset);
        return Tensor<T, 3>{ shape, input.Strides(), croppedData };
    }
};

}  // namespace libmlvc
