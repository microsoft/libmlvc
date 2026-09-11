// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/fp16.hpp"
#include "libmlvc/common/tensor.hpp"

#include <libmlvc/types.hpp>

#include <cstdint>

namespace libmlvc {

Nv12FrameView TransposeNv12(const Nv12FrameView& frame, std::vector<std::byte>& buffer);
void Nv12ToYuv444Fp16(const Nv12FrameView& frame, Tensor<uint16_t, 3>& output);
Nv12FrameView Yuv444Fp16ToNv12(const Tensor<uint16_t, 3>& yuv444, std::vector<std::byte>& buffer);
void Float16ToInt32(const Tensor<uint16_t, 3>& input, Tensor<int32_t, 3>& output);
void Int32ToFloat16(const Tensor<int32_t, 3>& input, Tensor<uint16_t, 3>& output);

}  // namespace libmlvc
