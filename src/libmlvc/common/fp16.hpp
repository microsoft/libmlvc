// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/macros.hpp"

#include <bit>
#include <cstdint>

#if defined(MLVC_ARCH_X86_64)
    #include <immintrin.h>
#elif defined(MLVC_ARCH_ARM64)
    #include <arm_neon.h>
#endif

namespace libmlvc {

using mlvc_f16_t = uint16_t;

inline float Fp16To32(uint16_t value)
{
#if defined(__clang__) && defined(MLVC_ARCH_ARM64)
    return static_cast<float>(std::bit_cast<_Float16>(value));
#elif defined(MLVC_ARCH_ARM64)
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(value))), 0);
#elif defined(MLVC_ARCH_X86_64)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(value)));
#else
    #error "Unsupported architecture"
#endif
}

inline uint16_t Fp32To16(float value)
{
#if defined(__clang__) && defined(MLVC_ARCH_ARM64)
    return std::bit_cast<uint16_t>(static_cast<_Float16>(value));
#elif defined(MLVC_ARCH_ARM64)
    return vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(vdupq_n_f32(value))), 0);
#elif defined(MLVC_ARCH_X86_64)
    return static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(value), _MM_FROUND_TO_NEAREST_INT)));
#else
    #error "Unsupported architecture"
#endif
}

}  // namespace libmlvc
