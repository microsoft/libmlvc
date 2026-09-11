// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/transforms.hpp"
#include "libmlvc/common/logging.hpp"

#include <algorithm>

namespace libmlvc {

// ============================================================================
// TransposeNv12
// ============================================================================

#if defined(MLVC_ARCH_X86_64)
namespace {

template <typename T>
MLVC_FORCE_INLINE T* ByteOffset(T* ptr, std::size_t bytes)
{
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + bytes);
}

MLVC_FORCE_INLINE __m128i Load64(const uint8_t* ptr)
{
    return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
}

constexpr int TRANSPOSE_TILE = 128;

// Transpose a single 8x8 byte block (XMM).
MLVC_FORCE_INLINE void Transpose8x8Y(const uint8_t* src, std::size_t ss, uint8_t* dst, std::size_t ds)
{
    __m128i r0 = Load64(src + 0 * ss);
    __m128i r1 = Load64(src + 1 * ss);
    __m128i r2 = Load64(src + 2 * ss);
    __m128i r3 = Load64(src + 3 * ss);
    __m128i r4 = Load64(src + 4 * ss);
    __m128i r5 = Load64(src + 5 * ss);
    __m128i r6 = Load64(src + 6 * ss);
    __m128i r7 = Load64(src + 7 * ss);

    __m128i t0 = _mm_unpacklo_epi8(r0, r1);
    __m128i t1 = _mm_unpacklo_epi8(r2, r3);
    __m128i t2 = _mm_unpacklo_epi8(r4, r5);
    __m128i t3 = _mm_unpacklo_epi8(r6, r7);

    __m128i u0 = _mm_unpacklo_epi16(t0, t1);
    __m128i u1 = _mm_unpackhi_epi16(t0, t1);
    __m128i u2 = _mm_unpacklo_epi16(t2, t3);
    __m128i u3 = _mm_unpackhi_epi16(t2, t3);

    __m128i v0 = _mm_unpacklo_epi32(u0, u2);
    __m128i v1 = _mm_unpackhi_epi32(u0, u2);
    __m128i v2 = _mm_unpacklo_epi32(u1, u3);
    __m128i v3 = _mm_unpackhi_epi32(u1, u3);

    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 0 * ds), v0);
    _mm_storeh_pd(reinterpret_cast<double*>(dst + 1 * ds), _mm_castsi128_pd(v0));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 2 * ds), v1);
    _mm_storeh_pd(reinterpret_cast<double*>(dst + 3 * ds), _mm_castsi128_pd(v1));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 4 * ds), v2);
    _mm_storeh_pd(reinterpret_cast<double*>(dst + 5 * ds), _mm_castsi128_pd(v2));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 6 * ds), v3);
    _mm_storeh_pd(reinterpret_cast<double*>(dst + 7 * ds), _mm_castsi128_pd(v3));
}

// Transpose two vertically-adjacent 8x8 Y blocks (YMM).
MLVC_FORCE_INLINE void Transpose2x8x8Y(const uint8_t* src0, const uint8_t* src1, std::size_t ss, uint8_t* dst, std::size_t ds)
{
    __m256i r0 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 0 * ss)), Load64(src1 + 0 * ss), 1);
    __m256i r1 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 1 * ss)), Load64(src1 + 1 * ss), 1);
    __m256i r2 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 2 * ss)), Load64(src1 + 2 * ss), 1);
    __m256i r3 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 3 * ss)), Load64(src1 + 3 * ss), 1);
    __m256i r4 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 4 * ss)), Load64(src1 + 4 * ss), 1);
    __m256i r5 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 5 * ss)), Load64(src1 + 5 * ss), 1);
    __m256i r6 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 6 * ss)), Load64(src1 + 6 * ss), 1);
    __m256i r7 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 7 * ss)), Load64(src1 + 7 * ss), 1);

    __m256i t0 = _mm256_unpacklo_epi8(r0, r1);
    __m256i t1 = _mm256_unpacklo_epi8(r2, r3);
    __m256i t2 = _mm256_unpacklo_epi8(r4, r5);
    __m256i t3 = _mm256_unpacklo_epi8(r6, r7);

    __m256i u0 = _mm256_unpacklo_epi16(t0, t1);
    __m256i u1 = _mm256_unpackhi_epi16(t0, t1);
    __m256i u2 = _mm256_unpacklo_epi16(t2, t3);
    __m256i u3 = _mm256_unpackhi_epi16(t2, t3);

    __m256i v0 = _mm256_unpacklo_epi32(u0, u2);
    __m256i v1 = _mm256_unpackhi_epi32(u0, u2);
    __m256i v2 = _mm256_unpacklo_epi32(u1, u3);
    __m256i v3 = _mm256_unpackhi_epi32(u1, u3);

    __m256i p0 = _mm256_permute4x64_epi64(v0, 0xD8);
    __m256i p1 = _mm256_permute4x64_epi64(v1, 0xD8);
    __m256i p2 = _mm256_permute4x64_epi64(v2, 0xD8);
    __m256i p3 = _mm256_permute4x64_epi64(v3, 0xD8);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 0 * ds), _mm256_castsi256_si128(p0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 1 * ds), _mm256_extracti128_si256(p0, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2 * ds), _mm256_castsi256_si128(p1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 3 * ds), _mm256_extracti128_si256(p1, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 4 * ds), _mm256_castsi256_si128(p2));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 5 * ds), _mm256_extracti128_si256(p2, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 6 * ds), _mm256_castsi256_si128(p3));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 7 * ds), _mm256_extracti128_si256(p3, 1));
}

// Transpose four vertically-adjacent 8x8 Y blocks with ILP.
MLVC_FORCE_INLINE void Transpose4x8x8Y(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2,
                                       const uint8_t* src3, std::size_t ss, uint8_t* dst0, uint8_t* dst1, std::size_t ds)
{
    // Pair A: blocks 0+1
    __m256i a0 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 0 * ss)), Load64(src1 + 0 * ss), 1);
    __m256i a1 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 1 * ss)), Load64(src1 + 1 * ss), 1);
    __m256i a2 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 2 * ss)), Load64(src1 + 2 * ss), 1);
    __m256i a3 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 3 * ss)), Load64(src1 + 3 * ss), 1);
    __m256i a4 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 4 * ss)), Load64(src1 + 4 * ss), 1);
    __m256i a5 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 5 * ss)), Load64(src1 + 5 * ss), 1);
    __m256i a6 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 6 * ss)), Load64(src1 + 6 * ss), 1);
    __m256i a7 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src0 + 7 * ss)), Load64(src1 + 7 * ss), 1);

    // Pair B: blocks 2+3
    __m256i b0 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src2 + 0 * ss)), Load64(src3 + 0 * ss), 1);
    __m256i b1 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src2 + 1 * ss)), Load64(src3 + 1 * ss), 1);
    __m256i b2 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src2 + 2 * ss)), Load64(src3 + 2 * ss), 1);
    __m256i b3 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src2 + 3 * ss)), Load64(src3 + 3 * ss), 1);
    __m256i b4 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src2 + 4 * ss)), Load64(src3 + 4 * ss), 1);
    __m256i b5 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src2 + 5 * ss)), Load64(src3 + 5 * ss), 1);
    __m256i b6 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src2 + 6 * ss)), Load64(src3 + 6 * ss), 1);
    __m256i b7 = _mm256_inserti128_si256(_mm256_castsi128_si256(Load64(src2 + 7 * ss)), Load64(src3 + 7 * ss), 1);

    // Interleaved transposes for ILP
    __m256i at0 = _mm256_unpacklo_epi8(a0, a1);
    __m256i bt0 = _mm256_unpacklo_epi8(b0, b1);
    __m256i at1 = _mm256_unpacklo_epi8(a2, a3);
    __m256i bt1 = _mm256_unpacklo_epi8(b2, b3);
    __m256i at2 = _mm256_unpacklo_epi8(a4, a5);
    __m256i bt2 = _mm256_unpacklo_epi8(b4, b5);
    __m256i at3 = _mm256_unpacklo_epi8(a6, a7);
    __m256i bt3 = _mm256_unpacklo_epi8(b6, b7);

    __m256i au0 = _mm256_unpacklo_epi16(at0, at1);
    __m256i bu0 = _mm256_unpacklo_epi16(bt0, bt1);
    __m256i au1 = _mm256_unpackhi_epi16(at0, at1);
    __m256i bu1 = _mm256_unpackhi_epi16(bt0, bt1);
    __m256i au2 = _mm256_unpacklo_epi16(at2, at3);
    __m256i bu2 = _mm256_unpacklo_epi16(bt2, bt3);
    __m256i au3 = _mm256_unpackhi_epi16(at2, at3);
    __m256i bu3 = _mm256_unpackhi_epi16(bt2, bt3);

    __m256i av0 = _mm256_unpacklo_epi32(au0, au2);
    __m256i bv0 = _mm256_unpacklo_epi32(bu0, bu2);
    __m256i av1 = _mm256_unpackhi_epi32(au0, au2);
    __m256i bv1 = _mm256_unpackhi_epi32(bu0, bu2);
    __m256i av2 = _mm256_unpacklo_epi32(au1, au3);
    __m256i bv2 = _mm256_unpacklo_epi32(bu1, bu3);
    __m256i av3 = _mm256_unpackhi_epi32(au1, au3);
    __m256i bv3 = _mm256_unpackhi_epi32(bu1, bu3);

    __m256i ap0 = _mm256_permute4x64_epi64(av0, 0xD8);
    __m256i bp0 = _mm256_permute4x64_epi64(bv0, 0xD8);
    __m256i ap1 = _mm256_permute4x64_epi64(av1, 0xD8);
    __m256i bp1 = _mm256_permute4x64_epi64(bv1, 0xD8);
    __m256i ap2 = _mm256_permute4x64_epi64(av2, 0xD8);
    __m256i bp2 = _mm256_permute4x64_epi64(bv2, 0xD8);
    __m256i ap3 = _mm256_permute4x64_epi64(av3, 0xD8);
    __m256i bp3 = _mm256_permute4x64_epi64(bv3, 0xD8);

    // Interleaved stores: pair A and pair B
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + 0 * ds), _mm256_castsi256_si128(ap0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + 1 * ds), _mm256_extracti128_si256(ap0, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + 0 * ds), _mm256_castsi256_si128(bp0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + 1 * ds), _mm256_extracti128_si256(bp0, 1));

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + 2 * ds), _mm256_castsi256_si128(ap1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + 3 * ds), _mm256_extracti128_si256(ap1, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + 2 * ds), _mm256_castsi256_si128(bp1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + 3 * ds), _mm256_extracti128_si256(bp1, 1));

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + 4 * ds), _mm256_castsi256_si128(ap2));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + 5 * ds), _mm256_extracti128_si256(ap2, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + 4 * ds), _mm256_castsi256_si128(bp2));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + 5 * ds), _mm256_extracti128_si256(bp2, 1));

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + 6 * ds), _mm256_castsi256_si128(ap3));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + 7 * ds), _mm256_extracti128_si256(ap3, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + 6 * ds), _mm256_castsi256_si128(bp3));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + 7 * ds), _mm256_extracti128_si256(bp3, 1));
}

// Tiled Y plane transpose with quad-block ILP, dual-block, and single-block fallback.
void TransposeYTiled(const uint8_t* src, std::size_t ss, uint8_t* dst, std::size_t ds, int w, int h)
{
    for (int ty = 0; ty < h; ty += TRANSPOSE_TILE) {
        const int th = (ty + TRANSPOSE_TILE <= h) ? TRANSPOSE_TILE : (h - ty);
        for (int tx = 0; tx < w; tx += TRANSPOSE_TILE) {
            const int tw = (tx + TRANSPOSE_TILE <= w) ? TRANSPOSE_TILE : (w - tx);
            const int tw8 = tw & ~7;
            const int th8 = th & ~7;
            const int th16 = th & ~15;
            const int th32 = th & ~31;

            int by = 0;

            // Quad blocks: 4 vertical x 8 rows = 32 source rows
            for (; by < th32; by += 32) {
                for (int bx = 0; bx < tw8; bx += 8)
                    Transpose4x8x8Y(src + (ty + by) * ss + (tx + bx), src + (ty + by + 8) * ss + (tx + bx),
                                    src + (ty + by + 16) * ss + (tx + bx), src + (ty + by + 24) * ss + (tx + bx), ss,
                                    dst + (tx + bx) * ds + (ty + by), dst + (tx + bx) * ds + (ty + by + 16), ds);
                for (int x = tw8; x < tw; x++)
                    for (int dy = 0; dy < 32; dy++)
                        dst[(tx + x) * ds + (ty + by + dy)] = src[(ty + by + dy) * ss + (tx + x)];
            }

            // Remaining dual blocks: 2 vertical x 8 rows = 16 rows
            for (; by + 16 <= th8; by += 16) {
                for (int bx = 0; bx < tw8; bx += 8)
                    Transpose2x8x8Y(src + (ty + by) * ss + (tx + bx), src + (ty + by + 8) * ss + (tx + bx), ss,
                                    dst + (tx + bx) * ds + (ty + by), ds);
                for (int x = tw8; x < tw; x++)
                    for (int dy = 0; dy < 16; dy++)
                        dst[(tx + x) * ds + (ty + by + dy)] = src[(ty + by + dy) * ss + (tx + x)];
            }

            // Remaining single blocks: 8 rows
            for (; by < th8; by += 8) {
                for (int bx = 0; bx < tw8; bx += 8)
                    Transpose8x8Y(src + (ty + by) * ss + (tx + bx), ss, dst + (tx + bx) * ds + (ty + by), ds);
                for (int x = tw8; x < tw; x++)
                    for (int dy = 0; dy < 8; dy++)
                        dst[(tx + x) * ds + (ty + by + dy)] = src[(ty + by + dy) * ss + (tx + x)];
            }

            // Scalar tail: remaining < 8 rows
            for (int y = by; y < th; y++)
                for (int x = 0; x < tw; x++)
                    dst[(tx + x) * ds + (ty + y)] = src[(ty + y) * ss + (tx + x)];
        }
    }
}

// Transpose a single 8x8 UV-pair block (XMM). Each "element" is a 16-bit UV pair.
MLVC_FORCE_INLINE void Transpose8x8Uv(const uint8_t* src, std::size_t ss, uint8_t* dst, std::size_t ds)
{
    __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0 * ss));
    __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 1 * ss));
    __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2 * ss));
    __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 3 * ss));
    __m128i r4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4 * ss));
    __m128i r5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 5 * ss));
    __m128i r6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 6 * ss));
    __m128i r7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 7 * ss));

    __m128i t0 = _mm_unpacklo_epi16(r0, r1);
    __m128i t1 = _mm_unpackhi_epi16(r0, r1);
    __m128i t2 = _mm_unpacklo_epi16(r2, r3);
    __m128i t3 = _mm_unpackhi_epi16(r2, r3);
    __m128i t4 = _mm_unpacklo_epi16(r4, r5);
    __m128i t5 = _mm_unpackhi_epi16(r4, r5);
    __m128i t6 = _mm_unpacklo_epi16(r6, r7);
    __m128i t7 = _mm_unpackhi_epi16(r6, r7);

    __m128i u0 = _mm_unpacklo_epi32(t0, t2);
    __m128i u1 = _mm_unpackhi_epi32(t0, t2);
    __m128i u2 = _mm_unpacklo_epi32(t1, t3);
    __m128i u3 = _mm_unpackhi_epi32(t1, t3);
    __m128i u4 = _mm_unpacklo_epi32(t4, t6);
    __m128i u5 = _mm_unpackhi_epi32(t4, t6);
    __m128i u6 = _mm_unpacklo_epi32(t5, t7);
    __m128i u7 = _mm_unpackhi_epi32(t5, t7);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 0 * ds), _mm_unpacklo_epi64(u0, u4));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 1 * ds), _mm_unpackhi_epi64(u0, u4));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2 * ds), _mm_unpacklo_epi64(u1, u5));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 3 * ds), _mm_unpackhi_epi64(u1, u5));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 4 * ds), _mm_unpacklo_epi64(u2, u6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 5 * ds), _mm_unpackhi_epi64(u2, u6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 6 * ds), _mm_unpacklo_epi64(u3, u7));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 7 * ds), _mm_unpackhi_epi64(u3, u7));
}

// Transpose two vertically-adjacent 8x8 UV-pair blocks (YMM).
MLVC_FORCE_INLINE void Transpose2x8x8Uv(const uint8_t* src0, const uint8_t* src1, std::size_t ss, uint8_t* dst,
                                        std::size_t ds)
{
    __m256i r0 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + 0 * ss))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + 0 * ss)), 1);
    __m256i r1 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + 1 * ss))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + 1 * ss)), 1);
    __m256i r2 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + 2 * ss))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + 2 * ss)), 1);
    __m256i r3 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + 3 * ss))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + 3 * ss)), 1);
    __m256i r4 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + 4 * ss))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + 4 * ss)), 1);
    __m256i r5 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + 5 * ss))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + 5 * ss)), 1);
    __m256i r6 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + 6 * ss))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + 6 * ss)), 1);
    __m256i r7 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + 7 * ss))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + 7 * ss)), 1);

    __m256i t0 = _mm256_unpacklo_epi16(r0, r1);
    __m256i t1 = _mm256_unpackhi_epi16(r0, r1);
    __m256i t2 = _mm256_unpacklo_epi16(r2, r3);
    __m256i t3 = _mm256_unpackhi_epi16(r2, r3);
    __m256i t4 = _mm256_unpacklo_epi16(r4, r5);
    __m256i t5 = _mm256_unpackhi_epi16(r4, r5);
    __m256i t6 = _mm256_unpacklo_epi16(r6, r7);
    __m256i t7 = _mm256_unpackhi_epi16(r6, r7);

    __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
    __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
    __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
    __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
    __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 0 * ds), _mm256_unpacklo_epi64(u0, u4));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 1 * ds), _mm256_unpackhi_epi64(u0, u4));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 2 * ds), _mm256_unpacklo_epi64(u1, u5));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 3 * ds), _mm256_unpackhi_epi64(u1, u5));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 4 * ds), _mm256_unpacklo_epi64(u2, u6));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 5 * ds), _mm256_unpackhi_epi64(u2, u6));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 6 * ds), _mm256_unpacklo_epi64(u3, u7));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 7 * ds), _mm256_unpackhi_epi64(u3, u7));
}

// Tiled UV plane transpose with vertical pairing for 32-byte stores.
void TransposeUvTiled(const uint8_t* src, std::size_t ss, uint8_t* dst, std::size_t ds, int pw,
                      int ph)  // pw, ph are in UV-pair counts (= W/2, H/2)
{
    for (int ty = 0; ty < ph; ty += TRANSPOSE_TILE) {
        const int th = (ty + TRANSPOSE_TILE <= ph) ? TRANSPOSE_TILE : (ph - ty);
        for (int tx = 0; tx < pw; tx += TRANSPOSE_TILE) {
            const int tw = (tx + TRANSPOSE_TILE <= pw) ? TRANSPOSE_TILE : (pw - tx);
            const int tw8 = tw & ~7;
            const int th8 = th & ~7;
            const int th16 = th & ~15;

            for (int by = 0; by < th8; by += 8) {
                // Vertical pairing: pair (bx, by) and (bx, by+8)
                if (by + 8 <= th8 && by < th16 && (by & 15) == 0) {
                    for (int bx = 0; bx < tw8; bx += 8)
                        Transpose2x8x8Uv(src + (ty + by) * ss + (tx + bx) * 2, src + (ty + by + 8) * ss + (tx + bx) * 2,
                                         ss, dst + (tx + bx) * ds + (ty + by) * 2, ds);
                    // Scalar cleanup for remaining columns
                    for (int px = tw8; px < tw; px++)
                        for (int dy = 0; dy < 16; dy++) {
                            int py = ty + by + dy, pxx = tx + px;
                            dst[pxx * ds + py * 2] = src[py * ss + pxx * 2];
                            dst[pxx * ds + py * 2 + 1] = src[py * ss + pxx * 2 + 1];
                        }
                    by += 8;  // skip second block (already processed)
                } else {
                    // Fallback: single XMM 8x8 UV block
                    for (int bx = 0; bx < tw8; bx += 8)
                        Transpose8x8Uv(src + (ty + by) * ss + (tx + bx) * 2, ss, dst + (tx + bx) * ds + (ty + by) * 2, ds);
                    for (int px = tw8; px < tw; px++)
                        for (int dy = 0; dy < 8; dy++) {
                            int py = ty + by + dy, pxx = tx + px;
                            dst[pxx * ds + py * 2] = src[py * ss + pxx * 2];
                            dst[pxx * ds + py * 2 + 1] = src[py * ss + pxx * 2 + 1];
                        }
                }
            }
            // Scalar tail: remaining < 8 rows
            for (int y = th8; y < th; y++)
                for (int x = 0; x < tw; x++) {
                    int py = ty + y, px = tx + x;
                    dst[px * ds + py * 2] = src[py * ss + px * 2];
                    dst[px * ds + py * 2 + 1] = src[py * ss + px * 2 + 1];
                }
        }
    }
}

// Transpose NV12 with separate Y and UV plane pointers and strides.
void TransposeNv12Impl(const uint8_t* srcY, std::size_t srcYStride, const uint8_t* srcUv, std::size_t srcUvStride,
                       uint8_t* dstY, std::size_t dstYStride, uint8_t* dstUv, std::size_t dstUvStride, int w, int h)
{
    TransposeYTiled(srcY, srcYStride, dstY, dstYStride, w, h);
    TransposeUvTiled(srcUv, srcUvStride, dstUv, dstUvStride, w / 2, h / 2);
}

}  // anonymous namespace
#endif  // MLVC_ARCH_X86_64

#if defined(MLVC_ARCH_ARM64)
namespace {

constexpr int TRANSPOSE_TILE = 512;

// 8×8 byte transpose using 64-bit NEON registers.
MLVC_FORCE_INLINE void Transpose8x8U8(uint8x8_t r0, uint8x8_t r1, uint8x8_t r2, uint8x8_t r3, uint8x8_t r4, uint8x8_t r5,
                                      uint8x8_t r6, uint8x8_t r7, uint8x8_t& c0, uint8x8_t& c1, uint8x8_t& c2,
                                      uint8x8_t& c3, uint8x8_t& c4, uint8x8_t& c5, uint8x8_t& c6, uint8x8_t& c7)
{
    uint8x8x2_t t01 = vtrn_u8(r0, r1);
    uint8x8x2_t t23 = vtrn_u8(r2, r3);
    uint8x8x2_t t45 = vtrn_u8(r4, r5);
    uint8x8x2_t t67 = vtrn_u8(r6, r7);

    uint16x4x2_t u02 = vtrn_u16(vreinterpret_u16_u8(t01.val[0]), vreinterpret_u16_u8(t23.val[0]));
    uint16x4x2_t u13 = vtrn_u16(vreinterpret_u16_u8(t01.val[1]), vreinterpret_u16_u8(t23.val[1]));
    uint16x4x2_t u46 = vtrn_u16(vreinterpret_u16_u8(t45.val[0]), vreinterpret_u16_u8(t67.val[0]));
    uint16x4x2_t u57 = vtrn_u16(vreinterpret_u16_u8(t45.val[1]), vreinterpret_u16_u8(t67.val[1]));

    uint32x2x2_t v04 = vtrn_u32(vreinterpret_u32_u16(u02.val[0]), vreinterpret_u32_u16(u46.val[0]));
    uint32x2x2_t v26 = vtrn_u32(vreinterpret_u32_u16(u02.val[1]), vreinterpret_u32_u16(u46.val[1]));
    uint32x2x2_t v15 = vtrn_u32(vreinterpret_u32_u16(u13.val[0]), vreinterpret_u32_u16(u57.val[0]));
    uint32x2x2_t v37 = vtrn_u32(vreinterpret_u32_u16(u13.val[1]), vreinterpret_u32_u16(u57.val[1]));

    c0 = vreinterpret_u8_u32(v04.val[0]);
    c1 = vreinterpret_u8_u32(v15.val[0]);
    c2 = vreinterpret_u8_u32(v26.val[0]);
    c3 = vreinterpret_u8_u32(v37.val[0]);
    c4 = vreinterpret_u8_u32(v04.val[1]);
    c5 = vreinterpret_u8_u32(v15.val[1]);
    c6 = vreinterpret_u8_u32(v26.val[1]);
    c7 = vreinterpret_u8_u32(v37.val[1]);
}

// 8×8 u16 in-place transpose using 128-bit NEON registers.
// Treats interleaved UV pairs as uint16_t elements.
MLVC_FORCE_INLINE void Transpose8x8U16(uint16x8_t& r0, uint16x8_t& r1, uint16x8_t& r2, uint16x8_t& r3, uint16x8_t& r4,
                                       uint16x8_t& r5, uint16x8_t& r6, uint16x8_t& r7)
{
    uint16x8x2_t t01 = vtrnq_u16(r0, r1);
    uint16x8x2_t t23 = vtrnq_u16(r2, r3);
    uint16x8x2_t t45 = vtrnq_u16(r4, r5);
    uint16x8x2_t t67 = vtrnq_u16(r6, r7);

    uint32x4x2_t u02 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[0]), vreinterpretq_u32_u16(t23.val[0]));
    uint32x4x2_t u13 = vtrnq_u32(vreinterpretq_u32_u16(t01.val[1]), vreinterpretq_u32_u16(t23.val[1]));
    uint32x4x2_t u46 = vtrnq_u32(vreinterpretq_u32_u16(t45.val[0]), vreinterpretq_u32_u16(t67.val[0]));
    uint32x4x2_t u57 = vtrnq_u32(vreinterpretq_u32_u16(t45.val[1]), vreinterpretq_u32_u16(t67.val[1]));

    uint8x16_t a0 = vreinterpretq_u8_u32(u02.val[0]);
    uint8x16_t a2 = vreinterpretq_u8_u32(u02.val[1]);
    uint8x16_t a1 = vreinterpretq_u8_u32(u13.val[0]);
    uint8x16_t a3 = vreinterpretq_u8_u32(u13.val[1]);
    uint8x16_t a4 = vreinterpretq_u8_u32(u46.val[0]);
    uint8x16_t a6 = vreinterpretq_u8_u32(u46.val[1]);
    uint8x16_t a5 = vreinterpretq_u8_u32(u57.val[0]);
    uint8x16_t a7 = vreinterpretq_u8_u32(u57.val[1]);

    r0 = vreinterpretq_u16_u8(vcombine_u8(vget_low_u8(a0), vget_low_u8(a4)));
    r1 = vreinterpretq_u16_u8(vcombine_u8(vget_low_u8(a1), vget_low_u8(a5)));
    r2 = vreinterpretq_u16_u8(vcombine_u8(vget_low_u8(a2), vget_low_u8(a6)));
    r3 = vreinterpretq_u16_u8(vcombine_u8(vget_low_u8(a3), vget_low_u8(a7)));
    r4 = vreinterpretq_u16_u8(vcombine_u8(vget_high_u8(a0), vget_high_u8(a4)));
    r5 = vreinterpretq_u16_u8(vcombine_u8(vget_high_u8(a1), vget_high_u8(a5)));
    r6 = vreinterpretq_u16_u8(vcombine_u8(vget_high_u8(a2), vget_high_u8(a6)));
    r7 = vreinterpretq_u16_u8(vcombine_u8(vget_high_u8(a3), vget_high_u8(a7)));
}

// 16×16 byte in-place transpose using 128-bit NEON registers.
MLVC_FORCE_INLINE void Transpose16x16U8(uint8x16_t& r0, uint8x16_t& r1, uint8x16_t& r2, uint8x16_t& r3, uint8x16_t& r4,
                                        uint8x16_t& r5, uint8x16_t& r6, uint8x16_t& r7, uint8x16_t& r8, uint8x16_t& r9,
                                        uint8x16_t& r10, uint8x16_t& r11, uint8x16_t& r12, uint8x16_t& r13,
                                        uint8x16_t& r14, uint8x16_t& r15)
{
    // Step 1: byte-level transpose (pairs)
    uint8x16x2_t t01 = vtrnq_u8(r0, r1);
    uint8x16x2_t t23 = vtrnq_u8(r2, r3);
    uint8x16x2_t t45 = vtrnq_u8(r4, r5);
    uint8x16x2_t t67 = vtrnq_u8(r6, r7);
    uint8x16x2_t t89 = vtrnq_u8(r8, r9);
    uint8x16x2_t tab = vtrnq_u8(r10, r11);
    uint8x16x2_t tcd = vtrnq_u8(r12, r13);
    uint8x16x2_t tef = vtrnq_u8(r14, r15);

    // Step 2: 16-bit level transpose
    uint16x8x2_t u02 = vtrnq_u16(vreinterpretq_u16_u8(t01.val[0]), vreinterpretq_u16_u8(t23.val[0]));
    uint16x8x2_t u13 = vtrnq_u16(vreinterpretq_u16_u8(t01.val[1]), vreinterpretq_u16_u8(t23.val[1]));
    uint16x8x2_t u46 = vtrnq_u16(vreinterpretq_u16_u8(t45.val[0]), vreinterpretq_u16_u8(t67.val[0]));
    uint16x8x2_t u57 = vtrnq_u16(vreinterpretq_u16_u8(t45.val[1]), vreinterpretq_u16_u8(t67.val[1]));
    uint16x8x2_t u8a = vtrnq_u16(vreinterpretq_u16_u8(t89.val[0]), vreinterpretq_u16_u8(tab.val[0]));
    uint16x8x2_t u9b = vtrnq_u16(vreinterpretq_u16_u8(t89.val[1]), vreinterpretq_u16_u8(tab.val[1]));
    uint16x8x2_t uce = vtrnq_u16(vreinterpretq_u16_u8(tcd.val[0]), vreinterpretq_u16_u8(tef.val[0]));
    uint16x8x2_t udf = vtrnq_u16(vreinterpretq_u16_u8(tcd.val[1]), vreinterpretq_u16_u8(tef.val[1]));

    // Step 3: 32-bit level transpose
    uint32x4x2_t v04 = vtrnq_u32(vreinterpretq_u32_u16(u02.val[0]), vreinterpretq_u32_u16(u46.val[0]));
    uint32x4x2_t v15 = vtrnq_u32(vreinterpretq_u32_u16(u13.val[0]), vreinterpretq_u32_u16(u57.val[0]));
    uint32x4x2_t v26 = vtrnq_u32(vreinterpretq_u32_u16(u02.val[1]), vreinterpretq_u32_u16(u46.val[1]));
    uint32x4x2_t v37 = vtrnq_u32(vreinterpretq_u32_u16(u13.val[1]), vreinterpretq_u32_u16(u57.val[1]));
    uint32x4x2_t v8c = vtrnq_u32(vreinterpretq_u32_u16(u8a.val[0]), vreinterpretq_u32_u16(uce.val[0]));
    uint32x4x2_t v9d = vtrnq_u32(vreinterpretq_u32_u16(u9b.val[0]), vreinterpretq_u32_u16(udf.val[0]));
    uint32x4x2_t vae = vtrnq_u32(vreinterpretq_u32_u16(u8a.val[1]), vreinterpretq_u32_u16(uce.val[1]));
    uint32x4x2_t vbf = vtrnq_u32(vreinterpretq_u32_u16(u9b.val[1]), vreinterpretq_u32_u16(udf.val[1]));

    // Step 4: 64-bit level — swap halves using vcombine
    uint8x16_t a0 = vreinterpretq_u8_u32(v04.val[0]);
    uint8x16_t a4 = vreinterpretq_u8_u32(v04.val[1]);
    uint8x16_t a1 = vreinterpretq_u8_u32(v15.val[0]);
    uint8x16_t a5 = vreinterpretq_u8_u32(v15.val[1]);
    uint8x16_t a2 = vreinterpretq_u8_u32(v26.val[0]);
    uint8x16_t a6 = vreinterpretq_u8_u32(v26.val[1]);
    uint8x16_t a3 = vreinterpretq_u8_u32(v37.val[0]);
    uint8x16_t a7 = vreinterpretq_u8_u32(v37.val[1]);
    uint8x16_t a8 = vreinterpretq_u8_u32(v8c.val[0]);
    uint8x16_t ac = vreinterpretq_u8_u32(v8c.val[1]);
    uint8x16_t a9 = vreinterpretq_u8_u32(v9d.val[0]);
    uint8x16_t ad = vreinterpretq_u8_u32(v9d.val[1]);
    uint8x16_t aa = vreinterpretq_u8_u32(vae.val[0]);
    uint8x16_t ae = vreinterpretq_u8_u32(vae.val[1]);
    uint8x16_t ab = vreinterpretq_u8_u32(vbf.val[0]);
    uint8x16_t af = vreinterpretq_u8_u32(vbf.val[1]);

    r0 = vcombine_u8(vget_low_u8(a0), vget_low_u8(a8));
    r1 = vcombine_u8(vget_low_u8(a1), vget_low_u8(a9));
    r2 = vcombine_u8(vget_low_u8(a2), vget_low_u8(aa));
    r3 = vcombine_u8(vget_low_u8(a3), vget_low_u8(ab));
    r4 = vcombine_u8(vget_low_u8(a4), vget_low_u8(ac));
    r5 = vcombine_u8(vget_low_u8(a5), vget_low_u8(ad));
    r6 = vcombine_u8(vget_low_u8(a6), vget_low_u8(ae));
    r7 = vcombine_u8(vget_low_u8(a7), vget_low_u8(af));
    r8 = vcombine_u8(vget_high_u8(a0), vget_high_u8(a8));
    r9 = vcombine_u8(vget_high_u8(a1), vget_high_u8(a9));
    r10 = vcombine_u8(vget_high_u8(a2), vget_high_u8(aa));
    r11 = vcombine_u8(vget_high_u8(a3), vget_high_u8(ab));
    r12 = vcombine_u8(vget_high_u8(a4), vget_high_u8(ac));
    r13 = vcombine_u8(vget_high_u8(a5), vget_high_u8(ad));
    r14 = vcombine_u8(vget_high_u8(a6), vget_high_u8(ae));
    r15 = vcombine_u8(vget_high_u8(a7), vget_high_u8(af));
}

// Tiled Y plane transpose: 16×16 blocks with 8×8 edge fallback.
void TransposeYTiled(const uint8_t* src, std::size_t ss, uint8_t* dst, std::size_t ds, int w, int h)
{
    for (int ty = 0; ty < h; ty += TRANSPOSE_TILE) {
        const int tyEnd = std::min(ty + TRANSPOSE_TILE, h);
        for (int tx = 0; tx < w; tx += TRANSPOSE_TILE) {
            const int txEnd = std::min(tx + TRANSPOSE_TILE, w);
            const int bxEnd16 = tx + ((txEnd - tx) & ~15);
            const int byEnd16 = ty + ((tyEnd - ty) & ~15);

            // 16×16 blocks (main body)
            for (int by = ty; by < byEnd16; by += 16) {
                for (int bx = tx; bx < bxEnd16; bx += 16) {
                    uint8x16_t r0 = vld1q_u8(src + (by + 0) * ss + bx);
                    uint8x16_t r1 = vld1q_u8(src + (by + 1) * ss + bx);
                    uint8x16_t r2 = vld1q_u8(src + (by + 2) * ss + bx);
                    uint8x16_t r3 = vld1q_u8(src + (by + 3) * ss + bx);
                    uint8x16_t r4 = vld1q_u8(src + (by + 4) * ss + bx);
                    uint8x16_t r5 = vld1q_u8(src + (by + 5) * ss + bx);
                    uint8x16_t r6 = vld1q_u8(src + (by + 6) * ss + bx);
                    uint8x16_t r7 = vld1q_u8(src + (by + 7) * ss + bx);
                    uint8x16_t r8 = vld1q_u8(src + (by + 8) * ss + bx);
                    uint8x16_t r9 = vld1q_u8(src + (by + 9) * ss + bx);
                    uint8x16_t r10 = vld1q_u8(src + (by + 10) * ss + bx);
                    uint8x16_t r11 = vld1q_u8(src + (by + 11) * ss + bx);
                    uint8x16_t r12 = vld1q_u8(src + (by + 12) * ss + bx);
                    uint8x16_t r13 = vld1q_u8(src + (by + 13) * ss + bx);
                    uint8x16_t r14 = vld1q_u8(src + (by + 14) * ss + bx);
                    uint8x16_t r15 = vld1q_u8(src + (by + 15) * ss + bx);
                    Transpose16x16U8(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
                    vst1q_u8(dst + (bx + 0) * ds + by, r0);
                    vst1q_u8(dst + (bx + 1) * ds + by, r1);
                    vst1q_u8(dst + (bx + 2) * ds + by, r2);
                    vst1q_u8(dst + (bx + 3) * ds + by, r3);
                    vst1q_u8(dst + (bx + 4) * ds + by, r4);
                    vst1q_u8(dst + (bx + 5) * ds + by, r5);
                    vst1q_u8(dst + (bx + 6) * ds + by, r6);
                    vst1q_u8(dst + (bx + 7) * ds + by, r7);
                    vst1q_u8(dst + (bx + 8) * ds + by, r8);
                    vst1q_u8(dst + (bx + 9) * ds + by, r9);
                    vst1q_u8(dst + (bx + 10) * ds + by, r10);
                    vst1q_u8(dst + (bx + 11) * ds + by, r11);
                    vst1q_u8(dst + (bx + 12) * ds + by, r12);
                    vst1q_u8(dst + (bx + 13) * ds + by, r13);
                    vst1q_u8(dst + (bx + 14) * ds + by, r14);
                    vst1q_u8(dst + (bx + 15) * ds + by, r15);
                }
            }

            // 8×8 fallback for right strip
            const int bxEnd8 = tx + ((txEnd - tx) & ~7);
            const int byEnd8 = ty + ((tyEnd - ty) & ~7);
            for (int by = ty; by < byEnd16; by += 8) {
                for (int bx = bxEnd16; bx < bxEnd8; bx += 8) {
                    uint8x8_t r0 = vld1_u8(src + (by + 0) * ss + bx);
                    uint8x8_t r1 = vld1_u8(src + (by + 1) * ss + bx);
                    uint8x8_t r2 = vld1_u8(src + (by + 2) * ss + bx);
                    uint8x8_t r3 = vld1_u8(src + (by + 3) * ss + bx);
                    uint8x8_t r4 = vld1_u8(src + (by + 4) * ss + bx);
                    uint8x8_t r5 = vld1_u8(src + (by + 5) * ss + bx);
                    uint8x8_t r6 = vld1_u8(src + (by + 6) * ss + bx);
                    uint8x8_t r7 = vld1_u8(src + (by + 7) * ss + bx);
                    uint8x8_t c0, c1, c2, c3, c4, c5, c6, c7;
                    Transpose8x8U8(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7);
                    vst1_u8(dst + (bx + 0) * ds + by, c0);
                    vst1_u8(dst + (bx + 1) * ds + by, c1);
                    vst1_u8(dst + (bx + 2) * ds + by, c2);
                    vst1_u8(dst + (bx + 3) * ds + by, c3);
                    vst1_u8(dst + (bx + 4) * ds + by, c4);
                    vst1_u8(dst + (bx + 5) * ds + by, c5);
                    vst1_u8(dst + (bx + 6) * ds + by, c6);
                    vst1_u8(dst + (bx + 7) * ds + by, c7);
                }
            }
            // 8×8 fallback for bottom strip
            for (int by = byEnd16; by < byEnd8; by += 8) {
                for (int bx = tx; bx < bxEnd8; bx += 8) {
                    uint8x8_t r0 = vld1_u8(src + (by + 0) * ss + bx);
                    uint8x8_t r1 = vld1_u8(src + (by + 1) * ss + bx);
                    uint8x8_t r2 = vld1_u8(src + (by + 2) * ss + bx);
                    uint8x8_t r3 = vld1_u8(src + (by + 3) * ss + bx);
                    uint8x8_t r4 = vld1_u8(src + (by + 4) * ss + bx);
                    uint8x8_t r5 = vld1_u8(src + (by + 5) * ss + bx);
                    uint8x8_t r6 = vld1_u8(src + (by + 6) * ss + bx);
                    uint8x8_t r7 = vld1_u8(src + (by + 7) * ss + bx);
                    uint8x8_t c0, c1, c2, c3, c4, c5, c6, c7;
                    Transpose8x8U8(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7);
                    vst1_u8(dst + (bx + 0) * ds + by, c0);
                    vst1_u8(dst + (bx + 1) * ds + by, c1);
                    vst1_u8(dst + (bx + 2) * ds + by, c2);
                    vst1_u8(dst + (bx + 3) * ds + by, c3);
                    vst1_u8(dst + (bx + 4) * ds + by, c4);
                    vst1_u8(dst + (bx + 5) * ds + by, c5);
                    vst1_u8(dst + (bx + 6) * ds + by, c6);
                    vst1_u8(dst + (bx + 7) * ds + by, c7);
                }
            }
            // Scalar fallback for remaining pixels
            for (int y = ty; y < tyEnd; y++)
                for (int x = bxEnd8; x < txEnd; x++)
                    dst[x * ds + y] = src[y * ss + x];
            for (int y = byEnd8; y < tyEnd; y++)
                for (int x = tx; x < bxEnd8; x++)
                    dst[x * ds + y] = src[y * ss + x];
        }
    }
}

// Tiled UV plane transpose: 8×8 u16 blocks with 4×4 edge fallback.
void TransposeUvTiled(const uint8_t* srcBytes, std::size_t ss, uint8_t* dstBytes, std::size_t ds, int pw, int ph)
{
    const uint16_t* src = reinterpret_cast<const uint16_t*>(srcBytes);
    uint16_t* dst = reinterpret_cast<uint16_t*>(dstBytes);
    const std::size_t ss16 = ss / 2;
    const std::size_t ds16 = ds / 2;
    const int uvTile = TRANSPOSE_TILE / 2;

    for (int ty = 0; ty < ph; ty += uvTile) {
        const int tyEnd = std::min(ty + uvTile, ph);
        for (int tx = 0; tx < pw; tx += uvTile) {
            const int txEnd = std::min(tx + uvTile, pw);
            const int bxEnd8 = tx + ((txEnd - tx) & ~7);
            const int byEnd8 = ty + ((tyEnd - ty) & ~7);

            // 8×8 u16 blocks (main body)
            for (int by = ty; by < byEnd8; by += 8) {
                for (int bx = tx; bx < bxEnd8; bx += 8) {
                    uint16x8_t r0 = vld1q_u16(src + (by + 0) * ss16 + bx);
                    uint16x8_t r1 = vld1q_u16(src + (by + 1) * ss16 + bx);
                    uint16x8_t r2 = vld1q_u16(src + (by + 2) * ss16 + bx);
                    uint16x8_t r3 = vld1q_u16(src + (by + 3) * ss16 + bx);
                    uint16x8_t r4 = vld1q_u16(src + (by + 4) * ss16 + bx);
                    uint16x8_t r5 = vld1q_u16(src + (by + 5) * ss16 + bx);
                    uint16x8_t r6 = vld1q_u16(src + (by + 6) * ss16 + bx);
                    uint16x8_t r7 = vld1q_u16(src + (by + 7) * ss16 + bx);

                    Transpose8x8U16(r0, r1, r2, r3, r4, r5, r6, r7);

                    vst1q_u16(dst + (bx + 0) * ds16 + by, r0);
                    vst1q_u16(dst + (bx + 1) * ds16 + by, r1);
                    vst1q_u16(dst + (bx + 2) * ds16 + by, r2);
                    vst1q_u16(dst + (bx + 3) * ds16 + by, r3);
                    vst1q_u16(dst + (bx + 4) * ds16 + by, r4);
                    vst1q_u16(dst + (bx + 5) * ds16 + by, r5);
                    vst1q_u16(dst + (bx + 6) * ds16 + by, r6);
                    vst1q_u16(dst + (bx + 7) * ds16 + by, r7);
                }
            }

            // 4×4 u16 fallback — right strip
            const int bxEnd4 = tx + ((txEnd - tx) & ~3);
            const int byEnd4 = ty + ((tyEnd - ty) & ~3);
            for (int by = ty; by < byEnd8; by += 4) {
                for (int bx = bxEnd8; bx < bxEnd4; bx += 4) {
                    uint16x4_t r0 = vld1_u16(src + (by + 0) * ss16 + bx);
                    uint16x4_t r1 = vld1_u16(src + (by + 1) * ss16 + bx);
                    uint16x4_t r2 = vld1_u16(src + (by + 2) * ss16 + bx);
                    uint16x4_t r3 = vld1_u16(src + (by + 3) * ss16 + bx);
                    uint16x4x2_t t01 = vtrn_u16(r0, r1);
                    uint16x4x2_t t23 = vtrn_u16(r2, r3);
                    uint32x2x2_t u02 = vtrn_u32(vreinterpret_u32_u16(t01.val[0]), vreinterpret_u32_u16(t23.val[0]));
                    uint32x2x2_t u13 = vtrn_u32(vreinterpret_u32_u16(t01.val[1]), vreinterpret_u32_u16(t23.val[1]));
                    vst1_u16(dst + (bx + 0) * ds16 + by, vreinterpret_u16_u32(u02.val[0]));
                    vst1_u16(dst + (bx + 1) * ds16 + by, vreinterpret_u16_u32(u13.val[0]));
                    vst1_u16(dst + (bx + 2) * ds16 + by, vreinterpret_u16_u32(u02.val[1]));
                    vst1_u16(dst + (bx + 3) * ds16 + by, vreinterpret_u16_u32(u13.val[1]));
                }
            }
            // 4×4 u16 fallback — bottom strip
            for (int by = byEnd8; by < byEnd4; by += 4) {
                for (int bx = tx; bx < bxEnd4; bx += 4) {
                    uint16x4_t r0 = vld1_u16(src + (by + 0) * ss16 + bx);
                    uint16x4_t r1 = vld1_u16(src + (by + 1) * ss16 + bx);
                    uint16x4_t r2 = vld1_u16(src + (by + 2) * ss16 + bx);
                    uint16x4_t r3 = vld1_u16(src + (by + 3) * ss16 + bx);
                    uint16x4x2_t t01 = vtrn_u16(r0, r1);
                    uint16x4x2_t t23 = vtrn_u16(r2, r3);
                    uint32x2x2_t u02 = vtrn_u32(vreinterpret_u32_u16(t01.val[0]), vreinterpret_u32_u16(t23.val[0]));
                    uint32x2x2_t u13 = vtrn_u32(vreinterpret_u32_u16(t01.val[1]), vreinterpret_u32_u16(t23.val[1]));
                    vst1_u16(dst + (bx + 0) * ds16 + by, vreinterpret_u16_u32(u02.val[0]));
                    vst1_u16(dst + (bx + 1) * ds16 + by, vreinterpret_u16_u32(u13.val[0]));
                    vst1_u16(dst + (bx + 2) * ds16 + by, vreinterpret_u16_u32(u02.val[1]));
                    vst1_u16(dst + (bx + 3) * ds16 + by, vreinterpret_u16_u32(u13.val[1]));
                }
            }
            // Scalar remainder
            for (int cy = ty; cy < tyEnd; cy++)
                for (int cx = bxEnd4; cx < txEnd; cx++)
                    dst[cx * ds16 + cy] = src[cy * ss16 + cx];
            for (int cy = byEnd4; cy < tyEnd; cy++)
                for (int cx = tx; cx < bxEnd4; cx++)
                    dst[cx * ds16 + cy] = src[cy * ss16 + cx];
        }
    }
}

void TransposeNv12Impl(const uint8_t* srcY, std::size_t srcYStride, const uint8_t* srcUv, std::size_t srcUvStride,
                       uint8_t* dstY, std::size_t dstYStride, uint8_t* dstUv, std::size_t dstUvStride, int w, int h)
{
    TransposeYTiled(srcY, srcYStride, dstY, dstYStride, w, h);
    TransposeUvTiled(srcUv, srcUvStride, dstUv, dstUvStride, w / 2, h / 2);
}

}  // anonymous namespace
#endif  // MLVC_ARCH_ARM64

Nv12FrameView TransposeNv12(const Nv12FrameView& frame, std::vector<std::byte>& buffer)
{
    const int width = frame.Width();
    const int height = frame.Height();
    const int stride = frame.Stride();

    buffer.resize(static_cast<std::size_t>(width) * height * 3 / 2);

    const uint8_t* srcY = reinterpret_cast<const uint8_t*>(frame.YPlane().data());
    const uint8_t* srcUv = reinterpret_cast<const uint8_t*>(frame.UvPlane().data());
    uint8_t* dstY = reinterpret_cast<uint8_t*>(buffer.data());
    uint8_t* dstUv = dstY + width * height;

    TransposeNv12Impl(srcY, static_cast<std::size_t>(stride), srcUv, static_cast<std::size_t>(stride), dstY,
                      static_cast<std::size_t>(height), dstUv, static_cast<std::size_t>(height), width, height);

    return Nv12FrameView{ height, width, buffer };
}

// ============================================================================
// Nv12ToYuv444Fp16
// ============================================================================

#if defined(MLVC_ARCH_X86_64)
namespace {

void ConvertNv12Aligned(const uint8_t* yPlane, const uint8_t* uvPlane, std::size_t nv12Stride, uint16_t* yOut,
                        uint16_t* uOut, uint16_t* vOut, std::size_t yuv444Stride, int width, int height)
{
    const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
    const __m128i shufU = _mm_setr_epi8(0, 1, 0, 1, 4, 5, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13);
    const __m128i shufV = _mm_setr_epi8(2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15);
    const int w16 = width & ~15;

    for (int y = 0; y < height; y += 2) {
        const uint8_t* srcY0 = yPlane + y * nv12Stride;
        const uint8_t* srcY1 = yPlane + (y + 1) * nv12Stride;
        const uint8_t* srcUv = uvPlane + (y / 2) * nv12Stride;

        uint16_t* dy0 = ByteOffset(yOut, static_cast<std::size_t>(y) * yuv444Stride);
        uint16_t* dy1 = ByteOffset(yOut, static_cast<std::size_t>(y + 1) * yuv444Stride);
        uint16_t* du0 = ByteOffset(uOut, static_cast<std::size_t>(y) * yuv444Stride);
        uint16_t* du1 = ByteOffset(uOut, static_cast<std::size_t>(y + 1) * yuv444Stride);
        uint16_t* dv0 = ByteOffset(vOut, static_cast<std::size_t>(y) * yuv444Stride);
        uint16_t* dv1 = ByteOffset(vOut, static_cast<std::size_t>(y + 1) * yuv444Stride);

        if (w16 < 16) goto scalar_tail;

        {
            __m128i pre_y0 = Load64(srcY0);
            __m128i pre_y0h = Load64(srcY0 + 8);
            __m128i pre_y1 = Load64(srcY1);
            __m128i pre_y1h = Load64(srcY1 + 8);
            __m128i pre_uv0 = Load64(srcUv);
            __m128i pre_uv1 = Load64(srcUv + 8);

            int x = 0;
            for (; x < w16 - 16; x += 16) {
                __m128i cur_y0 = pre_y0, cur_y0h = pre_y0h;
                __m128i cur_y1 = pre_y1, cur_y1h = pre_y1h;
                __m128i cur_uv0 = pre_uv0, cur_uv1 = pre_uv1;

                pre_y0 = Load64(srcY0 + x + 16);
                pre_y0h = Load64(srcY0 + x + 24);
                pre_y1 = Load64(srcY1 + x + 16);
                pre_y1h = Load64(srcY1 + x + 24);
                pre_uv0 = Load64(srcUv + x + 16);
                pre_uv1 = Load64(srcUv + x + 24);

                __m256 y0fLo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(cur_y0)), scale);
                __m256 y0fHi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(cur_y0h)), scale);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dy0 + x),
                                    _mm256_set_m128i(_mm256_cvtps_ph(y0fHi, _MM_FROUND_TO_NEAREST_INT),
                                                     _mm256_cvtps_ph(y0fLo, _MM_FROUND_TO_NEAREST_INT)));

                __m256 y1fLo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(cur_y1)), scale);
                __m256 y1fHi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(cur_y1h)), scale);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dy1 + x),
                                    _mm256_set_m128i(_mm256_cvtps_ph(y1fHi, _MM_FROUND_TO_NEAREST_INT),
                                                     _mm256_cvtps_ph(y1fLo, _MM_FROUND_TO_NEAREST_INT)));

                __m128i uvhLo = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(cur_uv0)), scale),
                                                _MM_FROUND_TO_NEAREST_INT);
                __m128i uvhHi = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(cur_uv1)), scale),
                                                _MM_FROUND_TO_NEAREST_INT);
                __m256i uf = _mm256_set_m128i(_mm_shuffle_epi8(uvhHi, shufU), _mm_shuffle_epi8(uvhLo, shufU));
                __m256i vf = _mm256_set_m128i(_mm_shuffle_epi8(uvhHi, shufV), _mm_shuffle_epi8(uvhLo, shufV));
                _mm256_stream_si256(reinterpret_cast<__m256i*>(du0 + x), uf);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(du1 + x), uf);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dv0 + x), vf);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dv1 + x), vf);
            }

            // Process last chunk with already-preloaded data
            {
                __m256 y0fLo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pre_y0)), scale);
                __m256 y0fHi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pre_y0h)), scale);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dy0 + x),
                                    _mm256_set_m128i(_mm256_cvtps_ph(y0fHi, _MM_FROUND_TO_NEAREST_INT),
                                                     _mm256_cvtps_ph(y0fLo, _MM_FROUND_TO_NEAREST_INT)));

                __m256 y1fLo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pre_y1)), scale);
                __m256 y1fHi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pre_y1h)), scale);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dy1 + x),
                                    _mm256_set_m128i(_mm256_cvtps_ph(y1fHi, _MM_FROUND_TO_NEAREST_INT),
                                                     _mm256_cvtps_ph(y1fLo, _MM_FROUND_TO_NEAREST_INT)));

                __m128i uvhLo = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pre_uv0)), scale),
                                                _MM_FROUND_TO_NEAREST_INT);
                __m128i uvhHi = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pre_uv1)), scale),
                                                _MM_FROUND_TO_NEAREST_INT);
                __m256i uf = _mm256_set_m128i(_mm_shuffle_epi8(uvhHi, shufU), _mm_shuffle_epi8(uvhLo, shufU));
                __m256i vf = _mm256_set_m128i(_mm_shuffle_epi8(uvhHi, shufV), _mm_shuffle_epi8(uvhLo, shufV));
                _mm256_stream_si256(reinterpret_cast<__m256i*>(du0 + x), uf);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(du1 + x), uf);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dv0 + x), vf);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dv1 + x), vf);
                x += 16;
            }
        }

    scalar_tail:
        for (int x = w16; x < width; x++) {
            dy0[x] = Fp32To16(srcY0[x] / 255.0f);
            dy1[x] = Fp32To16(srcY1[x] / 255.0f);
            uint16_t uVal = Fp32To16(srcUv[x & ~1] / 255.0f);
            uint16_t vVal = Fp32To16(srcUv[x | 1] / 255.0f);
            du0[x] = uVal;
            du1[x] = uVal;
            dv0[x] = vVal;
            dv1[x] = vVal;
        }
    }

    _mm_sfence();
}

void ConvertNv12Unaligned(const uint8_t* yPlane, const uint8_t* uvPlane, std::size_t nv12Stride, uint16_t* yOut,
                          uint16_t* uOut, uint16_t* vOut, std::size_t yuv444Stride, int width, int height)
{
    const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
    const __m128i shufU = _mm_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
    const __m128i shufV = _mm_setr_epi8(1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15);

    auto toFp16x16 = [&](__m128i u8x16) -> __m256i {
        __m256 flo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8x16)), scale);
        __m128i hlo = _mm256_cvtps_ph(flo, _MM_FROUND_TO_NEAREST_INT);
        __m256 fhi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u8x16, 8))), scale);
        __m128i hhi = _mm256_cvtps_ph(fhi, _MM_FROUND_TO_NEAREST_INT);
        return _mm256_set_m128i(hhi, hlo);
    };

    const int w16 = width & ~15;

    // Pass 1: Y plane
    for (int y = 0; y < height; y++) {
        const uint8_t* srcY = yPlane + y * nv12Stride;
        uint16_t* dstY = ByteOffset(yOut, static_cast<std::size_t>(y) * yuv444Stride);
        int x = 0;
        for (; x < w16; x += 16) {
            __m128i y8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcY + x));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstY + x), toFp16x16(y8));
        }
        for (; x < width; x++)
            dstY[x] = Fp32To16(srcY[x] / 255.0f);
    }

    // Pass 2: UV planes
    for (int y = 0; y < height; y++) {
        const uint8_t* srcUv = uvPlane + (y / 2) * nv12Stride;
        uint16_t* dstU = ByteOffset(uOut, static_cast<std::size_t>(y) * yuv444Stride);
        uint16_t* dstV = ByteOffset(vOut, static_cast<std::size_t>(y) * yuv444Stride);
        int x = 0;
        for (; x < w16; x += 16) {
            __m128i uv8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcUv + x));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstU + x), toFp16x16(_mm_shuffle_epi8(uv8, shufU)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstV + x), toFp16x16(_mm_shuffle_epi8(uv8, shufV)));
        }
        for (; x < width; x++) {
            dstU[x] = Fp32To16(srcUv[x & ~1] / 255.0f);
            dstV[x] = Fp32To16(srcUv[x | 1] / 255.0f);
        }
    }
}

void ConvertNv12ToYuv444Fp16Impl(const uint8_t* yPlane, const uint8_t* uvPlane, std::size_t nv12Stride, uint16_t* yOut,
                                 uint16_t* uOut, uint16_t* vOut, std::size_t yuv444Stride, int width, int height)
{
    MLVC_ASSERT(width % 2 == 0 && height % 2 == 0);

    const bool aligned = (yuv444Stride % 32 == 0) && (reinterpret_cast<std::uintptr_t>(yOut) % 32 == 0)
                         && (reinterpret_cast<std::uintptr_t>(uOut) % 32 == 0)
                         && (reinterpret_cast<std::uintptr_t>(vOut) % 32 == 0);

    if (aligned) {
        ConvertNv12Aligned(yPlane, uvPlane, nv12Stride, yOut, uOut, vOut, yuv444Stride, width, height);
    } else {
        ConvertNv12Unaligned(yPlane, uvPlane, nv12Stride, yOut, uOut, vOut, yuv444Stride, width, height);
    }
}

}  // anonymous namespace
#endif  // MLVC_ARCH_X86_64

#if defined(MLVC_ARCH_ARM64)
namespace {

void ConvertNv12ToYuv444Fp16Impl(const uint8_t* yPlane, const uint8_t* uvPlane, std::size_t nv12Stride, uint16_t* yOut,
                                 uint16_t* uOut, uint16_t* vOut, std::size_t yuv444Stride, int width, int height)
{
    MLVC_ASSERT(width % 2 == 0 && height % 2 == 0);

    // scale = RN_f16(1/255) = 0x1c04
    const float16x8_t scale = vreinterpretq_f16_u16(vdupq_n_u16(0x1c04));
    // bias = smallest positive f16 subnormal = 0x0001
    const float16x8_t bias = vreinterpretq_f16_u16(vdupq_n_u16(0x0001));
    const uint16x8_t zero = vdupq_n_u16(0);

    const int w16 = width & ~15;

    for (int y = 0; y < height; y += 2) {
        const uint8_t* srcY0 = yPlane + y * nv12Stride;
        const uint8_t* srcY1 = yPlane + (y + 1) * nv12Stride;
        const uint8_t* srcUv = uvPlane + (y / 2) * nv12Stride;

        auto row = [](uint16_t* base, int y, std::size_t stride) -> uint16_t* {
            return reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(base) + static_cast<std::size_t>(y) * stride);
        };

        uint16_t* dy0 = row(yOut, y, yuv444Stride);
        uint16_t* dy1 = row(yOut, y + 1, yuv444Stride);
        uint16_t* du0 = row(uOut, y, yuv444Stride);
        uint16_t* du1 = row(uOut, y + 1, yuv444Stride);
        uint16_t* dv0 = row(vOut, y, yuv444Stride);
        uint16_t* dv1 = row(vOut, y + 1, yuv444Stride);

        int x = 0;
        for (; x < w16; x += 16) {
            // Y row 0: 16 bytes -> 16 fp16
            {
                uint8x16_t y8 = vld1q_u8(srcY0 + x);
                uint16x8_t lo16 = vmovl_u8(vget_low_u8(y8));
                uint16x8_t hi16 = vmovl_u8(vget_high_u8(y8));
                float16x8_t lo_f = vcvtq_f16_u16(lo16);
                float16x8_t hi_f = vcvtq_f16_u16(hi16);
                uint16x8_t lo_r = vbicq_u16(vreinterpretq_u16_f16(vfmaq_f16(bias, lo_f, scale)), vceqq_u16(lo16, zero));
                uint16x8_t hi_r = vbicq_u16(vreinterpretq_u16_f16(vfmaq_f16(bias, hi_f, scale)), vceqq_u16(hi16, zero));
                vst1q_u16(dy0 + x, lo_r);
                vst1q_u16(dy0 + x + 8, hi_r);
            }
            // Y row 1
            {
                uint8x16_t y8 = vld1q_u8(srcY1 + x);
                uint16x8_t lo16 = vmovl_u8(vget_low_u8(y8));
                uint16x8_t hi16 = vmovl_u8(vget_high_u8(y8));
                float16x8_t lo_f = vcvtq_f16_u16(lo16);
                float16x8_t hi_f = vcvtq_f16_u16(hi16);
                uint16x8_t lo_r = vbicq_u16(vreinterpretq_u16_f16(vfmaq_f16(bias, lo_f, scale)), vceqq_u16(lo16, zero));
                uint16x8_t hi_r = vbicq_u16(vreinterpretq_u16_f16(vfmaq_f16(bias, hi_f, scale)), vceqq_u16(hi16, zero));
                vst1q_u16(dy1 + x, lo_r);
                vst1q_u16(dy1 + x + 8, hi_r);
            }
            // UV: load 8 interleaved (U,V) pairs, duplicate to 16 each
            {
                uint8x8x2_t uv = vld2_u8(srcUv + (x & ~1));
                uint16x8_t u16 = vmovl_u8(uv.val[0]);
                uint16x8_t v16 = vmovl_u8(uv.val[1]);
                float16x8_t u_f = vcvtq_f16_u16(u16);
                float16x8_t v_f = vcvtq_f16_u16(v16);
                uint16x8_t u_r = vbicq_u16(vreinterpretq_u16_f16(vfmaq_f16(bias, u_f, scale)), vceqq_u16(u16, zero));
                uint16x8_t v_r = vbicq_u16(vreinterpretq_u16_f16(vfmaq_f16(bias, v_f, scale)), vceqq_u16(v16, zero));
                // Horizontal upsampling: 8 chroma values -> 16
                uint16x8x2_t u_dup = vzipq_u16(u_r, u_r);
                uint16x8x2_t v_dup = vzipq_u16(v_r, v_r);
                vst1q_u16(du0 + x, u_dup.val[0]);
                vst1q_u16(du0 + x + 8, u_dup.val[1]);
                vst1q_u16(du1 + x, u_dup.val[0]);
                vst1q_u16(du1 + x + 8, u_dup.val[1]);
                vst1q_u16(dv0 + x, v_dup.val[0]);
                vst1q_u16(dv0 + x + 8, v_dup.val[1]);
                vst1q_u16(dv1 + x, v_dup.val[0]);
                vst1q_u16(dv1 + x + 8, v_dup.val[1]);
            }
        }
        // Scalar tail (0..15 remaining pixels) — still uses NEON single-lane for bit-exact results
        for (; x < width; x++) {
            auto cvt = [&](uint8_t v) -> uint16_t {
                if (v == 0) return 0;
                uint16x8_t u = vdupq_n_u16(v);
                return vgetq_lane_u16(vreinterpretq_u16_f16(vfmaq_f16(bias, vcvtq_f16_u16(u), scale)), 0);
            };
            dy0[x] = cvt(srcY0[x]);
            dy1[x] = cvt(srcY1[x]);
            uint16_t uVal = cvt(srcUv[x & ~1]);
            uint16_t vVal = cvt(srcUv[x | 1]);
            du0[x] = uVal;
            du1[x] = uVal;
            dv0[x] = vVal;
            dv1[x] = vVal;
        }
    }
}
}  // anonymous namespace
#endif  // MLVC_ARCH_ARM64

void Nv12ToYuv444Fp16(const Nv12FrameView& frame, Tensor<uint16_t, 3>& output)
{
    const int width = frame.Width();
    const int height = frame.Height();
    const int stride = frame.Stride();

    MLVC_ASSERT(output.Shape()[0] == 3);
    MLVC_ASSERT(output.Shape()[1] >= height);
    MLVC_ASSERT(output.Shape()[2] >= width);

    const auto outputStrides = output.Strides();
    MLVC_ASSERT(outputStrides[2] == 1);

    uint16_t* data = output.Data().data();
    uint16_t* yOut = data;
    uint16_t* uOut = data + outputStrides[0];
    uint16_t* vOut = data + 2 * outputStrides[0];
    const std::size_t yuv444Stride = static_cast<std::size_t>(outputStrides[1]) * sizeof(uint16_t);

    const uint8_t* yPlane = reinterpret_cast<const uint8_t*>(frame.YPlane().data());
    const uint8_t* uvPlane = reinterpret_cast<const uint8_t*>(frame.UvPlane().data());

    ConvertNv12ToYuv444Fp16Impl(yPlane, uvPlane, static_cast<std::size_t>(stride), yOut, uOut, vOut, yuv444Stride,
                                width, height);
}

// ============================================================================
// Yuv444Fp16ToNv12
// ============================================================================

#if defined(MLVC_ARCH_X86_64)
namespace {

MLVC_FORCE_INLINE uint8_t Fp16ToU8(uint16_t h)
{
    const float f = Fp16To32(h) * 255.0f;
    return static_cast<uint8_t>(std::clamp(std::roundf(f), 0.0f, 255.0f));
}

MLVC_FORCE_INLINE const uint16_t* RowPtr(const uint16_t* base, int y, std::size_t strideBytes)
{
    return reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(base)
                                             + static_cast<std::size_t>(y) * strideBytes);
}

MLVC_FORCE_INLINE __m256 LdFp16(const uint16_t* p)
{
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

MLVC_FORCE_INLINE __m256i FmaRound(__m256 val, __m256 scale, __m256 biasF, __m256i biasI)
{
    return _mm256_sub_epi32(_mm256_castps_si256(_mm256_fmadd_ps(val, scale, biasF)), biasI);
}

MLVC_FORCE_INLINE __m256i ReduceUv16(const uint16_t* su0, const uint16_t* su1, const uint16_t* sv0, const uint16_t* sv1,
                                     int x, __m256 scale, __m256 biasF, __m256i biasI)
{
    __m256 us = _mm256_add_ps(LdFp16(su0 + x), LdFp16(su1 + x));
    __m256 vs = _mm256_add_ps(LdFp16(sv0 + x), LdFp16(sv1 + x));
    __m256 ubs = _mm256_add_ps(LdFp16(su0 + x + 8), LdFp16(su1 + x + 8));
    __m256 vbs = _mm256_add_ps(LdFp16(sv0 + x + 8), LdFp16(sv1 + x + 8));
    __m256 a = _mm256_add_ps(_mm256_shuffle_ps(us, vs, 0x88), _mm256_shuffle_ps(us, vs, 0xDD));
    __m256 b = _mm256_add_ps(_mm256_shuffle_ps(ubs, vbs, 0x88), _mm256_shuffle_ps(ubs, vbs, 0xDD));
    return _mm256_packs_epi32(FmaRound(a, scale, biasF, biasI), FmaRound(b, scale, biasF, biasI));
}

void ConvertYuv444Fp16ToNv12Impl(const uint16_t* yIn, const uint16_t* uIn, const uint16_t* vIn, std::size_t yuv444Stride,
                                 uint8_t* nv12Out, std::size_t nv12Stride, int width, int height)
{
    uint8_t* nv12Y = nv12Out;
    uint8_t* nv12Uv = nv12Out + nv12Stride * static_cast<std::size_t>(height);

    const __m256 s255 = _mm256_set1_ps(255.0f);
    const __m256 sAvg = _mm256_set1_ps(255.0f / 4.0f);
    const __m256 biasF = _mm256_set1_ps(8388608.0f);      // 2^23
    const __m256i biasI = _mm256_set1_epi32(0x4B000000);  // same as float
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i uvSwap = _mm256_setr_epi8(0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15, 0, 2, 1, 3, 4, 6, 5,
                                            7, 8, 10, 9, 11, 12, 14, 13, 15);

    const int w32 = width & ~31;

    for (int y = 0; y < height; y += 2) {
        const uint16_t* sy0 = RowPtr(yIn, y, yuv444Stride);
        const uint16_t* sy1 = RowPtr(yIn, y + 1, yuv444Stride);
        const uint16_t* su0 = RowPtr(uIn, y, yuv444Stride);
        const uint16_t* su1 = RowPtr(uIn, y + 1, yuv444Stride);
        const uint16_t* sv0 = RowPtr(vIn, y, yuv444Stride);
        const uint16_t* sv1 = RowPtr(vIn, y + 1, yuv444Stride);
        uint8_t* dy0 = nv12Y + static_cast<std::size_t>(y) * nv12Stride;
        uint8_t* dy1 = nv12Y + static_cast<std::size_t>(y + 1) * nv12Stride;
        uint8_t* duv = nv12Uv + static_cast<std::size_t>(y / 2) * nv12Stride;

        int x = 0;
        for (; x < w32; x += 32) {
            // Y: 32 fp16 -> 32 uint8 per row
            __m256i y0a = _mm256_packs_epi32(FmaRound(LdFp16(sy0 + x), s255, biasF, biasI),
                                             FmaRound(LdFp16(sy0 + x + 8), s255, biasF, biasI));
            __m256i y0b = _mm256_packs_epi32(FmaRound(LdFp16(sy0 + x + 16), s255, biasF, biasI),
                                             FmaRound(LdFp16(sy0 + x + 24), s255, biasF, biasI));
            __m256i y1a = _mm256_packs_epi32(FmaRound(LdFp16(sy1 + x), s255, biasF, biasI),
                                             FmaRound(LdFp16(sy1 + x + 8), s255, biasF, biasI));
            __m256i y1b = _mm256_packs_epi32(FmaRound(LdFp16(sy1 + x + 16), s255, biasF, biasI),
                                             FmaRound(LdFp16(sy1 + x + 24), s255, biasF, biasI));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dy0 + x),
                                _mm256_permutevar8x32_epi32(_mm256_packus_epi16(y0a, y0b), perm));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dy1 + x),
                                _mm256_permutevar8x32_epi32(_mm256_packus_epi16(y1a, y1b), perm));

            // UV: 2x2 box average -> interleaved UVUV uint8
            __m256i uv1 = ReduceUv16(su0, su1, sv0, sv1, x, sAvg, biasF, biasI);
            __m256i uv2 = ReduceUv16(su0, su1, sv0, sv1, x + 16, sAvg, biasF, biasI);
            __m256i uv = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(uv1, uv2), perm);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(duv + x), _mm256_shuffle_epi8(uv, uvSwap));
        }

        // Scalar tail for widths not divisible by 32
        for (; x < width; x += 2) {
            dy0[x] = Fp16ToU8(sy0[x]);
            dy0[x + 1] = Fp16ToU8(sy0[x + 1]);
            dy1[x] = Fp16ToU8(sy1[x]);
            dy1[x + 1] = Fp16ToU8(sy1[x + 1]);
            float u00 = Fp16To32(su0[x]), u01 = Fp16To32(su0[x + 1]);
            float u10 = Fp16To32(su1[x]), u11 = Fp16To32(su1[x + 1]);
            float v00 = Fp16To32(sv0[x]), v01 = Fp16To32(sv0[x + 1]);
            float v10 = Fp16To32(sv1[x]), v11 = Fp16To32(sv1[x + 1]);
            int uAvg = static_cast<int>(std::roundf((u00 + u01 + u10 + u11) * 63.75f));
            int vAvg = static_cast<int>(std::roundf((v00 + v01 + v10 + v11) * 63.75f));
            duv[x] = static_cast<uint8_t>(uAvg < 0 ? 0 : uAvg > 255 ? 255 : uAvg);
            duv[x + 1] = static_cast<uint8_t>(vAvg < 0 ? 0 : vAvg > 255 ? 255 : vAvg);
        }
    }
}

}  // anonymous namespace
#endif  // MLVC_ARCH_X86_64

#if defined(MLVC_ARCH_ARM64)
namespace {
void ConvertYuv444Fp16ToNv12Impl(const uint16_t* yIn, const uint16_t* uIn, const uint16_t* vIn, std::size_t yuv444Stride,
                                 uint8_t* nv12Out, std::size_t nv12Stride, int width, int height)
{
    uint8_t* nv12Y = nv12Out;
    uint8_t* nv12Uv = nv12Out + nv12Stride * static_cast<std::size_t>(height);

    // FP16 constant: 255.0 = 0x5BF8 (used by FMLAL for FP32-precision Y multiply)
    const float16x8_t s255 = vreinterpretq_f16_u16(vdupq_n_u16(0x5BF8));
    // FP32 constant: 255.0 (for scalar tail Y path)
    const float32x4_t s255_f32 = vdupq_n_f32(255.0f);
    // FP32 constant: 63.75 = 255/4 (combines averaging and scaling)
    const float32x4_t s63_75 = vdupq_n_f32(63.75f);
    // FP16 constant: 1.0 = 0x3C00 (FMLAL multiplier for vertical add)
    const float16x8_t ones = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    // FP32 zero accumulator for FMLAL Y path
    const float32x4_t zero = vdupq_n_f32(0.0f);

    const int w16 = width & ~15;

    auto row = [](const uint16_t* base, int y, std::size_t stride) -> const uint16_t* {
        return reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(base)
                                                 + static_cast<std::size_t>(y) * stride);
    };

    for (int y = 0; y < height; y += 2) {
        const uint16_t* sy0 = row(yIn, y, yuv444Stride);
        const uint16_t* sy1 = row(yIn, y + 1, yuv444Stride);
        const uint16_t* su0 = row(uIn, y, yuv444Stride);
        const uint16_t* su1 = row(uIn, y + 1, yuv444Stride);
        const uint16_t* sv0 = row(vIn, y, yuv444Stride);
        const uint16_t* sv1 = row(vIn, y + 1, yuv444Stride);
        uint8_t* dy0 = nv12Y + static_cast<std::size_t>(y) * nv12Stride;
        uint8_t* dy1 = nv12Y + static_cast<std::size_t>(y + 1) * nv12Stride;
        uint8_t* duv = nv12Uv + static_cast<std::size_t>(y / 2) * nv12Stride;

        int x = 0;
        for (; x < w16; x += 16) {
            // Y row 0: FMLAL(0, y, 255) -> FP32-precision multiply -> round -> u8
            {
                float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(sy0 + x));
                float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(sy0 + x + 8));
                float32x4_t f0 = vfmlalq_low_f16(zero, h0, s255);
                float32x4_t f1 = vfmlalq_high_f16(zero, h0, s255);
                float32x4_t f2 = vfmlalq_low_f16(zero, h1, s255);
                float32x4_t f3 = vfmlalq_high_f16(zero, h1, s255);
                int16x8_t i01 = vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(f0)), vqmovn_s32(vcvtaq_s32_f32(f1)));
                int16x8_t i23 = vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(f2)), vqmovn_s32(vcvtaq_s32_f32(f3)));
                vst1q_u8(dy0 + x, vcombine_u8(vqmovun_s16(i01), vqmovun_s16(i23)));
            }

            // Y row 1
            {
                float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(sy1 + x));
                float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(sy1 + x + 8));
                float32x4_t f0 = vfmlalq_low_f16(zero, h0, s255);
                float32x4_t f1 = vfmlalq_high_f16(zero, h0, s255);
                float32x4_t f2 = vfmlalq_low_f16(zero, h1, s255);
                float32x4_t f3 = vfmlalq_high_f16(zero, h1, s255);
                int16x8_t i01 = vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(f0)), vqmovn_s32(vcvtaq_s32_f32(f1)));
                int16x8_t i23 = vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(f2)), vqmovn_s32(vcvtaq_s32_f32(f3)));
                vst1q_u8(dy1 + x, vcombine_u8(vqmovun_s16(i01), vqmovun_s16(i23)));
            }

            // UV: FMLAL vertical add (FP32, bit-exact)
            // Row 0: FCVTL to FP32.  Row 1: FMLAL(acc, row1, 1.0) adds f32(row1).
            // Then horizontal pairwise add + x63.75 in FP32.
            {
                float16x8_t u0lo = vreinterpretq_f16_u16(vld1q_u16(su0 + x));
                float16x8_t u0hi = vreinterpretq_f16_u16(vld1q_u16(su0 + x + 8));
                float16x8_t u1lo = vreinterpretq_f16_u16(vld1q_u16(su1 + x));
                float16x8_t u1hi = vreinterpretq_f16_u16(vld1q_u16(su1 + x + 8));

                float32x4_t us0 = vcvt_f32_f16(vget_low_f16(u0lo));
                float32x4_t us1 = vcvt_f32_f16(vget_high_f16(u0lo));
                float32x4_t us2 = vcvt_f32_f16(vget_low_f16(u0hi));
                float32x4_t us3 = vcvt_f32_f16(vget_high_f16(u0hi));
                us0 = vfmlalq_low_f16(us0, u1lo, ones);
                us1 = vfmlalq_high_f16(us1, u1lo, ones);
                us2 = vfmlalq_low_f16(us2, u1hi, ones);
                us3 = vfmlalq_high_f16(us3, u1hi, ones);

                float32x4_t uh01 = vmulq_f32(vpaddq_f32(us0, us1), s63_75);
                float32x4_t uh23 = vmulq_f32(vpaddq_f32(us2, us3), s63_75);
                int16x8_t ui = vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(uh01)), vqmovn_s32(vcvtaq_s32_f32(uh23)));

                float16x8_t v0lo = vreinterpretq_f16_u16(vld1q_u16(sv0 + x));
                float16x8_t v0hi = vreinterpretq_f16_u16(vld1q_u16(sv0 + x + 8));
                float16x8_t v1lo = vreinterpretq_f16_u16(vld1q_u16(sv1 + x));
                float16x8_t v1hi = vreinterpretq_f16_u16(vld1q_u16(sv1 + x + 8));

                float32x4_t vs0 = vcvt_f32_f16(vget_low_f16(v0lo));
                float32x4_t vs1 = vcvt_f32_f16(vget_high_f16(v0lo));
                float32x4_t vs2 = vcvt_f32_f16(vget_low_f16(v0hi));
                float32x4_t vs3 = vcvt_f32_f16(vget_high_f16(v0hi));
                vs0 = vfmlalq_low_f16(vs0, v1lo, ones);
                vs1 = vfmlalq_high_f16(vs1, v1lo, ones);
                vs2 = vfmlalq_low_f16(vs2, v1hi, ones);
                vs3 = vfmlalq_high_f16(vs3, v1hi, ones);

                float32x4_t vh01 = vmulq_f32(vpaddq_f32(vs0, vs1), s63_75);
                float32x4_t vh23 = vmulq_f32(vpaddq_f32(vs2, vs3), s63_75);
                int16x8_t vi = vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(vh01)), vqmovn_s32(vcvtaq_s32_f32(vh23)));

                uint8x8x2_t uv = { { vqmovun_s16(ui), vqmovun_s16(vi) } };
                vst2_u8(duv + x, uv);
            }
        }

        // Scalar tail (0..15 remaining pixels, must be even count)
        for (; x < width; x += 2) {
            // Y: widen to FP32, multiply by 255, round (bit-exact)
            auto cvtY = [&](uint16_t val) -> uint8_t {
                float32x4_t f = vcvt_f32_f16(vreinterpret_f16_u16(vcreate_u16(val)));
                float32x4_t scaled = vmulq_f32(f, s255_f32);
                int32x4_t rounded = vcvtaq_s32_f32(scaled);
                int32_t r = vgetq_lane_s32(rounded, 0);
                return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
            };

            // UV: widen to FP32, sum 4 values, x63.75, round (bit-exact)
            auto cvtUv = [&](uint16_t a, uint16_t b, uint16_t c, uint16_t d) -> uint8_t {
                uint16x4_t h = vcreate_u16(static_cast<uint64_t>(a) | (static_cast<uint64_t>(c) << 16)
                                           | (static_cast<uint64_t>(b) << 32) | (static_cast<uint64_t>(d) << 48));
                float32x4_t f = vcvt_f32_f16(vreinterpret_f16_u16(h));
                float32x2_t sums = vpadd_f32(vget_low_f32(f), vget_high_f32(f));
                float32x2_t total = vpadd_f32(sums, sums);
                float32x2_t scaled = vmul_f32(total, vget_low_f32(s63_75));
                int32x2_t rounded = vcvta_s32_f32(scaled);
                int32_t r = vget_lane_s32(rounded, 0);
                return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
            };

            dy0[x] = cvtY(sy0[x]);
            dy0[x + 1] = cvtY(sy0[x + 1]);
            dy1[x] = cvtY(sy1[x]);
            dy1[x + 1] = cvtY(sy1[x + 1]);

            duv[x] = cvtUv(su0[x], su0[x + 1], su1[x], su1[x + 1]);
            duv[x + 1] = cvtUv(sv0[x], sv0[x + 1], sv1[x], sv1[x + 1]);
        }
    }
}
}  // anonymous namespace
#endif  // MLVC_ARCH_ARM64

Nv12FrameView Yuv444Fp16ToNv12(const Tensor<uint16_t, 3>& yuv444, std::vector<std::byte>& buffer)
{
    const auto shape = yuv444.Shape();
    const auto strides = yuv444.Strides();
    MLVC_ASSERT(strides[2] == 1);

    const int height = shape[1];
    const int width = shape[2];
    MLVC_ASSERT(width % 2 == 0 && height % 2 == 0);

    const uint16_t* data = yuv444.Data().data();
    const uint16_t* yIn = data;
    const uint16_t* uIn = data + strides[0];
    const uint16_t* vIn = data + 2 * strides[0];
    const std::size_t yuv444Stride = static_cast<std::size_t>(strides[1]) * sizeof(uint16_t);

    buffer.resize(static_cast<std::size_t>(height) * width * 3 / 2);

    ConvertYuv444Fp16ToNv12Impl(yIn, uIn, vIn, yuv444Stride, reinterpret_cast<uint8_t*>(buffer.data()),
                                static_cast<std::size_t>(width), width, height);

    return Nv12FrameView{ width, height, buffer };
}

// ============================================================================
// Float16ToInt32
// ============================================================================

namespace {
MLVC_FORCE_INLINE void ConvertFp16ToInt32x8(const uint16_t* src, int32_t* dst)
{
#if defined(MLVC_ARCH_X86_64)
    const __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    const __m256 f0 = _mm256_cvtph_ps(h0);
    const __m256i i0 = _mm256_cvttps_epi32(f0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), i0);
#elif defined(MLVC_ARCH_ARM64)
    float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src));
    float32x4_t fLo = vcvt_f32_f16(vget_low_f16(h));
    float32x4_t fHi = vcvt_f32_f16(vget_high_f16(h));
    int32x4_t iLo = vcvtq_s32_f32(fLo);
    int32x4_t iHi = vcvtq_s32_f32(fHi);
    vst1q_s32(dst, iLo);
    vst1q_s32(dst + 4, iHi);
#else
    #error "Unsupported architecture: requires MLVC_ARCH_X86_64 or MLVC_ARCH_ARM64"
#endif
}
}  // namespace

void Float16ToInt32(const Tensor<uint16_t, 3>& input, Tensor<int32_t, 3>& output)
{
    output.Create(input.Shape());

    const auto inputShape = input.Shape();
    const auto inputStrides = input.Strides();
    const auto outputStrides = output.Strides();
    const uint16_t* srcData = input.Data().data();
    int32_t* dstData = output.Data().data();

    if (input.IsContiguous() && output.IsContiguous()) {
        const std::size_t n = input.Data().size();
        std::size_t i = 0;

        for (; i + 8 <= n; i += 8)
            ConvertFp16ToInt32x8(srcData + i, dstData + i);

        for (; i < n; ++i)
            dstData[i] = static_cast<int32_t>(Fp16To32(srcData[i]));
    } else {
        const int width = inputShape[2];
        for (int ch = 0; ch < inputShape[0]; ch++) {
            for (int y = 0; y < inputShape[1]; y++) {
                const uint16_t* srcRow = srcData + ch * inputStrides[0] + y * inputStrides[1];
                int32_t* dstRow = dstData + ch * outputStrides[0] + y * outputStrides[1];
                int x = 0;

                for (; x + 8 <= width; x += 8)
                    ConvertFp16ToInt32x8(srcRow + x, dstRow + x);

                for (; x < width; ++x)
                    dstRow[x] = static_cast<int32_t>(Fp16To32(srcRow[x]));
            }
        }
    }
}

// ============================================================================
// Int32ToFloat16
// ============================================================================

namespace {
MLVC_FORCE_INLINE void ConvertInt32ToFp16x8(const int32_t* src, uint16_t* dst)
{
#if defined(MLVC_ARCH_X86_64)
    const __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    const __m256 vf = _mm256_cvtepi32_ps(vi);
    const __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), vh);
#elif defined(MLVC_ARCH_ARM64)
    int32x4_t iLo = vld1q_s32(src);
    int32x4_t iHi = vld1q_s32(src + 4);
    float32x4_t fLo = vcvtq_f32_s32(iLo);
    float32x4_t fHi = vcvtq_f32_s32(iHi);
    float16x4_t hLo = vcvt_f16_f32(fLo);
    float16x4_t hHi = vcvt_f16_f32(fHi);
    vst1q_u16(dst, vreinterpretq_u16_f16(vcombine_f16(hLo, hHi)));
#else
    #error "Unsupported architecture: requires MLVC_ARCH_X86_64 or MLVC_ARCH_ARM64"
#endif
}
}  // namespace

void Int32ToFloat16(const Tensor<int32_t, 3>& input, Tensor<uint16_t, 3>& output)
{
    output.Create(input.Shape());

    const auto inputShape = input.Shape();
    const auto inputStrides = input.Strides();
    const auto outputStrides = output.Strides();
    const int32_t* srcData = input.Data().data();
    uint16_t* dstData = output.Data().data();

    if (input.IsContiguous() && output.IsContiguous()) {
        const std::size_t n = input.Data().size();
        std::size_t i = 0;

        for (; i + 8 <= n; i += 8)
            ConvertInt32ToFp16x8(srcData + i, dstData + i);

        for (; i < n; ++i) {
            const float f = static_cast<float>(srcData[i]);
            dstData[i] = Fp32To16(f);
        }
    } else {
        const int width = inputShape[2];
        for (int ch = 0; ch < inputShape[0]; ch++) {
            for (int y = 0; y < inputShape[1]; y++) {
                const int32_t* srcRow = srcData + ch * inputStrides[0] + y * inputStrides[1];
                uint16_t* dstRow = dstData + ch * outputStrides[0] + y * outputStrides[1];
                int x = 0;

                for (; x + 8 <= width; x += 8)
                    ConvertInt32ToFp16x8(srcRow + x, dstRow + x);

                for (; x < width; ++x) {
                    const float f = static_cast<float>(srcRow[x]);
                    dstRow[x] = Fp32To16(f);
                }
            }
        }
    }
}

}  // namespace libmlvc
