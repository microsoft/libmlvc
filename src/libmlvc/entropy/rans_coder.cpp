// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/entropy/rans_coder.hpp"
#include "libmlvc/entropy/rans.hpp"

#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
    #define MLVC_RANS_NOINLINE __attribute__((noinline))
    #define MLVC_RANS_NOSPECTRE
    #define MLVC_RANS_FORCEINLINE_LAMBDA __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define MLVC_RANS_NOINLINE __declspec(noinline)
    #define MLVC_RANS_NOSPECTRE __declspec(spectre(nomitigation))
    #define MLVC_RANS_FORCEINLINE_LAMBDA [[msvc::forceinline]]
#else
    #define MLVC_RANS_NOINLINE
    #define MLVC_RANS_NOSPECTRE
    #define MLVC_RANS_FORCEINLINE_LAMBDA
#endif

// UMULH + noinline MADD encoder optimization for ARM64 (single-stream)
// UMULH(3cy) -> noinline MADD(3cy) = 6cy critical path (vs 8cy with UMULL+LSR+MADD+ADD).
// The noinline barrier prevents MSVC from reassociating MADD operands.
#if defined(_M_ARM64) || defined(__aarch64__)
    #if defined(_MSC_VER)
        #include <intrin.h>
    #endif
namespace rans_arm64 {

    #if defined(_MSC_VER) && defined(_M_ARM64)
__forceinline uint64_t umulh64(uint64_t a, uint64_t b)
{
    return __umulh(a, b);
}
    #elif defined(__SIZEOF_INT128__)
__attribute__((always_inline)) inline uint64_t umulh64(uint64_t a, uint64_t b)
{
    return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) >> 64);
}
    #endif
// Noinline MADD helper - compiles to exactly: madd x0,x0,x1,x2 / ret
// BL/RET overhead is free on Oryon (perfect branch prediction for tiny functions)
MLVC_RANS_NOINLINE inline uint64_t madd64(uint64_t a, uint64_t b, uint64_t c)
{
    return a * b + c;
}
}  // namespace rans_arm64
#elif defined(_M_X64) || defined(__x86_64__)
    #include <emmintrin.h>  // SSE2: _mm_cvtsi32_si128, _mm_cvtsi128_si32
#endif

namespace libmlvc::rans {

// ----- error category
class RansErrorCategory final : public std::error_category {
    const char* name() const noexcept override { return "libmlvc.rans"; }

    std::string message(int ev) const noexcept override
    {
        switch (static_cast<error>(ev)) {
        case error::general_failure:
            return "general failure";
        case error::invalid_pmf:
            return "invalid PMF data";
        case error::invalid_params:
            return "invalid parameter value";
        case error::invalid_stream:
            return "invalid stream";
        case error::invalid_state:
            return "invalid state";
        }
        MLVC_ASSERT(false);
        return "unknown error code";
    }
};

const std::error_category& error_category() noexcept
{
    static const RansErrorCategory category;
    return category;
}

// ----- HeapResizableBuffer

HeapResizableBuffer::HeapResizableBuffer(size_t initialSize, size_t maxSizeStep) : m_newBufferSize(0)
{
    initialSize = std::max(AlignSize(initialSize, true), s_MinBufferSize);
    m_buffer = std::make_unique<std::byte[]>(initialSize);
    m_bufferSize = initialSize;
    m_maxSizeStep = std::max(AlignSize(maxSizeStep, false), s_MinBufferSize);
}

std::span<std::byte> HeapResizableBuffer::BeginToGrow()
{
    auto newSize = m_bufferSize + std::min(m_bufferSize, m_maxSizeStep);
    m_newBuffer = std::make_unique<std::byte[]>(newSize);
    m_newBufferSize = newSize;
    return { m_newBuffer.get(), m_newBufferSize };
}

void HeapResizableBuffer::Commit()
{
    if (m_newBuffer) {
        m_buffer = std::move(m_newBuffer);
        m_bufferSize = m_newBufferSize;
        m_newBufferSize = 0;
        m_newBuffer.release();
    }
}

void HeapResizableBuffer::Rollback()
{
    if (m_newBuffer) {
        m_newBufferSize = 0;
        m_newBuffer.release();
    }
}

// ----- helper functions and types

// Probability distribution descriptor
struct DistributionDesc {
    // offset of zero value symbol
    int32_t m_ValueOffset;
    // last encodable value used as bypass sentinel
    int32_t m_BypassSentinel;
    // rANS symbol offset in global symbol table
    size_t m_SymbolOffset;
};

static std::error_code intializeDistributionDesc(std::vector<DistributionDesc>& distributionDescs,
                                                 const std::span<const int32_t>& pmfLengths,
                                                 const std::span<const int32_t>& pmfOffsets, size_t pmfTableSize)
{
    auto distributionCount = pmfLengths.size();
    if (pmfOffsets.size() != distributionCount) {
        return make_error_code(error::invalid_pmf);
    }
    distributionDescs.resize(distributionCount);

    size_t symbolCursor = 0;
    for (size_t i = 0; i < distributionCount; i++) {
        // last PMF element is tail mass, to be used by bypass symbol
        auto length = pmfLengths[i];
        if (length <= 1 || pmfTableSize - symbolCursor < static_cast<size_t>(length)) {
            return make_error_code(error::invalid_pmf);
        }
        auto& desc = distributionDescs[i];
        desc.m_ValueOffset = pmfOffsets[i];
        desc.m_BypassSentinel = length - 1;
        desc.m_SymbolOffset = symbolCursor;

        symbolCursor += static_cast<size_t>(length);
    }

    if (symbolCursor != pmfTableSize) {
        return make_error_code(error::invalid_pmf);
    }
    return {};
}

template <typename StateType, typename UnitType>
static std::error_code checkBits(int probabilityBits)
{
    constexpr auto MaxScaleBits = details::RansConstants<StateType, UnitType>::MaxScaleBits;
    if (probabilityBits < 2 || static_cast<size_t>(probabilityBits) > MaxScaleBits) {
        return make_error_code(error::invalid_params);
    }
    return {};
}

template <typename T>
static bool isSame(const std::span<T>& s1, const std::span<T>& s2)
{
    return s1.data() == s2.data() && s1.size() == s2.size();
}

template <typename F, typename T>
static std::error_code intializeSymbols(F&& makeSymbol, std::vector<T>& symbols,
                                        const std::vector<DistributionDesc>& distributionDescs,
                                        const std::span<const int32_t>& pmfTable, rans_freq_t symbolBits)
{
    auto totalSymbolCount = pmfTable.size();
    symbols.reserve(totalSymbolCount);

    auto maxFreq = static_cast<int32_t>(1) << symbolBits;

    auto pmfTableIt = pmfTable.begin();
    for (auto& desc : distributionDescs) {
        int32_t start = 0;
        for (size_t i = 0; i <= static_cast<size_t>(desc.m_BypassSentinel); i++) {
            auto freq = *pmfTableIt;
            ++pmfTableIt;
            if (!(freq > 0 && freq <= maxFreq - start)) {
                return make_error_code(error::invalid_pmf);
            }
            symbols.push_back(makeSymbol(static_cast<rans_freq_t>(start),  //
                                         static_cast<rans_freq_t>(freq), symbolBits));
            start += freq;
        }
    }

    return {};
}

// ----- rANS encoder stream

namespace details {

template <typename UnitType>
class ResizableBufferSink {
public:
    using unit_t = UnitType;

    ResizableBufferSink(IResizableBuffer& buffer);

    static std::span<unit_t> ToUnit(const std::span<std::byte>& span)
    {
        return { reinterpret_cast<unit_t*>(span.data()), span.size() / sizeof(unit_t) };
    }

    void operator()(unit_t value)
    {
        if (m_encodePtr <= m_bufferPtr) {
            enlargeBuffer();
        }
        *(--m_encodePtr) = value;
    }

    std::span<const std::byte> EncodedSpan() const
    {
        return { reinterpret_cast<const std::byte*>(m_encodePtr), reinterpret_cast<const std::byte*>(m_endPtr) };
    }

    void Reset() { m_encodePtr = m_endPtr; }

private:
    // Buffer
    IResizableBuffer& m_buffer;
    // Start of current buffer
    unit_t* m_bufferPtr;
    // Encoded data pointer
    unit_t* m_encodePtr;
    // Buffer end pointer
    unit_t* m_endPtr;

    void enlargeBuffer();
};

template <typename UnitType>
inline ResizableBufferSink<UnitType>::ResizableBufferSink(IResizableBuffer& buffer) : m_buffer(buffer)
{
    auto bufferSpan = ToUnit(m_buffer.GetBuffer());
    m_bufferPtr = bufferSpan.data();
    // rANS message is created from end to start
    m_encodePtr = m_endPtr = bufferSpan.data() + bufferSpan.size();
}

template <typename UnitType>
MLVC_RANS_NOINLINE void ResizableBufferSink<UnitType>::enlargeBuffer()
{
    auto newBufferSpan = ToUnit(m_buffer.BeginToGrow());

    auto content = std::span{ m_encodePtr, m_endPtr };
    MLVC_ASSERT(content.size() < newBufferSpan.size());

    auto newContent = newBufferSpan.last(content.size());
    std::copy(content.begin(), content.end(), newContent.begin());

    m_buffer.Commit();
    MLVC_ASSERT(isSame(newBufferSpan, ToUnit(m_buffer.GetBuffer())));

    m_bufferPtr = newBufferSpan.data();
    m_endPtr = newBufferSpan.data() + newBufferSpan.size();
    m_encodePtr = newContent.data();

    MLVC_ASSERT(m_bufferPtr < m_encodePtr);
}

template <typename StateType, typename UnitType>
using RawRansEncoderStream = RansEncoder<StateType, UnitType, ResizableBufferSink<UnitType>>;

template <typename StateType, typename UnitType>
class RansEncoderStreamImpl final : public IRansEncoderStreamImpl {
public:
    using state_t = StateType;
    using unit_t = UnitType;

    using RawStreamType = RawRansEncoderStream<state_t, unit_t>;

    RansEncoderStreamImpl(IResizableBuffer& buffer) : m_rawStream(buffer) {}
    ~RansEncoderStreamImpl() = default;

    RawStreamType& RawStream() { return m_rawStream; }

    static RansEncoderStreamImpl* Downcast(IRansEncoderStreamImpl& baseRef)
    {
        return reinterpret_cast<RansEncoderStreamImpl*>(baseRef.QueryDowncast(&s_tag));
    }

    virtual void* QueryDowncast(const void* tag) override;
    virtual std::span<const std::byte> Flush(bool abort) override;

private:
    RawStreamType m_rawStream;

    // Tag for downcast
    static const std::byte s_tag;
};

template <typename StateType, typename UnitType>
const std::byte RansEncoderStreamImpl<StateType, UnitType>::s_tag{};

template <typename StateType, typename UnitType>
void* RansEncoderStreamImpl<StateType, UnitType>::QueryDowncast(const void* tag)
{
    if (tag == &s_tag) {
        return this;
    }
    return nullptr;
}

template <typename StateType, typename UnitType>
std::span<const std::byte> RansEncoderStreamImpl<StateType, UnitType>::Flush(bool abort)
{
    std::span<const std::byte> data;
    if (!abort) {
        RawStream().Flush();
        data = RawStream().GetSink().EncodedSpan();
    }
    RawStream().Reset();
    RawStream().GetSink().Reset();
    return data;
}

}  // namespace details

std::error_code RansEncoderStream::Initialize(RansVariant variant, IResizableBuffer& buffer)
{
    std::unique_ptr<details::IRansEncoderStreamImpl> impl;
    switch (variant) {
    case RansVariant::RansByte:
        impl = std::make_unique<details::RansEncoderStreamImpl<uint32_t, uint8_t>>(buffer);
        break;
    case RansVariant::Rans64:
        impl = std::make_unique<details::RansEncoderStreamImpl<uint64_t, uint32_t>>(buffer);
        break;
    default:
        return make_error_code(error::invalid_params);
    }
    m_impl = std::move(impl);
    return {};
}

// ----- entropy encoder

namespace details {

template <typename StateType, typename UnitType>
class EntropyEncoderImpl final : public IEntropyEncoderImpl {
public:
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    virtual std::error_code Initialize(const std::span<const int32_t>& pmfLengths,
                                       const std::span<const int32_t>& pmfOffsets,
                                       const std::span<const int32_t>& pmfTable, int symbolBits, int bypassBits) override;
    virtual std::span<const std::byte> Encode(IResizableBuffer& buffer, const std::span<const int32_t>& indices,
                                              const std::span<const int32_t>& values) const override;
    virtual std::error_code Encode(IRansEncoderStreamImpl& stream, const std::span<const int32_t>& indices,
                                   const std::span<const int32_t>& values) const override;

private:
    using symbol_t = RansEncSymbol<state_t, unit_t>;
    using RansEncoder = RawRansEncoderStream<state_t, unit_t>;

    // minimal bypass bits value is 2
    static constexpr inline size_t s_MaxBypassParts = sizeof(freq_t) * CHAR_BIT / 2;

    // Bits used for symbol encoding
    freq_t m_symbolBits = 0;
    // distribution descriptors
    std::vector<DistributionDesc> m_distributionDescs;
    // Concatenated symbol table for all distributions
    std::vector<symbol_t> m_symbols;
    // Zero fast-path: precomputed symbol for value=0 per distribution
    std::vector<symbol_t> m_zeroSymbols;
    // Bits used for bypass encoding
    freq_t m_bypassBits = 0;
    // Max bypass value
    freq_t m_bypassMaxValue = 0;

    void encode(RansEncoder& encoder, const std::span<const int32_t>& indices, const std::span<const int32_t>& values) const;
    void encodeBypassValue(RansEncoder& encoder, freq_t bypassValue) const;
};

template <typename StateType, typename UnitType>
std::error_code EntropyEncoderImpl<StateType, UnitType>::Initialize(const std::span<const int32_t>& pmfLengths,
                                                                    const std::span<const int32_t>& pmfOffsets,
                                                                    const std::span<const int32_t>& pmfTable,
                                                                    int symbolBits, int bypassBits)
{
    auto e = checkBits<state_t, unit_t>(symbolBits);
    if (e) {
        return e;
    }
    e = checkBits<state_t, unit_t>(bypassBits);
    if (e) {
        return e;
    }
    std::vector<DistributionDesc> distributionDescs;
    e = intializeDistributionDesc(distributionDescs, pmfLengths, pmfOffsets, pmfTable.size());
    if (e) {
        return e;
    }
    std::vector<symbol_t> symbols;
    e = intializeSymbols(
        [](freq_t start, freq_t freq, freq_t symbolBits) {  //
            return symbol_t(start, freq, symbolBits);
        },
        symbols, distributionDescs, pmfTable, static_cast<freq_t>(symbolBits));
    if (e) {
        return e;
    }
    m_distributionDescs = std::move(distributionDescs);
    m_symbols = std::move(symbols);
    m_symbolBits = static_cast<freq_t>(symbolBits);
    m_bypassBits = static_cast<freq_t>(bypassBits);
    m_bypassMaxValue = static_cast<freq_t>((1U << bypassBits) - 1);

    // Build zero fast-path symbol table
    m_zeroSymbols.clear();
    m_zeroSymbols.reserve(m_distributionDescs.size());
    for (size_t i = 0; i < m_distributionDescs.size(); i++) {
        m_zeroSymbols.push_back(m_symbols[m_distributionDescs[i].m_SymbolOffset + m_distributionDescs[i].m_ValueOffset]);
    }

    return {};
}

template <typename StateType, typename UnitType>
std::span<const std::byte> EntropyEncoderImpl<StateType, UnitType>::Encode(IResizableBuffer& buffer,
                                                                           const std::span<const int32_t>& indices,
                                                                           const std::span<const int32_t>& values) const
{
    if (!m_symbolBits) {
        // not initialized
        MLVC_ASSERT(false);
        return {};
    }
    RansEncoder encoder(buffer);
    encode(encoder, indices, values);
    encoder.Flush();
    return encoder.GetSink().EncodedSpan();
}

template <typename StateType, typename UnitType>
std::error_code EntropyEncoderImpl<StateType, UnitType>::Encode(IRansEncoderStreamImpl& stream,
                                                                const std::span<const int32_t>& indices,
                                                                const std::span<const int32_t>& values) const
{
    if (!m_symbolBits) {
        // not initialized
        MLVC_ASSERT(false);
        return make_error_code(error::invalid_state);
    }
    auto* streamImpl = RansEncoderStreamImpl<state_t, unit_t>::Downcast(stream);
    if (!streamImpl) {
        return make_error_code(error::invalid_params);
    }
    encode(streamImpl->RawStream(), indices, values);
    return {};
}

template <typename StateType, typename UnitType>
void MLVC_RANS_NOSPECTRE EntropyEncoderImpl<StateType, UnitType>::encode(RansEncoder& encoder,
                                                                         const std::span<const int32_t>& indices,
                                                                         const std::span<const int32_t>& values) const
{
    MLVC_ASSERT(m_symbolBits);

    auto dataSize = indices.size();
    MLVC_ASSERT(dataSize == values.size());

    // Register promotion: copy state to local to break aliasing
    constexpr auto StateBits = RansEncoder::constants::StateBits;

#if defined(_M_ARM64) || defined(__aarch64__)
    // Use uint64_t state on ARM64 to avoid UXTW zero-extension before UMULH
    using enc_t = std::conditional_t<std::is_same_v<state_t, uint32_t>, uint64_t, state_t>;
#else
    using enc_t = state_t;
#endif
    enc_t x = encoder.MutableState();
    auto& sink = encoder.GetSink();

    const auto* descsData = m_distributionDescs.data();
    const auto* symbolsData = m_symbols.data();
    const auto* zeroSymsData = m_zeroSymbols.data();
    const auto descsCount = static_cast<int32_t>(m_distributionDescs.size());

    // Inline renormalize for precomputed symbols
    auto renormLocal = [&](enc_t x, enc_t x_max) MLVC_RANS_FORCEINLINE_LAMBDA -> enc_t {
        while (x >= x_max) {
            sink(static_cast<unit_t>(x));
            x >>= CHAR_BIT * sizeof(unit_t);
        }
        return x;
    };

    // Inline put for precomputed symbols
    auto putSymLocal = [&](enc_t x, const symbol_t& sym) MLVC_RANS_FORCEINLINE_LAMBDA -> enc_t {
        auto x_max = static_cast<enc_t>(sym.m_x_max_hi);
        if constexpr (StateBits > CHAR_BIT * sizeof(freq_t) - 1) {
            x_max <<= StateBits - CHAR_BIT * sizeof(freq_t) + 1;
        }
        x = renormLocal(x, x_max);
#if defined(_M_ARM64) || defined(__aarch64__)
        if constexpr (std::is_same_v<state_t, uint32_t>) {
            // UMULH + noinline MADD: 6-cycle critical path
            uint64_t rcp64 = static_cast<uint64_t>(sym.m_freq_rcp) << (64u - sym.m_freq_rcp_shift);
            uint64_t q = rans_arm64::umulh64(x, rcp64);
            uint64_t xpb = x + sym.m_bias;
            return static_cast<enc_t>(rans_arm64::madd64(q, static_cast<uint64_t>(sym.m_freq_cmpl), xpb));
        } else {
            auto q = RansEncoder::Quotient(static_cast<state_t>(x), sym);
            return x + static_cast<enc_t>(q) * sym.m_freq_cmpl + sym.m_bias;
        }
#elif defined(_M_X64) || defined(__x86_64__)
        // x64 XMM barrier: 8cy critical path (vs 9cy without)
        // _mm_cvtsi32_si128(x+bias) prevents MSVC from reassociating q*cmpl+(x+bias)
        // into q*cmpl+x+bias (2 serial ADDs). MSVC removes the MOVD pair but preserves
        // the expression tree, emitting LEA(x+bias) before the multiply chain.
        // Only requires SSE2 (baseline x64).
        if constexpr (std::is_same_v<state_t, uint32_t>) {
            __m128i vxpb = _mm_cvtsi32_si128(static_cast<int>(x + sym.m_bias));
            auto q = static_cast<state_t>((static_cast<uint64_t>(x) * sym.m_freq_rcp) >> sym.m_freq_rcp_shift);
            return static_cast<enc_t>(q * sym.m_freq_cmpl + static_cast<state_t>(_mm_cvtsi128_si32(vxpb)));
        } else {
            auto q = RansEncoder::Quotient(static_cast<state_t>(x), sym);
            return x + static_cast<enc_t>(q) * sym.m_freq_cmpl + sym.m_bias;
        }
#else
        auto q = RansEncoder::Quotient(static_cast<state_t>(x), sym);
        return x + static_cast<enc_t>(q) * sym.m_freq_cmpl + sym.m_bias;
#endif
    };

    // data is encoded in reverse order
    auto* indicesStart = indices.data();
    auto* indexPtr = indicesStart + dataSize - 1;
    auto* valuesPtr = values.data() + dataSize - 1;

    for (; indexPtr >= indicesStart; --indexPtr, --valuesPtr) {
        auto index = *indexPtr;
        if (index < 0) {
            MLVC_ASSERT(!*valuesPtr);
            continue;
        }
        MLVC_ASSERT(index < descsCount);
        index = std::min(index, descsCount - 1);

        // Zero fast-path (~99% of symbols)
        if (*valuesPtr == 0) [[likely]] {
            x = putSymLocal(x, zeroSymsData[index]);
            continue;
        }

        const auto& distributionDesc = descsData[index];
        auto value = *valuesPtr + distributionDesc.m_ValueOffset;
        if (value < 0 || value >= distributionDesc.m_BypassSentinel) {
            freq_t bypassValue;
            if (value < 0) {
                bypassValue = 2 * static_cast<freq_t>(-value) - 1;
            } else {
                bypassValue = 2 * static_cast<freq_t>(value - distributionDesc.m_BypassSentinel);
            }
            // Write local state back for bypass (uses encoder.Put directly)
            encoder.MutableState() = static_cast<state_t>(x);
            encodeBypassValue(encoder, bypassValue);
            x = static_cast<enc_t>(encoder.MutableState());

            value = distributionDesc.m_BypassSentinel;
        }

        auto symbol = distributionDesc.m_SymbolOffset + static_cast<freq_t>(value);
        x = putSymLocal(x, symbolsData[symbol]);
    }

    encoder.MutableState() = static_cast<state_t>(x);
}

template <typename StateType, typename UnitType>
inline void EntropyEncoderImpl<StateType, UnitType>::encodeBypassValue(RansEncoder& encoder, freq_t bypassValue) const
{
    // Buffer for bypass values
    std::array<freq_t, s_MaxBypassParts> bypassBuffer;

    // Split bypassValue into bypassBits parts
    auto cursor = bypassBuffer.begin();
    for (; bypassValue != 0; ++cursor, bypassValue >>= m_bypassBits) {
        *cursor = bypassValue & m_bypassMaxValue;
    }

    auto bypassCount = static_cast<freq_t>(cursor - bypassBuffer.begin());

    // Put parts in reverse order
    while (cursor > bypassBuffer.begin()) {
        encoder.Put(*(--cursor), 1, m_bypassBits);
    }

    auto bypassPrefixCount = 0;
    // expected to be faster than division
    for (; bypassCount >= m_bypassMaxValue; bypassCount -= m_bypassMaxValue) {
        ++bypassPrefixCount;
    }
    // Put bypassCount remainder
    encoder.Put(bypassCount, 1, m_bypassBits);
    // Put bypassCount prefix
    for (; bypassPrefixCount > 0; --bypassPrefixCount) {
        encoder.Put(m_bypassMaxValue, 1, m_bypassBits);
    }
}

}  // namespace details

std::error_code EntropyEncoder::Initialize(RansVariant variant, const std::span<const int32_t>& pmfLengths,
                                           const std::span<const int32_t>& pmfOffsets,
                                           const std::span<const int32_t>& pmfTable, int symbolBits, int bypassBits)
{
    std::shared_ptr<details::IEntropyEncoderImpl> impl;
    switch (variant) {
    case RansVariant::RansByte:
        impl = std::make_shared<details::EntropyEncoderImpl<uint32_t, uint8_t>>();
        break;
    case RansVariant::Rans64:
        impl = std::make_shared<details::EntropyEncoderImpl<uint64_t, uint32_t>>();
        break;
    default:
        return make_error_code(error::invalid_params);
    }
    auto ec = impl->Initialize(pmfLengths, pmfOffsets, pmfTable, symbolBits, bypassBits);
    if (ec) {
        return ec;
    }

    m_impl = std::move(impl);
    return {};
}

// ----- rANS decoder stream

namespace details {

template <typename UnitType>
class ByteSpanSource {
public:
    using unit_t = UnitType;

    ByteSpanSource(const std::span<const unit_t>& data) : m_decodePtr(data.data()), m_endPtr(data.data() + data.size())
    {
    }

    bool IsOpen() const { return m_decodePtr != nullptr; }

    bool operator()(unit_t& value)
    {
        if (m_decodePtr == m_endPtr) {
            return false;
        }
        value = *(m_decodePtr++);
        return true;
    }

    bool OnOK() { return true; }
    bool OnInvalidStream() { return false; }

    bool IsEOF() const { return m_decodePtr == m_endPtr; }

    // Direct pointer access for register-promoted decode loops
    const unit_t*& MutableDecodePtr() { return m_decodePtr; }
    const unit_t* EndPtr() const { return m_endPtr; }

private:
    const unit_t* m_decodePtr;
    const unit_t* m_endPtr;
};

template <typename StateType, typename UnitType>
using RawRansDecoderStream = RansDecoder<StateType, UnitType, ByteSpanSource<UnitType>>;

template <typename StateType, typename UnitType>
class RansDecoderStreamImpl final : public IRansDecoderStreamImpl {
public:
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    using RawStreamType = RawRansDecoderStream<StateType, UnitType>;

    RansDecoderStreamImpl() : m_rawStream(std::span<const unit_t>{}) {}
    ~RansDecoderStreamImpl() = default;

    RawStreamType& RawStream() { return m_rawStream; }
    const RawStreamType& RawStream() const { return m_rawStream; }

    static RansDecoderStreamImpl* Downcast(IRansDecoderStreamImpl& baseRef)
    {
        return reinterpret_cast<RansDecoderStreamImpl*>(baseRef.QueryDowncast(&s_tag));
    }

    virtual void* QueryDowncast(const void* tag) override;
    virtual std::error_code Open(const std::span<const std::byte>& data) override;
    virtual void Close() override;
    virtual bool IsOpen() const override;
    virtual bool CheckEOF() const override;

private:
    RawStreamType m_rawStream;

    // Tag for downcast
    static const std::byte s_tag;
};

template <typename StateType, typename UnitType>
const std::byte RansDecoderStreamImpl<StateType, UnitType>::s_tag{};

template <typename StateType, typename UnitType>
void* RansDecoderStreamImpl<StateType, UnitType>::QueryDowncast(const void* tag)
{
    if (tag == &s_tag) {
        return this;
    }
    return nullptr;
}

template <typename StateType, typename UnitType>
std::error_code RansDecoderStreamImpl<StateType, UnitType>::Open(const std::span<const std::byte>& data)
{
    if (data.size() % sizeof(unit_t)) {
        return make_error_code(error::invalid_stream);
    }
    auto unitData = std::span{ reinterpret_cast<const unit_t*>(data.data()), data.size() / sizeof(unit_t) };

    auto stream = RawStreamType(unitData);
    if (!stream.Init()) {
        return make_error_code(error::invalid_stream);
    }
    m_rawStream = std::move(stream);
    return {};
}

template <typename StateType, typename UnitType>
void RansDecoderStreamImpl<StateType, UnitType>::Close()
{
    m_rawStream = RawStreamType(std::span<const unit_t>{});
}

template <typename StateType, typename UnitType>
bool RansDecoderStreamImpl<StateType, UnitType>::IsOpen() const
{
    return RawStream().GetSource().IsOpen();
}

template <typename StateType, typename UnitType>
bool RansDecoderStreamImpl<StateType, UnitType>::CheckEOF() const
{
    return RawStream().GetSource().IsOpen() && RawStream().GetSource().IsEOF() && RawStream().CheckEOF();
}

}  // namespace details

std::error_code RansDecoderStream::Initialize(RansVariant variant)
{
    std::unique_ptr<details::IRansDecoderStreamImpl> impl;
    switch (variant) {
    case RansVariant::RansByte:
        impl = std::make_unique<details::RansDecoderStreamImpl<uint32_t, uint8_t>>();
        break;
    case RansVariant::Rans64:
        impl = std::make_unique<details::RansDecoderStreamImpl<uint64_t, uint32_t>>();
        break;
    default:
        return make_error_code(error::invalid_params);
    }
    m_impl = std::move(impl);
    return {};
}

// ----- entropy decoder

namespace details {

template <typename StateType, typename UnitType>
class EntropyDecoderImpl final : public IEntropyDecoderImpl {
public:
    using freq_t = rans_freq_t;
    using state_t = StateType;
    using unit_t = UnitType;

    virtual std::error_code Initialize(const std::span<const int32_t>& pmfLengths,
                                       const std::span<const int32_t>& pmfOffsets,
                                       const std::span<const int32_t>& pmfTable, int symbolBits, int bypassBits) override;

    virtual std::error_code Decode(const std::span<int32_t>& values, const std::span<const int32_t>& indices,
                                   const std::span<const std::byte>& data) const override;
    virtual std::error_code Decode(const std::span<int32_t>& values, const std::span<const int32_t>& indices,
                                   IRansDecoderStreamImpl& stream) const override;

private:
    using RansDecoder = RawRansDecoderStream<state_t, unit_t>;

    // Bits used for symbol encoding
    freq_t m_symbolBits = 0;
    // Bits used for bypass encoding
    freq_t m_bypassBits = 0;
    // Max bypass value
    freq_t m_bypassMaxValue = 0;
    // Distribution descriptors
    std::vector<DistributionDesc> m_distributionDescs;
    // Concatenated CDF table
    std::vector<freq_t> m_cdfTable;
    // Zero fast-path tables (one entry per distribution)
    std::vector<freq_t> m_zeroStart;
    std::vector<freq_t> m_zeroFreq;

    std::error_code decode(RansDecoder& decoder, const std::span<int32_t>& values,
                           const std::span<const int32_t>& indices) const;
    bool decodeBypassValue(RansDecoder& decoder, freq_t& bypassValue) const;
};

template <typename StateType, typename UnitType>
std::error_code EntropyDecoderImpl<StateType, UnitType>::Initialize(const std::span<const int32_t>& pmfLengths,
                                                                    const std::span<const int32_t>& pmfOffsets,
                                                                    const std::span<const int32_t>& pmfTable,
                                                                    int symbolBits, int bypassBits)
{
    auto e = checkBits<state_t, unit_t>(symbolBits);
    if (e) {
        return e;
    }
    e = checkBits<state_t, unit_t>(bypassBits);
    if (e) {
        return e;
    }
    std::vector<DistributionDesc> distributionDescs;
    e = intializeDistributionDesc(distributionDescs, pmfLengths, pmfOffsets, pmfTable.size());
    if (e) {
        return e;
    }

    std::vector<freq_t> cdfTable(pmfTable.size() + distributionDescs.size());

    auto maxFreq = static_cast<int32_t>(1) << symbolBits;
    size_t cursor = 0;
    for (size_t index = 0; index < distributionDescs.size(); index++) {
        auto& desc = distributionDescs[index];

        // set symbol offset to offset in CDF table
        desc.m_SymbolOffset = cursor + index;

        int32_t start = 0;
        for (size_t i = 0; i <= static_cast<size_t>(desc.m_BypassSentinel); i++, cursor++) {
            auto freq = pmfTable[cursor];
            if (!(freq > 0 && freq <= maxFreq - start)) {
                return make_error_code(error::invalid_pmf);
            }
            cdfTable[cursor + index] = static_cast<freq_t>(start);
            start += freq;
        }
        cdfTable[cursor + index] = start;
    }

    m_distributionDescs = std::move(distributionDescs);
    m_cdfTable = std::move(cdfTable);
    m_symbolBits = static_cast<freq_t>(symbolBits);
    m_bypassBits = static_cast<freq_t>(bypassBits);
    m_bypassMaxValue = static_cast<freq_t>((1U << bypassBits) - 1);

    // Build zero fast-path tables
    m_zeroStart.resize(m_distributionDescs.size());
    m_zeroFreq.resize(m_distributionDescs.size());
    for (size_t i = 0; i < m_distributionDescs.size(); i++) {
        auto off = m_distributionDescs[i].m_SymbolOffset + m_distributionDescs[i].m_ValueOffset;
        m_zeroStart[i] = m_cdfTable[off];
        m_zeroFreq[i] = m_cdfTable[off + 1] - m_cdfTable[off];
    }

    return {};
}

template <typename StateType, typename UnitType>
std::error_code EntropyDecoderImpl<StateType, UnitType>::Decode(const std::span<int32_t>& values,
                                                                const std::span<const int32_t>& indices,
                                                                const std::span<const std::byte>& data) const
{
    if (data.size() % sizeof(unit_t)) {
        return make_error_code(error::invalid_stream);
    }
    auto unitData = std::span{ reinterpret_cast<const unit_t*>(data.data()), data.size() / sizeof(unit_t) };
    RansDecoder decoder(unitData);
    if (!decoder.Init()) {
        return make_error_code(error::invalid_stream);
    }
    auto ec = decode(decoder, values, indices);
    if (ec) {
        return ec;
    }
    if (!decoder.GetSource().IsEOF() || !decoder.CheckEOF()) {
        return make_error_code(error::invalid_stream);
    }
    return {};
}

template <typename StateType, typename UnitType>
std::error_code EntropyDecoderImpl<StateType, UnitType>::Decode(const std::span<int32_t>& values,
                                                                const std::span<const int32_t>& indices,
                                                                IRansDecoderStreamImpl& stream) const
{
    auto* streamImpl = RansDecoderStreamImpl<state_t, unit_t>::Downcast(stream);
    if (!streamImpl) {
        return make_error_code(error::invalid_params);
    }
    return decode(streamImpl->RawStream(), values, indices);
}

template <typename StateType, typename UnitType>
std::error_code MLVC_RANS_NOSPECTRE EntropyDecoderImpl<StateType, UnitType>::decode(
    RansDecoder& decoder, const std::span<int32_t>& values, const std::span<const int32_t>& indices) const
{
    if (!m_symbolBits) {
        MLVC_ASSERT(false);
        return make_error_code(error::invalid_state);
    }

    if (values.size() != indices.size()) {
        return make_error_code(error::invalid_params);
    }

    auto FindCdfBinWithHint = [](const freq_t* begin, const freq_t* end, uint32_t offset, uint32_t value) {
        if (value >= begin[offset] && value < begin[offset + 1]) [[likely]] {
            // 99+ % hit rate expected
            return begin + offset;
        } else [[unlikely]] {
            return std::upper_bound(begin + 1, end, value) - 1;
        }
    };

    // Register promotion: copy state and source pointer to locals
    // This breaks aliasing through decoder's m_pair tuple, allowing the compiler
    // to keep x and ptr in registers across loop iterations.
    constexpr auto LowerBound = RansDecoder::constants::LowerBound;
    const auto symbolBits = m_symbolBits;
    const freq_t symMask = (static_cast<freq_t>(1) << symbolBits) - 1;

    state_t x = decoder.MutableState();
    const unit_t* ptr = decoder.GetSource().MutableDecodePtr();
    const unit_t* const endPtr = decoder.GetSource().EndPtr();

    const auto* descsData = m_distributionDescs.data();
    const auto* cdfData = m_cdfTable.data();
    const auto* zeroStartData = m_zeroStart.data();
    const auto* zeroFreqData = m_zeroFreq.data();
    const auto descsCount = static_cast<int32_t>(m_distributionDescs.size());

    // Inline advance with MADD expression form and bounds-checked renormalization
    auto adv = [&](freq_t start, freq_t freq, freq_t scale_bits) MLVC_RANS_FORCEINLINE_LAMBDA {
        x = freq * (x >> scale_bits) + (x & ((static_cast<freq_t>(1) << scale_bits) - 1)) - start;
        while (x < LowerBound) {
            if (ptr >= endPtr) return false;
            x = (x << CHAR_BIT * sizeof(unit_t)) | *ptr++;
        }
        return true;
    };

    for (size_t i = 0; i < values.size(); i++) {
        auto index = indices[i];

        if (index < 0) {
            values[i] = 0;
            continue;
        }

        MLVC_ASSERT(index < descsCount);
        index = std::min(index, descsCount - 1);

        // Zero fast-path: ~99% of symbols are zero
        freq_t cumFreq = static_cast<freq_t>(x) & symMask;
        freq_t zs = zeroStartData[index];
        freq_t zf = zeroFreqData[index];
        freq_t diff = cumFreq - zs;
        if (diff < zf) [[likely]] {
            x = zf * (x >> symbolBits) + diff;
            while (x < LowerBound) {
                if (ptr >= endPtr) goto stream_error;
                x = (x << CHAR_BIT * sizeof(unit_t)) | *ptr++;
            }
            values[i] = 0;
            continue;
        }

        // Non-zero path: CDF bin search + advance
        {
            const auto& desc = descsData[index];
            const auto* basePtr = cdfData + desc.m_SymbolOffset;
            const auto* startPtr =
                FindCdfBinWithHint(basePtr, basePtr + desc.m_BypassSentinel + 1, desc.m_ValueOffset, cumFreq);

            if (!adv(startPtr[0], startPtr[1] - startPtr[0], symbolBits)) {
                goto stream_error;
            }

            auto symbol = static_cast<int32_t>(startPtr - basePtr);
            if (symbol == desc.m_BypassSentinel) [[unlikely]] {
                // Write locals back for bypass (uses decoder directly)
                decoder.MutableState() = x;
                decoder.GetSource().MutableDecodePtr() = ptr;

                freq_t bypassValue;
                if (!decodeBypassValue(decoder, bypassValue)) {
                    goto stream_error;
                }

                // Re-extract locals after bypass
                x = decoder.MutableState();
                ptr = decoder.GetSource().MutableDecodePtr();

                if (bypassValue & 1) {
                    symbol = -static_cast<int32_t>(bypassValue >> 1) - 1;
                } else {
                    symbol = static_cast<int32_t>(bypassValue >> 1) + desc.m_BypassSentinel;
                }
            }
            values[i] = symbol - desc.m_ValueOffset;
        }
    }

    // Write locals back to decoder
    decoder.MutableState() = x;
    decoder.GetSource().MutableDecodePtr() = ptr;
    return {};

stream_error:
    decoder.MutableState() = x;
    decoder.GetSource().MutableDecodePtr() = ptr;
    return make_error_code(error::invalid_stream);
}

template <typename StateType, typename UnitType>
bool EntropyDecoderImpl<StateType, UnitType>::decodeBypassValue(RansDecoder& decoder, freq_t& bypassValue) const
{
    // Step 1 : Read bypass count.
    freq_t value = decoder.Get(m_bypassBits);
    if (!decoder.Advance(value, 1, m_bypassBits)) {
        return false;
    }
    freq_t bypassCount = value;
    while (value == m_bypassMaxValue) {
        value = decoder.Get(m_bypassBits);
        if (!decoder.Advance(value, 1, m_bypassBits)) {
            return false;
        }
        bypassCount += value;
        if (bypassCount > sizeof(freq_t) * CHAR_BIT) {
            return false;
        }
    }

    // Step 2 : Read bypass value.
    freq_t encodedValue = 0;
    bypassCount *= m_bypassBits;
    for (freq_t bypassShift = 0; bypassShift < bypassCount; bypassShift += m_bypassBits) {
        value = decoder.Get(m_bypassBits);
        if (!decoder.Advance(value, 1, m_bypassBits)) {
            return false;
        }
        encodedValue |= value << bypassShift;
    }
    bypassValue = encodedValue;
    return true;
}

}  // namespace details

std::error_code EntropyDecoder::Initialize(RansVariant variant, const std::span<const int32_t>& pmfLengths,
                                           const std::span<const int32_t>& pmfOffsets,
                                           const std::span<const int32_t>& pmfTable, int symbolBits, int bypassBits)
{
    std::shared_ptr<details::IEntropyDecoderImpl> impl;
    switch (variant) {
    case RansVariant::RansByte:
        impl = std::make_shared<details::EntropyDecoderImpl<uint32_t, uint8_t>>();
        break;
    case RansVariant::Rans64:
        impl = std::make_shared<details::EntropyDecoderImpl<uint64_t, uint32_t>>();
        break;
    default:
        return make_error_code(error::invalid_params);
    }
    auto ec = impl->Initialize(pmfLengths, pmfOffsets, pmfTable, symbolBits, bypassBits);
    if (ec) {
        return ec;
    }

    m_impl = std::move(impl);
    return {};
}

}  // namespace libmlvc::rans
