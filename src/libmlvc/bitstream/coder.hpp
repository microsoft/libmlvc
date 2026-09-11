// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/bitstream/bit_io.hpp"

#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <array>
#include <climits>
#include <cstdint>
#include <optional>
#include <span>

namespace libmlvc {

// --------------------------------------------------------------------------------------
// NALU builder and parser
// --------------------------------------------------------------------------------------

// SODB (String Of Data Bits)
// RBSP (Raw Byte Sequence Payload) = SODB + 1 stop bit + trailing zero bits
// EBSP (Encapsulated Byte Sequence Payload) = RBSP + emulation prevention bytes
// NALU (Network Abstraction Layer Unit) = NALU header + EBSP

class NaluBuilder {
public:
    NaluBuilder();
    void Reset();
    void AppendSps(const SpsNalu& sps);
    void AppendPps(const PpsNalu& pps);
    void AppendFrame(const SpsNalu& sps, const FrameNalu& frame);
    std::span<const std::byte> GetOutput() const { return m_outputBuffer; }

private:
    BitWriter m_bitWriter;
    std::vector<std::byte> m_outputBuffer;

    void AppendStartCode();
    void WriteNaluHeader(const NaluHeader& naluHeader);
    void WriteFrameHeader(const SpsNalu& sps, const NaluHeader& naluHeader, const FrameHeader& frameHeader);
    void FlushBitWriter();
};

class NaluParser {
public:
    void Open(std::span<const std::byte> buffer);
    bool HasNext() const;
    expected<NaluHeader> PeekNaluHeader() const;
    expected<SpsNalu> ReadSps();
    expected<PpsNalu> ReadPps();
    expected<FrameNalu> ReadFrame(const SpsNalu& sps);
    expected<CustomNalu> ReadCustomNalu();

private:
    std::span<const std::byte> m_buffer;
    std::vector<std::byte> m_rbspBuffer;

    expected<std::span<const std::byte>> ReadNextNalu();
    BitReader PrepareBitReader(std::span<const std::byte> bytes);
    static expected<NaluHeader> ParseNaluHeader(BitReader& bitReader);
    static expected<FrameHeader> ParseFrameHeader(BitReader& bitReader, const SpsNalu& sps, const NaluHeader& naluHeader);
};

// --------------------------------------------------------------------------------------
// Bitstream encoder and decoder
// --------------------------------------------------------------------------------------

class BaseBitstreamCoder {
public:
    BaseBitstreamCoder()
    {
        // Initialize SPS and PPS with invalid IDs
        m_sps.spsId = -1;
        m_pps.ppsId = -1;
        m_pps.spsId = -2;
    }

protected:
    SpsNalu m_sps{};
    PpsNalu m_pps{};
};

class BitstreamEncoder : public BaseBitstreamCoder {
public:
    expected<void> Initialize();
    expected<std::span<const std::byte>> Encode(const FrameData& data);

private:
    NaluBuilder m_naluBuilder{};
};

class BitstreamDecoder : public BaseBitstreamCoder {
public:
    expected<void> Initialize();
    expected<FrameData> Decode(std::span<const std::byte> bytes);
    const SpsNalu& GetSps() { return m_sps; }
    const PpsNalu& GetPps() { return m_pps; }

private:
    NaluParser m_naluParser{};

    expected<std::optional<FrameData>> DecodeNextNalu();
};

}  // namespace libmlvc
