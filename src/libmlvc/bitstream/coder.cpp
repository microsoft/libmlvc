// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/bitstream/coder.hpp"

#include <libmlvc/error_codes.hpp>

#include <array>
#include <cstdint>
#include <functional>

namespace libmlvc {
namespace {

constexpr int NALU_TYPE_BITS = 6;
constexpr int LAYER_ID_BITS = 6;
constexpr int TEMPORAL_ID_PLUS1_BITS = 3;
constexpr int MLVC_VERSION_BITS = 8;
constexpr int MODEL_SIZE_BITS = 12;
constexpr int MAX_TEMPORAL_LAYERS_MINUS1_BITS = 3;
constexpr int LTR_NUM_SLOTS_BITS = 4;
constexpr int MAX_SPS_ID = 3;
constexpr int MAX_PPS_ID = 3;
constexpr int MIN_FRAME_IDX_BITS = 8;
constexpr int MAX_FRAME_IDX_BITS = 30;
constexpr std::array<std::byte, 4> NALU_START_CODE = {
    std::byte{ 0x00 },
    std::byte{ 0x00 },
    std::byte{ 0x00 },
    std::byte{ 0x01 },
};

}  // namespace

NaluBuilder::NaluBuilder() : m_bitWriter{ 64 * 1024 }
{
    m_outputBuffer.reserve(64 * 1024);
}

void NaluBuilder::Reset()
{
    m_outputBuffer.clear();
    m_bitWriter.Reset();
}

void NaluBuilder::AppendSps(const SpsNalu& sps)
{
    AppendStartCode();
    m_bitWriter.Reset();
    WriteNaluHeader(sps.naluHeader);
    m_bitWriter.WriteUint(sps.mlvcVersionMajor, MLVC_VERSION_BITS);
    m_bitWriter.WriteUint(sps.mlvcVersionMinor, MLVC_VERSION_BITS);
    m_bitWriter.WriteUe(sps.spsId);
    m_bitWriter.WriteUint(sps.modelWidthDiv2, MODEL_SIZE_BITS);
    m_bitWriter.WriteUint(sps.modelHeightDiv2, MODEL_SIZE_BITS);
    m_bitWriter.WriteBit(sps.transposeFlag);
    m_bitWriter.WriteBit(sps.cropFlag);
    if (sps.cropFlag) {
        m_bitWriter.WriteUe(sps.cropLeftDiv2);
        m_bitWriter.WriteUe(sps.cropRightDiv2);
        m_bitWriter.WriteUe(sps.cropTopDiv2);
        m_bitWriter.WriteUe(sps.cropBottomDiv2);
    }
    m_bitWriter.WriteUint(sps.maxTemporalLayersMinus1, MAX_TEMPORAL_LAYERS_MINUS1_BITS);
    m_bitWriter.WriteUe(sps.frameIdxBitsMinus8);
    FlushBitWriter();
}

void NaluBuilder::AppendPps(const PpsNalu& pps)
{
    AppendStartCode();
    m_bitWriter.Reset();
    WriteNaluHeader(pps.naluHeader);
    m_bitWriter.WriteUe(pps.ppsId);
    m_bitWriter.WriteUe(pps.spsId);
    m_bitWriter.WriteSe(pps.initQpMinus26);
    FlushBitWriter();
}

void NaluBuilder::AppendFrame(const SpsNalu& sps, const FrameNalu& frame)
{
    AppendStartCode();
    m_bitWriter.Reset();
    WriteNaluHeader(frame.naluHeader);
    WriteFrameHeader(sps, frame.naluHeader, frame.frameHeader);
    m_bitWriter.WriteBytes(frame.payload);
    FlushBitWriter();
}

void NaluBuilder::AppendStartCode()
{
    m_outputBuffer.insert(m_outputBuffer.end(), NALU_START_CODE.begin(), NALU_START_CODE.end());
}

void NaluBuilder::WriteNaluHeader(const NaluHeader& naluHeader)
{
    m_bitWriter.WriteBit(naluHeader.forbiddenZero);
    m_bitWriter.WriteUint(static_cast<int>(naluHeader.naluType), NALU_TYPE_BITS);
    m_bitWriter.WriteUint(naluHeader.layerId, LAYER_ID_BITS);
    m_bitWriter.WriteUint(naluHeader.temporalIdPlus1, TEMPORAL_ID_PLUS1_BITS);
}

void NaluBuilder::WriteFrameHeader(const SpsNalu& sps, const NaluHeader& naluHeader, const FrameHeader& frameHeader)
{
    m_bitWriter.WriteUe(frameHeader.ppsId);
    m_bitWriter.WriteSe(frameHeader.qpDelta);

    const int frameIdxBits = sps.frameIdxBitsMinus8 + MIN_FRAME_IDX_BITS;
    int numLtrSlots = 0;
    for (int i = 0; i < static_cast<int>(frameHeader.ltrSlots.size()); i++) {
        if (frameHeader.ltrSlots[i].HasValue()) {
            numLtrSlots = i + 1;
        }
    }
    m_bitWriter.WriteUint(numLtrSlots, LTR_NUM_SLOTS_BITS);
    for (int i = 0; i < numLtrSlots; i++) {
        const auto& slot = frameHeader.ltrSlots[i];
        m_bitWriter.WriteBit(slot.empty);
        if (slot.HasValue()) {
            m_bitWriter.WriteUint(slot.frameIdx, frameIdxBits);
        }
    }

    if (naluHeader.naluType != NaluType::IDR_N_LP) {
        m_bitWriter.WriteUint(frameHeader.frameIdx, frameIdxBits);
        m_bitWriter.WriteUint(frameHeader.refFrameIdx, frameIdxBits);
        m_bitWriter.WriteBit(frameHeader.featureResetFlag);
    }
}

void NaluBuilder::FlushBitWriter()
{
    m_bitWriter.WriteStopBit();
    AddEmulationPreventionBytes(m_bitWriter.GetBytes(), m_outputBuffer);
    m_bitWriter.Reset();
}

void NaluParser::Open(std::span<const std::byte> buffer)
{
    m_buffer = buffer;

    // Remove start code if it exists at the beginning of the buffer
    if (m_buffer.size() >= NALU_START_CODE.size()
        && std::equal(NALU_START_CODE.begin(), NALU_START_CODE.end(), m_buffer.begin())) {
        m_buffer = m_buffer.subspan(NALU_START_CODE.size());
    }
}

bool NaluParser::HasNext() const
{
    return m_buffer.size() > 0;
}

expected<NaluHeader> NaluParser::PeekNaluHeader() const
{
    if (m_buffer.empty()) {
        MLVC_LOG_ERROR("Buffer is empty");
        return make_error_code(Error::bit_stream_unexpected_error);
    }

    BitReader bitReader = BitReader(m_buffer);
    auto ret = ParseNaluHeader(bitReader);
    if (!ret) {
        MLVC_LOG_ERROR("Failed to parse NALU header");
        return ret.error();
    }
    return ret.value();
}

expected<SpsNalu> NaluParser::ReadSps()
{
    const auto nalu = ReadNextNalu();
    if (!nalu) return nalu.error();

    SpsNalu sps{};
    BitReader bitReader = PrepareBitReader(nalu.value());
    {
        auto ret = ParseNaluHeader(bitReader);
        if (!ret) return ret.error();
        sps.naluHeader = ret.value();
    }

    {
        auto mlvcVersionMajor = bitReader.ReadUint(MLVC_VERSION_BITS);
        if (!mlvcVersionMajor) return mlvcVersionMajor.error();
        sps.mlvcVersionMajor = mlvcVersionMajor.value();
    }

    {
        auto mlvcVersionMinor = bitReader.ReadUint(MLVC_VERSION_BITS);
        if (!mlvcVersionMinor) return mlvcVersionMinor.error();
        sps.mlvcVersionMinor = mlvcVersionMinor.value();
    }

    {
        auto spsId = bitReader.ReadUe();
        if (!spsId) return spsId.error();
        sps.spsId = spsId.value();
    }

    {
        auto modelWidthDiv2 = bitReader.ReadUint(MODEL_SIZE_BITS);
        if (!modelWidthDiv2) return modelWidthDiv2.error();
        sps.modelWidthDiv2 = modelWidthDiv2.value();
    }

    {
        auto modelHeightDiv2 = bitReader.ReadUint(MODEL_SIZE_BITS);
        if (!modelHeightDiv2) return modelHeightDiv2.error();
        sps.modelHeightDiv2 = modelHeightDiv2.value();
    }

    {
        auto transposeFlag = bitReader.ReadBit();
        if (!transposeFlag) return transposeFlag.error();
        sps.transposeFlag = transposeFlag.value();
    }

    {
        auto cropFlag = bitReader.ReadBit();
        if (!cropFlag) return cropFlag.error();
        sps.cropFlag = cropFlag.value();
    }

    if (sps.cropFlag) {
        {
            auto cropLeftDiv2 = bitReader.ReadUe();
            if (!cropLeftDiv2) return cropLeftDiv2.error();
            sps.cropLeftDiv2 = cropLeftDiv2.value();
        }

        {
            auto cropRightDiv2 = bitReader.ReadUe();
            if (!cropRightDiv2) return cropRightDiv2.error();
            sps.cropRightDiv2 = cropRightDiv2.value();
        }

        {
            auto cropTopDiv2 = bitReader.ReadUe();
            if (!cropTopDiv2) return cropTopDiv2.error();
            sps.cropTopDiv2 = cropTopDiv2.value();
        }

        {
            auto cropBottomDiv2 = bitReader.ReadUe();
            if (!cropBottomDiv2) return cropBottomDiv2.error();
            sps.cropBottomDiv2 = cropBottomDiv2.value();
        }
    } else {
        sps.cropLeftDiv2 = 0;
        sps.cropRightDiv2 = 0;
        sps.cropTopDiv2 = 0;
        sps.cropBottomDiv2 = 0;
    }

    // Widen the non-negative crop sums before checking that the remaining size is positive.
    // Runs for both crop paths so zero model dimensions are rejected either way, and it
    // bounds the later conversion from Div2 values to pixels.
    const auto cropWidthDiv2 = static_cast<int64_t>(sps.cropLeftDiv2) + sps.cropRightDiv2;
    const auto cropHeightDiv2 = static_cast<int64_t>(sps.cropTopDiv2) + sps.cropBottomDiv2;
    if (cropWidthDiv2 >= sps.modelWidthDiv2 || cropHeightDiv2 >= sps.modelHeightDiv2) {
        MLVC_LOG_ERROR("Invalid SPS geometry (div2): left=%d right=%d top=%d bottom=%d for model %dx%d", sps.cropLeftDiv2,
                       sps.cropRightDiv2, sps.cropTopDiv2, sps.cropBottomDiv2, sps.modelWidthDiv2, sps.modelHeightDiv2);
        return make_error_code(Error::bit_stream_unexpected_error);
    }

    {
        auto maxTemporalLayersMinus1 = bitReader.ReadUint(MAX_TEMPORAL_LAYERS_MINUS1_BITS);
        if (!maxTemporalLayersMinus1) return maxTemporalLayersMinus1.error();

        if (maxTemporalLayersMinus1.value() >= MAX_TEMPORAL_LAYERS) {
            MLVC_LOG_ERROR("Max temporal layers %d exceeds maximum %d", maxTemporalLayersMinus1.value() + 1,
                           MAX_TEMPORAL_LAYERS);
            return make_error_code(Error::bit_stream_unexpected_error);
        }
        sps.maxTemporalLayersMinus1 = maxTemporalLayersMinus1.value();
    }

    {
        auto frameIdxBitsMinus8 = bitReader.ReadUe();
        if (!frameIdxBitsMinus8) return frameIdxBitsMinus8.error();

        if (frameIdxBitsMinus8.value() > MAX_FRAME_IDX_BITS - MIN_FRAME_IDX_BITS) {
            MLVC_LOG_ERROR("Frame index bits (minus %d) %d out of range [0, %d]", MIN_FRAME_IDX_BITS,
                           frameIdxBitsMinus8.value(), MAX_FRAME_IDX_BITS - MIN_FRAME_IDX_BITS);
            return make_error_code(Error::bit_stream_unexpected_error);
        }
        sps.frameIdxBitsMinus8 = frameIdxBitsMinus8.value();
    }

    if (auto ret = bitReader.ReadStopBit(); !ret) {
        return ret.error();
    }
    return sps;
}

expected<PpsNalu> NaluParser::ReadPps()
{

    const auto nalu = ReadNextNalu();
    if (!nalu) return nalu.error();

    PpsNalu pps{};
    BitReader bitReader = PrepareBitReader(nalu.value());
    {
        auto naluHeader = ParseNaluHeader(bitReader);
        if (!naluHeader) return naluHeader.error();
        pps.naluHeader = naluHeader.value();
    }

    {
        auto ppsId = bitReader.ReadUe();
        if (!ppsId) return ppsId.error();
        pps.ppsId = ppsId.value();
    }

    {
        auto spsId = bitReader.ReadUe();
        if (!spsId) return spsId.error();
        pps.spsId = spsId.value();
    }

    {
        auto initQpMinus26 = bitReader.ReadSe();
        if (!initQpMinus26) return initQpMinus26.error();

        if (initQpMinus26.value() < MIN_QP - 26 || initQpMinus26.value() > MAX_QP - 26) {
            MLVC_LOG_ERROR("Initial QP (minus 26) %d out of range [%d, %d]", initQpMinus26.value(), MIN_QP - 26,
                           MAX_QP - 26);
            return make_error_code(Error::bit_stream_unexpected_error);
        }
        pps.initQpMinus26 = initQpMinus26.value();
    }

    if (auto ret = bitReader.ReadStopBit(); !ret) {
        return ret.error();
    }
    return pps;
}

expected<FrameNalu> NaluParser::ReadFrame(const SpsNalu& sps)
{

    const auto nalu = ReadNextNalu();
    if (!nalu) return nalu.error();

    FrameNalu frame{};
    BitReader bitReader = PrepareBitReader(nalu.value());
    {
        auto naluHeader = ParseNaluHeader(bitReader);
        if (!naluHeader) return naluHeader.error();
        frame.naluHeader = naluHeader.value();
    }

    {
        auto frameHeader = ParseFrameHeader(bitReader, sps, frame.naluHeader);
        if (!frameHeader) return frameHeader.error();
        frame.frameHeader = frameHeader.value();
    }

    {
        auto payload = bitReader.ReadBytes();
        if (!payload) return payload.error();
        frame.payload = payload.value();
    }

    if (auto ret = bitReader.ReadStopBit(); !ret) {
        return ret.error();
    }
    return frame;
}

expected<CustomNalu> NaluParser::ReadCustomNalu()
{
    const auto nalu = ReadNextNalu();
    if (!nalu) return nalu.error();

    CustomNalu customNalu;
    BitReader bitReader = PrepareBitReader(nalu.value());
    {
        auto naluHeader = ParseNaluHeader(bitReader);
        if (!naluHeader) return naluHeader.error();
        customNalu.naluHeader = naluHeader.value();
    }

    {
        // Read the payload bytes (guaranteed to be byte-aligned)
        auto payload = bitReader.ReadBytes();
        if (!payload) return payload.error();
        customNalu.payload = payload.value();
    }
    return customNalu;
}

expected<std::span<const std::byte>> NaluParser::ReadNextNalu()
{
    if (m_buffer.empty()) {
        return make_error_code(Error::bit_stream_unexpected_error);
    }

    auto isStartCode = [](std::span<const std::byte> data) {
        return data.size() >= NALU_START_CODE.size()
               && std::equal(NALU_START_CODE.begin(), NALU_START_CODE.end(), data.begin());
    };

    // Collect bytes until the next NALU start code
    size_t nextNaluStart = 0;
    while (nextNaluStart < m_buffer.size() && !isStartCode(m_buffer.subspan(nextNaluStart))) {
        nextNaluStart++;
    }
    auto naluBytes = m_buffer.subspan(0, nextNaluStart);

    // Update buffer
    if (nextNaluStart + NALU_START_CODE.size() <= m_buffer.size()) {
        m_buffer = m_buffer.subspan(nextNaluStart + NALU_START_CODE.size());
    } else {
        m_buffer = {};
    }

    return naluBytes;
}

BitReader NaluParser::PrepareBitReader(std::span<const std::byte> bytes)
{
    RemoveEmulationPreventionBytes(bytes, m_rbspBuffer);
    return BitReader{ m_rbspBuffer };
}

expected<NaluHeader> NaluParser::ParseNaluHeader(BitReader& bitReader)
{
    auto forbiddenZero = bitReader.ReadBit();
    if (!forbiddenZero) return forbiddenZero.error();
    if (forbiddenZero.value()) {
        MLVC_LOG_ERROR("Forbidden zero bit is set in NALU header");
        return make_error_code(Error::bit_stream_unexpected_error);
    }

    auto naluType = bitReader.ReadUint(NALU_TYPE_BITS);
    if (!naluType) return naluType.error();

    auto layerId = bitReader.ReadUint(LAYER_ID_BITS);
    if (!layerId) return layerId.error();

    auto temporalIdPlus1 = bitReader.ReadUint(TEMPORAL_ID_PLUS1_BITS);
    if (!temporalIdPlus1) return temporalIdPlus1.error();

    NaluHeader header;
    header.forbiddenZero = 0;
    header.naluType = static_cast<NaluType>(naluType.value());
    header.layerId = layerId.value();
    header.temporalIdPlus1 = temporalIdPlus1.value();
    return header;
}

expected<FrameHeader> NaluParser::ParseFrameHeader(BitReader& bitReader, const SpsNalu& sps, const NaluHeader& naluHeader)
{
    FrameHeader frameHeader;

    {
        auto ppsId = bitReader.ReadUe();
        if (!ppsId) return ppsId.error();
        frameHeader.ppsId = ppsId.value();
    }

    {
        auto qpDelta = bitReader.ReadSe();
        if (!qpDelta) return qpDelta.error();
        frameHeader.qpDelta = qpDelta.value();
    }

    const int frameIdxBits = sps.frameIdxBitsMinus8 + MIN_FRAME_IDX_BITS;
    {
        std::fill(frameHeader.ltrSlots.begin(), frameHeader.ltrSlots.end(), LtrSlotInfo{});
        auto numLtrSlots = bitReader.ReadUint(LTR_NUM_SLOTS_BITS);
        if (!numLtrSlots) return numLtrSlots.error();

        if (numLtrSlots.value() > MAX_LTR_SLOTS) {
            MLVC_LOG_ERROR("Number of LTR slots %d exceeds maximum %d", numLtrSlots.value(), MAX_LTR_SLOTS);
            return make_error_code(Error::bit_stream_unexpected_error);
        }

        for (int i = 0; i < numLtrSlots.value(); i++) {
            auto isEmpty = bitReader.ReadBit();
            if (!isEmpty) return isEmpty.error();

            if (!isEmpty.value()) {
                auto ltrFrameIdx = bitReader.ReadUint(frameIdxBits);
                if (!ltrFrameIdx) return ltrFrameIdx.error();
                frameHeader.ltrSlots[i] = LtrSlotInfo{ static_cast<int>(ltrFrameIdx.value()) };
            }
        }
    }

    if (naluHeader.naluType != NaluType::IDR_N_LP) {
        {
            auto frameIdx = bitReader.ReadUint(frameIdxBits);
            if (!frameIdx) return frameIdx.error();
            frameHeader.frameIdx = frameIdx.value();
        }

        {
            auto refFrameIdx = bitReader.ReadUint(frameIdxBits);
            if (!refFrameIdx) return refFrameIdx.error();
            frameHeader.refFrameIdx = refFrameIdx.value();
        }

        {
            auto featureResetFlag = bitReader.ReadBit();
            if (!featureResetFlag) return featureResetFlag.error();
            frameHeader.featureResetFlag = featureResetFlag.value();
        }

    } else {
        frameHeader.frameIdx = 0;
        frameHeader.refFrameIdx = 0;
    }

    return frameHeader;
}

expected<void> BitstreamEncoder::Initialize()
{
    return {};
}

expected<std::span<const std::byte>> BitstreamEncoder::Encode(const FrameData& data)
{
    if (data.maxTemporalLayers < 1 || data.maxTemporalLayers > MAX_TEMPORAL_LAYERS) {
        MLVC_LOG_ERROR("Max temporal layers %d is out of range [1, %d]", data.maxTemporalLayers, MAX_TEMPORAL_LAYERS);
        return make_error_code(Error::bit_stream_unexpected_error);
    }
    if (data.temporalId < 0 || data.temporalId >= data.maxTemporalLayers) {
        MLVC_LOG_ERROR("Temporal ID %d is out of range [0, %d]", data.temporalId, data.maxTemporalLayers - 1);
        return make_error_code(Error::bit_stream_unexpected_error);
    }

    m_naluBuilder.Reset();

    if (data.frameType == FrameType::I_FRAME) {
        SpsNalu newSps;
        newSps.naluHeader.naluType = NaluType::SPS;
        newSps.mlvcVersionMajor = data.mlvcVersion.major;
        newSps.mlvcVersionMinor = data.mlvcVersion.minor;
        newSps.spsId = m_sps.spsId;  // Will be updated below if needed
        newSps.modelWidthDiv2 = data.modelWidth / 2;
        newSps.modelHeightDiv2 = data.modelHeight / 2;
        newSps.cropFlag = !data.cropOffsets.IsEmpty();
        newSps.cropLeftDiv2 = data.cropOffsets.left / 2;
        newSps.cropRightDiv2 = data.cropOffsets.right / 2;
        newSps.cropTopDiv2 = data.cropOffsets.top / 2;
        newSps.cropBottomDiv2 = data.cropOffsets.bottom / 2;
        newSps.transposeFlag = data.transposeFlag;
        newSps.maxTemporalLayersMinus1 = data.maxTemporalLayers - 1;
        newSps.frameIdxBitsMinus8 = data.frameIdxBits - MIN_FRAME_IDX_BITS;

        const bool updateSps = m_sps.spsId < 0 || m_pps.ppsId < 0 || newSps != m_sps;
        if (updateSps) {
            const int newSpsId = (m_sps.spsId + 1) % MAX_SPS_ID;
            m_sps = newSps;
            m_sps.spsId = newSpsId;

            const int newPpsId = (m_pps.ppsId + 1) % MAX_PPS_ID;
            m_pps.ppsId = newPpsId;
            m_pps.spsId = m_sps.spsId;
            m_pps.initQpMinus26 = data.qp - 26;
        }

        m_naluBuilder.AppendSps(m_sps);
        m_naluBuilder.AppendPps(m_pps);
    }

    // Write frame data
    if (data.payload.size() > 0) {
        FrameNalu frameNalu{};
        frameNalu.naluHeader.temporalIdPlus1 = data.temporalId + 1;

        if (data.frameType == FrameType::I_FRAME) {
            frameNalu.naluHeader.naluType = NaluType::IDR_N_LP;
        } else if (data.frameType == FrameType::P_FRAME || data.frameType == FrameType::LTR_RECOVERY) {
            frameNalu.naluHeader.naluType = data.temporalId == 0 ? NaluType::TRAIL_R : NaluType::TSA_R;
        } else {
            MLVC_LOG_ERROR("Unsupported frame type %d", static_cast<int>(data.frameType));
            return make_error_code(Error::bit_stream_unexpected_error);
        }

        auto& frameHeader = frameNalu.frameHeader;
        frameHeader.ppsId = m_pps.ppsId;
        frameHeader.qpDelta = data.qp - (m_pps.initQpMinus26 + 26);
        frameHeader.ltrSlots = data.ltrSlots;
        frameHeader.frameIdx = data.curFrameIdx;
        frameHeader.refFrameIdx = data.refFrameIdx;
        frameHeader.featureResetFlag = data.featureResetFlag;

        frameNalu.payload = data.payload;
        m_naluBuilder.AppendFrame(m_sps, frameNalu);
    }

    return m_naluBuilder.GetOutput();
}

expected<void> BitstreamDecoder::Initialize()
{
    return {};
}

expected<FrameData> BitstreamDecoder::Decode(std::span<const std::byte> bytes)
{
    m_naluParser.Open(bytes);
    while (m_naluParser.HasNext()) {
        auto frame = DecodeNextNalu();
        if (!frame) return frame.error();

        // Video data should be the last NALU in the access unit
        if (frame.value().has_value() && m_naluParser.HasNext()) {
            MLVC_LOG_ERROR("Extra NALUs after frame NALUs are not supported");
            return make_error_code(Error::bit_stream_unexpected_error);
        }

        if (frame.value().has_value()) {
            return frame.value().value();
        }
    }

    return make_error_code(Error::bit_stream_partial_access_unit_error);
}

expected<std::optional<FrameData>> BitstreamDecoder::DecodeNextNalu()
{
    auto naluHeader = m_naluParser.PeekNaluHeader();
    if (!naluHeader) {
        MLVC_LOG_ERROR("Failed to peek NALU header: %s", naluHeader.error().message().c_str());
        return naluHeader.error();
    }
    const NaluType naluType = naluHeader.value().naluType;

    if (naluType == NaluType::SPS) {
        auto sps = m_naluParser.ReadSps();
        if (!sps) {
            MLVC_LOG_ERROR("Failed to parse SPS NALU: %s", sps.error().message().c_str());
            return sps.error();
        }
        m_sps = sps.value();
        return {};
    } else if (naluType == NaluType::PPS) {
        auto pps = m_naluParser.ReadPps();
        if (!pps) {
            MLVC_LOG_ERROR("Failed to parse PPS NALU: %s", pps.error().message().c_str());
            return pps.error();
        }
        m_pps = pps.value();
        if (m_pps.spsId != m_sps.spsId) {
            MLVC_LOG_WARN("PPS SPS ID does not match current SPS: %d vs %d", m_pps.spsId, m_sps.spsId);
            return make_error_code(Error::bit_stream_missing_sps_error);
        }
        return {};
    } else if (naluType == NaluType::IDR_N_LP || naluType == NaluType::TRAIL_R || naluType == NaluType::TSA_R
               || naluType == NaluType::TRAIL_N || naluType == NaluType::TSA_N) {
        auto frame = m_naluParser.ReadFrame(m_sps);
        if (!frame) {
            MLVC_LOG_ERROR("Failed to parse frame NALU: %s", frame.error().message().c_str());
            return frame.error();
        }

        const auto& frameHeader = frame.value().frameHeader;
        if (frameHeader.ppsId != m_pps.ppsId) {
            MLVC_LOG_WARN("Frame header PPS ID does not match current PPS: %d vs %d", frameHeader.ppsId, m_pps.ppsId);
            return make_error_code(Error::bit_stream_missing_pps_error);
        }

        if (m_pps.spsId != m_sps.spsId) {
            MLVC_LOG_WARN("Frame header SPS ID does not match current SPS: %d vs %d", m_pps.spsId, m_sps.spsId);
            return make_error_code(Error::bit_stream_missing_sps_error);
        }

        // Frame type
        if (naluHeader.value().temporalIdPlus1 == 0) {
            MLVC_LOG_ERROR("Temporal ID plus 1 must be greater than 0");
            return make_error_code(Error::bit_stream_unexpected_error);
        }
        const int temporalId = naluHeader.value().temporalIdPlus1 - 1;
        const int maxTemporalLayers = m_sps.maxTemporalLayersMinus1 + 1;
        if (temporalId >= maxTemporalLayers) {
            MLVC_LOG_ERROR("Temporal ID %d exceeds max temporal layers %d", temporalId, maxTemporalLayers);
            return make_error_code(Error::bit_stream_unexpected_error);
        }
        FrameType frameType;
        if (naluType == NaluType::IDR_N_LP) {
            frameType = FrameType::I_FRAME;
        } else if (temporalId == 0 && (frameHeader.frameIdx - frameHeader.refFrameIdx) > 2) {
            frameType = FrameType::LTR_RECOVERY;
        } else {
            frameType = FrameType::P_FRAME;
        }

        const int64_t qp = static_cast<int64_t>(frameHeader.qpDelta) + m_pps.initQpMinus26 + 26;
        if (qp < MIN_QP || qp > MAX_QP) {
            MLVC_LOG_ERROR("QP %lld out of range [%d, %d]", static_cast<long long>(qp), MIN_QP, MAX_QP);
            return make_error_code(Error::bit_stream_unexpected_error);
        }

        FrameData res{};
        res.mlvcVersion.major = m_sps.mlvcVersionMajor;
        res.mlvcVersion.minor = m_sps.mlvcVersionMinor;
        res.modelWidth = 2 * m_sps.modelWidthDiv2;
        res.modelHeight = 2 * m_sps.modelHeightDiv2;
        res.cropOffsets.left = 2 * m_sps.cropLeftDiv2;
        res.cropOffsets.right = 2 * m_sps.cropRightDiv2;
        res.cropOffsets.top = 2 * m_sps.cropTopDiv2;
        res.cropOffsets.bottom = 2 * m_sps.cropBottomDiv2;
        res.transposeFlag = m_sps.transposeFlag;
        res.frameIdxBits = m_sps.frameIdxBitsMinus8 + MIN_FRAME_IDX_BITS;
        res.temporalId = temporalId;
        res.maxTemporalLayers = maxTemporalLayers;
        res.frameType = frameType;
        res.qp = static_cast<int>(qp);
        res.featureResetFlag = frameHeader.featureResetFlag;
        res.curFrameIdx = frameHeader.frameIdx;
        res.refFrameIdx = frameHeader.refFrameIdx;
        res.ltrSlots = frameHeader.ltrSlots;
        res.payload = frame.value().payload;
        return res;
    } else {
        const auto nalu = m_naluParser.ReadCustomNalu();
        if (!nalu) {
            MLVC_LOG_ERROR("Failed to read unknown NALU: %s", nalu.error().message().c_str());
            return nalu.error();
        }
        return {};
    }

    // Should never reach here
    return make_error_code(Error::bit_stream_unexpected_error);
}

}  // namespace libmlvc
