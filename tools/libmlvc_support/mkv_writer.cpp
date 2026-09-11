// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/mkv_writer.hpp"

#include <cmath>
#include <ostream>

namespace libmlvc {

// EBML Element IDs for Matroska
namespace ebml {
constexpr uint32_t EBML = 0x1A45DFA3;
constexpr uint32_t EBMLVersion = 0x4286;
constexpr uint32_t EBMLReadVersion = 0x42F7;
constexpr uint32_t EBMLMaxIDLength = 0x42F2;
constexpr uint32_t EBMLMaxSizeLength = 0x42F3;
constexpr uint32_t DocType = 0x4282;
constexpr uint32_t DocTypeVersion = 0x4287;
constexpr uint32_t DocTypeReadVersion = 0x4285;

constexpr uint32_t Segment = 0x18538067;
constexpr uint32_t SegmentInfo = 0x1549A966;
constexpr uint32_t TimecodeScale = 0x2AD7B1;
constexpr uint32_t MuxingApp = 0x4D80;
constexpr uint32_t WritingApp = 0x5741;

constexpr uint32_t Tracks = 0x1654AE6B;
constexpr uint32_t TrackEntry = 0xAE;
constexpr uint32_t TrackNumber = 0xD7;
constexpr uint32_t TrackUID = 0x73C5;
constexpr uint32_t TrackType = 0x83;
constexpr uint32_t FlagEnabled = 0xB9;
constexpr uint32_t FlagDefault = 0x88;
constexpr uint32_t FlagLacing = 0x9C;
constexpr uint32_t CodecID = 0x86;
constexpr uint32_t DefaultDuration = 0x23E383;

constexpr uint32_t Video = 0xE0;
constexpr uint32_t PixelWidth = 0xB0;
constexpr uint32_t PixelHeight = 0xBA;
constexpr uint32_t ColourSpace = 0x2EB524;

constexpr uint32_t Cluster = 0x1F43B675;
constexpr uint32_t Timecode = 0xE7;
constexpr uint32_t SimpleBlock = 0xA3;
constexpr uint32_t timecodeScale = 1'000'000;
}  // namespace ebml

MkvWriter::MkvWriter(std::ostream& output, int width, int height, double frameRate)
    : m_stream{ output }
    , m_width{ width }
    , m_height{ height }
    , m_frameRate{ frameRate }
    , m_frameCount{ 0 }
    , m_clusterTimecode{ 0 }
{
    Initialize();
}

void MkvWriter::Initialize() noexcept
{
    if (!m_stream || m_stream.exceptions() != std::ios::goodbit) {
        m_closed = true;
        m_hasError = true;
        return;
    }
    WriteEbmlHeader();

    // Write Segment header with unknown size
    WriteEbmlId(ebml::Segment);
    // Write unknown size marker (0x01FFFFFFFFFFFFFF = 8 bytes)
    WriteU8(0x01);
    for (int i = 0; i < 7; i++)
        WriteU8(0xFF);

    WriteSegmentInfo();
    WriteTracks();
    m_hasError = !m_stream;
}

MkvWriter::~MkvWriter() noexcept
{
    Close();
}

void MkvWriter::WriteU8(uint8_t val)
{
    m_stream.write(reinterpret_cast<const char*>(&val), 1);
}

void MkvWriter::WriteU16BE(uint16_t val)
{
    uint8_t buf[2] = { static_cast<uint8_t>(val >> 8), static_cast<uint8_t>(val) };
    m_stream.write(reinterpret_cast<const char*>(buf), 2);
}

void MkvWriter::WriteU32BE(uint32_t val)
{
    uint8_t buf[4] = { static_cast<uint8_t>(val >> 24), static_cast<uint8_t>(val >> 16), static_cast<uint8_t>(val >> 8),
                       static_cast<uint8_t>(val) };
    m_stream.write(reinterpret_cast<const char*>(buf), 4);
}

void MkvWriter::WriteU64BE(uint64_t val)
{
    uint8_t buf[8] = {
        static_cast<uint8_t>(val >> 56), static_cast<uint8_t>(val >> 48), static_cast<uint8_t>(val >> 40),
        static_cast<uint8_t>(val >> 32), static_cast<uint8_t>(val >> 24), static_cast<uint8_t>(val >> 16),
        static_cast<uint8_t>(val >> 8),  static_cast<uint8_t>(val),
    };
    m_stream.write(reinterpret_cast<const char*>(buf), 8);
}

void MkvWriter::WriteString(const std::string& str)
{
    m_stream.write(str.data(), str.size());
}

size_t MkvWriter::GetVintSize(uint64_t value) const
{
    if (value < 0x7F) return 1;
    if (value < 0x3FFF) return 2;
    if (value < 0x1FFFFF) return 3;
    if (value < 0x0FFFFFFF) return 4;
    if (value < 0x07FFFFFFFFULL) return 5;
    if (value < 0x03FFFFFFFFFFULL) return 6;
    if (value < 0x01FFFFFFFFFFFFULL) return 7;
    return 8;
}

void MkvWriter::WriteEbmlId(uint32_t id)
{
    if (id > 0xFFFFFF) {
        WriteU32BE(id);
    } else if (id > 0xFFFF) {
        uint8_t buf[3] = { static_cast<uint8_t>(id >> 16), static_cast<uint8_t>(id >> 8), static_cast<uint8_t>(id) };
        m_stream.write(reinterpret_cast<const char*>(buf), 3);
    } else if (id > 0xFF) {
        WriteU16BE(static_cast<uint16_t>(id));
    } else {
        WriteU8(static_cast<uint8_t>(id));
    }
}

void MkvWriter::WriteVint(uint64_t value)
{
    size_t len = GetVintSize(value);
    uint8_t buf[8] = {};

    for (size_t i = 0; i < len; i++) {
        buf[len - 1 - i] = static_cast<uint8_t>(value >> (i * 8));
    }
    buf[0] |= (0x80 >> (len - 1));  // Set VINT marker

    m_stream.write(reinterpret_cast<const char*>(buf), len);
}

void MkvWriter::WriteVintSize(uint64_t size)
{
    WriteVint(size);
}

size_t MkvWriter::GetElementSize(uint32_t id, uint64_t contentSize) const
{
    size_t idSize = (id > 0xFFFFFF) ? 4 : (id > 0xFFFF) ? 3 : (id > 0xFF) ? 2 : 1;
    return idSize + GetVintSize(contentSize) + contentSize;
}

void MkvWriter::WriteEbmlHeader()
{
    // Calculate EBML header content size
    const std::string docType = "matroska";
    size_t headerContentSize = 0;
    headerContentSize += GetElementSize(ebml::EBMLVersion, 1);
    headerContentSize += GetElementSize(ebml::EBMLReadVersion, 1);
    headerContentSize += GetElementSize(ebml::EBMLMaxIDLength, 1);
    headerContentSize += GetElementSize(ebml::EBMLMaxSizeLength, 1);
    headerContentSize += GetElementSize(ebml::DocType, docType.size());
    headerContentSize += GetElementSize(ebml::DocTypeVersion, 1);
    headerContentSize += GetElementSize(ebml::DocTypeReadVersion, 1);

    WriteEbmlId(ebml::EBML);
    WriteVintSize(headerContentSize);

    WriteEbmlId(ebml::EBMLVersion);
    WriteVintSize(1);
    WriteU8(1);
    WriteEbmlId(ebml::EBMLReadVersion);
    WriteVintSize(1);
    WriteU8(1);
    WriteEbmlId(ebml::EBMLMaxIDLength);
    WriteVintSize(1);
    WriteU8(4);
    WriteEbmlId(ebml::EBMLMaxSizeLength);
    WriteVintSize(1);
    WriteU8(8);
    WriteEbmlId(ebml::DocType);
    WriteVintSize(docType.size());
    WriteString(docType);
    WriteEbmlId(ebml::DocTypeVersion);
    WriteVintSize(1);
    WriteU8(4);
    WriteEbmlId(ebml::DocTypeReadVersion);
    WriteVintSize(1);
    WriteU8(2);
}

void MkvWriter::WriteSegmentInfo()
{
    const std::string muxingApp = "libmlvc";
    const std::string writingApp = "libmlvc";

    size_t contentSize = 0;
    contentSize += GetElementSize(ebml::TimecodeScale, 4);
    contentSize += GetElementSize(ebml::MuxingApp, muxingApp.size());
    contentSize += GetElementSize(ebml::WritingApp, writingApp.size());

    WriteEbmlId(ebml::SegmentInfo);
    WriteVintSize(contentSize);

    WriteEbmlId(ebml::TimecodeScale);
    WriteVintSize(4);
    WriteU32BE(ebml::timecodeScale);
    WriteEbmlId(ebml::MuxingApp);
    WriteVintSize(muxingApp.size());
    WriteString(muxingApp);
    WriteEbmlId(ebml::WritingApp);
    WriteVintSize(writingApp.size());
    WriteString(writingApp);
}

void MkvWriter::WriteTracks()
{
    // V_UNCOMPRESSED for raw video
    const std::string codecId = "V_UNCOMPRESSED";
    const std::string colourSpace = "NV12";
    const auto defaultDuration = static_cast<uint64_t>(std::llround(1'000'000'000.0 / m_frameRate));

    // Calculate Video element size
    size_t videoSize = 0;
    videoSize += GetElementSize(ebml::PixelWidth, 2);
    videoSize += GetElementSize(ebml::PixelHeight, 2);
    videoSize += GetElementSize(ebml::ColourSpace, colourSpace.size());

    // Calculate TrackEntry size
    size_t trackEntrySize = 0;
    trackEntrySize += GetElementSize(ebml::TrackNumber, 1);
    trackEntrySize += GetElementSize(ebml::TrackUID, 4);
    trackEntrySize += GetElementSize(ebml::TrackType, 1);
    trackEntrySize += GetElementSize(ebml::FlagEnabled, 1);
    trackEntrySize += GetElementSize(ebml::FlagDefault, 1);
    trackEntrySize += GetElementSize(ebml::FlagLacing, 1);
    trackEntrySize += GetElementSize(ebml::CodecID, codecId.size());
    trackEntrySize += GetElementSize(ebml::DefaultDuration, 4);
    trackEntrySize += GetElementSize(ebml::Video, videoSize);

    // Calculate Tracks size
    size_t tracksSize = GetElementSize(ebml::TrackEntry, trackEntrySize);

    WriteEbmlId(ebml::Tracks);
    WriteVintSize(tracksSize);

    WriteEbmlId(ebml::TrackEntry);
    WriteVintSize(trackEntrySize);

    WriteEbmlId(ebml::TrackNumber);
    WriteVintSize(1);
    WriteU8(1);
    WriteEbmlId(ebml::TrackUID);
    WriteVintSize(4);
    WriteU32BE(1);
    WriteEbmlId(ebml::TrackType);
    WriteVintSize(1);
    WriteU8(1);  // Video
    WriteEbmlId(ebml::FlagEnabled);
    WriteVintSize(1);
    WriteU8(1);
    WriteEbmlId(ebml::FlagDefault);
    WriteVintSize(1);
    WriteU8(1);
    WriteEbmlId(ebml::FlagLacing);
    WriteVintSize(1);
    WriteU8(0);
    WriteEbmlId(ebml::CodecID);
    WriteVintSize(codecId.size());
    WriteString(codecId);
    WriteEbmlId(ebml::DefaultDuration);
    WriteVintSize(4);
    WriteU32BE(static_cast<uint32_t>(defaultDuration));

    WriteEbmlId(ebml::Video);
    WriteVintSize(videoSize);
    WriteEbmlId(ebml::PixelWidth);
    WriteVintSize(2);
    WriteU16BE(static_cast<uint16_t>(m_width));
    WriteEbmlId(ebml::PixelHeight);
    WriteVintSize(2);
    WriteU16BE(static_cast<uint16_t>(m_height));
    WriteEbmlId(ebml::ColourSpace);
    WriteVintSize(colourSpace.size());
    WriteString(colourSpace);
}

void MkvWriter::StartCluster(uint64_t timecode)
{
    m_clusterTimecode = timecode;

    // Write Cluster with unknown size
    WriteEbmlId(ebml::Cluster);
    WriteU8(0x01);
    for (int i = 0; i < 7; i++)
        WriteU8(0xFF);

    // Write cluster timecode
    const size_t tcSize = (timecode < 0x100) ? 1 : (timecode < 0x10000) ? 2 : (timecode <= UINT32_MAX) ? 4 : 8;
    WriteEbmlId(ebml::Timecode);
    WriteVintSize(tcSize);
    if (tcSize == 1)
        WriteU8(static_cast<uint8_t>(timecode));
    else if (tcSize == 2)
        WriteU16BE(static_cast<uint16_t>(timecode));
    else if (tcSize == 4)
        WriteU32BE(static_cast<uint32_t>(timecode));
    else
        WriteU64BE(timecode);
}

void MkvWriter::WriteFrame(const Nv12FrameView& frame) noexcept
{
    if (m_closed || m_hasError || m_stream.exceptions() != std::ios::goodbit) {
        m_hasError = true;
        return;
    }
    const auto frameTimecode = static_cast<uint64_t>(std::llround(m_frameCount * 1000.0 / m_frameRate));

    // Start new cluster periodically or on first frame
    if (m_frameCount % framesPerCluster == 0) {
        StartCluster(frameTimecode);
    }

    const size_t totalSize = static_cast<size_t>(m_width) * m_height * 3 / 2;

    // SimpleBlock: track number (1 byte VINT) + timecode (2 bytes) + flags (1 byte) + data
    const size_t blockHeaderSize = 1 + 2 + 1;
    const size_t blockSize = blockHeaderSize + totalSize;

    WriteEbmlId(ebml::SimpleBlock);
    WriteVintSize(blockSize);

    // Track number as VINT
    WriteU8(0x81);  // Track 1

    // Relative timecode (relative to cluster timecode)
    int16_t relativeTimecode = static_cast<int16_t>(frameTimecode - m_clusterTimecode);
    WriteU16BE(static_cast<uint16_t>(relativeTimecode));

    // Flags: keyframe (0x80), no lacing
    WriteU8(0x80);

    // Write frame data
    const auto yPlane = frame.YPlane();
    const auto uvPlane = frame.UvPlane();
    for (int row = 0; row < m_height; row++) {
        m_stream.write(reinterpret_cast<const char*>(yPlane.data() + static_cast<size_t>(row) * frame.Stride()), m_width);
    }
    for (int row = 0; row < m_height / 2; row++) {
        m_stream.write(reinterpret_cast<const char*>(uvPlane.data() + static_cast<size_t>(row) * frame.Stride()), m_width);
    }

    m_hasError = !m_stream;
    if (!m_hasError) {
        m_frameCount++;
    }
}

void MkvWriter::Close() noexcept
{
    if (m_closed) return;

    if (m_stream.exceptions() != std::ios::goodbit) {
        m_hasError = true;
        m_closed = true;
        return;
    }

    m_stream.flush();
    m_hasError = m_hasError || !m_stream;
    m_closed = true;
}

bool MkvWriter::HasError() const noexcept
{
    return m_hasError || !m_stream;
}

}  // namespace libmlvc
