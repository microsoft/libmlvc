// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/types.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace libmlvc {

class MkvWriter {
public:
    MkvWriter(std::ostream& output, int width, int height, double frameRate);
    ~MkvWriter() noexcept;

    void WriteFrame(const Nv12FrameView& frame) noexcept;
    void Close() noexcept;
    bool HasError() const noexcept;

private:
    std::ostream& m_stream;
    int m_width;
    int m_height;
    double m_frameRate;
    uint64_t m_frameCount;
    uint64_t m_clusterTimecode;
    bool m_closed{};
    bool m_hasError{};
    static constexpr uint32_t framesPerCluster = 25;

    void Initialize() noexcept;
    void WriteEbmlId(uint32_t id);
    void WriteVint(uint64_t value);
    void WriteVintSize(uint64_t size);
    void WriteU8(uint8_t val);
    void WriteU16BE(uint16_t val);
    void WriteU32BE(uint32_t val);
    void WriteU64BE(uint64_t val);
    void WriteString(const std::string& str);
    void WriteEbmlHeader();
    void WriteSegmentInfo();
    void WriteTracks();
    void StartCluster(uint64_t timecode);
    size_t GetVintSize(uint64_t value) const;
    size_t GetElementSize(uint32_t id, uint64_t contentSize) const;
};

}  // namespace libmlvc
