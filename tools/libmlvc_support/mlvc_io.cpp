// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/mlvc_io.hpp"

#include <libmlvc/error_codes.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

namespace libmlvc {

namespace {

constexpr size_t startCodeLength = 4;

bool IsNaluStartCode(const std::vector<std::byte>& buffer, const size_t pos)
{
    return pos + startCodeLength <= buffer.size() && buffer[pos] == std::byte{ 0x00 }
           && buffer[pos + 1] == std::byte{ 0x00 } && buffer[pos + 2] == std::byte{ 0x00 }
           && buffer[pos + 3] == std::byte{ 0x01 };
}

expected<bool> FillMore(std::istream* stream, std::streambuf* streamBuffer, std::ifstream* ownedFile,
                        std::vector<std::byte>& buffer, size_t& bufferOffset)
{
    constexpr size_t readChunkSize = 64 * 1024;
    if (!stream || stream->exceptions() != std::ios::goodbit) {
        return make_error_code(Error::invalid_argument);
    }
    if (stream->bad() || (stream->fail() && !stream->eof())) {
        return make_error_code(Error::io_error);
    }
    buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(bufferOffset));
    bufferOffset = 0;

    const size_t oldSize = buffer.size();
    buffer.resize(oldSize + readChunkSize);
    auto* output = reinterpret_cast<char*>(buffer.data() + oldSize);
    std::streamsize bytesRead;
    if (ownedFile) {
        if (ownedFile->fail() && !ownedFile->eof()) {
            buffer.resize(oldSize);
            return make_error_code(Error::io_error);
        }
        if (ownedFile->eof()) {
            buffer.resize(oldSize);
            return false;
        }
        ownedFile->read(output, static_cast<std::streamsize>(readChunkSize));
        bytesRead = ownedFile->gcount();
        if (ownedFile->bad() || (ownedFile->fail() && !ownedFile->eof())) {
            buffer.resize(oldSize);
            return make_error_code(Error::io_error);
        }
    } else {
        bytesRead = streamBuffer->sgetn(output, static_cast<std::streamsize>(readChunkSize));
    }
    buffer.resize(oldSize + (bytesRead > 0 ? static_cast<size_t>(bytesRead) : 0));
    return bytesRead > 0;
}

expected<bool> EnsureAvailable(std::istream* stream, std::streambuf* streamBuffer, std::ifstream* ownedFile,
                               std::vector<std::byte>& buffer, size_t& bufferOffset, const size_t count)
{
    while (buffer.size() - bufferOffset < count) {
        auto result = FillMore(stream, streamBuffer, ownedFile, buffer, bufferOffset);
        if (!result) {
            return result.error();
        }
        if (!*result) {
            return false;
        }
    }
    return true;
}

expected<size_t> FindNalEnd(std::istream* stream, std::streambuf* streamBuffer, std::ifstream* ownedFile,
                            std::vector<std::byte>& buffer, size_t& bufferOffset)
{
    size_t pos = bufferOffset + startCodeLength;
    while (true) {
        for (; pos + startCodeLength <= buffer.size(); pos++) {
            if (IsNaluStartCode(buffer, pos)) {
                return pos;
            }
        }
        pos -= bufferOffset;
        auto result = FillMore(stream, streamBuffer, ownedFile, buffer, bufferOffset);
        if (!result) {
            return result.error();
        }
        if (!*result) {
            return buffer.size();
        }
    }
}

}  // namespace

// ============================================================================
// MlvcReader
// ============================================================================

MlvcReader::MlvcReader(std::istream* stream, std::unique_ptr<std::ifstream> ownedFile)
    : m_ownedFile{ std::move(ownedFile) }, m_stream{ stream }, m_streamBuffer{ stream ? stream->rdbuf() : nullptr }
{
}

MlvcReader::~MlvcReader() = default;

// Move nulls the borrowed stream buffer so a moved-from reader reads nothing.
MlvcReader::MlvcReader(MlvcReader&& other) noexcept
    : m_ownedFile{ std::move(other.m_ownedFile) }
    , m_stream{ std::exchange(other.m_stream, nullptr) }
    , m_streamBuffer{ std::exchange(other.m_streamBuffer, nullptr) }
    , m_buffer{ std::move(other.m_buffer) }
    , m_bufferOffset{ std::exchange(other.m_bufferOffset, 0) }
{
}

MlvcReader& MlvcReader::operator=(MlvcReader&& other) noexcept
{
    m_ownedFile = std::move(other.m_ownedFile);
    m_stream = std::exchange(other.m_stream, nullptr);
    m_streamBuffer = std::exchange(other.m_streamBuffer, nullptr);
    m_buffer = std::move(other.m_buffer);
    m_bufferOffset = std::exchange(other.m_bufferOffset, 0);
    return *this;
}

expected<MlvcReader> MlvcReader::OpenFile(const std::filesystem::path& filename)
{
    auto file = std::make_unique<std::ifstream>(filename, std::ios::binary);
    if (!*file) {
        std::cerr << "Error: Failed to open access unit file: " << filename << '\n';
        return make_error_code(Error::io_error);
    }
    std::istream* stream = file.get();
    return MlvcReader(stream, std::move(file));
}

expected<MlvcReader> MlvcReader::OpenStream(std::istream& input)
{
    if (!input.rdbuf() || !input) {
        std::cerr << "Error: Invalid MLVC input stream\n";
        return make_error_code(Error::io_error);
    }
    if (input.exceptions() != std::ios::goodbit) {
        std::cerr << "Error: MLVC input stream must not enable exceptions\n";
        return make_error_code(Error::invalid_argument);
    }
    return MlvcReader(&input, nullptr);
}

expected<std::optional<MlvcAccessUnit>> MlvcReader::Read()
{
    if (!m_streamBuffer) {
        std::cerr << "Error: Cannot read from a moved-from access unit reader\n";
        return make_error_code(Error::invalid_argument);
    }
    std::vector<std::byte> accessUnit;
    while (true) {
        auto haveHeader =
            EnsureAvailable(m_stream, m_streamBuffer, m_ownedFile.get(), m_buffer, m_bufferOffset, startCodeLength + 1);
        if (!haveHeader) {
            return haveHeader.error();
        }
        if (m_bufferOffset == m_buffer.size()) {
            return std::optional<MlvcAccessUnit>{};
        }
        if (!*haveHeader || !IsNaluStartCode(m_buffer, m_bufferOffset)) {
            std::cerr << "Error: Failed to find NALU start code\n";
            return make_error_code(Error::io_error);
        }
        auto nalEnd = FindNalEnd(m_stream, m_streamBuffer, m_ownedFile.get(), m_buffer, m_bufferOffset);
        if (!nalEnd) {
            return nalEnd.error();
        }
        const int naluType = (static_cast<int>(m_buffer[m_bufferOffset + startCodeLength]) >> 1) & 0x3F;
        accessUnit.insert(accessUnit.end(), m_buffer.begin() + static_cast<std::ptrdiff_t>(m_bufferOffset),
                          m_buffer.begin() + static_cast<std::ptrdiff_t>(*nalEnd));
        m_bufferOffset = *nalEnd;
        if (naluType < 32) {  // NAL types 0..31 are VCL access-unit slices
            return std::optional<MlvcAccessUnit>{ MlvcAccessUnit{ std::move(accessUnit) } };
        }
    }
}

// ============================================================================
// MlvcWriter
// ============================================================================

MlvcWriter::MlvcWriter(std::ostream* stream, std::unique_ptr<std::ofstream> ownedFile)
    : m_ownedFile{ std::move(ownedFile) }, m_stream{ stream }
{
}

MlvcWriter::~MlvcWriter()
{
    if (m_stream) {
        (void)Close();
    }
}

MlvcWriter::MlvcWriter(MlvcWriter&& other) noexcept
    : m_ownedFile{ std::move(other.m_ownedFile) }
    , m_stream{ std::exchange(other.m_stream, nullptr) }
    , m_closed{ std::exchange(other.m_closed, true) }
    , m_failed{ std::exchange(other.m_failed, false) }
{
}

expected<MlvcWriter> MlvcWriter::OpenFile(const std::filesystem::path& filename)
{
    auto file = std::make_unique<std::ofstream>(filename, std::ios::binary);
    if (!*file) {
        std::cerr << "Error: Failed to open MLVC output file: " << filename << '\n';
        return make_error_code(Error::io_error);
    }
    std::ostream* stream = file.get();
    return MlvcWriter(stream, std::move(file));
}

expected<MlvcWriter> MlvcWriter::OpenStream(std::ostream& output)
{
    if (!output.rdbuf() || !output) {
        std::cerr << "Error: Invalid MLVC output stream\n";
        return make_error_code(Error::io_error);
    }
    if (output.exceptions() != std::ios::goodbit) {
        std::cerr << "Error: MLVC output stream must not enable exceptions\n";
        return make_error_code(Error::invalid_argument);
    }
    return MlvcWriter(&output, nullptr);
}

expected<void> MlvcWriter::Write(const std::span<const std::byte> accessUnit)
{
    if (!m_stream || m_closed) {
        std::cerr << "Error: Cannot write to a closed or moved-from MLVC writer\n";
        return make_error_code(Error::invalid_argument);
    }
    if (m_failed) {
        return make_error_code(Error::io_error);
    }
    if (m_stream->exceptions() != std::ios::goodbit) {
        std::cerr << "Error: MLVC output stream is configured to throw\n";
        return make_error_code(Error::invalid_argument);
    }
    m_stream->write(reinterpret_cast<const char*>(accessUnit.data()), static_cast<std::streamsize>(accessUnit.size()));
    if (!*m_stream) {
        std::cerr << "Error: Failed to write MLVC access unit\n";
        m_failed = true;
        return make_error_code(Error::io_error);
    }
    return {};
}

expected<void> MlvcWriter::Close()
{
    if (!m_stream) {
        std::cerr << "Error: Cannot close a moved-from MLVC writer\n";
        return make_error_code(Error::invalid_argument);
    }
    if (m_closed) {
        return m_failed ? expected<void>{ make_error_code(Error::io_error) } : expected<void>{};
    }
    if (m_stream->exceptions() != std::ios::goodbit) {
        std::cerr << "Error: MLVC output stream is configured to throw\n";
        return make_error_code(Error::invalid_argument);
    }

    if (m_ownedFile) {
        m_ownedFile->close();
        m_failed = m_failed || !*m_ownedFile;
    } else {
        m_stream->flush();
        m_failed = m_failed || !*m_stream;
    }
    m_closed = true;
    if (m_failed) {
        std::cerr << "Error: Failed to close MLVC output\n";
        return make_error_code(Error::io_error);
    }
    return {};
}

// ============================================================================
// Functions
// ============================================================================

expected<std::vector<MlvcAccessUnit>> ReadMlvcAccessUnits(const std::filesystem::path& filename,
                                                          const std::optional<size_t> maxNumAccessUnits)
{
    auto reader = MlvcReader::OpenFile(filename);
    if (!reader) {
        return reader.error();
    }

    std::vector<MlvcAccessUnit> accessUnits;
    while (!maxNumAccessUnits || accessUnits.size() < *maxNumAccessUnits) {
        auto accessUnit = reader->Read();
        if (!accessUnit) {
            return accessUnit.error();
        }
        if (!*accessUnit) {
            break;
        }
        accessUnits.push_back(std::move(**accessUnit));
    }
    return accessUnits;
}

}  // namespace libmlvc
