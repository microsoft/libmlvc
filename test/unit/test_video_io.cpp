// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/compress.hpp"
#include "libmlvc_support/video_io.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

using namespace libmlvc;

namespace {

VideoReader::Options Nv12Options(const int frameWidth, const int frameHeight)
{
    return {
        .rawFrameWidth = frameWidth,
        .rawFrameHeight = frameHeight,
        .rawPixelFormat = VideoReader::PixelFormat::NV12,
    };
}

class ChunkedStreamBuffer : public std::streambuf {
public:
    ChunkedStreamBuffer(std::string data, const size_t maxChunkSize)
        : m_data(std::move(data)), m_maxChunkSize(maxChunkSize)
    {
    }

protected:
    std::streamsize xsgetn(char* output, const std::streamsize count) override
    {
        const auto bytesToRead = std::min({ static_cast<size_t>(count), m_maxChunkSize, m_data.size() - m_offset });
        std::copy_n(m_data.data() + static_cast<std::ptrdiff_t>(m_offset), bytesToRead, output);
        m_offset += bytesToRead;
        return static_cast<std::streamsize>(bytesToRead);
    }

private:
    std::string m_data;
    size_t m_maxChunkSize{};
    size_t m_offset{};
};

class FailingOutputBuffer : public std::streambuf {
protected:
    std::streamsize xsputn(const char*, std::streamsize) override { return 0; }
};

class ScopedTempFile {
public:
    ScopedTempFile(std::string_view suffix, std::span<const std::byte> content)
        : path(std::filesystem::temp_directory_path()
               / ("libmlvc_video_reader_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                  + std::string(suffix)))
    {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(content.data()), static_cast<std::streamsize>(content.size()));
    }

    ~ScopedTempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    std::filesystem::path path;
};

void ExpectOpenStreamRejected(VideoReader::Options options)
{
    std::istringstream input({}, std::ios::in | std::ios::binary);
    auto reader = VideoReader::OpenStream(input, std::move(options));
    ASSERT_FALSE(reader);
    EXPECT_EQ(reader.error(), make_error_code(Error::invalid_argument));
}

}  // namespace

// ============================================================================
// Frame reading
// ============================================================================

TEST(UnitTestVideoReader, ReadsFramesIncrementally)
{
    constexpr int frameWidth = 2;
    constexpr int frameHeight = 2;
    constexpr size_t frameSize = frameWidth * frameHeight * 3 / 2;

    std::string inputBytes(frameSize * 2, '\0');
    for (size_t i = 0; i < inputBytes.size(); ++i) {
        inputBytes[i] = static_cast<char>(i);
    }
    std::istringstream input(inputBytes, std::ios::in | std::ios::binary);

    auto reader = VideoReader::OpenStream(input, Nv12Options(frameWidth, frameHeight));
    ASSERT_TRUE(reader);

    for (size_t frameIndex = 0; frameIndex < 2; ++frameIndex) {
        auto frame = reader->Read();
        ASSERT_TRUE(frame);
        ASSERT_TRUE(frame->has_value());
        EXPECT_EQ(frame->value().width, frameWidth);
        EXPECT_EQ(frame->value().height, frameHeight);
        ASSERT_EQ(frame->value().data.size(), frameSize);
        for (size_t i = 0; i < frameSize; ++i) {
            EXPECT_EQ(frame->value().data[i], static_cast<std::byte>(frameIndex * frameSize + i));
        }
    }

    auto end = reader->Read();
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
}

TEST(UnitTestVideoReader, ReadsFrameAcrossShortStreamReads)
{
    ChunkedStreamBuffer streamBuffer(std::string{ 0, 1, 2, 3, 4, 5 }, 2);
    std::istream input(&streamBuffer);
    auto reader = VideoReader::OpenStream(input, Nv12Options(2, 2));
    ASSERT_TRUE(reader);

    auto frame = reader->Read();
    ASSERT_TRUE(frame);
    ASSERT_TRUE(frame->has_value());
    EXPECT_EQ(frame->value().data, (std::vector<std::byte>{ std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 },
                                                            std::byte{ 3 }, std::byte{ 4 }, std::byte{ 5 } }));

    auto end = reader->Read();
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
}

TEST(UnitTestVideoReader, EmptyStreamEndsCleanly)
{
    std::istringstream input({}, std::ios::in | std::ios::binary);
    auto reader = VideoReader::OpenStream(input, Nv12Options(2, 2));
    ASSERT_TRUE(reader);

    auto frame = reader->Read();
    ASSERT_TRUE(frame);
    EXPECT_FALSE(frame->has_value());
}

TEST(UnitTestVideoReader, RejectsPartialFrame)
{
    std::istringstream input(std::string(5, '\0'), std::ios::in | std::ios::binary);
    auto reader = VideoReader::OpenStream(input, Nv12Options(2, 2));
    ASSERT_TRUE(reader);

    auto frame = reader->Read();
    ASSERT_FALSE(frame);
    EXPECT_EQ(frame.error(), make_error_code(Error::io_error));
}

TEST(UnitTestVideoReader, RejectsReadAfterMove)
{
    std::istringstream input(std::string(6, '\0'), std::ios::in | std::ios::binary);
    auto reader = VideoReader::OpenStream(input, Nv12Options(2, 2));
    ASSERT_TRUE(reader);
    auto movedReader = std::move(*reader);

    auto frame = reader->Read();
    ASSERT_FALSE(frame);
    EXPECT_EQ(frame.error(), make_error_code(Error::invalid_argument));

    auto movedFrame = movedReader.Read();
    ASSERT_TRUE(movedFrame);
    EXPECT_TRUE(movedFrame->has_value());
}

// ============================================================================
// Options validation
// ============================================================================

TEST(UnitTestVideoReader, RejectsInvalidDimensions)
{
    ExpectOpenStreamRejected(Nv12Options(3, 2));
}

TEST(UnitTestVideoReader, RejectsMissingDimension)
{
    ExpectOpenStreamRejected({ .rawFrameWidth = 2, .rawPixelFormat = VideoReader::PixelFormat::NV12 });
}

TEST(UnitTestVideoReader, RejectsStreamWithoutPixelFormat)
{
    ExpectOpenStreamRejected({ .rawFrameWidth = 2, .rawFrameHeight = 2 });
}

// ============================================================================
// File format detection
// ============================================================================

TEST(UnitTestVideoReader, StreamsGzipYuv420pAndConvertsToNv12)
{
    constexpr int frameWidth = 4;
    constexpr int frameHeight = 2;
    std::vector<std::byte> yuv420p = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 4 },  std::byte{ 5 },
        std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 }, std::byte{ 9 }, std::byte{ 10 }, std::byte{ 11 },
    };
    auto compressed = Compress(yuv420p);
    ASSERT_TRUE(compressed);

    ScopedTempFile file(".yuv.gz", *compressed);
    auto reader = VideoReader::OpenFile(file.path, { .rawFrameWidth = frameWidth, .rawFrameHeight = frameHeight });
    ASSERT_TRUE(reader);
    auto frame = reader->Read();
    ASSERT_TRUE(frame);
    ASSERT_TRUE(frame->has_value());

    const std::vector<std::byte> expectedNv12 = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 },  std::byte{ 4 }, std::byte{ 5 },
        std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 }, std::byte{ 10 }, std::byte{ 9 }, std::byte{ 11 },
    };
    EXPECT_EQ(frame->value().data, expectedNv12);

    auto end = reader->Read();
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
}

TEST(UnitTestVideoReader, InfersUppercaseGzipNv12Extension)
{
    const std::vector<std::byte> rawFrame(6);
    auto compressed = Compress(rawFrame);
    ASSERT_TRUE(compressed);

    ScopedTempFile file(".NV12.GZ", *compressed);
    auto reader = VideoReader::OpenFile(file.path, { .rawFrameWidth = 2, .rawFrameHeight = 2 });
    ASSERT_TRUE(reader);
    auto frame = reader->Read();
    ASSERT_TRUE(frame);
    ASSERT_TRUE(frame->has_value());
    EXPECT_EQ(frame->value().data, rawFrame);
}

TEST(UnitTestVideoReader, ExplicitFileFormatOverridesExtension)
{
    const std::vector<std::byte> i420 = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 4 },  std::byte{ 5 },
        std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 }, std::byte{ 9 }, std::byte{ 10 }, std::byte{ 11 },
    };
    ScopedTempFile file(".nv12", i420);
    auto reader = VideoReader::OpenFile(
        file.path, { .rawFrameWidth = 4, .rawFrameHeight = 2, .rawPixelFormat = VideoReader::PixelFormat::I420 });
    ASSERT_TRUE(reader);
    auto frame = reader->Read();
    ASSERT_TRUE(frame);
    ASSERT_TRUE(frame->has_value());
    EXPECT_EQ(frame->value().data[9], std::byte{ 10 });
    EXPECT_EQ(frame->value().data[10], std::byte{ 9 });
}

TEST(UnitTestVideoReader, ExplicitFormatAllowsUnknownExtension)
{
    ScopedTempFile file(".raw", std::vector<std::byte>(6));
    auto reader = VideoReader::OpenFile(
        file.path, { .rawFrameWidth = 2, .rawFrameHeight = 2, .rawPixelFormat = VideoReader::PixelFormat::NV12 });
    ASSERT_TRUE(reader);
    auto frame = reader->Read();
    ASSERT_TRUE(frame);
    ASSERT_TRUE(frame->has_value());
}

TEST(UnitTestVideoReader, RejectsUncompressedFileWithGzipExtension)
{
    ScopedTempFile file(".nv12.gz", std::vector<std::byte>(6));
    auto reader = VideoReader::OpenFile(file.path, { .rawFrameWidth = 2, .rawFrameHeight = 2 });
    ASSERT_FALSE(reader);
    EXPECT_EQ(reader.error(), make_error_code(Error::io_error));
}

TEST(UnitTestVideoReader, RejectsCorruptGzipStream)
{
    auto compressed = Compress(std::vector<std::byte>(6));
    ASSERT_TRUE(compressed);
    ASSERT_GE(compressed->size(), 8u);
    (*compressed)[compressed->size() - 8] ^= std::byte{ 0xFF };  // Corrupt gzip CRC

    ScopedTempFile file(".nv12.gz", *compressed);
    auto reader = VideoReader::OpenFile(file.path, { .rawFrameWidth = 2, .rawFrameHeight = 2 });
    ASSERT_TRUE(reader);

    auto frame = reader->Read();
    ASSERT_TRUE(frame);
    ASSERT_TRUE(frame->has_value());

    auto nextFrame = reader->Read();
    ASSERT_FALSE(nextFrame);
    EXPECT_EQ(nextFrame.error(), make_error_code(Error::io_error));
}

TEST(UnitTestVideoReader, RejectsUnsupportedFileFormat)
{
    ScopedTempFile file(".mp4", std::vector<std::byte>(16));
    auto reader = VideoReader::OpenFile(file.path, { .rawFrameWidth = 2, .rawFrameHeight = 2 });
    ASSERT_FALSE(reader);
    EXPECT_EQ(reader.error(), make_error_code(Error::invalid_argument));
}

// ============================================================================
// Frame writing
// ============================================================================

TEST(UnitTestVideoWriter, WritesNv12Frame)
{
    const std::vector<std::byte> frameData = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 4 }, std::byte{ 5 },
    };
    ScopedTempFile file(".nv12", {});
    auto writer = VideoWriter::OpenFile(file.path);
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->Write(Nv12FrameView{ 2, 2, frameData }));
    ASSERT_TRUE(writer->Close());

    std::vector<std::byte> written(frameData.size());
    std::ifstream input(file.path, std::ios::binary);
    input.read(reinterpret_cast<char*>(written.data()), static_cast<std::streamsize>(written.size()));
    EXPECT_EQ(written, frameData);
    EXPECT_EQ(input.peek(), std::char_traits<char>::eof());
}

TEST(UnitTestVideoWriter, WritesI420Frame)
{
    const std::vector<std::byte> nv12 = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 4 },  std::byte{ 5 },
        std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 }, std::byte{ 9 }, std::byte{ 10 }, std::byte{ 11 },
    };
    ScopedTempFile file(".yuv", {});
    auto writer = VideoWriter::OpenFile(file.path);
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->Write(Nv12FrameView{ 4, 2, nv12 }));
    ASSERT_TRUE(writer->Close());

    std::vector<std::byte> written(nv12.size());
    std::ifstream input(file.path, std::ios::binary);
    input.read(reinterpret_cast<char*>(written.data()), static_cast<std::streamsize>(written.size()));
    const std::vector<std::byte> expected = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 },  std::byte{ 4 }, std::byte{ 5 },
        std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 }, std::byte{ 10 }, std::byte{ 9 }, std::byte{ 11 },
    };
    EXPECT_EQ(written, expected);
}

TEST(UnitTestVideoWriter, WritesMkvFile)
{
    ScopedTempFile file(".mkv", {});
    auto writer = VideoWriter::OpenFile(file.path);
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->Write(Nv12FrameView{ 2, 2, std::vector<std::byte>(6) }));
    ASSERT_TRUE(writer->Close());

    std::array<unsigned char, 4> signature{};
    std::ifstream input(file.path, std::ios::binary);
    input.read(reinterpret_cast<char*>(signature.data()), static_cast<std::streamsize>(signature.size()));
    EXPECT_EQ(signature, (std::array<unsigned char, 4>{ 0x1A, 0x45, 0xDF, 0xA3 }));
}

TEST(UnitTestVideoWriter, OmitsStridePadding)
{
    const std::vector<std::byte> yPlane = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 0xEE }, std::byte{ 0xEE },
        std::byte{ 4 }, std::byte{ 5 }, std::byte{ 6 }, std::byte{ 7 }, std::byte{ 0xEE }, std::byte{ 0xEE },
    };
    const std::vector<std::byte> uvPlane = {
        std::byte{ 8 }, std::byte{ 9 }, std::byte{ 10 }, std::byte{ 11 }, std::byte{ 0xEE }, std::byte{ 0xEE },
    };
    std::ostringstream output(std::ios::binary);
    auto writer = VideoWriter::OpenStream(output, { .format = VideoWriter::Format::NV12 });
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->Write(Nv12FrameView{ 4, 2, 6, yPlane, uvPlane }));
    ASSERT_TRUE(writer->Close());

    const std::vector<std::byte> expected = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 4 },  std::byte{ 5 },
        std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 }, std::byte{ 9 }, std::byte{ 10 }, std::byte{ 11 },
    };
    const std::string expectedBytes(reinterpret_cast<const char*>(expected.data()), expected.size());
    EXPECT_EQ(output.str(), expectedBytes);

    std::ostringstream i420Output(std::ios::binary);
    auto i420Writer = VideoWriter::OpenStream(i420Output, { .format = VideoWriter::Format::I420 });
    ASSERT_TRUE(i420Writer);
    ASSERT_TRUE(i420Writer->Write(Nv12FrameView{ 4, 2, 6, yPlane, uvPlane }));
    ASSERT_TRUE(i420Writer->Close());
    const std::vector<std::byte> expectedI420 = {
        std::byte{ 0 }, std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 },  std::byte{ 4 }, std::byte{ 5 },
        std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 }, std::byte{ 10 }, std::byte{ 9 }, std::byte{ 11 },
    };
    EXPECT_EQ(i420Output.str(), std::string(reinterpret_cast<const char*>(expectedI420.data()), expectedI420.size()));

    std::ostringstream mkvOutput(std::ios::binary);
    auto mkvWriter = VideoWriter::OpenStream(mkvOutput, { .format = VideoWriter::Format::MKV });
    ASSERT_TRUE(mkvWriter);
    ASSERT_TRUE(mkvWriter->Write(Nv12FrameView{ 4, 2, 6, yPlane, uvPlane }));
    ASSERT_TRUE(mkvWriter->Close());
    const auto mkvBytes = mkvOutput.str();
    ASSERT_GE(mkvBytes.size(), expectedBytes.size());
    EXPECT_TRUE(mkvBytes.ends_with(expectedBytes));
}

TEST(UnitTestVideoWriter, WritesMkvStreamWithFrameRate)
{
    std::ostringstream output(std::ios::binary);
    auto writer = VideoWriter::OpenStream(output, { .format = VideoWriter::Format::MKV, .frameRate = 29.97 });
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->Write(Nv12FrameView{ 2, 2, std::vector<std::byte>(6) }));
    ASSERT_TRUE(writer->Close());

    const auto bytes = output.str();
    ASSERT_GE(bytes.size(), 4u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[0]), 0x1A);
    EXPECT_EQ(static_cast<unsigned char>(bytes[1]), 0x45);
    EXPECT_EQ(static_cast<unsigned char>(bytes[2]), 0xDF);
    EXPECT_EQ(static_cast<unsigned char>(bytes[3]), 0xA3);

    const std::array<unsigned char, 8> defaultDuration = { 0x23, 0xE3, 0x83, 0x84, 0x01, 0xFD, 0x22, 0xAC };
    const auto duration = std::search(
        bytes.begin(), bytes.end(), defaultDuration.begin(), defaultDuration.end(),
        [](const char actual, const unsigned char expected) { return static_cast<unsigned char>(actual) == expected; });
    EXPECT_NE(duration, bytes.end());
}

TEST(UnitTestVideoWriter, RejectsInvalidFrameGeometry)
{
    std::ostringstream output(std::ios::binary);
    auto writer = VideoWriter::OpenStream(output, { .format = VideoWriter::Format::NV12 });
    ASSERT_TRUE(writer);
    auto write = writer->Write(Nv12FrameView{ 3, 2, std::vector<std::byte>(9) });
    ASSERT_FALSE(write);
    EXPECT_EQ(write.error(), make_error_code(Error::invalid_argument));
}

TEST(UnitTestVideoWriter, RejectsMkvResolutionChange)
{
    std::ostringstream output(std::ios::binary);
    auto writer = VideoWriter::OpenStream(output, { .format = VideoWriter::Format::MKV });
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->Write(Nv12FrameView{ 2, 2, std::vector<std::byte>(6) }));
    auto write = writer->Write(Nv12FrameView{ 4, 2, std::vector<std::byte>(12) });
    ASSERT_FALSE(write);
    EXPECT_EQ(write.error(), make_error_code(Error::invalid_argument));
    EXPECT_TRUE(writer->Close());
}

TEST(UnitTestVideoWriter, MoveAssignsFileWriter)
{
    ScopedTempFile firstFile(".mkv", {});
    ScopedTempFile secondFile(".mkv", {});
    auto first = VideoWriter::OpenFile(firstFile.path);
    auto second = VideoWriter::OpenFile(secondFile.path);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_TRUE(first->Write(Nv12FrameView{ 2, 2, std::vector<std::byte>(6) }));
    ASSERT_TRUE(second->Write(Nv12FrameView{ 2, 2, std::vector<std::byte>(6) }));

    *second = std::move(*first);
    ASSERT_TRUE(second->Write(Nv12FrameView{ 2, 2, std::vector<std::byte>(6) }));
    EXPECT_TRUE(second->Close());
}

TEST(UnitTestVideoWriter, ReturnsErrorForFailingOutputStream)
{
    for (const auto format : { VideoWriter::Format::NV12, VideoWriter::Format::MKV }) {
        FailingOutputBuffer buffer;
        std::ostream output(&buffer);
        auto writer = VideoWriter::OpenStream(output, { .format = format });
        ASSERT_TRUE(writer);
        auto write = writer->Write(Nv12FrameView{ 2, 2, std::vector<std::byte>(6) });
        ASSERT_FALSE(write);
        EXPECT_EQ(write.error(), make_error_code(Error::io_error));
    }
}

TEST(UnitTestVideoWriter, RejectsInvalidOutputStreamAndFrameRate)
{
    std::ostringstream failedOutput;
    failedOutput.setstate(std::ios::badbit);
    auto failedStreamWriter = VideoWriter::OpenStream(failedOutput, { .format = VideoWriter::Format::NV12 });
    ASSERT_FALSE(failedStreamWriter);
    EXPECT_EQ(failedStreamWriter.error(), make_error_code(Error::io_error));

    std::ostringstream output;
    auto invalidFpsWriter = VideoWriter::OpenStream(
        output, { .format = VideoWriter::Format::MKV, .frameRate = std::numeric_limits<double>::quiet_NaN() });
    ASSERT_FALSE(invalidFpsWriter);
    EXPECT_EQ(invalidFpsWriter.error(), make_error_code(Error::invalid_argument));
}

TEST(UnitTestVideoWriter, RejectsChangedOutputStreamExceptionMask)
{
    for (const auto format : { VideoWriter::Format::NV12, VideoWriter::Format::MKV }) {
        std::ostringstream output;
        auto writer = VideoWriter::OpenStream(output, { .format = format });
        ASSERT_TRUE(writer);
        output.exceptions(std::ios::badbit);

        auto write = writer->Write(Nv12FrameView{ 2, 2, std::vector<std::byte>(6) });
        ASSERT_FALSE(write);
        EXPECT_EQ(write.error(), make_error_code(Error::invalid_argument));
    }
}

TEST(UnitTestVideoWriter, EmptyMkvFailsAndTruncatesExistingFile)
{
    const std::vector<std::byte> staleData = { std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 } };
    ScopedTempFile file(".mkv", staleData);
    auto writer = VideoWriter::OpenFile(file.path);
    ASSERT_TRUE(writer);
    auto close = writer->Close();
    ASSERT_FALSE(close);
    EXPECT_EQ(close.error(), make_error_code(Error::invalid_argument));
    auto secondClose = writer->Close();
    ASSERT_FALSE(secondClose);
    EXPECT_EQ(secondClose.error(), make_error_code(Error::invalid_argument));
    EXPECT_EQ(std::filesystem::file_size(file.path), 0u);
}

TEST(UnitTestVideoWriter, RejectsUnsupportedFileFormat)
{
    ScopedTempFile file(".mp4", {});
    auto writer = VideoWriter::OpenFile(file.path);
    ASSERT_FALSE(writer);
    EXPECT_EQ(writer.error(), make_error_code(Error::invalid_argument));
}

// ============================================================================
// LoadNv12Frames
// ============================================================================

TEST(UnitTestVideoReader, LoadNv12FramesWithoutLimitReadsToEnd)
{
    ScopedTempFile file(".nv12", std::vector<std::byte>(12));
    auto frames = LoadNv12Frames(file.path, { .rawFrameWidth = 2, .rawFrameHeight = 2 });
    ASSERT_TRUE(frames);
    EXPECT_EQ(frames->size(), 2);
}
