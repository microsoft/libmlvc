// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/video_io.hpp"

#include "libmlvc_support/mkv_writer.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

namespace libmlvc {

namespace detail {
class VideoSource {
public:
    virtual ~VideoSource() = default;
    virtual expected<std::optional<Nv12Frame>> Read() = 0;
};

class VideoSink {
public:
    virtual ~VideoSink() = default;
    virtual expected<void> Write(const Nv12FrameView& frame) = 0;
    virtual expected<void> Close() = 0;
};
}  // namespace detail

namespace {

constexpr double minMkvFrameRate = 1.0;
constexpr double maxMkvFrameRate = 1000.0;

bool HasExtension(const std::filesystem::path& filename, const std::string_view expectedExtension)
{
    const auto extension = filename.extension().native();
    if (extension.size() != expectedExtension.size()) {
        return false;
    }
    return std::equal(extension.begin(), extension.end(), expectedExtension.begin(), [](auto actual, const char expected) {
        using PathChar = std::filesystem::path::value_type;
        if (actual >= static_cast<PathChar>('A') && actual <= static_cast<PathChar>('Z')) {
            actual = static_cast<PathChar>(actual + static_cast<PathChar>('a' - 'A'));
        }
        return actual == static_cast<PathChar>(expected);
    });
}

std::optional<VideoReader::PixelFormat> DetectRawPixelFormat(const std::filesystem::path& filename,
                                                             const VideoReader::Options& options)
{
    if (options.rawPixelFormat) {
        return options.rawPixelFormat;
    }
    const auto rawFilename = HasExtension(filename, ".gz") ? filename.stem() : filename;
    if (HasExtension(rawFilename, ".nv12")) {
        return VideoReader::PixelFormat::NV12;
    }
    if (HasExtension(rawFilename, ".yuv")) {
        return VideoReader::PixelFormat::I420;
    }
    return std::nullopt;
}

// ============================================================================
// Byte input
// ============================================================================

struct GzipFileCloser {
    void operator()(gzFile_s* file) const noexcept { gzclose(file); }
};

using GzipFile = std::unique_ptr<gzFile_s, GzipFileCloser>;

// Decompresses a gzip file on the fly, exposing the plain bytes through a streambuf.
class GzipStreambuf : public std::streambuf {
public:
    explicit GzipStreambuf(GzipFile gzipFile) : m_gzipFile{ std::move(gzipFile) } {}

    bool HasError() const noexcept { return m_hasError; }

protected:
    int_type underflow() override
    {
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }
        const int bytesRead = gzread(m_gzipFile.get(), m_buffer.data(), static_cast<unsigned int>(m_buffer.size()));
        if (bytesRead <= 0) {
            int zlibError = Z_OK;
            gzerror(m_gzipFile.get(), &zlibError);
            m_hasError = bytesRead < 0 || zlibError != Z_OK;
            return traits_type::eof();
        }
        setg(m_buffer.data(), m_buffer.data(), m_buffer.data() + bytesRead);
        return traits_type::to_int_type(*gptr());
    }

private:
    GzipFile m_gzipFile;
    std::array<char, 64 * 1024> m_buffer;
    bool m_hasError{ false };
};

class ByteReader {
public:
    static expected<ByteReader> OpenFile(const std::filesystem::path& filename)
    {
        ByteReader reader;
        if (HasExtension(filename, ".gz")) {
            GzipFile gzipFile;
#if defined(_WIN32)
            gzipFile.reset(gzopen_w(filename.c_str(), "rb"));
#else
            gzipFile.reset(gzopen(filename.c_str(), "rb"));
#endif
            if (!gzipFile) {
                std::cerr << "Error: Failed to open gzip video file: " << filename << '\n';
                return make_error_code(Error::io_error);
            }
            if (gzdirect(gzipFile.get()) != 0) {
                std::cerr << "Error: File is not gzip-compressed: " << filename << '\n';
                return make_error_code(Error::io_error);
            }
            reader.m_gzipBuf = std::make_unique<GzipStreambuf>(std::move(gzipFile));
            reader.m_streamBuffer = reader.m_gzipBuf.get();
            return reader;
        }

        auto file = std::make_unique<std::ifstream>(filename, std::ios::binary);
        if (!*file) {
            std::cerr << "Error: Failed to open raw video file: " << filename << '\n';
            return make_error_code(Error::io_error);
        }
        reader.m_streamBuffer = file->rdbuf();
        reader.m_ownedFile = std::move(file);
        return reader;
    }

    static ByteReader FromStream(std::istream& input)
    {
        ByteReader reader;
        reader.m_streamBuffer = input.rdbuf();
        return reader;
    }

    expected<size_t> Read(std::span<std::byte> buffer)
    {
        if (!m_streamBuffer) {
            std::cerr << "Error: Video input stream has no stream buffer\n";
            return make_error_code(Error::io_error);
        }
        size_t totalBytesRead = 0;
        while (totalBytesRead < buffer.size()) {
            const auto bytesRead =
                m_streamBuffer->sgetn(reinterpret_cast<char*>(buffer.data() + static_cast<std::ptrdiff_t>(totalBytesRead)),
                                      static_cast<std::streamsize>(buffer.size() - totalBytesRead));
            if (bytesRead <= 0) {
                break;
            }
            totalBytesRead += static_cast<size_t>(bytesRead);
        }
        if (m_gzipBuf && m_gzipBuf->HasError()) {
            std::cerr << "Error: Failed to read gzip video stream\n";
            return make_error_code(Error::io_error);
        }
        return totalBytesRead;
    }

private:
    ByteReader() = default;

    std::unique_ptr<GzipStreambuf> m_gzipBuf;
    std::unique_ptr<std::ifstream> m_ownedFile;
    std::streambuf* m_streamBuffer{};
};

// ============================================================================
// Raw video source
// ============================================================================

std::vector<std::byte> ConvertI420ToNv12(std::span<const std::byte> i420, const int width, const int height)
{
    assert(i420.size() == static_cast<size_t>(width) * height * 3 / 2);
    const size_t ySize = static_cast<size_t>(width) * height;
    const size_t uvSize = ySize / 4;
    std::vector<std::byte> nv12(ySize + uvSize * 2);
    std::copy_n(i420.begin(), ySize, nv12.begin());
    for (size_t i = 0; i < uvSize; i++) {
        nv12[ySize + i * 2] = i420[ySize + i];
        nv12[ySize + i * 2 + 1] = i420[ySize + uvSize + i];
    }
    return nv12;
}

expected<void> ValidateNv12Frame(const Nv12FrameView& frame, const std::optional<int> maxDimension = {})
{
    if (frame.Width() <= 0 || frame.Height() <= 0 || frame.Width() % 2 != 0 || frame.Height() % 2 != 0
        || frame.Stride() < frame.Width()
        || (maxDimension && (frame.Width() > *maxDimension || frame.Height() > *maxDimension))) {
        std::cerr << "Error: Invalid NV12 frame geometry " << frame.Width() << "x" << frame.Height() << ", stride "
                  << frame.Stride() << '\n';
        return make_error_code(Error::invalid_argument);
    }
    const size_t stride = static_cast<size_t>(frame.Stride());
    const size_t height = static_cast<size_t>(frame.Height());
    if (frame.YPlane().size() < stride * height || frame.UvPlane().size() < stride * height / 2) {
        std::cerr << "Error: NV12 frame planes are smaller than the declared geometry\n";
        return make_error_code(Error::invalid_argument);
    }
    return {};
}

class RawVideoSource final : public detail::VideoSource {
public:
    static expected<std::unique_ptr<detail::VideoSource>> Create(ByteReader input, int frameWidth, int frameHeight,
                                                                 VideoReader::PixelFormat pixelFormat)
    {
        constexpr int maxDimension = 8192;  // 8K-capable; keeps the frame size trivially representable
        switch (pixelFormat) {
        case VideoReader::PixelFormat::NV12:
        case VideoReader::PixelFormat::I420:
            break;
        default:
            std::cerr << "Error: Invalid raw pixel format\n";
            return make_error_code(Error::invalid_argument);
        }
        if (frameWidth <= 0 || frameHeight <= 0 || frameWidth % 2 != 0 || frameHeight % 2 != 0
            || frameWidth > maxDimension || frameHeight > maxDimension) {
            std::cerr << "Error: Invalid raw frame dimensions " << frameWidth << "x" << frameHeight
                      << " (must be positive, even, and at most " << maxDimension << ")\n";
            return make_error_code(Error::invalid_argument);
        }
        const size_t frameSize = static_cast<size_t>(frameWidth) * frameHeight * 3 / 2;
        return std::make_unique<RawVideoSource>(std::move(input), frameWidth, frameHeight, frameSize, pixelFormat);
    }

    RawVideoSource(ByteReader input, int frameWidth, int frameHeight, size_t frameSize, VideoReader::PixelFormat pixelFormat)
        : m_input{ std::move(input) }
        , m_frameWidth{ frameWidth }
        , m_frameHeight{ frameHeight }
        , m_frameSize{ frameSize }
        , m_pixelFormat{ pixelFormat }
    {
    }

    expected<std::optional<Nv12Frame>> Read() override
    {
        std::vector<std::byte> rawFrameData(m_frameSize);
        auto bytesRead = m_input.Read(rawFrameData);
        if (!bytesRead) {
            return bytesRead.error();
        }
        if (*bytesRead == 0) {
            return std::optional<Nv12Frame>{};
        }
        if (*bytesRead != m_frameSize) {
            std::cerr << "Error: Incomplete raw video frame: expected " << m_frameSize << " bytes, got " << *bytesRead
                      << '\n';
            return make_error_code(Error::io_error);
        }

        Nv12Frame frame{
            .width = m_frameWidth,
            .height = m_frameHeight,
            .data = m_pixelFormat == VideoReader::PixelFormat::I420
                        ? ConvertI420ToNv12(rawFrameData, m_frameWidth, m_frameHeight)
                        : std::move(rawFrameData),
        };
        return std::optional<Nv12Frame>{ std::move(frame) };
    }

private:
    ByteReader m_input;
    int m_frameWidth{};
    int m_frameHeight{};
    size_t m_frameSize{};
    VideoReader::PixelFormat m_pixelFormat{ VideoReader::PixelFormat::NV12 };
};

class RawVideoSink final : public detail::VideoSink {
public:
    RawVideoSink(std::ostream& stream, const VideoWriter::Format format) : m_stream{ stream }, m_format{ format } {}

    expected<void> Write(const Nv12FrameView& frame) override
    {
        if (m_closed || m_stream.exceptions() != std::ios::goodbit) {
            std::cerr << "Error: Raw video output stream is closed or configured to throw\n";
            return make_error_code(Error::invalid_argument);
        }
        if (auto ret = ValidateNv12Frame(frame); !ret) {
            return ret.error();
        }
        const int width = frame.Width();
        const int height = frame.Height();
        const int stride = frame.Stride();
        const auto yPlane = frame.YPlane();
        for (int row = 0; row < height; row++) {
            m_stream.write(reinterpret_cast<const char*>(yPlane.data() + static_cast<size_t>(row) * stride), width);
        }
        const auto uvPlane = frame.UvPlane();
        if (m_format == VideoWriter::Format::NV12) {
            for (int row = 0; row < height / 2; row++) {
                m_stream.write(reinterpret_cast<const char*>(uvPlane.data() + static_cast<size_t>(row) * stride), width);
            }
        } else {
            const size_t planeSize = static_cast<size_t>(width) * height / 4;
            m_i420Chroma.resize(planeSize * 2);
            for (int row = 0; row < height / 2; row++) {
                for (int column = 0; column < width / 2; column++) {
                    const size_t sourceIndex = static_cast<size_t>(row) * stride + column * 2;
                    const size_t destinationIndex = static_cast<size_t>(row) * width / 2 + column;
                    m_i420Chroma[destinationIndex] = uvPlane[sourceIndex];
                    m_i420Chroma[planeSize + destinationIndex] = uvPlane[sourceIndex + 1];
                }
            }
            m_stream.write(reinterpret_cast<const char*>(m_i420Chroma.data()), m_i420Chroma.size());
        }
        if (!m_stream) {
            std::cerr << "Error: Failed to write raw video frame\n";
            return make_error_code(Error::io_error);
        }
        return {};
    }

    expected<void> Close() override
    {
        if (m_closed) {
            return m_stream ? expected<void>{} : make_error_code(Error::io_error);
        }
        if (m_stream.exceptions() != std::ios::goodbit) {
            std::cerr << "Error: Raw video output stream is configured to throw\n";
            return make_error_code(Error::invalid_argument);
        }
        m_stream.flush();
        m_closed = true;
        if (!m_stream) {
            std::cerr << "Error: Failed to flush raw video stream\n";
            return make_error_code(Error::io_error);
        }
        return {};
    }

private:
    std::ostream& m_stream;
    VideoWriter::Format m_format{};
    std::vector<std::byte> m_i420Chroma;
    bool m_closed{};
};

class MkvVideoSink final : public detail::VideoSink {
public:
    MkvVideoSink(std::ostream& output, const double frameRate) : m_stream{ output }, m_frameRate{ frameRate } {}

    expected<void> Write(const Nv12FrameView& frame) override
    {
        if (m_closed || m_stream.exceptions() != std::ios::goodbit) {
            std::cerr << "Error: MKV output stream is closed or configured to throw\n";
            return make_error_code(Error::invalid_argument);
        }
        if (auto ret = ValidateNv12Frame(frame, std::numeric_limits<uint16_t>::max()); !ret) {
            return ret.error();
        }
        if (!m_writer) {
            m_width = frame.Width();
            m_height = frame.Height();
            m_writer = std::make_unique<MkvWriter>(m_stream, m_width, m_height, m_frameRate);
            if (m_writer->HasError()) {
                std::cerr << "Error: Failed to initialize MKV output\n";
                return make_error_code(Error::io_error);
            }
        } else if (frame.Width() != m_width || frame.Height() != m_height) {
            std::cerr << "Error: MKV frame dimensions changed from " << m_width << "x" << m_height << " to "
                      << frame.Width() << "x" << frame.Height() << '\n';
            return make_error_code(Error::invalid_argument);
        }

        m_writer->WriteFrame(frame);
        if (m_writer->HasError()) {
            std::cerr << "Error: Failed to write MKV frame\n";
            return make_error_code(Error::io_error);
        }
        return {};
    }

    expected<void> Close() override
    {
        if (m_closed) {
            if (!m_stream || (m_writer && m_writer->HasError())) {
                return make_error_code(Error::io_error);
            }
            return m_writer ? expected<void>{} : make_error_code(Error::invalid_argument);
        }
        if (m_stream.exceptions() != std::ios::goodbit) {
            std::cerr << "Error: MKV output stream is configured to throw\n";
            return make_error_code(Error::invalid_argument);
        }
        if (!m_writer) {
            m_stream.flush();
            m_closed = true;
            if (!m_stream) {
                std::cerr << "Error: Failed to flush MKV output stream\n";
                return make_error_code(Error::io_error);
            }
            std::cerr << "Error: Cannot finalize an MKV file without frames\n";
            return make_error_code(Error::invalid_argument);
        }
        m_writer->Close();
        m_closed = true;
        if (m_writer->HasError()) {
            std::cerr << "Error: Failed to finalize MKV output\n";
            return make_error_code(Error::io_error);
        }
        return {};
    }

private:
    std::ostream& m_stream;
    double m_frameRate{};
    int m_width{};
    int m_height{};
    std::unique_ptr<MkvWriter> m_writer;
    bool m_closed{};
};

}  // namespace

// ============================================================================
// VideoReader
// ============================================================================

VideoReader::VideoReader(std::unique_ptr<detail::VideoSource> source) : m_source(std::move(source)) {}

VideoReader::~VideoReader() = default;

VideoReader::VideoReader(VideoReader&&) noexcept = default;

VideoReader& VideoReader::operator=(VideoReader&&) noexcept = default;

expected<VideoReader> VideoReader::OpenFile(const std::filesystem::path& filename, Options options)
{
    if (auto pixelFormat = DetectRawPixelFormat(filename, options)) {
        if (!options.rawFrameWidth || !options.rawFrameHeight) {
            std::cerr << "Error: Raw frame width and height are required\n";
            return make_error_code(Error::invalid_argument);
        }
        auto input = ByteReader::OpenFile(filename);
        if (!input) {
            return input.error();
        }
        auto source =
            RawVideoSource::Create(std::move(*input), *options.rawFrameWidth, *options.rawFrameHeight, *pixelFormat);
        if (!source) {
            return source.error();
        }
        return VideoReader(std::move(*source));
    }

    std::cerr << "Error: Unsupported video format: " << filename << '\n';
    return make_error_code(Error::invalid_argument);
}

expected<VideoReader> VideoReader::OpenStream(std::istream& input, Options options)
{
    if (options.rawPixelFormat) {
        if (!options.rawFrameWidth || !options.rawFrameHeight) {
            std::cerr << "Error: Raw frame width and height are required\n";
            return make_error_code(Error::invalid_argument);
        }
        auto source = RawVideoSource::Create(ByteReader::FromStream(input), *options.rawFrameWidth,
                                             *options.rawFrameHeight, *options.rawPixelFormat);
        if (!source) {
            return source.error();
        }
        return VideoReader(std::move(*source));
    }

    std::cerr << "Error: Unsupported video format\n";
    return make_error_code(Error::invalid_argument);
}

expected<std::optional<Nv12Frame>> VideoReader::Read()
{
    if (!m_source) {
        std::cerr << "Error: Cannot read from a moved-from video reader\n";
        return make_error_code(Error::invalid_argument);
    }
    return m_source->Read();
}

// ============================================================================
// VideoWriter
// ============================================================================

VideoWriter::VideoWriter(std::unique_ptr<detail::VideoSink> sink, std::unique_ptr<std::ofstream> ownedFile)
    : m_ownedFile{ std::move(ownedFile) }, m_sink{ std::move(sink) }
{
}

VideoWriter::~VideoWriter()
{
    if (m_sink) {
        (void)m_sink->Close();
    }
}

VideoWriter::VideoWriter(VideoWriter&&) noexcept = default;

VideoWriter& VideoWriter::operator=(VideoWriter&& other) noexcept
{
    if (this != &other) {
        if (m_sink) {
            (void)m_sink->Close();
        }
        m_sink.reset();
        m_ownedFile = std::move(other.m_ownedFile);
        m_sink = std::move(other.m_sink);
        m_fileCloseFailed = std::exchange(other.m_fileCloseFailed, false);
    }
    return *this;
}

expected<VideoWriter> VideoWriter::OpenFile(const std::filesystem::path& filename)
{
    return OpenFile(filename, Options{});
}

expected<VideoWriter> VideoWriter::OpenFile(const std::filesystem::path& filename, Options options)
{
    auto& format = options.format;
    if (!format) {
        if (HasExtension(filename, ".nv12")) {
            format = Format::NV12;
        } else if (HasExtension(filename, ".yuv")) {
            format = Format::I420;
        } else if (HasExtension(filename, ".mkv")) {
            format = Format::MKV;
        } else {
            std::cerr << "Error: Unsupported video format: " << filename << '\n';
            return make_error_code(Error::invalid_argument);
        }
    }
    const bool isRaw = *format == Format::NV12 || *format == Format::I420;
    const bool isMkv = *format == Format::MKV;
    if ((!isRaw && !isMkv)
        || (isMkv
            && (!std::isfinite(options.frameRate) || options.frameRate < minMkvFrameRate
                || options.frameRate > maxMkvFrameRate))) {
        std::cerr << "Error: Invalid video writer options\n";
        return make_error_code(Error::invalid_argument);
    }
    auto file = std::make_unique<std::ofstream>(filename, std::ios::binary);
    if (!*file) {
        std::cerr << "Error: Failed to open video file: " << filename << '\n';
        return make_error_code(Error::io_error);
    }
    std::unique_ptr<detail::VideoSink> sink;
    if (isRaw) {
        sink = std::make_unique<RawVideoSink>(*file, *format);
    } else {
        sink = std::make_unique<MkvVideoSink>(*file, options.frameRate);
    }
    return VideoWriter(std::move(sink), std::move(file));
}

expected<VideoWriter> VideoWriter::OpenStream(std::ostream& output, const Options options)
{
    if (!options.format) {
        std::cerr << "Error: Video output format is required for streams\n";
        return make_error_code(Error::invalid_argument);
    }
    if (output.exceptions() != std::ios::goodbit) {
        std::cerr << "Error: Video output stream must not enable exceptions\n";
        return make_error_code(Error::invalid_argument);
    }
    if (!output) {
        std::cerr << "Error: Video output stream is not writable\n";
        return make_error_code(Error::io_error);
    }
    const auto format = *options.format;
    if (format == Format::NV12 || format == Format::I420) {
        return VideoWriter(std::make_unique<RawVideoSink>(output, format));
    }
    if (format == Format::MKV) {
        if (!std::isfinite(options.frameRate) || options.frameRate < minMkvFrameRate || options.frameRate > maxMkvFrameRate) {
            std::cerr << "Error: Invalid MKV frame rate " << options.frameRate << " (must be between "
                      << minMkvFrameRate << " and " << maxMkvFrameRate << ")\n";
            return make_error_code(Error::invalid_argument);
        }
        return VideoWriter(std::make_unique<MkvVideoSink>(output, options.frameRate));
    }
    std::cerr << "Error: Unsupported video output format\n";
    return make_error_code(Error::invalid_argument);
}

expected<void> VideoWriter::Write(const Nv12FrameView& frame)
{
    if (!m_sink) {
        std::cerr << "Error: Cannot write with a moved-from video writer\n";
        return make_error_code(Error::invalid_argument);
    }
    return m_sink->Write(frame);
}

expected<void> VideoWriter::Close()
{
    if (!m_sink) {
        std::cerr << "Error: Cannot close a moved-from video writer\n";
        return make_error_code(Error::invalid_argument);
    }
    if (m_fileCloseFailed) {
        std::cerr << "Error: Video output file previously failed to close\n";
        return make_error_code(Error::io_error);
    }
    auto result = m_sink->Close();
    if (m_ownedFile && m_ownedFile->is_open()) {
        m_ownedFile->close();
        m_fileCloseFailed = !*m_ownedFile;
    }
    if (m_fileCloseFailed) {
        std::cerr << "Error: Failed to close video output file\n";
        return make_error_code(Error::io_error);
    }
    if (!result) {
        return result.error();
    }
    return {};
}

// ============================================================================
// Functions
// ============================================================================

expected<std::vector<Nv12Frame>> LoadNv12Frames(const std::filesystem::path& filename, VideoReader::Options options,
                                                const std::optional<size_t> maxNumFrames)
{
    auto reader = VideoReader::OpenFile(filename, std::move(options));
    if (!reader) {
        return reader.error();
    }

    std::vector<Nv12Frame> frames;
    while (!maxNumFrames || frames.size() < *maxNumFrames) {
        auto frame = reader->Read();
        if (!frame) {
            return frame.error();
        }
        if (!*frame) {
            break;
        }
        frames.push_back(std::move(**frame));
    }
    return frames;
}

}  // namespace libmlvc
