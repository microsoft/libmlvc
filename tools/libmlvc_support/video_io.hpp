// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <vector>

namespace libmlvc {

struct Nv12Frame {
    int width{};
    int height{};
    std::vector<std::byte> data;

    Nv12FrameView View() const { return { width, height, data }; }
};

namespace detail {
class VideoSource;
class VideoSink;
}  // namespace detail

class VideoReader {
public:
    enum class PixelFormat { NV12, I420 };

    struct Options {
        std::optional<int> rawFrameWidth;
        std::optional<int> rawFrameHeight;
        std::optional<PixelFormat> rawPixelFormat;
    };

    static expected<VideoReader> OpenFile(const std::filesystem::path& filename, Options options);
    static expected<VideoReader> OpenStream(std::istream& input, Options options);

    ~VideoReader();
    VideoReader(const VideoReader&) = delete;
    VideoReader& operator=(const VideoReader&) = delete;
    VideoReader(VideoReader&&) noexcept;
    VideoReader& operator=(VideoReader&&) noexcept;

    expected<std::optional<Nv12Frame>> Read();

private:
    explicit VideoReader(std::unique_ptr<detail::VideoSource> source);

    std::unique_ptr<detail::VideoSource> m_source;
};

class VideoWriter {
public:
    enum class Format { NV12, I420, MKV };

    struct Options {
        std::optional<Format> format;
        double frameRate = 30.0;
    };

    static expected<VideoWriter> OpenFile(const std::filesystem::path& filename);
    static expected<VideoWriter> OpenFile(const std::filesystem::path& filename, Options options);
    static expected<VideoWriter> OpenStream(std::ostream& output, Options options);

    ~VideoWriter();
    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;
    VideoWriter(VideoWriter&&) noexcept;
    VideoWriter& operator=(VideoWriter&&) noexcept;

    expected<void> Write(const Nv12FrameView& frame);
    expected<void> Close();

private:
    VideoWriter(std::unique_ptr<detail::VideoSink> sink, std::unique_ptr<std::ofstream> ownedFile = {});

    std::unique_ptr<std::ofstream> m_ownedFile;
    std::unique_ptr<detail::VideoSink> m_sink;
    bool m_fileCloseFailed{};
};

expected<std::vector<Nv12Frame>> LoadNv12Frames(const std::filesystem::path& filename, VideoReader::Options options,
                                                std::optional<size_t> maxNumFrames = {});

}  // namespace libmlvc
