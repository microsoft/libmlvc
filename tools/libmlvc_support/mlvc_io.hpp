// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace libmlvc {

struct MlvcAccessUnit {
    std::vector<std::byte> data;
};

class MlvcReader {
public:
    static expected<MlvcReader> OpenFile(const std::filesystem::path& filename);
    static expected<MlvcReader> OpenStream(std::istream& input);

    ~MlvcReader();
    MlvcReader(const MlvcReader&) = delete;
    MlvcReader& operator=(const MlvcReader&) = delete;
    MlvcReader(MlvcReader&&) noexcept;
    MlvcReader& operator=(MlvcReader&&) noexcept;

    expected<std::optional<MlvcAccessUnit>> Read();

private:
    MlvcReader(std::istream* stream, std::unique_ptr<std::ifstream> ownedFile);

    std::unique_ptr<std::ifstream> m_ownedFile;
    std::istream* m_stream{};
    std::streambuf* m_streamBuffer{};
    std::vector<std::byte> m_buffer;
    size_t m_bufferOffset{};
};

class MlvcWriter {
public:
    static expected<MlvcWriter> OpenFile(const std::filesystem::path& filename);
    static expected<MlvcWriter> OpenStream(std::ostream& output);

    ~MlvcWriter();
    MlvcWriter(const MlvcWriter&) = delete;
    MlvcWriter& operator=(const MlvcWriter&) = delete;
    MlvcWriter(MlvcWriter&&) noexcept;
    MlvcWriter& operator=(MlvcWriter&&) noexcept = delete;

    expected<void> Write(std::span<const std::byte> accessUnit);
    expected<void> Close();

private:
    MlvcWriter(std::ostream* stream, std::unique_ptr<std::ofstream> ownedFile);

    std::unique_ptr<std::ofstream> m_ownedFile;
    std::ostream* m_stream{};
    bool m_closed{};
    bool m_failed{};
};

expected<std::vector<MlvcAccessUnit>> ReadMlvcAccessUnits(const std::filesystem::path& filename,
                                                          std::optional<size_t> maxNumAccessUnits = std::nullopt);

}  // namespace libmlvc
