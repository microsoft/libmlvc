// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/mlvc_io.hpp"

#include <libmlvc/error_codes.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace libmlvc;

namespace {

std::vector<std::byte> MakeBytes(std::initializer_list<int> values)
{
    std::vector<std::byte> bytes;
    bytes.reserve(values.size());
    for (const int value : values) {
        bytes.push_back(static_cast<std::byte>(value));
    }
    return bytes;
}

std::string AsString(std::span<const std::byte> bytes)
{
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

class ScopedTempFile {
public:
    ScopedTempFile(std::string_view suffix, std::span<const std::byte> content)
        : path(std::filesystem::temp_directory_path()
               / ("libmlvc_access_unit_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
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

class FailingOutputBuffer : public std::streambuf {
protected:
    std::streamsize xsputn(const char*, std::streamsize) override { return 0; }
};

class FailingSyncBuffer : public std::stringbuf {
protected:
    int sync() override { return -1; }
};

// NAL header 0x02 -> type 1 (VCL); 0x40 -> type 32 (non-VCL).
const std::vector<std::byte> kVclAccessUnit1 = MakeBytes({ 0, 0, 0, 1, 0x02, 0xAA });
const std::vector<std::byte> kVclAccessUnit2 = MakeBytes({ 0, 0, 0, 1, 0x02, 0xBB });

}  // namespace

// ============================================================================
// Access unit reading
// ============================================================================

TEST(UnitTestMlvcReader, ReadsAccessUnitsIncrementally)
{
    auto data = kVclAccessUnit1;
    data.insert(data.end(), kVclAccessUnit2.begin(), kVclAccessUnit2.end());
    std::istringstream input(AsString(data), std::ios::in | std::ios::binary);

    auto reader = MlvcReader::OpenStream(input);
    ASSERT_TRUE(reader);

    auto first = reader->Read();
    ASSERT_TRUE(first);
    ASSERT_TRUE(first->has_value());
    EXPECT_EQ(first->value().data, kVclAccessUnit1);

    auto second = reader->Read();
    ASSERT_TRUE(second);
    ASSERT_TRUE(second->has_value());
    EXPECT_EQ(second->value().data, kVclAccessUnit2);

    auto end = reader->Read();
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
}

TEST(UnitTestMlvcReader, GroupsNonVclNalsWithVcl)
{
    auto data = MakeBytes({ 0, 0, 0, 1, 0x40, 0xCC, 0, 0, 0, 1, 0x02, 0xDD });
    std::istringstream input(AsString(data), std::ios::in | std::ios::binary);

    auto reader = MlvcReader::OpenStream(input);
    ASSERT_TRUE(reader);

    auto accessUnit = reader->Read();
    ASSERT_TRUE(accessUnit);
    ASSERT_TRUE(accessUnit->has_value());
    EXPECT_EQ(accessUnit->value().data, data);

    auto end = reader->Read();
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
}

TEST(UnitTestMlvcReader, RejectsInvalidStream)
{
    std::istringstream input;
    input.setstate(std::ios::badbit);

    auto reader = MlvcReader::OpenStream(input);
    ASSERT_FALSE(reader);
    EXPECT_EQ(reader.error(), make_error_code(Error::io_error));
}

TEST(UnitTestMlvcReader, RejectsChangedInputStreamState)
{
    {
        std::istringstream input(AsString(kVclAccessUnit1), std::ios::in | std::ios::binary);
        auto reader = MlvcReader::OpenStream(input);
        ASSERT_TRUE(reader);
        input.setstate(std::ios::badbit);

        auto accessUnit = reader->Read();
        ASSERT_FALSE(accessUnit);
        EXPECT_EQ(accessUnit.error(), make_error_code(Error::io_error));
    }
    {
        std::istringstream input(AsString(kVclAccessUnit1), std::ios::in | std::ios::binary);
        auto reader = MlvcReader::OpenStream(input);
        ASSERT_TRUE(reader);
        input.exceptions(std::ios::badbit);

        auto accessUnit = reader->Read();
        ASSERT_FALSE(accessUnit);
        EXPECT_EQ(accessUnit.error(), make_error_code(Error::invalid_argument));
    }
}

TEST(UnitTestMlvcReader, RejectsMissingStartCode)
{
    auto data = MakeBytes({ 0xDE, 0xAD, 0xBE, 0xEF });
    std::istringstream input(AsString(data), std::ios::in | std::ios::binary);

    auto reader = MlvcReader::OpenStream(input);
    ASSERT_TRUE(reader);

    auto accessUnit = reader->Read();
    ASSERT_FALSE(accessUnit);
    EXPECT_EQ(accessUnit.error(), make_error_code(Error::io_error));
}

TEST(UnitTestMlvcReader, RejectsReadAfterMove)
{
    std::istringstream input(AsString(kVclAccessUnit1), std::ios::in | std::ios::binary);
    auto reader = MlvcReader::OpenStream(input);
    ASSERT_TRUE(reader);
    auto movedReader = std::move(*reader);

    auto afterMove = reader->Read();
    ASSERT_FALSE(afterMove);
    EXPECT_EQ(afterMove.error(), make_error_code(Error::invalid_argument));

    auto movedRead = movedReader.Read();
    ASSERT_TRUE(movedRead);
    EXPECT_TRUE(movedRead->has_value());
}

TEST(UnitTestMlvcReader, ReadsManyAccessUnitsAcrossReadChunks)
{
    // Total size spans several internal read chunks to exercise incremental parsing.
    constexpr int accessUnitCount = 200;
    constexpr size_t payloadSize = 1024;
    std::vector<std::byte> data;
    for (int i = 0; i < accessUnitCount; i++) {
        for (const int b : { 0x00, 0x00, 0x00, 0x01, 0x02 }) {
            data.push_back(static_cast<std::byte>(b));
        }
        for (size_t j = 0; j < payloadSize; j++) {
            data.push_back(static_cast<std::byte>((i + j) & 0xFF));
        }
    }
    std::istringstream input(AsString(data), std::ios::in | std::ios::binary);

    auto reader = MlvcReader::OpenStream(input);
    ASSERT_TRUE(reader);

    int count = 0;
    while (true) {
        auto accessUnit = reader->Read();
        ASSERT_TRUE(accessUnit);
        if (!accessUnit->has_value()) {
            break;
        }
        EXPECT_EQ(accessUnit->value().data.size(), 5 + payloadSize);
        count++;
    }
    EXPECT_EQ(count, accessUnitCount);
}

TEST(UnitTestMlvcReader, ReadsStartCodeAcrossReadChunkBoundary)
{
    constexpr size_t readChunkSize = 64 * 1024;
    constexpr size_t secondStartCodeOffset = readChunkSize - 3;
    std::vector<std::byte> data(secondStartCodeOffset, std::byte{ 0xAA });
    std::copy(kVclAccessUnit1.begin(), kVclAccessUnit1.begin() + 5, data.begin());
    data.insert(data.end(), kVclAccessUnit2.begin(), kVclAccessUnit2.end());
    ScopedTempFile file(".mlvc", data);

    auto reader = MlvcReader::OpenFile(file.path);
    ASSERT_TRUE(reader);

    auto first = reader->Read();
    ASSERT_TRUE(first);
    ASSERT_TRUE(first->has_value());
    EXPECT_EQ(first->value().data.size(), secondStartCodeOffset);

    auto second = reader->Read();
    ASSERT_TRUE(second);
    ASSERT_TRUE(second->has_value());
    EXPECT_EQ(second->value().data, kVclAccessUnit2);
}

// ============================================================================
// Access unit writing
// ============================================================================

TEST(UnitTestMlvcWriter, WritesAccessUnitsToStream)
{
    std::ostringstream output(std::ios::binary);
    auto writer = MlvcWriter::OpenStream(output);
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->Write(kVclAccessUnit1));
    ASSERT_TRUE(writer->Write(kVclAccessUnit2));
    ASSERT_TRUE(writer->Close());

    auto expected = kVclAccessUnit1;
    expected.insert(expected.end(), kVclAccessUnit2.begin(), kVclAccessUnit2.end());
    EXPECT_EQ(output.str(), AsString(expected));
}

TEST(UnitTestMlvcWriter, WritesFileReadableByMlvcReader)
{
    ScopedTempFile file(".mlvc", {});
    auto writer = MlvcWriter::OpenFile(file.path);
    ASSERT_TRUE(writer);
    ASSERT_TRUE(writer->Write(kVclAccessUnit1));
    ASSERT_TRUE(writer->Write(kVclAccessUnit2));
    ASSERT_TRUE(writer->Close());

    auto accessUnits = ReadMlvcAccessUnits(file.path);
    ASSERT_TRUE(accessUnits);
    ASSERT_EQ(accessUnits->size(), 2u);
    EXPECT_EQ((*accessUnits)[0].data, kVclAccessUnit1);
    EXPECT_EQ((*accessUnits)[1].data, kVclAccessUnit2);
}

TEST(UnitTestMlvcWriter, ReturnsErrorForFailingOutputStream)
{
    {
        FailingOutputBuffer buffer;
        std::ostream output(&buffer);
        auto writer = MlvcWriter::OpenStream(output);
        ASSERT_TRUE(writer);
        auto write = writer->Write(kVclAccessUnit1);
        ASSERT_FALSE(write);
        EXPECT_EQ(write.error(), make_error_code(Error::io_error));
        auto close = writer->Close();
        ASSERT_FALSE(close);
        EXPECT_EQ(close.error(), make_error_code(Error::io_error));
    }
    {
        FailingSyncBuffer buffer;
        std::ostream output(&buffer);
        auto writer = MlvcWriter::OpenStream(output);
        ASSERT_TRUE(writer);
        ASSERT_TRUE(writer->Write(kVclAccessUnit1));
        auto close = writer->Close();
        ASSERT_FALSE(close);
        EXPECT_EQ(close.error(), make_error_code(Error::io_error));
    }
    {
        std::ostringstream output;
        auto writer = MlvcWriter::OpenStream(output);
        ASSERT_TRUE(writer);
        output.exceptions(std::ios::badbit);
        auto write = writer->Write(kVclAccessUnit1);
        ASSERT_FALSE(write);
        EXPECT_EQ(write.error(), make_error_code(Error::invalid_argument));
    }
}

// ============================================================================
// ReadMlvcAccessUnits
// ============================================================================

TEST(UnitTestMlvcReader, ReadMlvcAccessUnitsRespectsLimit)
{
    auto data = kVclAccessUnit1;
    data.insert(data.end(), kVclAccessUnit2.begin(), kVclAccessUnit2.end());
    ScopedTempFile file(".mlvc", data);

    auto accessUnits = ReadMlvcAccessUnits(file.path, 1);
    ASSERT_TRUE(accessUnits);
    EXPECT_EQ(accessUnits->size(), 1u);
}

TEST(UnitTestMlvcReader, ReadMlvcAccessUnitsRejectsMissingFile)
{
    auto accessUnits = ReadMlvcAccessUnits("nonexistent_access_unit_file.mlvc");
    ASSERT_FALSE(accessUnits);
    EXPECT_EQ(accessUnits.error(), make_error_code(Error::io_error));
}
