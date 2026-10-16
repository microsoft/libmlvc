// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/tar.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/libmlvc.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace libmlvc;

namespace {

std::filesystem::path FindBundle()
{
    for (const auto& entry : std::filesystem::directory_iterator(GetDefaultModelBundlesDir())) {
        if (entry.path().extension() == ".tar") return entry.path();
    }
    return {};
}

std::vector<std::byte> ReadBytes(const std::filesystem::path& path)
{
    std::vector<std::byte> data(std::filesystem::file_size(path));
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return data;
}

class UnitTestTar : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        s_bundlePath = FindBundle();
        if (!s_bundlePath.empty()) s_archive = ReadBytes(s_bundlePath);
    }

    static void TearDownTestSuite() { s_archive = {}; }

    void SetUp() override
    {
        ASSERT_FALSE(s_bundlePath.empty()) << "No model bundle in " << GetDefaultModelBundlesDir();
    }

    static inline std::filesystem::path s_bundlePath;
    static inline std::vector<std::byte> s_archive;
};

}  // namespace

TEST_F(UnitTestTar, ReadsEntriesLikeExtractTar)
{
    const auto entries = ExtractTar(s_archive);
    ASSERT_TRUE(entries) << entries.error().message();

    for (const auto& [name, data] : entries.value()) {
        SCOPED_TRACE(name);
        const auto entry = ReadTarEntry(s_bundlePath, name);
        ASSERT_TRUE(entry) << entry.error().message();
        EXPECT_TRUE(std::ranges::equal(entry.value(), data));
    }
}

TEST_F(UnitTestTar, ReportsMissingEntryOrArchive)
{
    const auto missingEntry = ReadTarEntry(s_bundlePath, "missing.json");
    ASSERT_FALSE(missingEntry);
    EXPECT_EQ(missingEntry.error(), make_error_code(Error::io_error));

    const auto missingArchive = ReadTarEntry(s_bundlePath.parent_path() / "missing.tar", "bundle_manifest.json");
    ASSERT_FALSE(missingArchive);
    EXPECT_EQ(missingArchive.error(), make_error_code(Error::io_error));
}

TEST_F(UnitTestTar, ReportsTruncatedEntry)
{
    const auto entries = ExtractTar(s_archive);
    ASSERT_TRUE(entries) << entries.error().message();

    // Cut the archive in the middle of its first entry
    const auto first =
        std::ranges::min_element(entries.value(), {}, [](const auto& entry) { return entry.second.data(); });
    ASSERT_FALSE(first->second.empty());
    const auto cut = static_cast<size_t>(first->second.data() - s_archive.data()) + first->second.size() / 2;
    const auto truncatedPath = std::filesystem::temp_directory_path() / "mlvc_truncated_bundle.tar";
    std::ofstream(truncatedPath, std::ios::binary).write(reinterpret_cast<const char*>(s_archive.data()), cut);

    const auto entry = ReadTarEntry(truncatedPath, first->first);
    std::filesystem::remove(truncatedPath);
    ASSERT_FALSE(entry);
    EXPECT_EQ(entry.error(), make_error_code(Error::general_failure));
}

TEST_F(UnitTestTar, ReportsEntryLargerThanArchive)
{
    // A header declaring the maximum entry size (~8 GiB), followed by the end-of-archive blocks
    std::vector<std::byte> archive(3 * 512);
    constexpr std::string_view name = "bundle_manifest.json";
    constexpr std::string_view size = "77777777777";
    std::ranges::copy(std::as_bytes(std::span{ name }), archive.begin());
    std::ranges::copy(std::as_bytes(std::span{ size }), archive.begin() + 124);
    archive[156] = std::byte{ '0' };
    const auto oversizedPath = std::filesystem::temp_directory_path() / "mlvc_oversized_bundle.tar";
    std::ofstream(oversizedPath, std::ios::binary)
        .write(reinterpret_cast<const char*>(archive.data()), static_cast<std::streamsize>(archive.size()));

    const auto entry = ReadTarEntry(oversizedPath, name);
    std::filesystem::remove(oversizedPath);
    ASSERT_FALSE(entry);
    EXPECT_EQ(entry.error(), make_error_code(Error::general_failure));
}
