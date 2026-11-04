// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/utils.hpp"
#include "libmlvc/inference/model_cache.hpp"
#include "support/test_config.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/libmlvc.hpp>

#include <gtest/gtest.h>

#if defined(MLVC_PLATFORM_WINCLASSIC)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <latch>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace libmlvc;
namespace fs = std::filesystem;

static_assert(!std::is_constructible_v<ModelCache, const fs::path&>);

namespace {

using Files = std::map<std::string, std::string>;  // Relative path -> contents

// Literals rather than the implementation's constants, so that changing the on-disk format fails these tests
constexpr auto kKeyPrefix = "v2_";
constexpr auto kManifestFilename = "mlvc_manifest.json";
constexpr auto kAccessTimeFilename = "last_access.time";
constexpr auto kManifestSizeLimit = size_t{ 1 } << 20;

const auto kModelFiles = Files{ { "model.bin", "compiled model" } };

const auto kInvalidArgument = make_error_code(Error::invalid_argument);
const auto kCacheError = make_error_code(Error::model_cache_error);
const auto kModelInitError = make_error_code(Error::model_init_error);
const auto kIoError = make_error_code(Error::io_error);

template <typename T>
std::error_code ErrorOf(const expected<T>& result)
{
    return result ? std::error_code{} : result.error();
}

void WriteFile(const fs::path& path, const std::string& contents)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file) << path;
    file << contents;
    file.close();
    ASSERT_TRUE(file) << path;
}

std::string ReadFile(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::string ManifestFor(std::string_view path, uint64_t size)
{
    return R"({"version":1,"files":[{"path":")" + std::string(path) + R"(","size":)" + std::to_string(size) + "}]}";
}

int64_t SecondsAgo(std::chrono::seconds age)
{
    return std::chrono::duration_cast<std::chrono::seconds>((std::chrono::system_clock::now() - age).time_since_epoch()).count();
}

int64_t DaysAgo(int days)
{
    return SecondsAgo(std::chrono::days(days));
}

void WriteTime(const fs::path& dir, int64_t seconds)
{
    // Little-endian int64 seconds since 1970, as older versions write it.
    std::string bytes(8, '\0');
    for (size_t i = 0; i < bytes.size(); i++) {
        bytes[i] = static_cast<char>((static_cast<uint64_t>(seconds) >> (8 * i)) & 0xff);
    }
    WriteFile(dir / kAccessTimeFilename, bytes);
}

std::optional<int64_t> ReadTime(const fs::path& dir)
{
    const auto bytes = ReadFile(dir / kAccessTimeFilename);
    if (bytes.size() < 8) {
        return std::nullopt;
    }
    uint64_t value = 0;
    for (size_t i = 0; i < 8; i++) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return static_cast<int64_t>(value);
}

testing::AssertionResult IsRecent(std::optional<int64_t> time)
{
    const auto now = SecondsAgo({});
    if (time && *time >= now - 60 && *time <= now + 60) {
        return testing::AssertionSuccess();
    }
    return testing::AssertionFailure()
           << "last-access time " << (time ? std::to_string(*time) : "missing") << ", now " << now;
}

size_t CountItems(const fs::path& dir)
{
    if (!fs::exists(dir)) {
        return 0;
    }
    return static_cast<size_t>(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}));
}

class QuietLog {
public:
    QuietLog() { SetLogLevel(LogLevel::Fatal); }
    QuietLog(const QuietLog&) = delete;
    QuietLog& operator=(const QuietLog&) = delete;
    QuietLog(QuietLog&&) = delete;
    QuietLog& operator=(QuietLog&&) = delete;
    ~QuietLog() { SetLogLevel(GetTestConfig().logLevel); }
};

bool CreateDirectoryLink(const fs::path& target, const fs::path& link)
{
#if defined(MLVC_PLATFORM_WINCLASSIC)
    // Junctions need no special privileges, unlike symbolic links on Windows.
    const auto command = L"mklink /J \"" + link.wstring() + L"\" \"" + target.wstring() + L"\" >nul 2>&1";
    return _wsystem(command.c_str()) == 0;
#else
    std::error_code ec;
    fs::create_directory_symlink(target, link, ec);
    return !ec;
#endif
}

#if defined(MLVC_PLATFORM_WINCLASSIC)
std::thread HoldFile(const fs::path& path, DWORD share, std::function<void()> until)
{
    // Simulate antivirus/indexer handles holding a newly written file open.
    std::promise<bool> opened;
    auto openedFuture = opened.get_future();
    std::thread holder([path, share, until = std::move(until), opened = std::move(opened)]() mutable {
        auto* const file =
            ::CreateFileW(path.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        opened.set_value(file != INVALID_HANDLE_VALUE);
        until();
        if (file != INVALID_HANDLE_VALUE) {
            ::CloseHandle(file);
        }
    });
    EXPECT_TRUE(openedFuture.get());
    return holder;
}
#endif

}  // namespace

class UnitTestModelCache : public ::testing::Test {
protected:
    void SetUp() override
    {
        m_baseDir = fs::temp_directory_path() / ("mlvc_model_cache_test_" + std::to_string(std::random_device{}()));
        std::error_code ec;
        ASSERT_TRUE(fs::create_directories(m_baseDir, ec)) << ec.message();
        m_entriesDir = m_baseDir / "models";
        m_stagingDir = m_baseDir / "models.staging";
        m_trashDir = m_baseDir / "models.trash";
        auto cache = ModelCache::Create(m_entriesDir);
        ASSERT_TRUE(cache) << cache.error().message();
        m_cache = std::move(cache.value());
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(m_baseDir, ec);
    }

    ModelCache::CompileFunction Compiler(Files files = kModelFiles)
    {
        return [this, files = std::move(files)](const fs::path& dir) -> expected<void> {
            m_compiles++;
            for (const auto& [path, contents] : files) {
                WriteFile(dir / path, contents);
                if (HasFatalFailure()) {
                    return kIoError;
                }
            }
            return {};
        };
    }

    std::vector<expected<fs::path>> GetOrCreateConcurrently(const std::vector<std::shared_ptr<ModelCache>>& caches)
    {
        constexpr auto kCalls = 8;
        auto results = std::vector<expected<fs::path>>(kCalls);
        auto start = std::latch{ kCalls };
        const auto compile = Compiler();
        auto threads = std::vector<std::thread>{};
        for (size_t i = 0; i < results.size(); i++) {
            threads.emplace_back([&, i] {
                start.arrive_and_wait();
                results[i] = caches[i % caches.size()]->GetOrCreate(
                    i % 2 == 0 ? "Key" : "KEY", [&](const fs::path& dir) -> expected<void> {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        return compile(dir);
                    });
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return results;
    }

    fs::path EntryDir(const std::string& key) const { return m_entriesDir / (kKeyPrefix + key); }

    fs::path AddEntry(const std::string& key, int64_t time, const std::string& model = "compiled model")
    {
        const auto entryDir = EntryDir(key);
        WriteFile(entryDir / "model.bin", model);
        WriteFile(entryDir / kManifestFilename, ManifestFor("model.bin", model.size()));
        WriteTime(entryDir, time);
        return entryDir;
    }

    fs::path AddOlderEntry(const std::string& key, int64_t time, const std::string& model = "old model")
    {
        // Legacy entries use the key verbatim and have no manifest.
        const auto entryDir = m_entriesDir / key;
        WriteFile(entryDir / "model.onnx", model);
        WriteTime(entryDir, time);
        return entryDir;
    }

    fs::path AddStagingDir(const std::string& name, int64_t time)
    {
        const auto stagingDir = m_stagingDir / name;
        WriteFile(stagingDir / "model.bin", "partial");
        WriteTime(stagingDir, time);
        return stagingDir;
    }

    fs::path m_baseDir;
    fs::path m_entriesDir;
    fs::path m_stagingDir;
    fs::path m_trashDir;
    std::shared_ptr<ModelCache> m_cache;
    std::atomic<int> m_compiles{ 0 };
};

//------------------------------------------------------------------------------
// Creation and lookup
//------------------------------------------------------------------------------

TEST_F(UnitTestModelCache, PipelineErrorCodesAreStable)
{
    struct Case {
        Error m_error{};
        int m_value{};
        const char* m_message{};
    };
    const auto cases = std::array{
        Case{ Error::model_cache_error, 24, "model cache error" },
        Case{ Error::model_compile_error, 25, "model compilation error" },
        Case{ Error::model_cache_load_error, 26, "model cache load error" },
    };
    for (const auto& test : cases) {
        const auto error = make_error_code(test.m_error);
        EXPECT_EQ(error.value(), test.m_value);
        EXPECT_EQ(&error.category(), &error_category());
        EXPECT_STREQ(error.category().name(), "libmlvc");
        EXPECT_EQ(error.message(), test.m_message);
    }
}

TEST_F(UnitTestModelCache, PreservesCompileCallbackErrors)
{
    const auto errors = {
        make_error_code(Error::io_error),
        make_error_code(Error::general_failure),
        make_error_code(Error::json_parse_error),
        make_error_code(Error::model_init_error),
        make_error_code(Error::operation_cancelled),
        make_error_code(Error::invalid_argument),
        make_error_code(Error::ep_download_error),
        make_error_code(Error::ep_register_error),
        make_error_code(Error::model_inference_error),
        make_error_code(Error::model_cache_error),
        make_error_code(Error::model_compile_error),
        make_error_code(Error::model_cache_load_error),
        std::make_error_code(std::errc::no_space_on_device),
    };
    for (const auto error : errors) {
        SCOPED_TRACE(error);
        const auto created = m_cache->GetOrCreate("key", [&](const fs::path& dir) -> expected<void> {
            WriteFile(dir / "model.bin", "partial");
            return error;
        });
        EXPECT_EQ(ErrorOf(created), error);
        EXPECT_FALSE(fs::exists(EntryDir("key")));
        EXPECT_EQ(CountItems(m_stagingDir), 0u);
    }
}

TEST_F(UnitTestModelCache, GetOrCreateCompilesOnceAndPublishes)
{
    {
        SCOPED_TRACE("Not cached: Get fails and creates nothing");
        EXPECT_EQ(ErrorOf(m_cache->Get("key")), kCacheError);
        EXPECT_FALSE(fs::exists(m_entriesDir));
    }

    auto entryDir =
        m_cache->GetOrCreate("key", Compiler({ { "model.bin", "compiled model" }, { "Sub/data.bin", "data" } }));
    ASSERT_TRUE(entryDir) << entryDir.error().message();
    EXPECT_EQ(*entryDir, EntryDir("key"));
    EXPECT_EQ(m_compiles, 1);
    EXPECT_EQ(ReadFile(*entryDir / "model.bin"), "compiled model");
    EXPECT_EQ(ReadFile(*entryDir / "Sub" / "data.bin"), "data");
    // Manifest ordering is case-sensitive, even on a case-insensitive filesystem.
    EXPECT_EQ(ReadFile(*entryDir / kManifestFilename),
              R"({"version":1,"files":[{"path":"Sub/data.bin","size":4},{"path":"model.bin","size":14}]})");
    EXPECT_TRUE(IsRecent(ReadTime(*entryDir)));
    EXPECT_EQ(CountItems(m_stagingDir), 0u);

    {
        SCOPED_TRACE("Cached: not compiled again");
        auto cached = m_cache->GetOrCreate("key", Compiler());
        ASSERT_TRUE(cached) << cached.error().message();
        EXPECT_EQ(*cached, *entryDir);
        auto found = m_cache->Get("key");
        ASSERT_TRUE(found) << found.error().message();
        EXPECT_EQ(*found, *entryDir);
        EXPECT_EQ(m_compiles, 1);
    }
}

//------------------------------------------------------------------------------
// Arguments, paths, and keys
//------------------------------------------------------------------------------

TEST_F(UnitTestModelCache, RejectsInvalidArguments)
{
    // Keys become directory names: only letters, digits, '_' and '-'
    for (const auto* key : { "", "a/b", "a\\b", "..", "a b", "a.b", "k\xc3\xa9y" }) {
        SCOPED_TRACE(key);
        EXPECT_EQ(ErrorOf(m_cache->GetOrCreate(key, Compiler())), kInvalidArgument);
        EXPECT_EQ(ErrorOf(m_cache->Get(key)), kInvalidArgument);
    }
    EXPECT_EQ(ErrorOf(m_cache->GetOrCreate("key", nullptr)), kInvalidArgument);
    {
        SCOPED_TRACE("No entries directory");
        EXPECT_EQ(ErrorOf(ModelCache::Create({})), kInvalidArgument);
    }
    EXPECT_EQ(m_compiles, 0);
    EXPECT_FALSE(fs::exists(m_entriesDir));
    EXPECT_TRUE(m_cache->GetOrCreate("Az09_-", Compiler()));
}

TEST_F(UnitTestModelCache, UsesTemporaryFoldersNextToTheEntriesDirectory)
{
    // Trailing separators must not change sibling staging/trash locations.
    const auto cache = ModelCache::Create(m_entriesDir / "");
    ASSERT_TRUE(cache) << cache.error().message();
    auto created = cache.value()->GetOrCreate("key", [this](const fs::path& dir) -> expected<void> {
        EXPECT_EQ(dir.parent_path(), m_stagingDir);
        EXPECT_FALSE(fs::exists(m_entriesDir));
        EXPECT_TRUE(IsRecent(ReadTime(dir)));
        WriteFile(dir / "model.bin", "compiled model");
        return {};
    });
    ASSERT_TRUE(created) << created.error().message();
    EXPECT_EQ(*created, EntryDir("key"));
    const auto expired = AddEntry("expired", DaysAgo(30));
    cache.value()->Cleanup();
    EXPECT_FALSE(fs::exists(expired));
    EXPECT_EQ(CountItems(m_baseDir), 3u);  // The entries, staging and trash folders
    EXPECT_TRUE(fs::is_directory(m_stagingDir));
    EXPECT_TRUE(fs::is_directory(m_trashDir));
}

TEST_F(UnitTestModelCache, RelativeRootIsAnchoredAtConstruction)
{
    const auto originalDir = fs::current_path();
    const auto restoreDirectory = ScopeExit{ [&originalDir]() noexcept {
        std::error_code ec;
        fs::current_path(originalDir, ec);
        EXPECT_FALSE(ec) << ec.message();
    } };
    fs::current_path(m_baseDir);
    const auto cache = ModelCache::Create(fs::path{ "." } / "models" / "");
    ASSERT_TRUE(cache) << cache.error().message();
    const auto kept = AddEntry("key", DaysAgo(0));
    const auto expired = AddEntry("expired", DaysAgo(30));
    const auto abandoned = AddStagingDir("abandoned", DaysAgo(1));
    const auto trashed = m_trashDir / "leftover";
    WriteFile(trashed / "model.bin", "partial");

    const auto elsewhere = m_baseDir / "elsewhere";
    for (const auto* folder : { "models", "models.staging", "models.trash" }) {
        const auto unrelated = elsewhere / folder / "unrelated";
        WriteFile(unrelated / "data.bin", "keep");
        fs::last_write_time(unrelated, fs::file_time_type::clock::now() - std::chrono::hours(2));
    }
    fs::current_path(elsewhere);
    cache.value()->Cleanup();
    EXPECT_FALSE(fs::exists(expired));
    EXPECT_FALSE(fs::exists(abandoned));
    EXPECT_FALSE(fs::exists(trashed));
    for (const auto* folder : { "models", "models.staging", "models.trash" }) {
        EXPECT_EQ(ReadFile(elsewhere / folder / "unrelated" / "data.bin"), "keep") << folder;
    }

    const auto found = cache.value()->Get("key");
    ASSERT_TRUE(found) << found.error().message();
    EXPECT_TRUE(fs::equivalent(*found, kept));
    const auto added = cache.value()->GetOrCreate("other", [this](const fs::path& dir) -> expected<void> {
        EXPECT_TRUE(fs::equivalent(dir.parent_path(), m_stagingDir));
        return Compiler()(dir);
    });
    ASSERT_TRUE(added) << added.error().message();
    EXPECT_TRUE(fs::equivalent(*added, EntryDir("other")));
}

#if defined(MLVC_PLATFORM_WINCLASSIC)
TEST_F(UnitTestModelCache, CreationReturnsPathResolutionError)
{
    const auto root = fs::path{ std::string(1, '\0') };
    auto error = std::error_code{};
    ASSERT_TRUE(fs::absolute(root, error).empty());
    ASSERT_TRUE(error);
    const auto cache = ModelCache::Create(root);
    ASSERT_FALSE(cache);
    EXPECT_EQ(cache.error(), kCacheError);
    EXPECT_EQ(CountItems(m_baseDir), 0u);
}

TEST_F(UnitTestModelCache, ExtendedLengthRootsShareNestedEntries)
{
    const auto extendedDir = fs::path{ L"\\\\?\\" + m_entriesDir.native() };
    const auto extendedCache = ModelCache::Create(extendedDir);
    ASSERT_TRUE(extendedCache) << extendedCache.error().message();
    const auto names = std::vector<fs::path>{ "model.bin", fs::path{ "sub" } / "data.bin",
                                              fs::path{ "sub" } / fs::path{ u8"m\u00f3del \u65e5\u672c.bin" } };
    const auto compile = [&](const fs::path& dir) -> expected<void> {
        ++m_compiles;
        for (const auto& name : names) {
            WriteFile(dir / name, "data");
            if (HasFatalFailure()) {
                return kIoError;
            }
        }
        return {};
    };

    for (const auto* key : { "normal", "extended" }) {
        SCOPED_TRACE(key);
        const auto& creator = std::string_view{ key } == "normal" ? m_cache : extendedCache.value();
        const auto created = creator->GetOrCreate(key, compile);
        ASSERT_TRUE(created) << created.error().message();
        const auto normal = m_cache->Get(key);
        ASSERT_TRUE(normal) << normal.error().message();
        const auto extended = extendedCache.value()->Get(key);
        ASSERT_TRUE(extended) << extended.error().message();
        EXPECT_TRUE(fs::equivalent(*normal, *extended));
        for (const auto& name : names) {
            EXPECT_EQ(ReadFile(*extended / name), "data");
        }
        EXPECT_NE(ReadFile(*extended / kManifestFilename).find(R"("path":"sub/data.bin")"), std::string::npos);
        EXPECT_TRUE(m_cache->GetOrCreate(key, compile));
        EXPECT_TRUE(extendedCache.value()->GetOrCreate(key, compile));
    }
    EXPECT_EQ(m_compiles, 2);
}

TEST_F(UnitTestModelCache, ExtendedLengthRootsSupportLongNestedPaths)
{
    const auto root = m_baseDir / "long" / std::string(120, 'a') / std::string(120, 'b') / "models";
    ASSERT_GT(root.native().size(), static_cast<size_t>(MAX_PATH));
    const auto extendedDir = fs::path{ L"\\\\?\\" + root.native() };
    const auto longParent = fs::path{ L"\\\\?\\" + (m_baseDir / "long").native() };
    const auto cleanup = ScopeExit{ [&longParent]() noexcept {
        auto ec = std::error_code{};
        fs::remove_all(longParent, ec);
        EXPECT_FALSE(ec) << ec.message();
    } };
    const auto cache = ModelCache::Create(extendedDir);
    ASSERT_TRUE(cache) << cache.error().message();
    const auto created = cache.value()->GetOrCreate("key", [this](const fs::path& dir) -> expected<void> {
        ++m_compiles;
        WriteFile(dir / "nested" / "model.bin", "compiled model");
        if (HasFatalFailure()) {
            return kIoError;
        }
        return {};
    });
    ASSERT_TRUE(created) << created.error().message();
    const auto found = cache.value()->Get("key");
    ASSERT_TRUE(found) << found.error().message();
    EXPECT_EQ(ReadFile(*found / "nested" / "model.bin"), "compiled model");
    EXPECT_EQ(ReadFile(*found / kManifestFilename), ManifestFor("nested/model.bin", 14));
    EXPECT_TRUE(cache.value()->GetOrCreate("key", Compiler()));
    EXPECT_EQ(m_compiles, 1);

    cache.value()->Cleanup(std::chrono::seconds{ 0 });
    EXPECT_FALSE(fs::exists(*created));
}

#endif

TEST_F(UnitTestModelCache, KeysAreCaseInsensitive)
{
    auto created = m_cache->GetOrCreate("Key", Compiler());
    ASSERT_TRUE(created) << created.error().message();
    EXPECT_EQ(*created, EntryDir("key"));
    auto found = m_cache->Get("KEY");
    ASSERT_TRUE(found) << found.error().message();
    EXPECT_EQ(*found, *created);
    auto cached = m_cache->GetOrCreate("kEy", Compiler());
    ASSERT_TRUE(cached) << cached.error().message();
    EXPECT_EQ(*cached, *created);
    EXPECT_EQ(m_compiles, 1);
}

TEST_F(UnitTestModelCache, NewKeysNeverMatchOlderVersionsEntries)
{
    const auto olderEntry = AddOlderEntry("key", DaysAgo(0));
    EXPECT_EQ(ErrorOf(m_cache->Get("key")), kCacheError);
    auto created = m_cache->GetOrCreate("key", Compiler());
    ASSERT_TRUE(created) << created.error().message();
    EXPECT_EQ(*created, EntryDir("key"));
    EXPECT_EQ(ReadFile(olderEntry / "model.onnx"), "old model");
}

//------------------------------------------------------------------------------
// Manifest and access-time validation
//------------------------------------------------------------------------------

TEST_F(UnitTestModelCache, ManifestCoversEveryCompiledFile)
{
    const auto names = std::vector<fs::path>{ "model.bin", fs::path("sub") / "data.bin", "with spaces.bin",
                                              fs::path(u8"m\u00f3del \u65e5\u672c.bin") };
    auto created = m_cache->GetOrCreate("key", [&](const fs::path& dir) -> expected<void> {
        for (const auto& name : names) {
            WriteFile(dir / name, "data");
        }
        return {};
    });
    ASSERT_TRUE(created) << created.error().message();
    ASSERT_TRUE(m_cache->Get("key"));
    for (size_t i = 0; i < names.size(); i++) {
        SCOPED_TRACE(i);
        WriteFile(*created / names[i], "another size");
        EXPECT_EQ(ErrorOf(m_cache->Get("key")), kCacheError);
        WriteFile(*created / names[i], "data");
        EXPECT_TRUE(m_cache->Get("key"));
    }
}

TEST_F(UnitTestModelCache, ManifestsAreBackwardAndForwardCompatible)
{
    const auto entryDir = AddEntry("key", DaysAgo(0));
    const auto manifestPath = entryDir / kManifestFilename;
    {
        SCOPED_TRACE("Written by an earlier version: only the fields of version 1");
        WriteFile(manifestPath, ManifestFor("model.bin", 14));
        EXPECT_TRUE(m_cache->Get("key"));
    }
    {
        SCOPED_TRACE("Written by a later version: the fields this version doesn't know are ignored");
        WriteFile(manifestPath, R"({"version":2,"compiler":"next","files":[{"path":"model.bin","size":14,"hash":"0123"}]})");
        EXPECT_TRUE(m_cache->Get("key"));
    }
    {
        SCOPED_TRACE("Written by a later version: the fields this version knows are still checked");
        WriteFile(manifestPath, R"({"version":2,"files":[{"path":"model.bin","size":15,"hash":"0123"}]})");
        EXPECT_EQ(ErrorOf(m_cache->Get("key")), kCacheError);
    }
}

TEST_F(UnitTestModelCache, ManifestSizeBoundary)
{
    const auto entryDir = AddEntry("key", 1000);
    auto text = ManifestFor("model.bin", 14);
    text.resize(kManifestSizeLimit, ' ');
    WriteFile(entryDir / kManifestFilename, text);
    ASSERT_TRUE(m_cache->Get("key"));
    EXPECT_TRUE(IsRecent(ReadTime(entryDir)));

    text.push_back(' ');
    WriteFile(entryDir / kManifestFilename, text);
    WriteTime(entryDir, 1000);
    EXPECT_EQ(ErrorOf(m_cache->Get("key")), kCacheError);
    EXPECT_EQ(ErrorOf(m_cache->GetOrCreate("key", Compiler())), kCacheError);
    EXPECT_EQ(ReadTime(entryDir), std::optional<int64_t>(1000));
    EXPECT_EQ(m_compiles, 0);
}

TEST_F(UnitTestModelCache, InvalidEntriesAreNeverRepaired)
{
    // Only files inside the entry count, even if one outside it matches
    const auto outside = m_baseDir / "outside.bin";
    WriteFile(outside, "compiled model");
    const auto absolute = outside.generic_u8string();
    const auto writeManifest = [](std::string text) {
        return [text = std::move(text)](const fs::path& dir) { WriteFile(dir / kManifestFilename, text); };
    };
    const auto damages = std::vector<std::pair<std::string, std::function<void(const fs::path&)>>>{
        { "Compiled file with another size", [](const fs::path& dir) { WriteFile(dir / "model.bin", "corrupt"); } },
        { "Compiled file missing", [](const fs::path& dir) { fs::remove(dir / "model.bin"); } },
        { "Manifest missing", [](const fs::path& dir) { fs::remove(dir / kManifestFilename); } },
        { "Manifest empty", writeManifest("") },
        { "Manifest not JSON", writeManifest("14 model.bin\n") },
        { "Manifest not an object", writeManifest(R"([{"version":1}])") },
        { "Manifest file not an object", writeManifest(R"({"version":1,"files":["model.bin"]})") },
        { "Manifest without a version", writeManifest(R"({"files":[{"path":"model.bin","size":14}]})") },
        { "Manifest without files", writeManifest(R"({"version":1,"files":[]})") },
        { "Manifest too large", writeManifest(ManifestFor("model.bin", 14) + std::string(1 << 20, ' ')) },
        { "Manifest size not a number", writeManifest(R"({"version":1,"files":[{"path":"model.bin","size":"14"}]})") },
        { "Manifest path not UTF-8", writeManifest(ManifestFor("model\xff.bin", 14)) },
        { "Manifest path contains NUL", writeManifest(ManifestFor(R"(model.bin\u0000suffix)", 14)) },
        { "Manifest path ends with NUL", writeManifest(ManifestFor(R"(model.bin\u0000)", 14)) },
        { "Manifest path outside the entry", writeManifest(ManifestFor("../../outside.bin", 14)) },
        { "Manifest path absolute", writeManifest(ManifestFor(std::string(absolute.begin(), absolute.end()), 14)) },
    };
    for (size_t i = 0; i < damages.size(); i++) {
        SCOPED_TRACE(damages[i].first);
        const auto key = "key" + std::to_string(i);
        const auto entryDir = AddEntry(key, 1000);
        damages[i].second(entryDir);
        EXPECT_EQ(ErrorOf(m_cache->Get(key)), kCacheError);
        EXPECT_EQ(ErrorOf(m_cache->GetOrCreate(key, Compiler())), kCacheError);
        EXPECT_EQ(m_compiles, 0);
        EXPECT_TRUE(fs::exists(entryDir));
        // Not refreshed, so that it ages out
        EXPECT_EQ(ReadTime(entryDir), std::optional<int64_t>(1000));
    }

    m_cache->Cleanup();
    EXPECT_EQ(CountItems(m_entriesDir), 0u);
    ASSERT_TRUE(m_cache->GetOrCreate("key0", Compiler()));
    EXPECT_EQ(m_compiles, 1);
}

TEST_F(UnitTestModelCache, LastAccessTimeIsUpdatedInPlace)
{
    const auto time1000 = std::string("\xe8\x03\0\0\0\0\0\0", 8);  // Little-endian
    const auto entryDir = AddEntry("key", 1000);
    const auto timePath = entryDir / kAccessTimeFilename;

    ASSERT_TRUE(m_cache->Get("key"));
    EXPECT_TRUE(IsRecent(ReadTime(entryDir)));
    WriteTime(entryDir, 1000);
    ASSERT_TRUE(m_cache->GetOrCreate("key", Compiler()));
    EXPECT_TRUE(IsRecent(ReadTime(entryDir)));
    {
        SCOPED_TRACE("Updates never truncate the file");
        WriteFile(timePath, time1000 + "tail");
        ASSERT_TRUE(m_cache->Get("key"));
        EXPECT_EQ(ReadFile(timePath).substr(8), "tail");
        EXPECT_TRUE(IsRecent(ReadTime(entryDir)));
    }
    {
        SCOPED_TRACE("Updates never create the file");
        fs::remove(timePath);
        ASSERT_TRUE(m_cache->Get("key"));
        ASSERT_TRUE(m_cache->GetOrCreate("key", Compiler()));
        EXPECT_FALSE(fs::exists(timePath));
        EXPECT_EQ(m_compiles, 0);
    }
}

TEST_F(UnitTestModelCache, LastAccessTimeIsNeverWrittenThroughLinks)
{
    const auto timePath = AddEntry("key", 1000) / kAccessTimeFilename;
    const auto other = m_baseDir / "other.bin";
    WriteFile(other, "other file");
    {
        SCOPED_TRACE("Hard link");
        fs::remove(timePath);
        std::error_code ec;
        fs::create_hard_link(other, timePath, ec);
        ASSERT_FALSE(ec) << ec.message();
        ASSERT_TRUE(m_cache->Get("key"));
        EXPECT_EQ(ReadFile(other), "other file");
    }
    {
        SCOPED_TRACE("Symbolic link, where creating one needs no privileges");
        fs::remove(timePath);
        std::error_code ec;
        fs::create_symlink(other, timePath, ec);
        if (!ec) {
            ASSERT_TRUE(m_cache->Get("key"));
            EXPECT_EQ(ReadFile(other), "other file");
        }
    }
}

//------------------------------------------------------------------------------
// Compilation and publication
//------------------------------------------------------------------------------

TEST_F(UnitTestModelCache, UnwritableCacheFailsWithoutCompiling)
{
    WriteFile(m_baseDir / "file", "");
    const auto cache = ModelCache::Create(m_baseDir / "file" / "models");
    ASSERT_TRUE(cache) << cache.error().message();
    EXPECT_EQ(ErrorOf(cache.value()->GetOrCreate("key", Compiler())), kCacheError);
    EXPECT_EQ(m_compiles, 0);
    EXPECT_EQ(ErrorOf(cache.value()->Get("key")), kCacheError);
    cache.value()->Cleanup();
}

TEST_F(UnitTestModelCache, FailedCompilesPublishNothing)
{
    {
        SCOPED_TRACE("Compile error");
        auto created = m_cache->GetOrCreate("key", [](const fs::path& dir) -> expected<void> {
            WriteFile(dir / "model.bin", "partial");
            return kModelInitError;
        });
        EXPECT_EQ(ErrorOf(created), kModelInitError);
    }
    {
        SCOPED_TRACE("No compiled files");
        EXPECT_EQ(ErrorOf(m_cache->GetOrCreate("key", Compiler({}))), kCacheError);
    }
    {
        SCOPED_TRACE("Staging directory removed during the compile, e.g. by Cleanup, and recreated by the compiler");
        auto created = m_cache->GetOrCreate("key", [](const fs::path& dir) -> expected<void> {
            fs::remove_all(dir);
            WriteFile(dir / "model.bin", "partial");
            return {};
        });
        EXPECT_EQ(ErrorOf(created), kCacheError);
    }
#if GTEST_HAS_EXCEPTIONS
    {
        SCOPED_TRACE("Exception");
        const auto compile = [](const fs::path& dir) -> expected<void> {
            WriteFile(dir / "model.bin", "partial");
            throw std::runtime_error("compile");
        };
        EXPECT_THROW(m_cache->GetOrCreate("key", compile), std::runtime_error);
    }
#endif
    {
        SCOPED_TRACE("Publishing fails after the compile: a file in place of the entries directory");
        int compiles = 0;
        auto created = m_cache->GetOrCreate("key", [&](const fs::path& dir) -> expected<void> {
            compiles++;
            WriteFile(dir / "model.bin", "compiled model");
            WriteFile(m_entriesDir, "");
            return {};
        });
        EXPECT_EQ(ErrorOf(created), kCacheError);
        EXPECT_EQ(compiles, 1);
        fs::remove(m_entriesDir);
    }
#if defined(MLVC_PLATFORM_WINCLASSIC)
    {
        SCOPED_TRACE("Compiled file name that isn't valid UTF-16, so the UTF-8 manifest can't list it");
        auto created = m_cache->GetOrCreate("key", [](const fs::path& dir) -> expected<void> {
            WriteFile(dir / std::wstring(L"model\xD800.bin"), "compiled model");
            return {};
        });
        EXPECT_EQ(ErrorOf(created), kCacheError);
    }
#endif
    EXPECT_FALSE(fs::exists(EntryDir("key")));
    EXPECT_EQ(CountItems(m_stagingDir), 0u);
}

TEST_F(UnitTestModelCache, OversizedManifestIsNeverPublished)
{
    constexpr auto kFileCount = size_t{ 6000 };
    auto prefix = std::u8string{};
    // UTF-8 keeps the manifest large without long native paths on Windows.
    for (auto i = 0; i < 60; ++i) {
        prefix += u8"\u65e5";
    }
    ASSERT_GT(prefix.size() * kFileCount, kManifestSizeLimit);

    const auto created = m_cache->GetOrCreate("key", [&](const fs::path& dir) -> expected<void> {
        ++m_compiles;
        for (auto i = size_t{}; i < kFileCount; ++i) {
            const auto suffix = std::to_string(i) + ".bin";
            auto name = prefix;
            name.append(suffix.begin(), suffix.end());
            WriteFile(dir / fs::path{ name }, "");
            if (HasFatalFailure()) {
                return kIoError;
            }
        }
        return {};
    });
    EXPECT_EQ(ErrorOf(created), kCacheError);
    EXPECT_EQ(m_compiles, 1);
    EXPECT_FALSE(fs::exists(EntryDir("key")));
    EXPECT_EQ(CountItems(m_stagingDir), 0u);

    const auto retry = m_cache->GetOrCreate("key", Compiler());
    ASSERT_TRUE(retry) << retry.error().message();
    EXPECT_EQ(m_compiles, 2);
    EXPECT_EQ(ReadFile(*retry / "model.bin"), "compiled model");
}

#if defined(MLVC_PLATFORM_WINCLASSIC)
TEST_F(UnitTestModelCache, PublishWaitsForBrieflyHeldFiles)
{
    int index = 0;
    // Without write sharing, flushing waits too; both modes block publication.
    for (const DWORD share : { DWORD{ FILE_SHARE_READ }, DWORD{ FILE_SHARE_READ | FILE_SHARE_WRITE } }) {
        SCOPED_TRACE(share);
        std::thread holder;
        const auto compile = [&](const fs::path& dir) -> expected<void> {
            WriteFile(dir / "model.bin", "compiled model");
            holder =
                HoldFile(dir / "model.bin", share, [] { std::this_thread::sleep_for(std::chrono::milliseconds(100)); });
            return {};
        };
        auto created = m_cache->GetOrCreate("key" + std::to_string(index++), compile);
        if (holder.joinable()) {
            holder.join();
        }
        ASSERT_TRUE(created) << created.error().message();
        EXPECT_EQ(ReadFile(*created / "model.bin"), "compiled model");
    }
}

TEST_F(UnitTestModelCache, PublishGivesUpOnFilesHeldTooLong)
{
    std::promise<void> release;
    std::thread holder;
    const auto compile = [&](const fs::path& dir) -> expected<void> {
        WriteFile(dir / "model.bin", "compiled model");
        holder = HoldFile(dir / "model.bin", FILE_SHARE_READ | FILE_SHARE_WRITE,
                          [released = release.get_future().share()] { released.wait(); });
        return {};
    };
    auto created = m_cache->GetOrCreate("key", compile);
    release.set_value();
    if (holder.joinable()) {
        holder.join();
    }
    EXPECT_EQ(ErrorOf(created), kCacheError);
    EXPECT_FALSE(fs::exists(EntryDir("key")));
    for (const auto& stagingDir : fs::directory_iterator(m_stagingDir)) {
        WriteTime(stagingDir.path(), SecondsAgo(std::chrono::hours(2)));
    }
    m_cache->Cleanup();
    EXPECT_EQ(CountItems(m_stagingDir), 0u);
}

#endif

TEST_F(UnitTestModelCache, AnotherCallersEntryWins)
{
    // Another process publishes the entry while this one compiles it
    auto created = m_cache->GetOrCreate("key", [this](const fs::path& dir) -> expected<void> {
        WriteFile(dir / "model.bin", "mine");
        const auto otherEntry = EntryDir("key");
        WriteFile(otherEntry / "model.bin", "theirs");
        WriteFile(otherEntry / kManifestFilename, ManifestFor("model.bin", 6));
        WriteTime(otherEntry, 1000);
        return {};
    });
    ASSERT_TRUE(created) << created.error().message();
    EXPECT_EQ(ReadFile(*created / "model.bin"), "theirs");
    EXPECT_TRUE(IsRecent(ReadTime(*created)));
    EXPECT_EQ(CountItems(m_stagingDir), 0u);
}

//------------------------------------------------------------------------------
// Concurrency and cache roots
//------------------------------------------------------------------------------

TEST_F(UnitTestModelCache, SameKeyInDifferentRootsKeepsSeparateEntries)
{
    const auto otherDir = m_baseDir / "other_models";
    const auto otherCache = ModelCache::Create(otherDir);
    ASSERT_TRUE(otherCache) << otherCache.error().message();
    const auto first = m_cache->GetOrCreate("key", Compiler({ { "model.bin", "first" } }));
    ASSERT_TRUE(first) << first.error().message();
    const auto second = otherCache.value()->GetOrCreate("key", Compiler({ { "model.bin", "second" } }));
    ASSERT_TRUE(second) << second.error().message();
    EXPECT_EQ(*first, EntryDir("key"));
    EXPECT_EQ(*second, otherDir / EntryDir("key").filename());
    EXPECT_EQ(ReadFile(*first / "model.bin"), "first");
    EXPECT_EQ(ReadFile(*second / "model.bin"), "second");
    EXPECT_EQ(m_compiles, 2);
}

TEST_F(UnitTestModelCache, ConcurrentCallersCompileOnce)
{
    // The key's lock is shared by every cache in the process
    const auto otherCache = ModelCache::Create(m_entriesDir);
    ASSERT_TRUE(otherCache) << otherCache.error().message();
    const auto results = GetOrCreateConcurrently({ m_cache, otherCache.value() });
    for (const auto& result : results) {
        ASSERT_TRUE(result) << result.error().message();
        EXPECT_EQ(*result, EntryDir("key"));
    }
    EXPECT_EQ(m_compiles, 1);
}

TEST_F(UnitTestModelCache, AliasedRootsCompileOnce)
{
    ASSERT_TRUE(fs::create_directories(m_entriesDir));
    const auto aliasDir = m_baseDir / "models_alias";
    if (!CreateDirectoryLink(m_entriesDir, aliasDir)) {
        GTEST_SKIP() << "Failed to create a directory link";
    }
    ASSERT_TRUE(fs::equivalent(m_entriesDir, aliasDir));
    const auto aliasCache = ModelCache::Create(aliasDir);
    ASSERT_TRUE(aliasCache) << aliasCache.error().message();
    auto caches = std::vector<std::shared_ptr<ModelCache>>{ m_cache, aliasCache.value() };

    const auto caseAlias = m_baseDir / "MODELS";
    if (fs::exists(caseAlias)) {
        ASSERT_TRUE(fs::equivalent(m_entriesDir, caseAlias));
        auto cache = ModelCache::Create(caseAlias);
        ASSERT_TRUE(cache) << cache.error().message();
        caches.push_back(std::move(cache.value()));
    }

    const auto results = GetOrCreateConcurrently(caches);
    for (const auto& result : results) {
        ASSERT_TRUE(result) << result.error().message();
        EXPECT_TRUE(fs::equivalent(*result, EntryDir("key")));
    }
    EXPECT_EQ(m_compiles, 1);
}

TEST_F(UnitTestModelCache, DifferentKeysCompileInParallel)
{
    std::promise<void> firstStarted;
    std::promise<void> secondDone;
    auto firstStartedFuture = firstStarted.get_future();
    auto secondDoneFuture = secondDone.get_future();
    std::thread first([&] {
        auto created = m_cache->GetOrCreate("first", [&](const fs::path& dir) -> expected<void> {
            firstStarted.set_value();
            // Times out if the other key's compile waits for this one
            EXPECT_EQ(secondDoneFuture.wait_for(std::chrono::seconds(10)), std::future_status::ready);
            WriteFile(dir / "model.bin", "first");
            return {};
        });
        EXPECT_TRUE(created);
    });
    EXPECT_EQ(firstStartedFuture.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    auto second = m_cache->GetOrCreate("second", Compiler());
    secondDone.set_value();
    first.join();
    ASSERT_TRUE(second) << second.error().message();
}

//------------------------------------------------------------------------------
// Cleanup and retention
//------------------------------------------------------------------------------

TEST_F(UnitTestModelCache, CreatingRemovesOnlyLeftovers)
{
    const auto old = AddEntry("old", DaysAgo(60));
    const auto olderVersion = AddOlderEntry("0123abcd_0123456789abcdef", DaysAgo(60));
    const auto untimed = AddEntry("untimed", DaysAgo(0));
    fs::remove(untimed / kAccessTimeFilename);
    const auto abandoned = AddStagingDir("abandoned", SecondsAgo(std::chrono::hours(2)));
    const auto compiling = AddStagingDir("compiling", SecondsAgo(std::chrono::minutes(1)));
    WriteFile(m_trashDir / "leftover" / "model.bin", "partial");

    // Preserve old entries until startup has a chance to refresh them.
    const auto cache = ModelCache::Create(m_entriesDir);
    ASSERT_TRUE(cache) << cache.error().message();
    EXPECT_TRUE(fs::exists(old));
    EXPECT_TRUE(fs::exists(olderVersion));
    EXPECT_TRUE(fs::exists(compiling));
    EXPECT_FALSE(fs::exists(untimed));
    EXPECT_FALSE(fs::exists(abandoned));
    EXPECT_EQ(CountItems(m_trashDir), 0u);

    ASSERT_TRUE(cache.value()->Get("old"));
    cache.value()->Cleanup();
    EXPECT_TRUE(fs::exists(old));
    EXPECT_FALSE(fs::exists(olderVersion));
}

TEST_F(UnitTestModelCache, OnlyCleanupRemovesDirectories)
{
    const auto expired = AddEntry("expired", DaysAgo(30));
    const auto abandoned = AddStagingDir("abandoned", DaysAgo(1));
    ASSERT_TRUE(m_cache->GetOrCreate("other", Compiler()));
    ASSERT_TRUE(m_cache->Get("other"));
    EXPECT_EQ(ErrorOf(m_cache->Get("missing")), kCacheError);
    EXPECT_TRUE(fs::exists(expired));
    EXPECT_TRUE(fs::exists(abandoned));
    m_cache->Cleanup();
    EXPECT_FALSE(fs::exists(expired));
    EXPECT_FALSE(fs::exists(abandoned));
}

TEST_F(UnitTestModelCache, CleanupRemovesExpiredAndUntimedEntries)
{
    const auto fresh = AddEntry("fresh", DaysAgo(13));
    const auto expired = AddEntry("expired", DaysAgo(14));
    const auto untimed = AddEntry("untimed", DaysAgo(0));
    fs::remove(untimed / kAccessTimeFilename);
    const auto unreadable = AddEntry("unreadable", DaysAgo(0));
    WriteFile(unreadable / kAccessTimeFilename, "abc");
    // A garbage time far in the past must not overflow into the future
    const auto garbage = AddEntry("garbage", std::numeric_limits<int64_t>::min());
    // A time in the future, e.g. after the clock was set back, isn't expired
    const auto future = AddEntry("future", DaysAgo(-1000));
    // Older versions' entries, which have no manifest, are kept until they expire
    const auto oldFresh = AddOlderEntry("0123abcd_0123456789abcdef", DaysAgo(1));
    const auto oldExpired = AddOlderEntry("4567abcd_0123456789abcdef", DaysAgo(59));

    m_cache->Cleanup();
    EXPECT_TRUE(fs::exists(fresh));
    EXPECT_TRUE(fs::exists(future));
    EXPECT_TRUE(fs::exists(oldFresh));
    EXPECT_FALSE(fs::exists(expired));
    EXPECT_FALSE(fs::exists(untimed));
    EXPECT_FALSE(fs::exists(unreadable));
    EXPECT_FALSE(fs::exists(garbage));
    EXPECT_FALSE(fs::exists(oldExpired));
    EXPECT_EQ(CountItems(m_trashDir), 0u);
    ASSERT_TRUE(m_cache->Get("fresh"));
}

TEST_F(UnitTestModelCache, CleanupRemovesAbandonedStagingDirectories)
{
    const auto abandoned = AddStagingDir("abandoned", SecondsAgo(std::chrono::hours(1)));
    const auto compiling = AddStagingDir("compiling", SecondsAgo(std::chrono::minutes(59)));
    // A time in the future comes from a clock that was wrong, so it counts by its distance from now
    const auto ahead = AddStagingDir("ahead", SecondsAgo(-std::chrono::hours(2)));
    const auto slightlyAhead = AddStagingDir("slightly_ahead", SecondsAgo(-std::chrono::minutes(59)));
    const auto garbagePast = AddStagingDir("garbage_past", std::numeric_limits<int64_t>::min());
    const auto garbageFuture = AddStagingDir("garbage_future", std::numeric_limits<int64_t>::max());
    m_cache->Cleanup();
    EXPECT_FALSE(fs::exists(abandoned));
    EXPECT_TRUE(fs::exists(compiling));
    EXPECT_FALSE(fs::exists(ahead));
    EXPECT_TRUE(fs::exists(slightlyAhead));
    EXPECT_FALSE(fs::exists(garbagePast));
    EXPECT_FALSE(fs::exists(garbageFuture));

    // Before the timestamp file exists, staging age falls back to directory mtime.
    const auto created = m_stagingDir / "created";
    for (const auto minutes : { 0, 59, -59, 120, -120 }) {
        SCOPED_TRACE("Modified " + std::to_string(minutes) + " minutes from now");
        fs::create_directories(created);
        std::error_code ec;
        fs::last_write_time(created, fs::file_time_type::clock::now() + std::chrono::minutes(minutes), ec);
        ASSERT_FALSE(ec) << ec.message();
        m_cache->Cleanup();
        EXPECT_EQ(fs::exists(created), std::abs(minutes) < 60);
    }
    EXPECT_EQ(CountItems(m_trashDir), 0u);
}

TEST_F(UnitTestModelCache, CleanupEmptiesTheTrash)
{
    {
        SCOPED_TRACE("Nothing cached: no folders are created");
        m_cache->Cleanup();
        EXPECT_EQ(CountItems(m_baseDir), 0u);
    }
    {
        SCOPED_TRACE("Left in the trash by an interrupted deletion");
        WriteFile(m_trashDir / "leftover" / "model.bin", "partial");
        m_cache->Cleanup();
        EXPECT_EQ(CountItems(m_trashDir), 0u);
    }
}

TEST_F(UnitTestModelCache, CleanupEnforcesSizeLimit)
{
    // Each entry holds about 1 KB: a 1000-byte file, its manifest and its last-access time
    const auto model = std::string(1000, 'x');
    const auto oldest = AddEntry("oldest", DaysAgo(3), model);
    const auto older = AddEntry("older", DaysAgo(2), model);
    const auto newest = AddEntry("newest", DaysAgo(1), model);
    // A time in the future comes from a clock that was wrong, so it counts by its distance from now
    const auto future = AddEntry("future", SecondsAgo(-std::chrono::hours(60)), model);
    const auto invalid = AddEntry("invalid", DaysAgo(0), model);
    WriteFile(invalid / "model.bin", model + "x");
    const auto oldVersion = AddOlderEntry("0123abcd_0123456789abcdef", DaysAgo(0), model);
    const auto recent = AddEntry("recent", SecondsAgo(std::chrono::minutes(30)), model);
    // Files that aren't entries don't count
    const auto other = m_entriesDir / "other.bin";
    WriteFile(other, std::string(100000, 'x'));

    m_cache->Cleanup(ModelCache::kRetention, 3500);  // Room for three entries
    EXPECT_FALSE(fs::exists(invalid));
    EXPECT_FALSE(fs::exists(oldVersion));
    EXPECT_FALSE(fs::exists(oldest));
    EXPECT_FALSE(fs::exists(future));
    EXPECT_TRUE(fs::exists(older));
    EXPECT_TRUE(fs::exists(newest));
    EXPECT_TRUE(fs::exists(recent));
    EXPECT_TRUE(fs::exists(other));

    m_cache->Cleanup(ModelCache::kRetention, 0);
    EXPECT_FALSE(fs::exists(older));
    EXPECT_FALSE(fs::exists(newest));
    EXPECT_FALSE(fs::exists(recent));
    EXPECT_TRUE(fs::exists(other));
}

TEST_F(UnitTestModelCache, CleanupEvictsRecentEntriesAboveSizeLimit)
{
    const auto model = std::string(1000, 'x');
    const auto ages = std::vector{ std::chrono::minutes{ 30 }, std::chrono::minutes{ 0 }, std::chrono::minutes{ -30 } };
    for (const auto age : ages) {
        SCOPED_TRACE(age.count());
        const auto entry = AddEntry("recent", SecondsAgo(age), model);
        const auto entrySize = fs::file_size(entry / "model.bin") + fs::file_size(entry / kManifestFilename)
                               + fs::file_size(entry / kAccessTimeFilename);

        m_cache->Cleanup(ModelCache::kRetention, entrySize);
        EXPECT_TRUE(fs::exists(entry));

        m_cache->Cleanup(ModelCache::kRetention, entrySize - 1);
        EXPECT_FALSE(fs::exists(entry));
        EXPECT_EQ(CountItems(m_entriesDir), 0u);
    }
}

TEST_F(UnitTestModelCache, CleanupNeverFollowsLinks)
{
    // Untimed and old enough to be removed from any of the cache's folders
    const auto victim = m_baseDir / "victim";
    WriteFile(victim / "untimed" / "data.bin", "data");
    std::error_code ec;
    fs::last_write_time(victim / "untimed", fs::file_time_type::clock::now() - std::chrono::hours(2), ec);
    ASSERT_FALSE(ec) << ec.message();
    for (const auto* folder : { "models", "models.staging", "models.trash" }) {
        SCOPED_TRACE(std::string("A link in place of ") + folder + ": nothing is removed through it");
        const auto cacheDir = m_baseDir / (std::string("link_as_") + folder);
        ASSERT_TRUE(fs::create_directories(cacheDir));
        if (!CreateDirectoryLink(victim, cacheDir / folder)) {
            GTEST_SKIP() << "Failed to create a directory link";
        }
        // A leftover in another folder, removed unless the trash is the link, as removing goes through the trash
        const auto leftover = cacheDir / (folder == std::string("models") ? "models.staging" : "models") / "untimed";
        WriteFile(leftover / "data.bin", "data");
        fs::last_write_time(leftover, fs::file_time_type::clock::now() - std::chrono::hours(2), ec);
        ASSERT_FALSE(ec) << ec.message();
        const auto cache = ModelCache::Create(cacheDir / "models");
        ASSERT_TRUE(cache) << cache.error().message();
        cache.value()->Cleanup();
        EXPECT_TRUE(fs::exists(victim / "untimed" / "data.bin"));
        EXPECT_EQ(fs::exists(leftover), folder == std::string("models.trash"));
    }
    {
        SCOPED_TRACE("Links in the entries directory are left alone");
        ASSERT_TRUE(fs::create_directories(m_entriesDir));
        ASSERT_TRUE(CreateDirectoryLink(victim, m_entriesDir / "link"));
        m_cache->Cleanup();
        EXPECT_TRUE(fs::exists(m_entriesDir / "link"));
        EXPECT_TRUE(fs::exists(victim / "untimed" / "data.bin"));
    }
    {
        SCOPED_TRACE("Links in a removed entry are removed, not followed");
        const auto entry = m_entriesDir / "untimed_entry";
        ASSERT_TRUE(fs::create_directories(entry));
        ASSERT_TRUE(CreateDirectoryLink(victim, entry / "link"));
        m_cache->Cleanup();
        EXPECT_FALSE(fs::exists(entry));
        EXPECT_TRUE(fs::exists(victim / "untimed" / "data.bin"));
    }
}

#if defined(MLVC_PLATFORM_WINCLASSIC)
TEST_F(UnitTestModelCache, CleanupSkipsEntriesInUse)
{
    // Expired entries listed before and after the one in use, and an unexpired one that only the size limit removes
    const auto first = AddEntry("a_expired", DaysAgo(30));
    const auto inUse = AddEntry("in_use", DaysAgo(30));
    const auto last = AddEntry("z_expired", DaysAgo(30));
    const auto fresh = AddEntry("fresh", DaysAgo(1));
    {
        // Windows only: a directory can't be renamed while one of its files is open
        const auto file = std::ifstream{ inUse / "model.bin" };
        ASSERT_TRUE(file);
        m_cache->Cleanup();
        EXPECT_FALSE(fs::exists(first));
        EXPECT_FALSE(fs::exists(last));
        EXPECT_TRUE(fs::exists(fresh));
        WriteFile(m_trashDir / "leftover" / "model.bin", "partial");
        m_cache->Cleanup(ModelCache::kRetention, 0);
        EXPECT_FALSE(fs::exists(fresh));
        EXPECT_EQ(CountItems(m_trashDir), 0u);
        // Left complete, not partially deleted
        for (const auto* name : { "model.bin", kManifestFilename, kAccessTimeFilename }) {
            EXPECT_TRUE(fs::exists(inUse / name)) << name;
        }
    }
    m_cache->Cleanup();
    EXPECT_FALSE(fs::exists(inUse));
}

#endif

TEST_F(UnitTestModelCache, ConcurrentCleanupKeepsEntriesComplete)
{
    const QuietLog quiet;
    const auto retention = std::chrono::seconds(0);
    std::atomic<bool> stop{ false };
    std::thread cleaner([&] {
        while (!stop) {
            m_cache->Cleanup(retention);
        }
    });
    // Require repeated eviction/recompilation, not just the initial three compiles.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::vector<std::thread> callers;
    for (int t = 0; t < 4; t++) {
        callers.emplace_back([&, t] {
            for (int i = t; m_compiles < 9 && std::chrono::steady_clock::now() < deadline; i++) {
                const auto key = "key" + std::to_string(i % 3);
                const auto entry = i % 4 == 0 ? m_cache->Get(key) : m_cache->GetOrCreate(key, Compiler());
                if (!entry) {
                    EXPECT_EQ(entry.error(), kCacheError) << key;
                }
            }
        });
    }
    for (auto& caller : callers) {
        caller.join();
    }
    stop = true;
    cleaner.join();

    EXPECT_GE(m_compiles, 9) << "Cleanup never removed the entries";
    for (const auto& item : fs::directory_iterator(m_entriesDir)) {
        EXPECT_EQ(ReadFile(item.path() / "model.bin"), "compiled model") << item.path();
        EXPECT_EQ(ReadFile(item.path() / kManifestFilename), ManifestFor("model.bin", 14));
    }
    EXPECT_EQ(CountItems(m_stagingDir), 0u);
}
