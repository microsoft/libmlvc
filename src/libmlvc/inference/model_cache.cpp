// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/inference/model_cache.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/utils.hpp"

#include <libmlvc/error_codes.hpp>

#include <boost/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(MLVC_PLATFORM_WINCLASSIC)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <cerrno>
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace libmlvc {

namespace fs = std::filesystem;

namespace {

// Antivirus and indexers can briefly hold new files, making flushes and renames fail
constexpr auto kLockedFileRetryTime = std::chrono::seconds(1);
constexpr auto kLockedFileRetryInterval = std::chrono::milliseconds(50);

// Creation locks shared by normalized key across cache roots.
// All cache operations, including background work, must finish before static teardown destroys these locks.
std::mutex g_registryMutex;
std::map<std::string, std::mutex> g_keyMutexes;

std::mutex& KeyMutex(const std::string& key)
{
    std::lock_guard lock(g_registryMutex);
    return g_keyMutexes[key];
}

// ----------------------------------------------------------------
// Filesystem helpers
// ----------------------------------------------------------------

bool IsAbsent(const fs::path& path)
{
    std::error_code ec;
    return !fs::exists(path, ec) && !ec;
}

std::vector<fs::path> ListDirectories(const fs::path& dir)
{
    std::vector<fs::path> dirs;
    std::error_code ec;
    // Skips symbolic links and junctions, here and in the folder, as they could point anywhere
    if (const auto type = fs::symlink_status(dir, ec).type(); type != fs::file_type::directory) {
        if (type != fs::file_type::not_found) {
            MLVC_LOG_WARN("Skipping model cache directory %s: it is a link or not a directory",
                          ToLogString(dir.filename()).c_str());
        }
        return dirs;
    }
    for (fs::directory_iterator it(dir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        std::error_code typeEc;
        if (it->symlink_status(typeEc).type() == fs::file_type::directory) {
            dirs.push_back(it->path());
        }
    }
    return dirs;
}

uintmax_t DirectorySize(const fs::path& dir)
{
    uintmax_t size = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code fileEc;
        if (it->is_regular_file(fileEc)) {
            const auto fileSize = it->file_size(fileEc);
            if (!fileEc) {
                size += fileSize;
            }
        }
    }
    return size;
}

expected<void> FlushToDisk(const fs::path& path)
{
#if defined(MLVC_PLATFORM_WINCLASSIC)
    const auto deadline = std::chrono::steady_clock::now() + kLockedFileRetryTime;
    HANDLE file = INVALID_HANDLE_VALUE;
    while ((file = ::CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr))
           == INVALID_HANDLE_VALUE) {
        const auto error = ::GetLastError();
        if (error != ERROR_SHARING_VIOLATION || std::chrono::steady_clock::now() >= deadline) {
            return std::error_code(static_cast<int>(error), std::system_category());
        }
        std::this_thread::sleep_for(kLockedFileRetryInterval);
    }
    const bool flushed = ::FlushFileBuffers(file) != FALSE;
    const auto error = ::GetLastError();
    ::CloseHandle(file);
    if (!flushed) {
        return std::error_code(static_cast<int>(error), std::system_category());
    }
    return {};
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return std::error_code(errno, std::generic_category());
    }
    #if defined(MLVC_PLATFORM_APPLE)
    // fsync() leaves the data in the drive's cache on Apple platforms
    const bool flushed = ::fcntl(fd, F_FULLFSYNC) == 0 || ::fsync(fd) == 0;
    #else
    const bool flushed = ::fsync(fd) == 0;
    #endif
    const int error = errno;
    ::close(fd);
    if (!flushed) {
        return std::error_code(error, std::generic_category());
    }
    return {};
#endif
}

expected<void> FlushFiles(const fs::path& dir)
{
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        const bool isFile = it->is_regular_file(ec);
        if (ec) {
            return ec;
        }
        if (isFile) {
            if (auto ret = FlushToDisk(it->path()); !ret) {
                return ret;
            }
        }
    }
    if (ec) {
        return ec;
    }
    return {};
}

// ----------------------------------------------------------------
// AccessTimeFile
// ----------------------------------------------------------------

class AccessTimeFile {
public:
    static constexpr auto kFilename = "last_access.time";  // Shared with other MLVC versions
    static_assert(std::endian::native == std::endian::little);

    explicit AccessTimeFile(const fs::path& dir) : m_path(dir / kFilename) {}

    expected<void> Create() const { return Write(std::ios::trunc); }

    std::optional<int64_t> Read() const
    {
        std::ifstream file(m_path, std::ios::binary);
        std::array<char, sizeof(int64_t)> bytes{};
        if (!file.read(bytes.data(), bytes.size())) {
            return std::nullopt;
        }
        return std::bit_cast<int64_t>(bytes);
    }

    expected<void> Update() const
    {
        // Never creates the file, so it can't recreate a removed entry, and never writes through a link
        std::error_code ec;
        if (fs::symlink_status(m_path, ec).type() != fs::file_type::regular || fs::hard_link_count(m_path, ec) != 1) {
            return std::make_error_code(std::errc::operation_not_permitted);
        }
        return Write(std::ios::in);
    }

    static int64_t Now()
    {
        return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    static uint64_t DistanceFromNow(std::optional<int64_t> time, int64_t now)
    {
        if (!time) {
            return std::numeric_limits<uint64_t>::max();
        }
        // The difference of any two int64_t values fits in uint64_t
        return *time < now ? static_cast<uint64_t>(now) - static_cast<uint64_t>(*time)
                           : static_cast<uint64_t>(*time) - static_cast<uint64_t>(now);
    }

private:
    fs::path m_path;

    expected<void> Write(std::ios::openmode mode) const
    {
        std::fstream file(m_path, std::ios::binary | std::ios::out | mode);
        const auto bytes = std::bit_cast<std::array<char, sizeof(int64_t)>>(Now());
        file.write(bytes.data(), bytes.size());
        file.close();
        if (file.fail()) {
            return make_error_code(Error::io_error);
        }
        return {};
    }
};

// ----------------------------------------------------------------
// Manifest
// ----------------------------------------------------------------

// Manifest versions add fields; incompatible changes require a new cache-key prefix.
class Manifest {
public:
    static constexpr auto kFilename = "mlvc_manifest.json";

    static expected<Manifest> List(const fs::path& dir)
    {
        Manifest manifest;
        std::error_code ec;
        for (fs::recursive_directory_iterator it(dir, ec); !ec && it != fs::recursive_directory_iterator();
             it.increment(ec)) {
            const bool isFile = it->is_regular_file(ec);
            if (ec) {
                break;
            }
            const auto relative = it->path().lexically_relative(dir);
            if (!isFile || relative == AccessTimeFile::kFilename || relative == kFilename) {
                continue;
            }
            // Not the listing's size, which can be stale on Windows
            const auto size = fs::file_size(it->path(), ec);
            if (ec) {
                break;
            }
            auto path = ToText(relative);
            if (!path) {
                return path.error();
            }
            manifest.m_files.push_back({ .path = std::move(*path), .size = size });
        }
        if (ec) {
            return ec;
        }
        std::ranges::sort(manifest.m_files, {}, &File::path);
        return manifest;
    }

    bool IsEmpty() const { return m_files.empty(); }

    expected<void> Write(const fs::path& dir) const
    {
        constexpr uint64_t kVersion = 1;
        boost::json::array files;
        for (const auto& file : m_files) {
            files.push_back(boost::json::object{ { "path", file.path }, { "size", file.size } });
        }
        const auto text =
            boost::json::serialize(boost::json::object{ { "version", kVersion }, { "files", std::move(files) } });
        if (!IsSizeValid(text.size())) {
            MLVC_LOG_ERROR("Model cache manifest is too large (%zu bytes)", text.size());
            return make_error_code(Error::model_init_error);
        }
        std::ofstream file(dir / kFilename, std::ios::binary | std::ios::trunc);
        file << text;
        file.close();
        if (file.fail()) {
            return make_error_code(Error::io_error);
        }
        return {};
    }

    static expected<void> Validate(const fs::path& dir)
    {
        const auto manifest = Read(dir);
        if (!manifest) {
            return manifest.error();
        }
        return manifest->Matches(dir);
    }

private:
    struct File {
        std::string path;  // Relative to the entry, in UTF-8 with '/' separators
        uint64_t size{};
    };
    std::vector<File> m_files;

    static bool IsSizeValid(uintmax_t size)
    {
        constexpr auto kMaxSize = uintmax_t{ 1 } << 20;
        return size <= kMaxSize;
    }

    static expected<Manifest> Read(const fs::path& dir)
    {
        const auto path = dir / kFilename;
        std::error_code ec;
        const auto size = fs::file_size(path, ec);
        if (ec) {
            return ec;
        }
        if (!IsSizeValid(size)) {
            return make_error_code(Error::json_parse_error);
        }
        std::string text(static_cast<size_t>(size), '\0');
        std::ifstream file(path, std::ios::binary);
        if (!file.read(text.data(), static_cast<std::streamsize>(text.size()))) {
            return make_error_code(Error::io_error);
        }

        const auto json = boost::json::parse(text, ec);
        if (ec) {
            return make_error_code(Error::json_parse_error);
        }
        const auto* version = json.find_pointer("/version", ec);
        const auto* files = json.find_pointer("/files", ec);
        const auto* fileList = files ? files->if_array() : nullptr;
        if (!version || ToUnsigned(*version).value_or(0) < 1 || !fileList || fileList->empty()) {
            return make_error_code(Error::json_parse_error);
        }
        Manifest manifest;
        for (const auto& item : *fileList) {
            const auto* itemPath = item.find_pointer("/path", ec);
            const auto* pathText = itemPath ? itemPath->if_string() : nullptr;
            const auto* itemSize = item.find_pointer("/size", ec);
            const auto sizeValue = itemSize ? ToUnsigned(*itemSize) : std::nullopt;
            if (!pathText || !sizeValue) {
                return make_error_code(Error::json_parse_error);
            }
            manifest.m_files.push_back({ .path = std::string(*pathText), .size = *sizeValue });
        }
        return manifest;
    }

    expected<void> Matches(const fs::path& dir) const
    {
        for (const auto& file : m_files) {
            const auto relative = FromText(file.path);
            if (!relative) {
                return relative.error();
            }
            if (relative->empty() || relative->has_root_path()
                || std::any_of(relative->begin(), relative->end(),
                               [](const fs::path& part) { return part == "." || part == ".."; })) {
                return std::make_error_code(std::errc::invalid_argument);
            }
            std::error_code ec;
            const auto path = dir / *relative;
            if (!fs::is_regular_file(path, ec)) {
                return ec ? ec : std::make_error_code(std::errc::invalid_argument);
            }
            const auto size = fs::file_size(path, ec);
            if (ec) {
                return ec;
            }
            if (size != file.size) {
                return make_error_code(Error::model_init_error);
            }
        }
        return {};
    }

    static expected<std::string> ToText(const fs::path& relative)
    {
#if defined(MLVC_PLATFORM_WINCLASSIC)
        // Fails instead of throwing, unlike path::u8string() for unpaired surrogates
        const auto wide = relative.generic_wstring();
        const auto wideSize = static_cast<int>(wide.size());
        const int size =
            ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), wideSize, nullptr, 0, nullptr, nullptr);
        if (size <= 0) {
            return std::make_error_code(std::errc::illegal_byte_sequence);
        }
        std::string text(static_cast<size_t>(size), '\0');
        ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), wideSize, text.data(), size, nullptr, nullptr);
        return text;
#else
        return relative.generic_string();
#endif
    }

    static expected<fs::path> FromText(std::string_view text)
    {
        if (text.find('\0') != std::string_view::npos) {
            return std::make_error_code(std::errc::invalid_argument);
        }
#if defined(MLVC_PLATFORM_WINCLASSIC)
        // Fails instead of throwing for invalid UTF-8, unlike fs::path(std::u8string)
        const auto size = static_cast<int>(text.size());
        const int wideSize = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), size, nullptr, 0);
        if (wideSize <= 0) {
            return std::make_error_code(std::errc::illegal_byte_sequence);
        }
        std::wstring wide(static_cast<size_t>(wideSize), L'\0');
        ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), size, wide.data(), wideSize);
        auto path = fs::path{ std::move(wide) };
        // Extended-length Windows paths do not translate '/' separators.
        path.make_preferred();
        return path;
#else
        return fs::path(text);
#endif
    }

    static std::optional<uint64_t> ToUnsigned(const boost::json::value& value)
    {
        if (const auto* number = value.if_uint64()) {
            return *number;
        }
        if (const auto* number = value.if_int64(); number && *number >= 0) {
            return static_cast<uint64_t>(*number);
        }
        return std::nullopt;
    }
};

}  // namespace

// ----------------------------------------------------------------
// ModelCache
// ----------------------------------------------------------------

expected<std::shared_ptr<ModelCache>> ModelCache::Create(const fs::path& entriesDir)
{
    if (entriesDir.empty()) {
        MLVC_LOG_ERROR("Model cache directory is empty");
        return make_error_code(Error::invalid_argument);
    }

    auto resolvedDir = entriesDir.lexically_normal();
    if (resolvedDir.is_relative()) {
        std::error_code ec;
        resolvedDir = fs::absolute(resolvedDir, ec);
        if (ec) {
            MLVC_LOG_ERROR("Failed to resolve model cache directory %s: %s:%d (%s)", ToLogString(entriesDir).c_str(),
                           ec.category().name(), ec.value(), ec.message().c_str());
            return make_error_code(Error::model_cache_error);
        }
        resolvedDir = resolvedDir.lexically_normal();
    }

    // Strip the trailing separator so suffixes create sibling directories.
    if (!resolvedDir.has_filename()) {
        resolvedDir = resolvedDir.parent_path();
    }
    return std::shared_ptr<ModelCache>(new ModelCache(std::move(resolvedDir)));
}

expected<fs::path> ModelCache::GetOrCreate(std::string_view key, const CompileFunction& compile) const
{
    return Acquire(key, &compile);
}

expected<fs::path> ModelCache::Get(std::string_view key) const
{
    return Acquire(key, nullptr);
}

void ModelCache::Cleanup(std::chrono::seconds retention, uintmax_t sizeLimit) const noexcept
{
    if (m_entriesDir.empty()) {
        return;
    }
    CleanLeftovers();
    ApplyRetention(retention, sizeLimit);
}

ModelCache::ModelCache(fs::path entriesDir) : m_entriesDir(std::move(entriesDir))
{
    // Older versions' cleanup would delete temporary directories inside the entries directory.
    m_stagingDir = m_entriesDir;
    m_stagingDir += ".staging";
    m_trashDir = m_entriesDir;
    m_trashDir += ".trash";
    CleanLeftovers();
}

expected<fs::path> ModelCache::Acquire(std::string_view key, const CompileFunction* compile) const
{
    const auto isKeyChar = [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '-';
    };
    if (m_entriesDir.empty() || key.empty() || !std::ranges::all_of(key, isKeyChar) || (compile && !*compile)) {
        MLVC_LOG_ERROR("Invalid model cache directory, key '%.*s' or compile function", static_cast<int>(key.size()),
                       key.data());
        return make_error_code(Error::invalid_argument);
    }

    constexpr auto kKeyPrefix = "v2_";
    auto name = std::string(kKeyPrefix) + std::string(key);
    // Case variants of a key must share the same entry and lock.
    for (auto& c : name) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    auto entryDir = m_entriesDir / name;

    const auto createIfMissing = [&]() -> expected<void> {
        std::lock_guard lock(KeyMutex(name));
        if (IsAbsent(entryDir)) {
            return CreateEntry(entryDir, *compile);
        }
        return {};
    };

    auto validationError = std::error_code{};
    const auto refresh = [&] {
        // Keep invalid entries eligible for retention cleanup.
        if (const auto valid = Manifest::Validate(entryDir); !valid) {
            validationError = valid.error();
            return false;
        }
        if (auto ret = AccessTimeFile(entryDir).Update(); !ret) {
            MLVC_LOG_WARN("Failed to update the last-access time of model cache entry %s: %s", name.c_str(),
                          ret.error().message().c_str());
        }
        // Revalidate in case cleanup raced with the refresh.
        const auto valid = Manifest::Validate(entryDir);
        if (!valid) {
            validationError = valid.error();
        }
        return static_cast<bool>(valid);
    };

    // 1. Compile the entry if it is missing
    if (compile) {
        if (auto ret = createIfMissing(); !ret) {
            return ret.error();
        }
    }

    // 2. Validate and refresh the entry before returning it
    if (refresh()) {
        return entryDir;
    }

    // 3. Retry once if concurrent cleanup removed the entry
    if (compile && IsAbsent(entryDir)) {
        if (auto ret = createIfMissing(); !ret) {
            return ret.error();
        }
        if (refresh()) {
            return entryDir;
        }
    }

    MLVC_LOG_ERROR("Model cache entry %s is %s: %s:%d (%s)", name.c_str(), IsAbsent(entryDir) ? "missing" : "invalid",
                   validationError.category().name(), validationError.value(), validationError.message().c_str());
    return make_error_code(Error::model_cache_error);
}

expected<void> ModelCache::CreateEntry(const fs::path& entryDir, const CompileFunction& compile) const
{
    const auto name = ToLogString(entryDir.filename());

    // 1. Create staging with cleanup on failure
    const auto stagingDir = m_stagingDir / UniqueName();
    std::error_code ec;
    if (!fs::create_directories(stagingDir, ec)) {
        MLVC_LOG_ERROR("Failed to create a staging directory for model cache entry %s: %s:%d (%s)", name.c_str(),
                       ec.category().name(), ec.value(), ec.message().c_str());
        return make_error_code(Error::model_cache_error);
    }
    auto stagingGuard = ScopeExit{ [&stagingDir]() noexcept {
        std::error_code ec;
        if (fs::remove_all(stagingDir, ec) == static_cast<uintmax_t>(-1)) {
            MLVC_LOG_WARN("Failed to delete model cache staging directory %s: %s",
                          ToLogString(stagingDir.filename()).c_str(), ec.message().c_str());
        }
    } };
    const auto accessTime = AccessTimeFile(stagingDir);

    // 2. Timestamp staging before compilation so cleanup can identify abandoned work
    if (auto ret = accessTime.Create(); !ret) {
        MLVC_LOG_ERROR("Failed to write the last-access time of model cache entry %s: %s:%d (%s)", name.c_str(),
                       ret.error().category().name(), ret.error().value(), ret.error().message().c_str());
        return make_error_code(Error::model_cache_error);
    }

    // 3. Compile the model
    MLVC_LOG_INFO("Compiling model cache entry %s", name.c_str());
    if (auto ret = compile(stagingDir); !ret) {
        MLVC_LOG_ERROR("Compilation callback for model cache entry %s failed: %s:%d (%s)", name.c_str(),
                       ret.error().category().name(), ret.error().value(), ret.error().message().c_str());
        return ret.error();
    }

    // 4. Detect staging removed by cleanup and only partly recreated by the compiler
    if (!accessTime.Read()) {
        MLVC_LOG_ERROR("The staging directory of model cache entry %s was removed during compilation", name.c_str());
        return make_error_code(Error::model_cache_error);
    }

    // 5. Write the compiled-file manifest
    auto manifest = Manifest::List(stagingDir);
    if (!manifest) {
        MLVC_LOG_ERROR("Failed to list the files of model cache entry %s: %s:%d (%s)", name.c_str(),
                       manifest.error().category().name(), manifest.error().value(), manifest.error().message().c_str());
        return make_error_code(Error::model_cache_error);
    }
    if (manifest->IsEmpty()) {
        MLVC_LOG_ERROR("Model cache entry %s has no compiled files", name.c_str());
        return make_error_code(Error::model_cache_error);
    }
    if (auto ret = manifest->Write(stagingDir); !ret) {
        MLVC_LOG_ERROR("Failed to write the manifest of model cache entry %s: %s:%d (%s)", name.c_str(),
                       ret.error().category().name(), ret.error().value(), ret.error().message().c_str());
        return make_error_code(Error::model_cache_error);
    }

    // 6. Flush file contents before publication
    if (auto ret = FlushFiles(stagingDir); !ret) {
        MLVC_LOG_ERROR("Failed to flush model cache entry %s to disk: %s:%d (%s)", name.c_str(),
                       ret.error().category().name(), ret.error().value(), ret.error().message().c_str());
        return make_error_code(Error::model_cache_error);
    }

    // 7. Publish, allowing a brief retry window for file locks
    fs::create_directories(m_entriesDir, ec);
    if (ec) {
        MLVC_LOG_ERROR("Failed to create the model cache directory: %s:%d (%s)", ec.category().name(), ec.value(),
                       ec.message().c_str());
        return make_error_code(Error::model_cache_error);
    }
    const auto deadline = std::chrono::steady_clock::now() + kLockedFileRetryTime;
    while (true) {
        fs::rename(stagingDir, entryDir, ec);
        if (!ec) {
            stagingGuard.Release();
            MLVC_LOG_INFO("Added model cache entry %s", name.c_str());
            return {};
        }
        if (!IsAbsent(entryDir)) {
            MLVC_LOG_INFO("Model cache entry %s was added by another caller first", name.c_str());
            return {};
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(kLockedFileRetryInterval);
    }
    MLVC_LOG_ERROR("Failed to add model cache entry %s: %s:%d (%s)", name.c_str(), ec.category().name(), ec.value(),
                   ec.message().c_str());
    return make_error_code(Error::model_cache_error);
}

void ModelCache::CleanLeftovers() const noexcept
{
    size_t removed = 0;
    for (const auto& dir : ListDirectories(m_entriesDir)) {
        if (!AccessTimeFile(dir).Read() && Remove(dir, std::nullopt)) {
            removed++;
        }
    }

    for (const auto& dir : ListDirectories(m_stagingDir)) {
        if (const auto time = AccessTimeFile(dir).Read(); IsStagingAbandoned(dir, time) && Remove(dir, time)) {
            removed++;
        }
    }

    for (const auto& dir : ListDirectories(m_trashDir)) {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    if (removed > 0) {
        MLVC_LOG_INFO("Model cache cleanup removed %zu leftovers", removed);
    }
}

void ModelCache::ApplyRetention(std::chrono::seconds retention, uintmax_t sizeLimit) const noexcept
{
    struct Entry {
        fs::path dir;
        std::optional<int64_t> time;
        uint64_t distance{};
        uintmax_t size{};
        bool valid{};
    };
    std::vector<Entry> entries;
    uintmax_t totalSize = 0;
    size_t removed = 0;
    const auto now = AccessTimeFile::Now();

    // Avoid subtracting potentially corrupt access timestamps.
    for (auto& dir : ListDirectories(m_entriesDir)) {
        const auto time = AccessTimeFile(dir).Read();
        if (time && *time <= now - retention.count() && Remove(dir, time)) {
            removed++;
            continue;
        }
        const auto size = DirectorySize(dir);
        totalSize += size;
        entries.push_back(
            { .dir = std::move(dir), .time = time, .distance = AccessTimeFile::DistanceFromNow(time, now), .size = size });
    }

    if (totalSize > sizeLimit) {
        for (auto& entry : entries) {
            entry.valid = static_cast<bool>(Manifest::Validate(entry.dir));
        }
        std::ranges::sort(entries, [](const Entry& a, const Entry& b) {
            if (a.valid != b.valid) {
                return !a.valid;
            }
            return a.distance > b.distance;
        });
        for (const auto& entry : entries) {
            if (totalSize <= sizeLimit) {
                break;
            }
            if (Remove(entry.dir, entry.time)) {
                totalSize -= entry.size;
                removed++;
            } else if (IsAbsent(entry.dir)) {
                totalSize -= entry.size;  // Removed by Cleanup in another process
            }
        }
        if (totalSize > sizeLimit) {
            MLVC_LOG_WARN("Model cache size %.1f MiB exceeds its limit of %.1f MiB",
                          static_cast<double>(totalSize) / (1 << 20), static_cast<double>(sizeLimit) / (1 << 20));
        }
    }

    if (removed > 0) {
        MLVC_LOG_INFO("Model cache cleanup removed %zu entries", removed);
    }
}

bool ModelCache::Remove(const fs::path& dir, std::optional<int64_t> chosenTime) const
{
    if (AccessTimeFile(dir).Read() != chosenTime) {
        return false;
    }
    std::error_code ec;
    fs::create_directories(m_trashDir, ec);
    if (!ec && fs::symlink_status(m_trashDir, ec).type() != fs::file_type::directory) {
        ec = std::make_error_code(std::errc::not_a_directory);
    }
    // Trashed first, so that it's never seen partially deleted
    const auto trashed = m_trashDir / UniqueName();
    if (!ec) {
        fs::rename(dir, trashed, ec);
    }
    if (ec) {
        if (!IsAbsent(dir)) {
            MLVC_LOG_WARN("Failed to remove model cache directory %s, it may be in use: %s",
                          ToLogString(dir.filename()).c_str(), ec.message().c_str());
        }
        return false;
    }
    MLVC_LOG_DEBUG("Removed model cache directory %s", ToLogString(dir.filename()).c_str());
    // Best effort: later cleanups empty the trash
    fs::remove_all(trashed, ec);
    return true;
}

std::string ModelCache::UniqueName()
{
    thread_local std::mt19937_64 random(std::random_device{}());
    return std::to_string(random());
}

bool ModelCache::IsStagingAbandoned(const fs::path& stagingDir, std::optional<int64_t> time)
{
    constexpr std::chrono::seconds kStagingMaxAge = std::chrono::hours(1);
    if (time) {
        return AccessTimeFile::DistanceFromNow(time, AccessTimeFile::Now())
               >= static_cast<uint64_t>(kStagingMaxAge.count());
    }
    std::error_code ec;
    const auto modified = fs::last_write_time(stagingDir, ec);
    const auto now = fs::file_time_type::clock::now();
    return !ec && (modified <= now - kStagingMaxAge || modified >= now + kStagingMaxAge);
}

}  // namespace libmlvc
