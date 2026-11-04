// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/logging.hpp"

#if defined(_WIN32)
    #define NOMINMAX
    #include <windows.h>
#elif defined(__APPLE__)
    #include <pthread.h>
#endif

#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>

namespace libmlvc {

namespace {

std::string GetLocalTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf;
#if defined(_WIN32)
    localtime_s(&tmBuf, &timeNow);
#else
    localtime_r(&timeNow, &tmBuf);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tmBuf, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

inline uint64_t GetNativeThreadId()
{
#if defined(_WIN32)
    return static_cast<uint64_t>(::GetCurrentThreadId());
#elif defined(__APPLE__)
    uint64_t tid = 0;
    (void)pthread_threadid_np(nullptr, &tid);
    return tid;
#else
    std::thread::id id = std::this_thread::get_id();
    return std::hash<std::thread::id>{}(id);
#endif
}

}  // namespace

void DefaultLogHandler(LogLevel level, const std::source_location& loc, std::string_view msg) noexcept
{
    const auto timestamp = GetLocalTimestamp();
    const auto threadId = GetNativeThreadId();
    const char* basename = loc.file_name();
    for (const char* p = loc.file_name(); *p; ++p)
        if (*p == '/' || *p == '\\') basename = p + 1;

    std::fprintf(stderr, "%s [T#%llu][%5s][%s:%u]: %.*s\n", timestamp.c_str(),
                 static_cast<unsigned long long>(threadId), LogLevelToString(level), basename,
                 static_cast<unsigned>(loc.line()), static_cast<int>(msg.size()), msg.data());
}

std::string ToLogString(const std::filesystem::path& path) noexcept
{
#if defined(_WIN32)
    const std::wstring& wide = path.native();
    if (wide.empty()) return {};
    const int wideLen = static_cast<int>(wide.size());
    const int len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wideLen, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return "<unprintable path>";
    std::string result(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), wideLen, result.data(), len, nullptr, nullptr);
    return result;
#else
    return path.native();
#endif
}

Logger& Logger::Instance() noexcept
{
    static Logger instance;
    return instance;
}

void Logger::SetLogLevel(LogLevel level) noexcept
{
    m_logLevel.store(level, std::memory_order_relaxed);
}

void Logger::SetHandler(LogHandler handler) noexcept
{
    std::lock_guard<std::mutex> lock(m_logMutex);
    m_logHandler = std::move(handler);
}

void Logger::LogMsg(LogLevel level, std::string_view msg, const std::source_location& loc) noexcept
{
    std::lock_guard<std::mutex> lock(m_logMutex);
    if (m_logHandler) {
        m_logHandler(level, loc, msg);
    }
}

}  // namespace libmlvc
