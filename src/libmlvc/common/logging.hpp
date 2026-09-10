// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/types.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <source_location>
#include <string_view>
#include <utility>

#define MLVC_LOG_IMPL(level, fmt, ...)                                                               \
    {                                                                                                \
        libmlvc::Logger::Instance().Log(level, std::source_location::current(), fmt, ##__VA_ARGS__); \
    }
#define MLVC_LOG_DEBUG(...) MLVC_LOG_IMPL(libmlvc::LogLevel::Debug, ##__VA_ARGS__)
#define MLVC_LOG_INFO(...) MLVC_LOG_IMPL(libmlvc::LogLevel::Info, ##__VA_ARGS__)
#define MLVC_LOG_WARN(...) MLVC_LOG_IMPL(libmlvc::LogLevel::Warn, ##__VA_ARGS__)
#define MLVC_LOG_ERROR(...) MLVC_LOG_IMPL(libmlvc::LogLevel::Error, ##__VA_ARGS__)
#define MLVC_LOG_ABORT(...)                                   \
    {                                                         \
        MLVC_LOG_IMPL(libmlvc::LogLevel::Fatal, __VA_ARGS__); \
        std::fflush(stderr);                                  \
        std::abort();                                         \
    }

namespace libmlvc {

void DefaultLogHandler(LogLevel level, const std::source_location& loc, std::string_view msg) noexcept;

class Logger {
public:
    static Logger& Instance() noexcept;
    void SetLogLevel(LogLevel level) noexcept;
    void SetHandler(LogHandler handler) noexcept;
    void LogMsg(LogLevel level, std::string_view msg,
                const std::source_location& loc = std::source_location::current()) noexcept;

    void Log(LogLevel level, std::source_location loc, const char* msg) noexcept
    {
        if (level < m_logLevel.load(std::memory_order_relaxed)) return;
        LogMsg(level, msg, loc);
    }

    template <typename... Args>
    void Log(LogLevel level, std::source_location loc, const char* fmt, Args&&... args) noexcept
    {
        if (level < m_logLevel.load(std::memory_order_relaxed)) return;

        // TODO: use safer std::format once available on all platforms
        static constexpr size_t kMaxLogMsgSize = 512;
        char msg[kMaxLogMsgSize];
        std::snprintf(msg, kMaxLogMsgSize, fmt, std::forward<Args>(args)...);
        LogMsg(level, msg, loc);
    }

private:
    std::atomic<LogLevel> m_logLevel{ LogLevel::Info };
    std::mutex m_logMutex;
    LogHandler m_logHandler{ DefaultLogHandler };
};

}  // namespace libmlvc
