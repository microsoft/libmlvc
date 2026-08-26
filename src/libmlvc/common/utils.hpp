// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/types.hpp>

#include <chrono>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace libmlvc {

class ManualOpTimer {
public:
    ManualOpTimer(const bool autostart = true) : m_timerStats(std::nullopt)
    {
        if (autostart) Start();
    }

    ManualOpTimer(OpTimerStats& timerStats, const bool autostart = true) : m_timerStats(std::ref(timerStats))
    {
        if (autostart) Start();
    }

    void Start() { m_start = std::chrono::steady_clock::now(); }
    double Stop()
    {
        const std::chrono::duration<double, std::milli> durationMs = std::chrono::steady_clock::now() - m_start;
        if (m_timerStats) {
            m_timerStats.value().get().Append(durationMs.count());
        }
        return durationMs.count();
    }

    const std::chrono::steady_clock::time_point& GetStartTime() const { return m_start; }

protected:
    std::chrono::steady_clock::time_point m_start;
    std::optional<std::reference_wrapper<OpTimerStats>> m_timerStats;
};

class ScopedOpTimer : public ManualOpTimer {
public:
    ScopedOpTimer(OpTimerStats& timerStats) : ManualOpTimer(timerStats, true) {}
    ~ScopedOpTimer() { ManualOpTimer::Stop(); }
    void Start() = delete;
    double Stop() = delete;
};

class IntervalOpTimer {
public:
    IntervalOpTimer(OpTimerStats& timerStats) : m_timerStats(timerStats) {}

    double Tick()
    {
        auto now = std::chrono::steady_clock::now();
        double intervalMs = 0.0;
        if (m_started) {
            intervalMs = std::chrono::duration<double, std::milli>(now - m_lastTick).count();
            m_timerStats.Append(intervalMs);
        }
        m_lastTick = now;
        m_started = true;
        return intervalMs;
    }

private:
    OpTimerStats& m_timerStats;
    std::chrono::steady_clock::time_point m_lastTick{};
    bool m_started{ false };
};

inline std::string TimerStats2Str(const OpTimerStats& s)
{
    if (s.Count() == 0) return "n/a";
    char buf[80];
    std::snprintf(buf, sizeof(buf), "%.2f/%.2f/%.2f/%.2f (n=%d)", s.AverageMs(), s.StdDevMs(), s.MinMs(), s.MaxMs(),
                  s.Count());
    return buf;
}

inline std::string TimerStats2StrShort(const OpTimerStats& s, const bool showCount = false)
{
    if (s.Count() == 0) return "n/a";
    char buf[80];
    if (showCount) {
        std::snprintf(buf, sizeof(buf), "%.2f/%.2f (n=%d)", s.AverageMs(), s.StdDevMs(), s.Count());
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f/%.2f", s.AverageMs(), s.StdDevMs());
    }
    return buf;
}

// Compare two dot-separated numeric version strings (e.g. "32.0.100.4778").
// Returns true if `actual` >= `required`. Non-numeric or missing components are treated as 0.
bool VersionAtLeast(std::string_view actual, std::string_view required);

}  // namespace libmlvc
