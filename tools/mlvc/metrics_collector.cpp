// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "metrics_collector.hpp"

#include <libmlvc/error_codes.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <iostream>
#include <limits>
#include <system_error>
#include <utility>

// clang-format off
#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <psapi.h>
#elif defined(__APPLE__)
    #include <mach/mach.h>
    #include <mach/mach_error.h>
    #include <sys/resource.h>
#endif
// clang-format on

namespace libmlvc {

ProcessMetrics ReadProcessMetrics()
{
    constexpr double bytesPerMb = 1024.0 * 1024.0;
    constexpr double unavailable = std::numeric_limits<double>::quiet_NaN();
    ProcessMetrics metrics{ .time = std::chrono::steady_clock::now(),
                            .cpuSeconds = unavailable,
                            .commitMb = unavailable,
                            .sharedCommitMb = unavailable,
                            .workingSetMb = unavailable };
#if defined(_WIN32)
    FILETIME creationTime{}, exitTime{}, kernelTime{}, userTime{};
    if (GetProcessTimes(GetCurrentProcess(), &creationTime, &exitTime, &kernelTime, &userTime)) {
        const auto toSeconds = [](const FILETIME& time) {
            return static_cast<double>((static_cast<uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime) / 1e7;
        };
        metrics.cpuSeconds = toSeconds(userTime) + toSeconds(kernelTime);
    } else {
        const auto error = GetLastError();
        std::cerr << "Warning: GetProcessTimes failed: " << std::system_category().message(static_cast<int>(error)) << '\n';
    }
    // Older Windows versions don't know the extended structure with the shared commit
    PROCESS_MEMORY_COUNTERS_EX2 counters{};
    auto* const basicCounters = reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters);
    const bool hasSharedCommit = GetProcessMemoryInfo(GetCurrentProcess(), basicCounters, sizeof(counters));
    if (hasSharedCommit || GetProcessMemoryInfo(GetCurrentProcess(), basicCounters, sizeof(PROCESS_MEMORY_COUNTERS_EX))) {
        metrics.commitMb = static_cast<double>(counters.PrivateUsage) / bytesPerMb;
        if (hasSharedCommit) metrics.sharedCommitMb = static_cast<double>(counters.SharedCommitUsage) / bytesPerMb;
        metrics.workingSetMb = static_cast<double>(counters.WorkingSetSize) / bytesPerMb;
    } else {
        const auto error = GetLastError();
        std::cerr << "Warning: GetProcessMemoryInfo failed: " << std::system_category().message(static_cast<int>(error))
                  << '\n';
    }
#elif defined(__APPLE__)
    rusage resourceUsage{};
    if (getrusage(RUSAGE_SELF, &resourceUsage) == 0) {
        const auto toSeconds = [](const timeval& time) { return static_cast<double>(time.tv_sec) + time.tv_usec / 1e6; };
        metrics.cpuSeconds = toSeconds(resourceUsage.ru_utime) + toSeconds(resourceUsage.ru_stime);
    } else {
        const auto error = errno;
        std::cerr << "Warning: getrusage failed: " << std::generic_category().message(error) << '\n';
    }
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const auto result = task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count);
    if (result == KERN_SUCCESS) {
        if (count >= TASK_VM_INFO_REV1_COUNT) metrics.commitMb = static_cast<double>(info.phys_footprint) / bytesPerMb;
        metrics.workingSetMb = static_cast<double>(info.resident_size) / bytesPerMb;
    } else {
        std::cerr << "Warning: task_info(TASK_VM_INFO) failed: " << mach_error_string(result) << '\n';
    }
#else
    #error "Unsupported platform"
#endif
    return metrics;
}

MetricsCollector::~MetricsCollector()
{
    Stop();
}

expected<void> MetricsCollector::Start(std::chrono::milliseconds interval)
{
    if (interval <= std::chrono::milliseconds::zero()) {
        std::cerr << "Error: Metrics collection interval must be positive\n";
        return make_error_code(Error::invalid_argument);
    }
    const std::lock_guard lock(m_mutex);
    if (m_thread.joinable() || m_stopping) {
        std::cerr << "Error: Metrics collector is already running or stopped\n";
        return make_error_code(Error::invalid_argument);
    }
    // Reserve up front to reduce allocation noise in the profile
    m_samples.reserve(static_cast<size_t>(std::chrono::minutes(10) / interval));
    m_thread = std::thread([this, interval] { Run(interval); });
    return {};
}

expected<void> MetricsCollector::Record(std::string phase, std::string event)
{
    return Add(std::move(phase), std::move(event), false);
}

expected<void> MetricsCollector::RecordWait(std::string phase, std::string event)
{
    return Add(std::move(phase), std::move(event), true);
}

expected<void> MetricsCollector::Add(std::string phase, std::string event, bool waiting)
{
    if (phase.empty()) {
        std::cerr << "Error: Metrics event phase must not be empty\n";
        return make_error_code(Error::invalid_argument);
    }
    const std::lock_guard lock(m_mutex);
    if (m_stopping || !m_thread.joinable()) {
        std::cerr << "Error: Cannot record metrics while the collector is not running\n";
        return make_error_code(Error::not_initialized);
    }
    m_samples.push_back({ std::move(phase), std::move(event), ReadProcessMetrics(), waiting });
    return {};
}

void MetricsCollector::Stop()
{
    {
        const std::lock_guard lock(m_mutex);
        if (m_thread.joinable() && !m_stopping) m_samples.push_back({ .metrics = ReadProcessMetrics() });
        m_stopping = true;
    }
    m_wakeUp.notify_one();
    if (m_thread.joinable()) m_thread.join();
}

std::vector<MetricsPhaseSummary> MetricsCollector::AggregateByPhase() const
{
    std::vector<MetricsPhaseSummary> summaries;
    const MetricsSample* previous = nullptr;
    for (const auto& sample : m_samples) {
        if (sample.phase.empty() && &sample != &m_samples.back()) continue;
        if (previous) {
            auto summary = std::ranges::find(summaries, previous->phase, &MetricsPhaseSummary::phase);
            if (summary == summaries.end()) summary = summaries.insert(summaries.end(), { .phase = previous->phase });
            const auto& from = previous->metrics;
            const auto& to = sample.metrics;
            if (!previous->waiting) {
                summary->seconds += std::chrono::duration<double>(to.time - from.time).count();
                summary->cpuSeconds += to.cpuSeconds - from.cpuSeconds;
            }
            summary->commitMb += to.commitMb - from.commitMb;
            summary->sharedCommitMb += to.sharedCommitMb - from.sharedCommitMb;
            summary->workingSetMb += to.workingSetMb - from.workingSetMb;
        }
        previous = &sample;
    }
    return summaries;
}

void MetricsCollector::Run(std::chrono::milliseconds interval)
{
    auto next = std::chrono::steady_clock::now();
    std::unique_lock lock(m_mutex);
    while (!m_wakeUp.wait_until(lock, next, [this] { return m_stopping; })) {
        m_samples.push_back({ .metrics = ReadProcessMetrics() });
        next += interval * ((std::chrono::steady_clock::now() - next) / interval + 1);
    }
}

}  // namespace libmlvc
