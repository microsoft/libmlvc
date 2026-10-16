// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <libmlvc/expected.hpp>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace libmlvc {

struct ProcessMetrics {
    std::chrono::steady_clock::time_point time{};
    double cpuSeconds{};      // User and kernel CPU time
    double commitMb{};        // Commit charge; on macOS, the physical footprint
    double sharedCommitMb{};  // Pagefile-backed shared sections, such as NPU memory; Windows only
    double workingSetMb{};    // On macOS, the resident size
};

ProcessMetrics ReadProcessMetrics();

struct MetricsSample {
    std::string phase;
    std::string event;
    ProcessMetrics metrics;
    bool waiting{};
};

struct MetricsPhaseSummary {
    std::string phase;
    double seconds{};
    double cpuSeconds{};
    double commitMb{};
    double sharedCommitMb{};
    double workingSetMb{};
};

class MetricsCollector {
public:
    MetricsCollector() = default;
    ~MetricsCollector();
    MetricsCollector(const MetricsCollector&) = delete;
    MetricsCollector& operator=(const MetricsCollector&) = delete;

    expected<void> Start(std::chrono::milliseconds interval);
    expected<void> Record(std::string phase, std::string event);
    expected<void> RecordWait(std::string phase, std::string event);

    // Owner-thread only; takes a final sample, which ends the last phase.
    void Stop();

    // Read only after Stop(): all periodic and event samples in time order.
    const std::vector<MetricsSample>& Samples() const { return m_samples; }

    // Read only after Stop(): the changes from each event to the next event or Stop(), summed per phase, in the
    // order the phases started.
    std::vector<MetricsPhaseSummary> AggregateByPhase() const;

private:
    expected<void> Add(std::string phase, std::string event, bool waiting);
    void Run(std::chrono::milliseconds interval);

    std::mutex m_mutex;
    std::condition_variable m_wakeUp;
    bool m_stopping{};
    std::vector<MetricsSample> m_samples;
    std::thread m_thread;
};

}  // namespace libmlvc
