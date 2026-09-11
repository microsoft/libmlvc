// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "arguments.hpp"
#include "libmlvc_support/mlvc_io.hpp"
#include "libmlvc_support/video_io.hpp"

#include <boost/json.hpp>
#include <libmlvc/libmlvc.hpp>

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace libmlvc {

class BenchmarkResults {
public:
    void Add(std::string_view streamName, const EncoderStats& stats, double avgFps);
    void Add(std::string_view streamName, const DecoderStats& stats, double avgFps);

    std::map<std::string, EncoderStats> GetEncoderStats() const;
    std::map<std::string, DecoderStats> GetDecoderStats() const;
    std::map<std::string, double> GetAvgFps() const;

private:
    std::map<std::string, EncoderStats> m_encoderStats;
    std::map<std::string, DecoderStats> m_decoderStats;
    std::map<std::string, double> m_avgFps;
    mutable std::mutex m_mutex;
};

class StatsDump {
public:
    void DumpStats(std::string_view stream_name, const std::chrono::system_clock::time_point& timestamp,
                   const EncoderStats& stats);
    void DumpStats(std::string_view stream_name, const std::chrono::system_clock::time_point& timestamp,
                   const DecoderStats& stats);
    void Save(std::string_view filename);

private:
    boost::json::array m_frameStats;
    std::mutex m_frameStatsMutex;

    static boost::json::object DumpToJson(std::string_view stream_name,
                                          const std::chrono::system_clock::time_point& timestamp);
    static boost::json::object DumpToJson(std::string_view stream_name,
                                          const std::chrono::system_clock::time_point& timestamp,
                                          const EncoderStats& stats);
    static boost::json::object DumpToJson(std::string_view stream_name,
                                          const std::chrono::system_clock::time_point& timestamp,
                                          const DecoderStats& stats);
};

class IBenchmarkRunner {
public:
    virtual ~IBenchmarkRunner() = default;
    virtual expected<void> Initialize() = 0;
    virtual expected<void> Run() = 0;
    virtual void Join() = 0;
};

class BaseBenchmarkRunner : public IBenchmarkRunner {
public:
    BaseBenchmarkRunner(std::string_view name, const BenchmarkArguments& args, const MlvcManager& manager, size_t numFrames,
                        StatsDump& dump, BenchmarkResults& benchmarkResults, std::atomic<bool>& stopFlag);
    ~BaseBenchmarkRunner();

    expected<void> Initialize() override;
    expected<void> Run() override;
    void Join() override;

protected:
    const std::string m_name;
    const BenchmarkArguments& m_args;
    const MlvcManager& m_manager;
    const size_t m_numFrames;
    StatsDump& m_dump;
    BenchmarkResults& m_benchmarkResults;
    std::atomic<bool>& m_stopFlag;
    std::thread m_thread;

    void ThreadFunc();
    virtual void ProcessFrame(size_t frameIndex) = 0;
    virtual void DumpStats(const std::chrono::system_clock::time_point& timestamp) = 0;
    virtual void SaveBenchmarkResults(double avgFps) = 0;
};

class EncoderBenchmarkRunner : public BaseBenchmarkRunner {
public:
    EncoderBenchmarkRunner(std::string_view name, const BenchmarkArguments& args, const MlvcManager& manager,
                           const std::vector<Nv12Frame>& frames, StatsDump& dump, BenchmarkResults& benchmarkResults,
                           std::atomic<bool>& stopFlag);

    expected<void> Initialize() override;

private:
    const std::vector<Nv12Frame>& m_frames;
    std::optional<MlvcEncoder> m_encoder;
    EncoderConfig m_encoderConfig{};

    void ProcessFrame(size_t frameIndex) override;
    void DumpStats(const std::chrono::system_clock::time_point& timestamp) override;
    void SaveBenchmarkResults(double avgFps) override;
};

class DecoderBenchmarkRunner : public BaseBenchmarkRunner {
public:
    DecoderBenchmarkRunner(std::string_view name, const BenchmarkArguments& args, const MlvcManager& manager,
                           const std::vector<MlvcAccessUnit>& accessUnits, StatsDump& dump,
                           BenchmarkResults& benchmarkResults, std::atomic<bool>& stopFlag);

private:
    const std::vector<MlvcAccessUnit>& m_accessUnits;
    std::optional<MlvcDecoder> m_decoder;

    void ProcessFrame(size_t frameIndex) override;
    void DumpStats(const std::chrono::system_clock::time_point& timestamp) override;
    void SaveBenchmarkResults(double avgFps) override;
};

}  // namespace libmlvc
