// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>

namespace libmlvc {

void StatsDump::DumpStats(std::string_view stream_name, const std::chrono::system_clock::time_point& timestamp,
                          const EncoderStats& stats)
{
    boost::json::object obj = DumpToJson(stream_name, timestamp, stats);
    {
        std::lock_guard lock(m_frameStatsMutex);
        m_frameStats.emplace_back(std::move(obj));
    }
}

void StatsDump::DumpStats(std::string_view stream_name, const std::chrono::system_clock::time_point& timestamp,
                          const DecoderStats& stats)
{
    boost::json::object obj = DumpToJson(stream_name, timestamp, stats);
    {
        std::lock_guard lock(m_frameStatsMutex);
        m_frameStats.emplace_back(std::move(obj));
    }
}

void BenchmarkResults::Add(std::string_view streamName, const EncoderStats& stats, double avgFps)
{
    std::lock_guard lock(m_mutex);
    auto key = std::string(streamName);
    m_encoderStats.emplace(key, stats);
    m_avgFps[key] = avgFps;
}

void BenchmarkResults::Add(std::string_view streamName, const DecoderStats& stats, double avgFps)
{
    std::lock_guard lock(m_mutex);
    auto key = std::string(streamName);
    m_decoderStats.emplace(key, stats);
    m_avgFps[key] = avgFps;
}

std::map<std::string, EncoderStats> BenchmarkResults::GetEncoderStats() const
{
    std::lock_guard lock(m_mutex);
    return m_encoderStats;
}

std::map<std::string, DecoderStats> BenchmarkResults::GetDecoderStats() const
{
    std::lock_guard lock(m_mutex);
    return m_decoderStats;
}

std::map<std::string, double> BenchmarkResults::GetAvgFps() const
{
    std::lock_guard lock(m_mutex);
    return m_avgFps;
}

boost::json::object StatsDump::DumpToJson(std::string_view stream_name, const std::chrono::system_clock::time_point& timestamp)
{
    // FIXME: Library loses precision by using scientific format for floats
    std::chrono::duration<double> epoch{ timestamp.time_since_epoch() };
    std::ostringstream timestampStr;
    timestampStr << std::fixed << std::setprecision(6) << epoch.count();

    boost::json::object obj;
    obj["stream_name"] = std::string(stream_name);
    obj["timestamp_str"] = timestampStr.str();
    return obj;
}

boost::json::object StatsDump::DumpToJson(std::string_view stream_name,
                                          const std::chrono::system_clock::time_point& timestamp, const EncoderStats& stats)
{
    boost::json::object obj = DumpToJson(stream_name, timestamp);
    obj["reconfigure_ms"] = stats.reconfigure.LatestMs();
    obj["preprocess_ms"] = stats.preprocess.LatestMs();
    obj["inference_ms"] = stats.inference.LatestMs();
    obj["scale_decoder_ms"] = stats.scaleDecoder.LatestMs();
    obj["entropy_coding_ms"] = stats.entropyCoding.LatestMs();
    obj["total_ms"] = stats.total.LatestMs();
    return obj;
}
boost::json::object StatsDump::DumpToJson(std::string_view stream_name,
                                          const std::chrono::system_clock::time_point& timestamp, const DecoderStats& stats)
{
    boost::json::object obj = DumpToJson(stream_name, timestamp);
    obj["reconfigure_ms"] = stats.reconfigure.LatestMs();
    obj["entropy_coding_ms"] = stats.entropyCoding.LatestMs();
    obj["scale_decoder_ms"] = stats.scaleDecoder.LatestMs();
    obj["inference_ms"] = stats.inference.LatestMs();
    obj["postprocess_ms"] = stats.postprocess.LatestMs();
    obj["total_ms"] = stats.total.LatestMs();
    return obj;
}

void StatsDump::Save(std::string_view filename)
{
    std::lock_guard lock(m_frameStatsMutex);

    boost::json::object root;
    root["frame_stats"] = m_frameStats;
    const std::string jsonString = boost::json::serialize(root);

    std::ofstream file{ filename.data() };
    file << jsonString;
}

BaseBenchmarkRunner::BaseBenchmarkRunner(std::string_view name, const BenchmarkArguments& args,
                                         const MlvcManager& manager, const size_t numFrames, StatsDump& dump,
                                         BenchmarkResults& benchmarkResults, std::atomic<bool>& stopFlag)
    : m_name(name)
    , m_args(args)
    , m_manager(manager)
    , m_numFrames(numFrames)
    , m_dump(dump)
    , m_benchmarkResults(benchmarkResults)
    , m_stopFlag(stopFlag)
{
}

BaseBenchmarkRunner::~BaseBenchmarkRunner()
{
    Join();
}

expected<void> BaseBenchmarkRunner::Initialize()
{
    return {};
}

expected<void> BaseBenchmarkRunner::Run()
{
    m_thread = std::thread(&BaseBenchmarkRunner::ThreadFunc, this);
    return {};
}

void BaseBenchmarkRunner::Join()
{
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void BaseBenchmarkRunner::ThreadFunc()
{
    const std::chrono::duration<double, std::milli> targetFramePeriod{ 1e3 / m_args.targetFps };
    constexpr auto sleepCorrection =
#ifdef _WIN32
        std::chrono::milliseconds(8);  // Compensate for Windows default timer resolution (~15.6ms)
#else
        std::chrono::milliseconds(3);
#endif
    int totalFrameCount = 0;
    double totalWallClockMs = 0.0;
    for (int iteration = 0; iteration < m_args.numIterations && !m_stopFlag; iteration++) {
        const auto runStart = std::chrono::steady_clock::now();
        const auto runEnd = runStart + std::chrono::seconds(m_args.durationSeconds);
        OpTimerStats runTimerStats;
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<double> jitterDist(0.0, 3.0);
        auto nextFrameTime = runStart + targetFramePeriod;
        for (size_t i = 0; !m_stopFlag && std::chrono::steady_clock::now() < runEnd; i++) {
            const auto frameStart = std::chrono::steady_clock::now();
            const auto timestamp = std::chrono::system_clock::now();

            ProcessFrame(i % m_numFrames);
            DumpStats(timestamp);

            // Wait to match target fps
            if (m_args.targetFps < 999.0) {
                const auto jitter = std::chrono::duration<double, std::milli>(jitterDist(rng));
                nextFrameTime += targetFramePeriod + jitter;
                const auto sleepDuration = nextFrameTime - std::chrono::steady_clock::now() - sleepCorrection;
                if (!m_stopFlag && sleepDuration > std::chrono::milliseconds(0)) {
                    std::this_thread::sleep_for(sleepDuration);
                }
            }
            const auto frameDuration =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart);
            runTimerStats.Append(frameDuration.count());
        }
        totalFrameCount += runTimerStats.Count();
        totalWallClockMs += runTimerStats.TotalMs();
    }

    const double avgFps = totalWallClockMs > 0 ? 1000.0 * totalFrameCount / totalWallClockMs : 0.0;
    SaveBenchmarkResults(avgFps);
    m_stopFlag = true;
}

EncoderBenchmarkRunner::EncoderBenchmarkRunner(std::string_view name, const BenchmarkArguments& args,
                                               const MlvcManager& manager, const std::vector<Nv12Frame>& frames,
                                               StatsDump& dump, BenchmarkResults& benchmarkResults,
                                               std::atomic<bool>& stopFlag)
    : BaseBenchmarkRunner(name, args, manager, frames.size(), dump, benchmarkResults, stopFlag), m_frames(frames)
{
}

expected<void> EncoderBenchmarkRunner::Initialize()
{
    auto encoderConfig = m_manager.GetDefaultEncoderConfig(m_args.mlvcVersion);
    if (!encoderConfig) {
        std::cerr << "Error: BenchmarkRunner(" << m_name
                  << "): Failed to get default encoder config: " << encoderConfig.error().message() << '\n';
        return encoderConfig.error();
    }
    m_encoderConfig = encoderConfig.value();
    m_args.Apply(m_encoderConfig);

    auto encoder = m_manager.CreateEncoder(m_encoderConfig);
    if (!encoder) {
        std::cerr << "Error: BenchmarkRunner(" << m_name << "): Failed to create encoder: " << encoder.error().message()
                  << '\n';
        return encoder.error();
    }
    m_encoder = std::move(encoder.value());
    return {};
}

void EncoderBenchmarkRunner::ProcessFrame(const size_t frameIndex)
{
    const auto& frame = m_frames[frameIndex];
    m_encoderConfig.SetSize(frame.width, frame.height);
    if (auto ret = m_encoder->Configure(m_encoderConfig); !ret) {
        std::cerr << "Fatal: BenchmarkRunner(" << m_name << "): Failed to configure encoder: " << ret.error().message()
                  << '\n';
        std::abort();
    }
    auto encodeRes = m_encoder->Encode(frame.View(), { .qp = m_args.qp });
    if (!encodeRes) {
        std::cerr << "Fatal: BenchmarkRunner(" << m_name << "): Failed to encode frame: " << encodeRes.error().message()
                  << '\n';
        std::abort();
    }
}

void EncoderBenchmarkRunner::DumpStats(const std::chrono::system_clock::time_point& timestamp)
{
    const auto& stats = m_encoder->GetStats();
    m_dump.DumpStats(m_name, timestamp, stats);
}

void EncoderBenchmarkRunner::SaveBenchmarkResults(double avgFps)
{
    const auto& stats = m_encoder->GetStats();
    m_benchmarkResults.Add(m_name, stats, avgFps);
}

DecoderBenchmarkRunner::DecoderBenchmarkRunner(std::string_view name, const BenchmarkArguments& args,
                                               const MlvcManager& manager,
                                               const std::vector<MlvcAccessUnit>& accessUnits, StatsDump& dump,
                                               BenchmarkResults& benchmarkResults, std::atomic<bool>& stopFlag)
    : BaseBenchmarkRunner(name, args, manager, accessUnits.size(), dump, benchmarkResults, stopFlag)
    , m_accessUnits(accessUnits)
{
}

void DecoderBenchmarkRunner::ProcessFrame(const size_t frameIndex)
{
    if (!m_decoder) {
        auto decoder = m_manager.CreateDecoder();
        if (!decoder) {
            std::cerr << "Fatal: BenchmarkRunner(" << m_name
                      << "): Failed to initialize decoder: " << decoder.error().message() << '\n';
            std::abort();
        }
        m_decoder = std::move(decoder.value());
    }

    auto decodeRes = m_decoder->Decode(m_accessUnits[frameIndex].data);
    if (!decodeRes) {
        std::cerr << "Fatal: BenchmarkRunner(" << m_name << "): Failed to decode frame: " << decodeRes.error().message()
                  << '\n';
        std::abort();
    }
}

void DecoderBenchmarkRunner::DumpStats(const std::chrono::system_clock::time_point& timestamp)
{
    const auto& stats = m_decoder->GetStats();
    m_dump.DumpStats(m_name, timestamp, stats);
}

void DecoderBenchmarkRunner::SaveBenchmarkResults(double avgFps)
{
    if (!m_decoder) return;
    const auto& stats = m_decoder->GetStats();
    m_benchmarkResults.Add(m_name, stats, avgFps);
}

}  // namespace libmlvc
