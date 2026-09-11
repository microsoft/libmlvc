// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reporting.hpp"
#include "benchmark_runner.hpp"

#include "libmlvc_support/dataset.hpp"
#include "libmlvc_support/interop.hpp"
#include "libmlvc_support/validate.hpp"
#include <libmlvc/build_info.hpp>
#include <libmlvc/platform_info.hpp>
#include <libmlvc/types.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

#if defined(_WIN32)
    #include <io.h>
    #define isatty _isatty
    #define fileno _fileno
#else
    #include <unistd.h>
#endif

namespace libmlvc {

namespace {
bool UseColor()
{
    static const bool sUseColor = isatty(fileno(stderr)) != 0;
    return sUseColor;
}

std::string FormatCell(const OpTimerStats& s, int width, int precision = 1)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << s.AverageMs() << "/" << s.StdDevMs();
    std::string str = oss.str();
    if (static_cast<int>(str.size()) < width) {
        str = std::string(width - static_cast<int>(str.size()), ' ') + str;
    }
    return str;
}

template <typename StatsType>
void PrintBenchmarkTable(
    const char* label, const std::map<std::string, StatsType>& statsMap,
    const std::vector<std::tuple<const char*, std::function<const OpTimerStats&(const StatsType&)>, int>>& timerCols,
    std::function<int(const StatsType&)> reconfCountGetter, std::function<int(const StatsType&)> totalCountGetter,
    const std::map<std::string, double>& avgFpsMap)
{
    if (statsMap.empty()) return;

    // Column widths (display chars)
    const int streamW = 11;
    const int countW = 11;
    const int timerW = 12;
    const int fpsW = 8;

    // Build separator
    std::ostringstream sepSs;
    sepSs << "+" << std::string(streamW, '-');
    sepSs << "+" << std::string(countW, '-');
    for (size_t i = 0; i < timerCols.size(); i++)
        sepSs << "+" << std::string(timerW, '-');
    sepSs << "+" << std::string(fpsW, '-') << "+\n";
    const auto sep = sepSs.str();

    std::cerr << label << " (ms, mean/std, count=reconfigurations/frames):\n";
    std::cerr << sep;

    // Header
    std::cerr << "| " << std::left << std::setw(streamW - 1) << "Stream";
    std::cerr << "| " << std::setw(countW - 1) << "Count";
    for (const auto& [name, getter, prec] : timerCols) {
        std::cerr << "| " << std::setw(timerW - 1) << name;
    }
    std::cerr << "| " << std::setw(fpsW - 1) << "FPS" << "|\n";
    std::cerr << sep;

    // Rows
    for (const auto& [streamName, s] : statsMap) {
        // Count column: reconf/total
        std::ostringstream countSs;
        countSs << reconfCountGetter(s) << "/" << totalCountGetter(s);
        std::cerr << "| " << std::left << std::setw(streamW - 1) << streamName;
        std::cerr << "| " << std::right << std::setw(countW - 2) << countSs.str() << " ";
        for (const auto& [name, getter, prec] : timerCols) {
            std::cerr << "| " << FormatCell(getter(s), timerW - 2, prec) << " ";
        }
        const auto it = avgFpsMap.find(streamName);
        const double avgFps = it != avgFpsMap.end() ? it->second : 0.0;
        std::cerr << "| " << std::right << std::fixed << std::setprecision(1) << std::setw(fpsW - 2) << avgFps << " |\n";
    }
    std::cerr << sep;
}
}  // namespace

const char* ColorCode(const char* code)
{
    return UseColor() ? code : "";
}

void PrintRuntimeInfo()
{
    PrintBuildInfo(GetBuildInfo());
    PrintPlatformInfo(GetPlatformInfo());
}

void PrintBuildInfo(const BuildInfo& buildInfo)
{
    std::cerr << "libmlvc " << buildInfo.libmlvcVersion << " (" << buildInfo.gitShortHash << ", " << buildInfo.gitBranch
              << ")\n";
}

void PrintPlatformInfo(const PlatformInfo& platform)
{
    std::cerr << "Platform: " << platform.hwManufacturer << " " << platform.hwModel << ", " << platform.osName << " "
              << platform.osVersion.ToString() << "\n";
    std::cerr << "  CPU: " << platform.cpuName << " (" << platform.cpuSeries << ", " << platform.cpuArch;
    if (platform.hasFP16) std::cerr << ", fp16";
    if (platform.hasAVX2) std::cerr << ", avx2";
    if (platform.hasAVXVNNI) std::cerr << ", avxvnni";
    if (platform.hasDotProd) std::cerr << ", dotprod";
    std::cerr << ")\n";
    auto toUpper = [](const char* s) {
        std::string r(s);
        for (auto& c : r)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return r;
    };
    for (const auto& acc : platform.accelerators) {
        auto label = toUpper(AcceleratorTypeToString(acc.type));
        std::cerr << "  " << label << ": " << acc.name << " (" << acc.vendorId << ":" << acc.deviceId
                  << ", driver=" << acc.driverVersion << "/" << acc.driverDate << ")\n";
    }
}

void PrintEnvironment(const char* label, const BuildInfo& buildInfo, const PlatformInfo& platform, const ManagerInfo& config)
{
    std::cerr << "  ";
    if (label[0] != '\0') std::cerr << label << ": ";
    std::cerr << platform.hwManufacturer << " " << platform.hwModel << " (" << platform.cpuSeries << "), " << platform.osName
              << " " << platform.osVersion.ToString() << ", " << InferenceBackendToString(config.inferenceBackend);
    if (config.inferenceBackend == InferenceBackend::WINDOWSML) {
        std::cerr << "/" << OnnxExecutionProviderToString(config.onnxExecutionProvider);
    }
    std::cerr << "/" << ComputeUnitToString(config.computeUnit);
    if (!config.onnxRuntimeVersion.empty()) {
        std::cerr << ", ort " << config.onnxRuntimeVersion;
        if (!config.onnxRuntimeEpVersion.empty()) std::cerr << "/" << config.onnxRuntimeEpVersion;
    }
    if (!config.driverVersion.empty()) {
        std::cerr << ", driver " << config.driverVersion;
    }
    std::cerr << ", libmlvc " << buildInfo.libmlvcVersion << " (" << buildInfo.gitShortHash << ")";

    std::cerr << "\n";
}

void PrintManagerInfo(const ManagerInfo& config)
{
    std::cerr << "Manager: " << InferenceBackendToString(config.inferenceBackend);
    if (config.inferenceBackend == InferenceBackend::WINDOWSML) {
        std::cerr << "/" << OnnxExecutionProviderToString(config.onnxExecutionProvider);
    }
    std::cerr << "/" << ComputeUnitToString(config.computeUnit);
    if (config.enableModelCache || config.enableSessionCaching) {
        std::cerr << ", cache=" << (config.enableModelCache ? "model" : "")
                  << (config.enableModelCache && config.enableSessionCaching ? "+" : "")
                  << (config.enableSessionCaching ? "session" : "");
    }
    if (config.inferenceBackend == InferenceBackend::WINDOWSML) {
        if (!config.windowsAppRuntimeVersion.empty()) {
            std::cerr << ", app_runtime " << config.windowsAppRuntimeVersion;
            if (!config.windowsAppRuntimeEpVersion.empty()) std::cerr << "/" << config.windowsAppRuntimeEpVersion;
        }
        if (!config.onnxRuntimeVersion.empty()) {
            std::cerr << ", ort " << config.onnxRuntimeVersion;
            if (!config.onnxRuntimeEpVersion.empty()) std::cerr << "/" << config.onnxRuntimeEpVersion;
        }
    }
    if (!config.driverVersion.empty()) {
        std::cerr << ", driver=" << config.driverVersion;
        if (!config.driverDate.empty()) std::cerr << " (" << config.driverDate << ")";
    }
    std::cerr << "\n";
}

void PrintCapabilities(const std::vector<MlvcVersion>& versions, const std::vector<Capabilities>& capabilities)
{
    for (size_t i = 0; i < versions.size(); i++) {
        const auto& v = versions[i];
        const auto& c = capabilities[i];
        std::cerr << "  " << v.ToString() << ": " << c.maxWidth << "x" << c.maxHeight << ", max_fps=" << c.maxFps
                  << ", max_temporal=" << c.maxNumOfTemporalLayers << ", max_ltr=" << c.maxNumOfLtrFrames << "\n";
    }
}

void PrintEncoderConfig(const EncoderConfig& config, const char* indent)
{
    std::cerr << indent << "Encoder: " << config.mlvcVersion.ToString() << ", " << config.width << "x" << config.height
              << ", iframe=" << config.iframePeriod << ", temporal=" << config.numTemporalLayers << ", ltr=("
              << LtrModeToString(config.ltrMode) << ", start=" << config.ltrStartIdx << ", period=" << config.ltrPeriod
              << ", slots=" << config.ltrNumSlots << ", recovery=" << config.ltrRecoveryPeriod << ")\n";
}

void PrintDataset(const Dataset& dataset)
{
    std::cerr << "Dataset: " << dataset.GetName() << ", " << dataset.NumClips() << " clips";
    if (!dataset.Clips().empty()) {
        const auto& clip = dataset.Clips().front();
        std::cerr << ", " << clip.width << "x" << clip.height;
    }
    std::cerr << ", " << dataset.GetFps() << " fps, scenarios:";
    for (const auto& s : dataset.GetScenarios())
        std::cerr << " " << s;
    std::cerr << "\n";
}

void PrintSnapshotResult(const SnapshotResult& result)
{
    std::cerr << "\nSnapshot\n";
    PrintEnvironment("Environment", result.buildInfo, result.platformInfo, result.managerInfo);
    PrintEncoderConfig(result.encoderConfig, "  ");
    for (int qp : result.GetQps()) {
        const auto metrics = result.AggregateMetricsAtQp(qp);
        if (metrics) {
            std::cerr << "  QP=" << std::setw(2) << qp << std::fixed << "  PSNR mean: " << std::setprecision(2)
                      << metrics->psnr << " dB\n"
                      << "         PSNR min:  " << std::setprecision(2) << metrics->psnrMin << " dB\n"
                      << "         kbps: " << std::setprecision(1) << metrics->kbps << "\n";

            // Print timing stats (first clip at this QP, if available)
            auto it = std::ranges::find_if(result.clipResults, [qp](const auto& c) { return c.qp == qp; });
            if (it != result.clipResults.end()) {
                const auto& enc = it->encoderStats;
                const auto& dec = it->decoderStats;
                if (enc.total.Count() > 0) {
                    std::cerr << "         encode:    " << std::setprecision(2) << enc.total.AverageMs()
                              << " ms (inference " << enc.inference.AverageMs() << " ms)\n";
                }
                if (dec.total.Count() > 0) {
                    std::cerr << "         decode:    " << std::setprecision(2) << dec.total.AverageMs()
                              << " ms (inference " << dec.inference.AverageMs() << " ms)\n";
                }
            }
        }
    }
}

void PrintInteropResult(const InteropResult& interopResult)
{
    const auto& reference = interopResult.reference;
    const auto& localDecode = interopResult.localDecode;
    std::cerr << "\nReference snapshot: " << reference.snapshotName << "\n";
    PrintEnvironment("Encoded by", reference.buildInfo, reference.platformInfo, reference.managerInfo);
    PrintEnvironment("Decoded by", localDecode.buildInfo, localDecode.platformInfo, localDecode.managerInfo);
    PrintEncoderConfig(reference.encoderConfig, "  ");
    // Build lookup: (clipName, qp) -> reconstructionHash for O(log n) reference lookups
    std::map<std::pair<std::string, int>, std::string_view> refHashMap;
    for (const auto& ref : reference.clipResults) {
        if (!ref.reconstructionHash.empty()) {
            refHashMap.emplace(std::pair{ ref.clipName, ref.qp }, ref.reconstructionHash);
        }
    }
    auto bitExactAtQp = [&](int qp) {
        int total = 0, matches = 0;
        for (const auto& local : localDecode.clipResults) {
            if (local.qp != qp || local.reconstructionHash.empty()) continue;
            auto it = refHashMap.find({ local.clipName, qp });
            if (it == refHashMap.end()) continue;
            if (it->second == local.reconstructionHash) matches++;
            total++;
        }
        return total > 0 ? 100.0 * matches / total : 0.0;
    };
    for (int qp : localDecode.GetQps()) {
        const auto localMetrics = localDecode.AggregateMetricsAtQp(qp);
        if (localMetrics) {
            const auto refMetrics = reference.AggregateMetricsAtQp(qp);
            std::cerr << "  QP=" << std::setw(2) << qp << std::fixed;
            if (refMetrics) {
                const double meanDrop = refMetrics->psnr - localMetrics->psnr;
                const double minDrop = refMetrics->psnrMin - localMetrics->psnrMin;
                std::cerr << "  PSNR mean: " << std::setprecision(2) << localMetrics->psnr << " dB (ref "
                          << refMetrics->psnr << " dB, drop " << meanDrop << " dB)\n";
                std::cerr << "         PSNR min:  " << std::setprecision(2) << localMetrics->psnrMin << " dB (ref "
                          << refMetrics->psnrMin << " dB, drop " << minDrop << " dB)\n";
                std::cerr << "         kbps: " << std::setprecision(1) << localMetrics->kbps
                          << ", bit-exact: " << std::setprecision(1) << bitExactAtQp(qp) << "%\n";
            } else {
                std::cerr << "  PSNR mean: " << std::setprecision(2) << localMetrics->psnr << " dB\n"
                          << "         PSNR min:  " << std::setprecision(2) << localMetrics->psnrMin << " dB\n"
                          << "         kbps: " << std::setprecision(1) << localMetrics->kbps << "\n";
            }

            // Print decode timing stats (local vs reference)
            auto localIt = std::ranges::find_if(localDecode.clipResults, [qp](const auto& c) { return c.qp == qp; });
            if (localIt != localDecode.clipResults.end() && localIt->decoderStats.total.Count() > 0) {
                const auto& localDec = localIt->decoderStats;
                auto refIt = std::ranges::find_if(reference.clipResults, [qp](const auto& c) { return c.qp == qp; });
                const bool hasRef = refIt != reference.clipResults.end() && refIt->decoderStats.total.Count() > 0
                                    && refIt->decoderStats.inference.Count() > 0;
                if (hasRef) {
                    const auto& refDec = refIt->decoderStats;
                    const auto latencyChange = (localDec.inference.AverageMs() - refDec.inference.AverageMs())
                                               / refDec.inference.AverageMs() * 100.0;
                    const auto decLatencyChange =
                        (localDec.total.AverageMs() - refDec.total.AverageMs()) / refDec.total.AverageMs() * 100.0;
                    std::cerr << "         inference: " << std::setprecision(2) << localDec.inference.AverageMs()
                              << " ms (ref " << refDec.inference.AverageMs() << " ms, " << std::showpos
                              << std::setprecision(1) << latencyChange << "%" << std::noshowpos << ")\n";
                    std::cerr << "         decode:    " << std::setprecision(2) << localDec.total.AverageMs()
                              << " ms (ref " << refDec.total.AverageMs() << " ms, " << std::showpos
                              << std::setprecision(1) << decLatencyChange << "%" << std::noshowpos << ")\n";
                } else {
                    std::cerr << "         inference: " << std::setprecision(2) << localDec.inference.AverageMs() << " ms\n";
                    std::cerr << "         decode:    " << std::setprecision(2) << localDec.total.AverageMs() << " ms\n";
                }
            }
        }
    }
    const auto& t = interopResult.thresholds;
    const auto& m = interopResult.metrics;
    const auto color = [](bool ok) { return ColorCode(ok ? "\033[32m" : "\033[31m"); };
    const auto reset = ColorCode("\033[0m");
    std::cerr << "  Checks (QP=0): " << color(m.meanPsnr >= t.meanPsnrFloor) << "mean  PSNR " << std::fixed
              << std::setprecision(2) << m.meanPsnr << " >= " << std::setprecision(1) << t.meanPsnrFloor << " dB"
              << reset << "\n";
    std::cerr << "                 " << color(m.minPsnr >= t.minPsnrFloor) << "min   PSNR " << std::setprecision(2)
              << m.minPsnr << " >= " << std::setprecision(1) << t.minPsnrFloor << " dB" << reset << "\n";
    std::cerr << "                 " << color(m.psnrDrop <= t.maxPsnrDrop) << "PSNR drop  " << std::setprecision(2)
              << m.psnrDrop << " <= " << std::setprecision(2) << t.maxPsnrDrop << " dB" << reset << "\n";
    if (t.requireBitExact) {
        std::cerr << "                 " << color(m.bitExact) << "bit-exact  decoder " << std::setprecision(2)
                  << m.decoderBitExactPct << "% (required)" << reset << "\n";
    }
    std::cerr << "  Result: "
              << (interopResult.evaluationPassed ? (std::string(ColorCode("\033[32m")) + "PASS" + ColorCode("\033[0m"))
                                                 : (std::string(ColorCode("\033[31m")) + "FAIL" + ColorCode("\033[0m")))
              << "\n";
}

void PrintInteropSummaryTable(const std::vector<InteropResult>& results)
{
    if (results.empty()) return;

    // Build row labels: "cpuSeries COMPUTE_UNIT" (e.g. "Snapdragon X1 NPU", "M5 NPU")
    std::vector<std::string> labels;
    size_t maxLabelLen = 9;  // minimum width for "Reference" header
    for (const auto& r : results) {
        std::string cu = ComputeUnitToString(r.reference.managerInfo.computeUnit);
        for (auto& c : cu)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        std::string label = r.reference.platformInfo.cpuSeries + " " + cu;
        maxLabelLen = std::max(maxLabelLen, label.size());
        labels.push_back(std::move(label));
    }

    // Helper: format a double into a right-aligned string of given width
    auto fmt = [](double val, int width, int precision) {
        std::ostringstream ss;
        ss << std::right << std::setw(width) << std::fixed << std::setprecision(precision) << val;
        return ss.str();
    };

    // Helper: right-align a string in given width
    auto rpad = [](const std::string& s, int width) {
        std::ostringstream ss;
        ss << std::right << std::setw(width) << s;
        return ss.str();
    };

    // Helper: wrap a pre-padded string with ANSI color
    auto colorWrap = [](const std::string& s, bool ok) {
        return std::string(ColorCode(ok ? "\033[32m" : "\033[31m")) + s + ColorCode("\033[0m");
    };

    // Build separator line
    std::string sep =
        "+-" + std::string(maxLabelLen, '-') + "-+-------+-------+-------+-----------+---------+---------+--------+\n";

    // Header
    std::cerr << "\nInterop Summary:\n";
    std::cerr << sep;
    std::cerr << "| " << std::left << std::setw(static_cast<int>(maxLabelLen)) << "" << std::right
              << " |       PSNR (dB)       |           |     Inference     |        |\n";
    std::cerr << sep;
    std::cerr << "| " << std::left << std::setw(static_cast<int>(maxLabelLen)) << "Reference" << std::right
              << " |  Mean |   Min |  Drop | Bit-exact |      ms |       % | Result |\n";
    std::cerr << sep;

    // Rows
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const auto& m = r.metrics;
        const auto& t = r.thresholds;

        // PSNR columns (format + color in one step)
        auto meanC = colorWrap(fmt(m.meanPsnr, 5, 2), m.meanPsnr >= t.meanPsnrFloor);
        auto minC = colorWrap(fmt(m.minPsnr, 5, 2), m.minPsnr >= t.minPsnrFloor);
        auto dropC = colorWrap(fmt(m.psnrDrop, 5, 2), m.psnrDrop <= t.maxPsnrDrop);

        // Bit-exact: color only when requireBitExact
        std::ostringstream beSs;
        beSs << std::fixed << std::setprecision(1) << m.decoderBitExactPct << "%";
        auto bitExactStr = rpad(beSs.str(), 9);
        auto bitExactC = t.requireBitExact ? colorWrap(bitExactStr, m.bitExact) : bitExactStr;

        // Inference time and latency change % (QP=0 decode)
        std::string inferMsStr = rpad("-", 7), inferPctStr = rpad("-", 7);
        auto localIt = std::ranges::find_if(r.localDecode.clipResults, [](const auto& c) { return c.qp == 0; });
        auto refIt = std::ranges::find_if(r.reference.clipResults, [](const auto& c) { return c.qp == 0; });
        if (localIt != r.localDecode.clipResults.end() && localIt->decoderStats.inference.Count() > 0) {
            inferMsStr = fmt(localIt->decoderStats.inference.AverageMs(), 7, 2);
            if (refIt != r.reference.clipResults.end() && refIt->decoderStats.inference.Count() > 0) {
                const double refMs = refIt->decoderStats.inference.AverageMs();
                const double localMs = localIt->decoderStats.inference.AverageMs();
                std::ostringstream ss;
                ss << std::showpos << std::fixed << std::setprecision(1) << (localMs - refMs) / refMs * 100.0 << "%";
                inferPctStr = rpad(ss.str(), 7);
            }
        }

        auto resultC = colorWrap(r.evaluationPassed ? "PASS  " : "FAIL  ", r.evaluationPassed);

        std::cerr << "| " << std::left << std::setw(static_cast<int>(maxLabelLen)) << labels[i] << std::right << " | "
                  << meanC << " | " << minC << " | " << dropC << " | " << bitExactC << " | " << inferMsStr << " | "
                  << inferPctStr << " | " << resultC << " |\n";
    }
    std::cerr << sep;
}

void PrintValidationResultsTable(const std::map<std::string, ValidationTestSummary>& summaries)
{
    std::cerr << "\n";

    constexpr auto sep =
        "+----------+-------+-------------+-------------+-------------+-------------+-------+-------+-------+-------+"
        "\n";
    std::cerr
        << sep << "|          |       |           kbps            |           PSNR            |             BD-rate           |\n"
        << sep << "| Scenario | Clips |   Anchor    |    Test     |   Anchor    |    Test     |   Y   |   U   |   V   |  YUV  |\n"
        << sep;

    for (const auto& [scenario, s] : summaries) {
        std::ostringstream anchorPsnr, testPsnr, anchorKbps, testKbps;
        anchorPsnr << std::fixed << std::setprecision(1) << s.anchorPsnrMin << "-" << s.anchorPsnrMax;
        testPsnr << std::fixed << std::setprecision(1) << s.testPsnrMin << "-" << s.testPsnrMax;
        anchorKbps << std::fixed << std::setprecision(0) << s.anchorKbpsMin << "-" << s.anchorKbpsMax;
        testKbps << std::fixed << std::setprecision(0) << s.testKbpsMin << "-" << s.testKbpsMax;

        std::cerr << "| " << std::left << std::setw(8) << scenario << " | " << std::right << std::setw(5) << s.numCommonClips
                  << " | " << std::setw(11) << anchorKbps.str() << " | " << std::setw(11) << testKbps.str() << " | "
                  << std::setw(11) << anchorPsnr.str() << " | " << std::setw(11) << testPsnr.str() << " | "
                  << std::fixed << std::setprecision(1) << std::setw(5) << s.bdRateY << " | " << std::setw(5)
                  << s.bdRateU << " | " << std::setw(5) << s.bdRateV << " | " << std::setw(5) << s.bdRate << " |\n";
    }
    std::cerr << sep;
}

void PrintBenchmarkSummary(const BenchmarkResults& results)
{
    const auto avgFpsMap = results.GetAvgFps();

    // Encoder table
    {
        using S = const EncoderStats&;
        std::vector<std::tuple<const char*, std::function<const OpTimerStats&(S)>, int>> cols = {
            { "Reconf", [](S s) -> const OpTimerStats& { return s.reconfigure; }, 1 },
            { "Preproc", [](S s) -> const OpTimerStats& { return s.preprocess; }, 2 },
            { "ScaleDec", [](S s) -> const OpTimerStats& { return s.scaleDecoder; }, 2 },
            { "Entropy", [](S s) -> const OpTimerStats& { return s.entropyCoding; }, 2 },
            { "Inference", [](S s) -> const OpTimerStats& { return s.inference; }, 1 },
            { "Total", [](S s) -> const OpTimerStats& { return s.total; }, 1 },
        };
        PrintBenchmarkTable<EncoderStats>(
            "Encoders", results.GetEncoderStats(), cols, [](const EncoderStats& s) { return s.reconfigure.Count(); },
            [](const EncoderStats& s) { return s.total.Count(); }, avgFpsMap);
    }

    // Decoder table
    {
        using S = const DecoderStats&;
        std::vector<std::tuple<const char*, std::function<const OpTimerStats&(S)>, int>> cols = {
            { "Reconf", [](S s) -> const OpTimerStats& { return s.reconfigure; }, 1 },
            { "Postproc", [](S s) -> const OpTimerStats& { return s.postprocess; }, 2 },
            { "ScaleDec", [](S s) -> const OpTimerStats& { return s.scaleDecoder; }, 2 },
            { "Entropy", [](S s) -> const OpTimerStats& { return s.entropyCoding; }, 2 },
            { "Inference", [](S s) -> const OpTimerStats& { return s.inference; }, 1 },
            { "Total", [](S s) -> const OpTimerStats& { return s.total; }, 1 },
        };
        PrintBenchmarkTable<DecoderStats>(
            "Decoders", results.GetDecoderStats(), cols, [](const DecoderStats& s) { return s.reconfigure.Count(); },
            [](const DecoderStats& s) { return s.total.Count(); }, avgFpsMap);
    }
}

void PrintHelp()
{
    std::cerr << "Usage: mlvc [command] [options]\n\n";
    std::cerr << "Commands:\n";
    std::cerr << "  encode      Encode a video file\n";
    std::cerr << "  decode      Decode a video file\n";
    std::cerr << "  benchmark   Run benchmark tests\n";
    std::cerr << "  validate    Run validation test\n";
    std::cerr << "  interop     Run interop tests against reference snapshots\n";
    std::cerr << "  help        Print help\n";
}

void PrintPassFail(bool passed, const char* suffix)
{
    const char* label = passed ? "PASSED" : "FAILED";
    const char* color = ColorCode(passed ? "\033[32m" : "\033[31m");
    const char* reset = ColorCode("\033[0m");
    std::cerr << "\n" << color << label << reset;
    if (suffix) std::cerr << ": " << suffix;
    std::cerr << "\n";
}

}  // namespace libmlvc
