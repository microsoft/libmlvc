// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "arguments.hpp"

#include <boost/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>

namespace libmlvc {

expected<void> IArguments::ParseCommandLine(int argc, char** argv)
{
    std::vector<std::string_view> argsVec;
    argsVec.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        argsVec.emplace_back(argv[i]);
    }

    for (size_t i = 0; i < argsVec.size();) {
        const auto remainingArgs = std::span<std::string_view>{ argsVec.begin() + i, argsVec.end() };
        const auto numParsed = Parse(remainingArgs);
        if (!numParsed) {
            return numParsed.error();
        }
        if (numParsed.value() == 0) {
            std::cerr << "Unexpected error parsing argument: " << argsVec[i] << std::endl;
            return make_error_code(Error::command_line_parse_error);
        }
        i += numParsed.value();
    }

    if (auto ret = Verify(); !ret) {
        return ret.error();
    }

    return {};
}

void IArguments::PrintHelp()
{
    PrintHelpImpl();
}

expected<int> CommonArguments::Parse(std::span<std::string_view> args)
{
    if (args[0] == "--model-bundles-dir") {
        auto ret = ParseValue(args.subspan(1), modelBundlesDir);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--mlvc-version") {
        auto ret = ParseValue(args.subspan(1), mlvcVersion);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--compute-unit") {
        auto ret = ParseValue(args.subspan(1), computeUnit);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--enable-model-cache") {
        auto ret = ParseValue(args.subspan(1), enableModelCache);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--enable-session-caching") {
        auto ret = ParseValue(args.subspan(1), enableSessionCaching);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--winml-init") {
        auto ret = ParseValue(args.subspan(1), winmlInitMode);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--log-level") {
        auto ret = ParseValue(args.subspan(1), logLevel);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else {
        std::cerr << "Unknown argument: " << args[0] << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
}

expected<void> CommonArguments::Verify()
{
    SetLogLevel(logLevel);
    return {};
}

void CommonArguments::PrintHelpImpl()
{
    std::cerr << "\nUsage: mlvc [command] [options]\n\n";
    std::cerr << "Commands:\n";
    std::cerr << "  encode                  Encode a video file\n";
    std::cerr << "  decode                  Decode a video file\n";
    std::cerr << "  benchmark               Run benchmark tests\n";
    std::cerr << "  validate                Run validation test\n";
    std::cerr << "  interop                 Run interop tests against reference snapshots\n\n";
    std::cerr << "Options:\n";
    std::cerr << "Common Arguments:\n";
    std::cerr << "  --model-bundles-dir             Specify the directory for models (default: " << modelBundlesDir << ")\n";
    std::cerr << "  --mlvc-version                  Specify the MLVC version (default: " << mlvcVersion.ToString() << ")\n";
    std::cerr << "  --compute-unit                  Specify the compute unit auto, cpu, gpu, npu (default: "
              << ComputeUnitToString(computeUnit) << ")\n";
    std::cerr << "  --enable-model-cache            Enable model cache (default: " << enableModelCache << ")\n";
    std::cerr << "  --enable-session-caching        Enable session caching (default: " << enableSessionCaching << ")\n";
    std::cerr << "  --winml-init                    WinML init mode: appsdk, self-contained (default: "
              << WinMlInitModeToString(winmlInitMode) << ")\n";
    std::cerr
        << "  --log-level                     Specify log level: debug, info, warn, error, fatal (default: error)\n";
    std::cerr << std::endl;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, MlvcVersion& output)
{
    if (args.empty() || args[0].empty()) {
        std::cerr << "Missing value for --mlvc-version" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }

    std::smatch match;
    std::string strValue{ args[0] };
    static const std::regex versionRegex("^v(\\d+)\\.(\\d+)$");
    if (!std::regex_match(strValue, match, versionRegex) || match.size() != 3) {
        std::cerr << "Invalid format for --mlvc-version. Expected v<major>.<minor>, got " << strValue << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }

    int major = 0;
    int minor = 0;
    const auto majorStr = match[1].str();
    const auto minorStr = match[2].str();
    if (std::from_chars(majorStr.data(), majorStr.data() + majorStr.size(), major).ec != std::errc{}
        || std::from_chars(minorStr.data(), minorStr.data() + minorStr.size(), minor).ec != std::errc{}) {
        std::cerr << "Invalid version number in '" << strValue << "'" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    output = MlvcVersion{ major, minor };
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, ComputeUnit& output)
{
    if (args.empty()) {
        std::cerr << "Missing value for --compute-unit" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }

    const auto value = args[0];
    if (value == "auto") {
        output = ComputeUnit::AUTO;
    } else if (value == "cpu") {
        output = ComputeUnit::CPU;
    } else if (value == "gpu") {
        output = ComputeUnit::GPU;
    } else if (value == "npu") {
        output = ComputeUnit::NPU;
    } else {
        std::cerr << "Unknown value for --compute-unit (" << value << ")" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, LogLevel& output)
{
    if (args.empty()) {
        std::cerr << "Missing value for --log-level" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }

    const auto value = args[0];
    if (value == "debug") {
        output = LogLevel::Debug;
    } else if (value == "info") {
        output = LogLevel::Info;
    } else if (value == "warn") {
        output = LogLevel::Warn;
    } else if (value == "error") {
        output = LogLevel::Error;
    } else if (value == "fatal") {
        output = LogLevel::Fatal;
    } else {
        std::cerr << "Unknown value for --log-level (" << value << ")" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, WinMlInitMode& output)
{
    if (args.empty()) {
        std::cerr << "Missing value for --winml-init" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    auto parsed = StringToWinMlInitMode(args[0]);
    if (!parsed) {
        std::cerr << "Unknown value for --winml-init (" << args[0] << "), expected 'appsdk' or 'self-contained'"
                  << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    output = *parsed;
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, LtrMode& output)
{
    if (args.empty()) {
        std::cerr << "Missing value for --ltr-mode" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }

    const auto value = args[0];
    if (value == "internal") {
        output = LtrMode::INTERNAL;
    } else if (value == "external") {
        output = LtrMode::EXTERNAL;
    } else {
        std::cerr << "Unknown value for --ltr-mode (" << value << "), expected 'internal' or 'external'" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, int& output)
{
    if (args.empty() || args[0].empty()) {
        std::cerr << "Missing value for argument" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    const auto sv = args[0];
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), output);
    if (ec != std::errc{} || ptr != sv.data() + sv.size()) {
        std::cerr << "Invalid integer value '" << sv << "'" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, double& output)
{
    if (args.empty() || args[0].empty()) {
        std::cerr << "Missing value for argument" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    // Note: std::from_chars<double> is unavailable on our minimum macOS deployment target (macOS 13).
    // strtod is locale-sensitive (LC_NUMERIC); consider switching to from_chars once macOS 13.3+ is the floor.
    const std::string str(args[0]);
    char* end = nullptr;
    errno = 0;
    output = std::strtod(str.c_str(), &end);
    if (end != str.c_str() + str.size() || errno == ERANGE) {
        std::cerr << "Invalid numeric value '" << args[0] << "'" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, bool& output)
{
    if (args.empty() || args[0].empty()) {
        std::cerr << "Missing value for argument" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }

    const auto value = args[0];
    if (value == "1") {
        output = true;
        return 1;
    }
    if (value == "0") {
        output = false;
        return 1;
    }
    std::cerr << "Invalid value for bool argument (expected 0 or 1)" << std::endl;
    return make_error_code(Error::command_line_parse_error);
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, VideoReader::PixelFormat& output)
{
    if (args.empty() || args[0].empty()) {
        std::cerr << "Missing value for --input-format" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (args[0] == "nv12") {
        output = VideoReader::PixelFormat::NV12;
    } else if (args[0] == "i420") {
        output = VideoReader::PixelFormat::I420;
    } else {
        std::cerr << "Invalid value for --input-format (expected nv12 or i420)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, VideoWriter::Format& output)
{
    if (args.empty() || args[0].empty()) {
        std::cerr << "Missing value for --output-format" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (args[0] == "nv12") {
        output = VideoWriter::Format::NV12;
    } else if (args[0] == "i420") {
        output = VideoWriter::Format::I420;
    } else if (args[0] == "mkv") {
        output = VideoWriter::Format::MKV;
    } else {
        std::cerr << "Invalid value for --output-format (expected nv12, i420, or mkv)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, std::string& output)
{
    if (args.empty()) {
        std::cerr << "Missing value for argument" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    output = std::string{ args[0] };
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, std::filesystem::path& output)
{
    if (args.empty()) {
        std::cerr << "Missing value for argument" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    output = std::filesystem::path{ args[0] };
    return 1;
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, std::vector<int>& output)
{
    output.clear();
    size_t i = 0;
    for (; i < args.size() && !args[i].starts_with("-"); ++i) {
        int value{};
        auto ret = ParseValue(args.subspan(i), value);
        if (!ret) return ret.error();
        output.push_back(value);
    }
    return static_cast<int>(i);
}

expected<int> CommonArguments::ParseValue(std::span<std::string_view> args, std::vector<std::string>& output)
{
    output.clear();
    size_t i = 0;
    for (; i < args.size() && !args[i].starts_with("-"); ++i) {
        output.emplace_back(args[i]);
    }
    return static_cast<int>(i);
}

expected<int> EncoderConfigArguments::Parse(std::span<std::string_view> args)
{
    if (args[0] == "--iframe-period") {
        int value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        iframePeriod = value;
        return 1 + ret.value();
    } else if (args[0] == "--num-temporal-layers") {
        int value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        numTemporalLayers = value;
        return 1 + ret.value();
    } else if (args[0] == "--ltr-mode") {
        LtrMode value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        ltrMode = value;
        return 1 + ret.value();
    } else if (args[0] == "--ltr-start-idx") {
        int value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        ltrStartIdx = value;
        return 1 + ret.value();
    } else if (args[0] == "--ltr-period") {
        int value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        ltrPeriod = value;
        return 1 + ret.value();
    } else if (args[0] == "--ltr-num-slots") {
        int value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        ltrNumSlots = value;
        return 1 + ret.value();
    } else if (args[0] == "--ltr-recovery-period") {
        int value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        ltrRecoveryPeriod = value;
        return 1 + ret.value();
    } else {
        return CommonArguments::Parse(args);
    }
}

expected<void> EncoderConfigArguments::Verify()
{
    if (auto ret = CommonArguments::Verify(); !ret) {
        return ret.error();
    }

    if (iframePeriod.has_value() && *iframePeriod < 0) {
        std::cerr << "Invalid value for --iframe-period (must be >= 0)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (numTemporalLayers.has_value() && *numTemporalLayers < 1) {
        std::cerr << "Invalid value for --num-temporal-layers (must be >= 1)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (ltrStartIdx.has_value() && *ltrStartIdx < 0) {
        std::cerr << "Invalid value for --ltr-start-idx (must be >= 0)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (ltrPeriod.has_value() && *ltrPeriod < 0) {
        std::cerr << "Invalid value for --ltr-period (must be >= 0)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (ltrNumSlots.has_value() && (*ltrNumSlots < 1 || *ltrNumSlots > MAX_LTR_SLOTS)) {
        std::cerr << "Invalid value for --ltr-num-slots (must be 1-" << MAX_LTR_SLOTS << ")" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (ltrRecoveryPeriod.has_value() && *ltrRecoveryPeriod < 0) {
        std::cerr << "Invalid value for --ltr-recovery-period (must be >= 0)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }

    return {};
}

void EncoderConfigArguments::PrintHelpImpl()
{
    CommonArguments::PrintHelpImpl();
    std::cerr << "Encoder Config Overrides (optional, uses model defaults if not specified):\n";
    std::cerr << "  --iframe-period             I-frame period, 0 = no periodic I-frames\n";
    std::cerr << "  --num-temporal-layers       Number of temporal layers (>= 1)\n";
    std::cerr << "  --ltr-mode                  LTR mode: 'internal' or 'external'\n";
    std::cerr << "  --ltr-start-idx             Skip first n frames before marking LTR frames\n";
    std::cerr << "  --ltr-period                LTR period, 0 = no periodic LTR frames\n";
    std::cerr << "  --ltr-num-slots             Number of LTR slots (1-" << MAX_LTR_SLOTS << ")\n";
    std::cerr << "  --ltr-recovery-period       Proactive LTR recovery period (in LTR period units), 0 = disabled\n";
    std::cerr << std::endl;
}

void BenchmarkArguments::Save(const std::filesystem::path& filename) const
{
    boost::json::object root;
    root["model_bundles_dir"] = modelBundlesDir.string();
    root["mlvc_version"] = mlvcVersion.ToString();
    root["compute_unit"] = ComputeUnitToString(computeUnit);
    root["enable_model_cache"] = enableModelCache;
    root["enable_session_caching"] = enableSessionCaching;
    root["input_file"] = inputFile.string();
    root["image_width"] = inputWidth;
    root["image_height"] = inputHeight;
    root["output_directory"] = outputDirectory.string();
    root["num_encoders"] = numEncoders;
    root["num_decoders"] = numDecoders;
    root["target_fps"] = targetFps;
    root["duration_seconds"] = durationSeconds;
    root["num_iterations"] = numIterations;
    root["strict_fps_target"] = strictFpsTarget;
    root["wait_seconds"] = waitSeconds;
    root["qp"] = qp;

    const std::string jsonString = boost::json::serialize(root);
    std::ofstream file{ filename };
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file for writing: " << filename.string() << '\n';
        return;
    }
    file << jsonString;
}

expected<int> BenchmarkArguments::Parse(std::span<std::string_view> args)
{
    if (args[0] == "--input") {
        auto ret = ParseValue(args.subspan(1), inputFile);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--input-width") {
        auto ret = ParseValue(args.subspan(1), inputWidth);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--input-height") {
        auto ret = ParseValue(args.subspan(1), inputHeight);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--output-dir") {
        auto ret = ParseValue(args.subspan(1), outputDirectory);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--num-encoders") {
        auto ret = ParseValue(args.subspan(1), numEncoders);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--num-decoders") {
        auto ret = ParseValue(args.subspan(1), numDecoders);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--target-fps") {
        auto ret = ParseValue(args.subspan(1), targetFps);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--duration-seconds") {
        auto ret = ParseValue(args.subspan(1), durationSeconds);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--num-iterations") {
        auto ret = ParseValue(args.subspan(1), numIterations);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--strict-fps-target") {
        auto ret = ParseValue(args.subspan(1), strictFpsTarget);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--wait-seconds") {
        auto ret = ParseValue(args.subspan(1), waitSeconds);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--qp") {
        auto ret = ParseValue(args.subspan(1), qp);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else {
        return EncoderConfigArguments::Parse(args);
    }
}

expected<void> BenchmarkArguments::Verify()
{
    if (auto ret = EncoderConfigArguments::Verify(); !ret) {
        return ret.error();
    }
    if (inputFile.empty()) {
        std::cerr << "Missing value for --input" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (inputWidth <= 0 || inputWidth % 2 != 0) {
        std::cerr << "Invalid value for --input-width (must be positive and even)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (inputHeight <= 0 || inputHeight % 2 != 0) {
        std::cerr << "Invalid value for --input-height (must be positive and even)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (outputDirectory.empty()) {
        std::cerr << "Missing value for --output-dir" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (numEncoders < 0) {
        std::cerr << "Invalid value for --num-encoders (must be >= 0)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (numDecoders < 0) {
        std::cerr << "Invalid value for --num-decoders (must be >= 0)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (numEncoders == 0 && numDecoders == 0) {
        std::cerr << "At least one encoder or decoder must be specified" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (targetFps <= 0.0) {
        std::cerr << "Invalid value for --target-fps" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (durationSeconds <= 0) {
        std::cerr << "Invalid value for --duration-seconds" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (numIterations <= 0) {
        std::cerr << "Invalid value for --num-iterations" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }

    return {};
}

void BenchmarkArguments::PrintHelpImpl()
{
    EncoderConfigArguments::PrintHelpImpl();
    std::cerr << "Command-Specific Arguments:\n";
    std::cerr << "  --input                 Input file (*.nv12, *.yuv, *.nv12.gz, *.yuv.gz) (default: " << inputFile
              << ")\n";
    std::cerr << "  --input-width           Input frame width (default: " << inputWidth << ")\n";
    std::cerr << "  --input-height          Input frame height (default: " << inputHeight << ")\n";
    std::cerr << "  --output-dir            Specify the output directory (default: " << outputDirectory << ")\n";
    std::cerr << "  --num-encoders          Specify the number of encoders to run in parallel (default: " << numEncoders
              << ")\n";
    std::cerr << "  --num-decoders          Specify the number of decoders to run in parallel (default: " << numDecoders
              << ")\n";
    std::cerr << "  --target-fps            Specify the target frames per second (default: " << targetFps << ")\n";
    std::cerr << "  --duration-seconds      Duration of each iteration (default: " << durationSeconds << ")\n";
    std::cerr << "  --num-iterations        Number of full benchmark iterations (default: " << numIterations << ")\n";
    std::cerr << "  --strict-fps-target     Strictly enforce the target fps (default: " << strictFpsTarget << ")\n";
    std::cerr << "  --wait-seconds          Seconds to wait between benchmark phases (default: " << waitSeconds << ")\n";
    std::cerr << "  --qp                    Quantization parameter for encoding (default: " << qp << ")\n";
}

expected<int> EncodeArguments::Parse(std::span<std::string_view> args)
{
    if (args[0] == "--input") {
        auto ret = ParseValue(args.subspan(1), inputFile);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--input-width") {
        int value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        inputWidth = value;
        return 1 + ret.value();
    } else if (args[0] == "--input-height") {
        int value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        inputHeight = value;
        return 1 + ret.value();
    } else if (args[0] == "--input-format") {
        VideoReader::PixelFormat value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        inputPixelFormat = value;
        return 1 + ret.value();
    } else if (args[0] == "--output") {
        auto ret = ParseValue(args.subspan(1), outputFile);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--qp") {
        auto ret = ParseValue(args.subspan(1), qp);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else {
        return EncoderConfigArguments::Parse(args);
    }
}

expected<void> EncodeArguments::Verify()
{
    if (auto ret = EncoderConfigArguments::Verify(); !ret) {
        return ret.error();
    }
    if (inputFile.empty()) {
        std::cerr << "Missing value for --input" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (!inputWidth || *inputWidth <= 0 || *inputWidth % 2 != 0) {
        std::cerr << "Invalid value for --input-width (must be positive and even)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (!inputHeight || *inputHeight <= 0 || *inputHeight % 2 != 0) {
        std::cerr << "Invalid value for --input-height (must be positive and even)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (inputFile == "-" && !inputPixelFormat) {
        inputPixelFormat = VideoReader::PixelFormat::NV12;
    }
    if (outputFile.empty()) {
        std::cerr << "Missing value for --output" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (qp < 0 || qp > 51) {
        std::cerr << "Invalid value for --qp" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return {};
}

void EncodeArguments::PrintHelpImpl()
{
    EncoderConfigArguments::PrintHelpImpl();
    std::cerr << "Command-Specific Arguments:\n";
    std::cerr << "  --input             Input file (*.nv12, *.yuv, *.nv12.gz, *.yuv.gz), or - for raw stdin\n";
    std::cerr << "  --input-width       Input frame width (required)\n";
    std::cerr << "  --input-height      Input frame height (required)\n";
    std::cerr << "  --input-format      Raw pixel format: nv12 or i420 (inferred for files, default for stdin: nv12)\n";
    std::cerr << "  --output            Output file (*.mlvc), or - for stdout\n";
    std::cerr << "  --qp                Specify the quantization parameter (0-51) (default: " << qp << ")\n";
}

expected<int> DecodeArguments::Parse(std::span<std::string_view> args)
{
    if (args[0] == "--input") {
        auto ret = ParseValue(args.subspan(1), inputFile);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--output") {
        auto ret = ParseValue(args.subspan(1), outputFile);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--output-format") {
        VideoWriter::Format value{};
        auto ret = ParseValue(args.subspan(1), value);
        if (!ret) return ret.error();
        outputFormat = value;
        return 1 + ret.value();
    } else if (args[0] == "--fps") {
        auto ret = ParseValue(args.subspan(1), outputFps);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else {
        return CommonArguments::Parse(args);
    }
}

expected<void> DecodeArguments::Verify()
{
    if (auto ret = CommonArguments::Verify(); !ret) {
        return ret.error();
    }
    if (inputFile.empty()) {
        std::cerr << "Missing value for --input" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (outputFile.empty()) {
        std::cerr << "Missing value for --output" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (outputFile == "-" && !outputFormat) {
        outputFormat = VideoWriter::Format::NV12;
    }
    if (!std::isfinite(outputFps) || outputFps < 1.0 || outputFps > 1000.0) {
        std::cerr << "Invalid value for --fps (must be between 1 and 1000)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return {};
}

void DecodeArguments::PrintHelpImpl()
{
    CommonArguments::PrintHelpImpl();
    std::cerr << "Command-Specific Arguments:\n";
    std::cerr << "  --input             Input file (*.mlvc), or - for stdin\n";
    std::cerr << "  --output            Output file (*.nv12, *.yuv, *.mkv), or - for stdout\n";
    std::cerr << "  --output-format     Output format: nv12, i420, or mkv (default for stdout: nv12)\n";
    std::cerr << "  --fps               Frame rate for MKV output (default: " << outputFps << ")\n";
}

expected<int> ValidationArguments::Parse(std::span<std::string_view> args)
{
    // Dataset / input selection
    if (args[0] == "--dataset-path") {
        auto ret = ParseValue(args.subspan(1), datasetPath);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--scenarios-list") {
        auto ret = ParseValue(args.subspan(1), scenariosList);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--num-clips-limit") {
        auto ret = ParseValue(args.subspan(1), numClipsLimit);
        if (!ret) return ret.error();
        return 1 + ret.value();
    }
    // Test configuration
    else if (args[0] == "--qp-list") {
        auto ret = ParseValue(args.subspan(1), qpList);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--exclude-overhead") {
        excludeOverhead = true;
        return 1;
    }
    // Comparison
    else if (args[0] == "--anchor-path") {
        auto ret = ParseValue(args.subspan(1), anchorPath);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else {
        return EncoderConfigArguments::Parse(args);
    }
}

expected<void> ValidationArguments::Verify()
{
    if (auto ret = EncoderConfigArguments::Verify(); !ret) {
        return ret.error();
    }
    if (datasetPath.empty()) {
        std::cerr << "Missing value for --dataset-path" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    if (qpList.empty()) {
        std::cerr << "--qp-list must contain at least one QP value" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    for (int qp : qpList) {
        if (qp < 0 || qp > 51) {
            std::cerr << "Invalid QP value " << qp << " in --qp-list (must be 0-51)" << std::endl;
            return make_error_code(Error::command_line_parse_error);
        }
    }
    if (numClipsLimit < 0) {
        std::cerr << "Invalid value for --num-clips-limit (must be >= 0)" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    return {};
}

void ValidationArguments::PrintHelpImpl()
{
    EncoderConfigArguments::PrintHelpImpl();
    std::cerr << "Command-Specific Arguments:\n";
    std::cerr << "  --dataset-path          Path to dataset configuration JSON file (default: " << datasetPath << ")\n";
    std::cerr << "  --qp-list               List of QP values to test, space-separated (default:";
    for (int qp : qpList)
        std::cerr << " " << qp;
    std::cerr << ")\n";
    std::cerr << "  --scenarios-list        List of scenarios to run, space-separated (default: all)\n";
    std::cerr << "  --num-clips-limit       Maximum number of clips to process per scenario (default: " << numClipsLimit
              << ")\n";
    std::cerr << "  --anchor-path           Path to anchor JSON file, can be empty (default: " << anchorPath << ")\n";
    std::cerr
        << "  --exclude-overhead      Exclude NALU overhead from kbps/bpp, use payload bytes only (default: off)\n";
}

expected<int> InteropArguments::Parse(std::span<std::string_view> args)
{
    if (args[0] == "--dataset-path") {
        auto ret = ParseValue(args.subspan(1), datasetPath);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--snapshots-dir") {
        auto ret = ParseValue(args.subspan(1), snapshotsDir);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--qp-list") {
        auto ret = ParseValue(args.subspan(1), qpList);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else if (args[0] == "--snapshot") {
        createSnapshot = true;
        return 1;
    } else if (args[0] == "--snapshot-qp-list") {
        auto ret = ParseValue(args.subspan(1), snapshotQpList);
        if (!ret) return ret.error();
        return 1 + ret.value();
    } else {
        return EncoderConfigArguments::Parse(args);
    }
}

expected<void> InteropArguments::Verify()
{
    if (auto ret = EncoderConfigArguments::Verify(); !ret) {
        return ret.error();
    }
    if (datasetPath.empty()) {
        std::cerr << "Missing value for --dataset-path" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    {
        std::error_code ec;
        if (!std::filesystem::exists(datasetPath, ec) || ec) {
            std::cerr << "Dataset path does not exist: " << datasetPath << std::endl;
            return make_error_code(Error::command_line_parse_error);
        }
    }
    if (!snapshotsDir.empty()) {
        std::error_code ec;
        if (!std::filesystem::is_directory(snapshotsDir, ec) || ec) {
            std::cerr << "Snapshots directory does not exist: " << snapshotsDir << std::endl;
            return make_error_code(Error::command_line_parse_error);
        }
    }
    if (qpList.empty()) {
        std::cerr << "--qp-list must contain at least one QP value" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    for (int qp : qpList) {
        if (qp < 0 || qp > 51) {
            std::cerr << "Invalid QP value " << qp << " in --qp-list (must be 0-51)" << std::endl;
            return make_error_code(Error::command_line_parse_error);
        }
    }
    if (snapshotQpList.empty()) {
        std::cerr << "--snapshot-qp-list must contain at least one QP value" << std::endl;
        return make_error_code(Error::command_line_parse_error);
    }
    for (int qp : snapshotQpList) {
        if (qp < 0 || qp > 51) {
            std::cerr << "Invalid QP value " << qp << " in --snapshot-qp-list (must be 0-51)" << std::endl;
            return make_error_code(Error::command_line_parse_error);
        }
    }
    return {};
}

void InteropArguments::PrintHelpImpl()
{
    EncoderConfigArguments::PrintHelpImpl();
    std::cerr << "Command-Specific Arguments:\n";
    std::cerr << "  --dataset-path          Path to dataset configuration JSON file (default: " << datasetPath << ")\n";
    std::cerr << "  --snapshots-dir         Override snapshots directory (default: auto-detected from test data)\n";
    std::cerr << "  --qp-list               List of QP values for testing, space-separated (default:";
    for (int qp : qpList)
        std::cerr << " " << qp;
    std::cerr << ")\n";
    std::cerr << "  --snapshot              Save snapshot as a new interop reference for this device\n";
    std::cerr << "  --snapshot-qp-list      List of QP values for snapshot, space-separated (default:";
    for (int qp : snapshotQpList)
        std::cerr << " " << qp;
    std::cerr << ")\n";
}

}  // namespace libmlvc
