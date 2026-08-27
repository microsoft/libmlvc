// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc_support/encoder_overrides.hpp"
#include "libmlvc_support/test_data.hpp"
#include "libmlvc_support/video_io.hpp"

#include <libmlvc/libmlvc.hpp>

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace libmlvc {

class IArguments {
public:
    virtual ~IArguments() = default;
    expected<void> ParseCommandLine(int argc, char** argv);
    void PrintHelp();

protected:
    virtual expected<int> Parse(std::span<std::string_view> args) = 0;
    virtual expected<void> Verify() = 0;
    virtual void PrintHelpImpl() = 0;
};

class CommonArguments : public IArguments {
public:
    std::filesystem::path modelBundlesDir = GetDefaultModelBundlesDir();
    MlvcVersion mlvcVersion = GetDefaultModelVersion();
    ComputeUnit computeUnit = ComputeUnit::NPU;
    LogLevel logLevel = LogLevel::Error;
    bool enableModelCache = true;
    bool enableSessionCaching = true;
    WinMlInitMode winmlInitMode = WinMlInitMode::AppSdk;

protected:
    expected<int> Parse(std::span<std::string_view> args) override;
    expected<void> Verify() override;
    void PrintHelpImpl() override;

    static expected<int> ParseValue(std::span<std::string_view> args, MlvcVersion& output);
    static expected<int> ParseValue(std::span<std::string_view> args, ComputeUnit& output);
    static expected<int> ParseValue(std::span<std::string_view> args, LogLevel& output);
    static expected<int> ParseValue(std::span<std::string_view> args, LtrMode& output);
    static expected<int> ParseValue(std::span<std::string_view> args, WinMlInitMode& output);
    static expected<int> ParseValue(std::span<std::string_view> args, int& output);
    static expected<int> ParseValue(std::span<std::string_view> args, double& output);
    static expected<int> ParseValue(std::span<std::string_view> args, bool& output);
    static expected<int> ParseValue(std::span<std::string_view> args, VideoReader::PixelFormat& output);
    static expected<int> ParseValue(std::span<std::string_view> args, VideoWriter::Format& output);
    static expected<int> ParseValue(std::span<std::string_view> args, std::filesystem::path& output);
    static expected<int> ParseValue(std::span<std::string_view> args, std::string& output);
    static expected<int> ParseValue(std::span<std::string_view> args, std::vector<int>& output);
    static expected<int> ParseValue(std::span<std::string_view> args, std::vector<std::string>& output);
};

class EncoderConfigArguments : public CommonArguments, public EncoderConfigOverrides {
protected:
    expected<int> Parse(std::span<std::string_view> args) override;
    expected<void> Verify() override;
    void PrintHelpImpl() override;
};

class BenchmarkArguments : public EncoderConfigArguments {
public:
    std::filesystem::path inputFile = GetTestDataDir() / "clips/VCD_s1_0380a3_960x540_30fps.nv12.gz";
    int inputWidth = 960;
    int inputHeight = 540;
    std::filesystem::path outputDirectory = "./";
    int numEncoders = 0;
    int numDecoders = 0;
    double targetFps = 1000.0;
    int durationSeconds = 20;
    int numIterations = 1;
    bool strictFpsTarget = true;
    int waitSeconds = 0;
    int qp = 26;

    void Save(const std::filesystem::path& filename) const;

protected:
    expected<int> Parse(std::span<std::string_view> args) override;
    expected<void> Verify() override;
    void PrintHelpImpl() override;
};

class EncodeArguments : public EncoderConfigArguments {
public:
    std::filesystem::path inputFile;
    std::optional<int> inputWidth;
    std::optional<int> inputHeight;
    std::optional<VideoReader::PixelFormat> inputPixelFormat;
    std::filesystem::path outputFile;
    int qp = 26;

protected:
    expected<int> Parse(std::span<std::string_view> args) override;
    expected<void> Verify() override;
    void PrintHelpImpl() override;
};

class DecodeArguments : public CommonArguments {
public:
    std::filesystem::path inputFile;
    std::filesystem::path outputFile;
    std::optional<VideoWriter::Format> outputFormat;
    double outputFps = 30.0;

protected:
    expected<int> Parse(std::span<std::string_view> args) override;
    expected<void> Verify() override;
    void PrintHelpImpl() override;
};

class ValidationArguments : public EncoderConfigArguments {
public:
    // Dataset / input selection
    std::filesystem::path datasetPath = GetTestDataDir() / "datasets/VCD-960x540-s1-3s/test_config.json";
    std::vector<std::string> scenariosList{};
    int numClipsLimit = 1000;

    // Test configuration
    std::vector<int> qpList = { 16, 24, 32, 40 };
    bool excludeOverhead = false;

    // Comparison
    std::filesystem::path anchorPath = GetTestDataDir() / "datasets/VCD-960x540-s1-3s/intel_hw_hevc_lp.json";

protected:
    expected<int> Parse(std::span<std::string_view> args) override;
    expected<void> Verify() override;
    void PrintHelpImpl() override;
};

class InteropArguments : public EncoderConfigArguments {
public:
    std::filesystem::path datasetPath = GetTestDataDir() / "datasets/VCD-960x540-div/test_config.json";
    std::filesystem::path snapshotsDir;
    std::vector<int> qpList = { 0 };
    bool createSnapshot = false;
    std::vector<int> snapshotQpList = { 0, 24, 32, 40 };

protected:
    expected<int> Parse(std::span<std::string_view> args) override;
    expected<void> Verify() override;
    void PrintHelpImpl() override;
};

}  // namespace libmlvc
