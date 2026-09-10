// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "arguments.hpp"
#include "benchmark_runner.hpp"
#include "reporting.hpp"

#include "libmlvc_support/dataset.hpp"
#include "libmlvc_support/interop.hpp"
#include "libmlvc_support/mlvc_io.hpp"
#include "libmlvc_support/validate.hpp"
#include "libmlvc_support/video_io.hpp"
#include <libmlvc/libmlvc.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
    #include <fcntl.h>
    #include <io.h>
#endif

namespace libmlvc {

namespace {

expected<void> SetBinaryMode(std::FILE* file)
{
#if defined(_WIN32)
    if (_setmode(_fileno(file), _O_BINARY) == -1) {
        return make_error_code(Error::io_error);
    }
#endif
    return {};
}

expected<MlvcManager> CreateManager(const CommonArguments& args)
{
    std::cerr << "Creating manager...\n";
    const ManagerParams managerParams{
        .computeUnit = args.computeUnit,
        .enableModelCache = args.enableModelCache,
        .enableSessionCaching = args.enableSessionCaching,
        .winmlInitMode = args.winmlInitMode,
    };

    const std::filesystem::path modelBundlesDir(args.modelBundlesDir);
    const auto versionsToLoad = std::array{ args.mlvcVersion };
    auto manager = MlvcManager::CreateFromDirectory(managerParams, modelBundlesDir, versionsToLoad);
    if (!manager) {
        std::cerr << "Error: Failed to initialize MLVC manager: " << manager.error().message() << '\n';
        return manager.error();
    }
    PrintManagerInfo(manager.value().GetInfo());

    const auto versions = manager.value().GetAvailableVersions();
    std::vector<Capabilities> capabilities;
    capabilities.reserve(versions.size());
    for (const auto& v : versions) {
        auto caps = manager.value().GetCapabilities(v);
        if (!caps) {
            std::cerr << "Error: Failed to get capabilities for " << v.ToString() << ": " << caps.error().message() << '\n';
            return caps.error();
        }
        capabilities.push_back(caps.value());
    }
    PrintCapabilities(versions, capabilities);

    return manager;
}

}  // namespace

expected<void> RunEncode(const EncodeArguments& args)
{
    PrintRuntimeInfo();
    const VideoReader::Options readerOptions{
        .rawFrameWidth = args.inputWidth,
        .rawFrameHeight = args.inputHeight,
        .rawPixelFormat = args.inputPixelFormat,
    };
    if (args.inputFile == "-") {
        if (auto ret = SetBinaryMode(stdin); !ret) {
            std::cerr << "Error: Failed to configure stdin: " << ret.error().message() << '\n';
            return ret.error();
        }
    }
    auto reader = args.inputFile == "-" ? VideoReader::OpenStream(std::cin, readerOptions)
                                        : VideoReader::OpenFile(args.inputFile, readerOptions);
    if (!reader) {
        std::cerr << "Error: Failed to open input: " << reader.error().message() << '\n';
        return reader.error();
    }
    auto frame = reader->Read();
    if (!frame) {
        std::cerr << "Error: Failed to read input frame: " << frame.error().message() << '\n';
        return frame.error();
    }
    if (!*frame) {
        std::cerr << "Error: Input contains no complete frames\n";
        return make_error_code(Error::io_error);
    }

    auto managerResult = CreateManager(args);
    if (!managerResult) return managerResult.error();
    const auto& manager = managerResult.value();

    auto encoderConfig = manager.GetDefaultEncoderConfig(args.mlvcVersion);
    if (!encoderConfig) {
        std::cerr << "Error: Failed to get default encoder config: " << encoderConfig.error().message() << '\n';
        return encoderConfig.error();
    }
    encoderConfig.value().SetSize(*args.inputWidth, *args.inputHeight);
    args.Apply(encoderConfig.value());
    PrintEncoderConfig(encoderConfig.value());

    auto encoder = manager.CreateEncoder(encoderConfig.value());
    if (!encoder) {
        std::cerr << "Error: Failed to create encoder: " << encoder.error().message() << '\n';
        return encoder.error();
    }

    std::cerr << "Encoding...\n";
    if (args.outputFile == "-") {
        if (auto ret = SetBinaryMode(stdout); !ret) {
            std::cerr << "Error: Failed to configure stdout: " << ret.error().message() << '\n';
            return ret.error();
        }
    }
    auto writer = args.outputFile == "-" ? MlvcWriter::OpenStream(std::cout) : MlvcWriter::OpenFile(args.outputFile);
    if (!writer) {
        std::cerr << "Error: Failed to open output file: " << args.outputFile << '\n';
        return writer.error();
    }
    while (true) {
        auto encodedFrame = encoder.value().Encode((**frame).View(), { .qp = args.qp });
        if (!encodedFrame) {
            std::cerr << "Error: Failed to encode frame: " << encodedFrame.error().message() << '\n';
            return encodedFrame.error();
        }
        const auto& bitStream = encodedFrame.value().bitStream;
        if (auto ret = writer->Write(bitStream); !ret) {
            std::cerr << "Error: Failed to write output file: " << args.outputFile << '\n';
            return ret.error();
        }
        frame = reader->Read();
        if (!frame) {
            std::cerr << "Error: Failed to read input frame: " << frame.error().message() << '\n';
            return frame.error();
        }
        if (!*frame) {
            break;
        }
    }
    if (auto ret = writer->Close(); !ret) {
        std::cerr << "Error: Failed to close output file: " << args.outputFile << '\n';
        return ret.error();
    }
    std::cerr << "Encoding completed.\n";
    return {};
}

expected<void> RunDecode(const DecodeArguments& args)
{
    PrintRuntimeInfo();
    if (args.inputFile == "-") {
        if (auto ret = SetBinaryMode(stdin); !ret) {
            std::cerr << "Error: Failed to configure stdin: " << ret.error().message() << '\n';
            return ret.error();
        }
    }
    std::cerr << "Reading access units...\n";
    auto reader = args.inputFile == "-" ? MlvcReader::OpenStream(std::cin) : MlvcReader::OpenFile(args.inputFile);
    if (!reader) {
        std::cerr << "Error: Failed to open access units: " << reader.error().message() << '\n';
        return reader.error();
    }
    auto accessUnit = reader->Read();
    if (!accessUnit) {
        std::cerr << "Error: Failed to read access unit: " << accessUnit.error().message() << '\n';
        return accessUnit.error();
    }
    if (!*accessUnit) {
        std::cerr << "Error: Input contains no complete access units\n";
        return make_error_code(Error::io_error);
    }

    auto managerResult = CreateManager(args);
    if (!managerResult) return managerResult.error();
    const auto& manager = managerResult.value();

    auto decoder = manager.CreateDecoder();
    if (!decoder) {
        std::cerr << "Error: Failed to create decoder: " << decoder.error().message() << '\n';
        return decoder.error();
    }

    std::cerr << "Decoding...\n";

    if (args.outputFile == "-") {
        if (auto ret = SetBinaryMode(stdout); !ret) {
            std::cerr << "Error: Failed to configure stdout: " << ret.error().message() << '\n';
            return ret.error();
        }
    }
    const VideoWriter::Options writerOptions{ .format = args.outputFormat, .frameRate = args.outputFps };
    auto writer = args.outputFile == "-" ? VideoWriter::OpenStream(std::cout, writerOptions)
                                         : VideoWriter::OpenFile(args.outputFile, writerOptions);
    if (!writer) {
        std::cerr << "Error: Failed to open output file: " << args.outputFile << '\n';
        return writer.error();
    }

    while (true) {
        auto decodedFrame = decoder.value().Decode((*accessUnit)->data);
        if (!decodedFrame) {
            std::cerr << "Error: Failed to decode frame: " << decodedFrame.error().message() << '\n';
            return decodedFrame.error();
        }
        const auto& nv12Frame = decodedFrame.value().frame;

        if (auto ret = writer->Write(nv12Frame); !ret) {
            std::cerr << "Error: Failed to write output file: " << args.outputFile << '\n';
            return ret.error();
        }
        accessUnit = reader->Read();
        if (!accessUnit) {
            std::cerr << "Error: Failed to read access unit: " << accessUnit.error().message() << '\n';
            return accessUnit.error();
        }
        if (!*accessUnit) {
            break;
        }
    }

    if (auto ret = writer->Close(); !ret) {
        std::cerr << "Error: Failed to close output file: " << args.outputFile << '\n';
        return ret.error();
    }
    std::cerr << "Decoding completed.\n";
    return {};
}

expected<void> RunBenchmark(const BenchmarkArguments& args)
{
    PrintRuntimeInfo();
    auto managerResult = CreateManager(args);
    if (!managerResult) return managerResult.error();
    const auto& manager = managerResult.value();

    if (args.waitSeconds > 0) {
        std::cerr << "Waiting " << args.waitSeconds << " seconds before loading benchmark data...\n";
        std::this_thread::sleep_for(std::chrono::seconds(args.waitSeconds));
    }

    const auto& outputDir = args.outputDirectory;
    std::error_code ec;
    if (!std::filesystem::exists(outputDir, ec)) {
        std::cerr << "Creating output directory: " << outputDir.string() << "\n";
        std::filesystem::create_directories(outputDir, ec);
        if (ec) {
            std::cerr << "Error: Failed to create output directory " << outputDir.string() << ": " << ec.message() << '\n';
            return ec;
        }
    }

    args.Save(outputDir / "benchmark_args.json");

    static constexpr size_t maxNumFrames = 128;

    std::cerr << "Reading benchmark input frames...\n";
    auto frames = LoadNv12Frames(args.inputFile,
                                 { .rawFrameWidth = args.inputWidth, .rawFrameHeight = args.inputHeight }, maxNumFrames);
    if (!frames) {
        std::cerr << "Error: Failed to read input frames: " << frames.error().message() << '\n';
        return frames.error();
    }
    if (frames->empty()) {
        std::cerr << "Error: Input contains no complete frames.\n";
        return make_error_code(Error::general_failure);
    }
    auto inputFrames = std::move(frames.value());

    std::vector<MlvcAccessUnit> decoderInputAccessUnits;
    if (args.numDecoders > 0) {
        std::cerr << "Pre-encoding decoder input frames (qp=" << args.qp << ")...\n";
        auto encoderConfig = manager.GetDefaultEncoderConfig(args.mlvcVersion);
        if (!encoderConfig) {
            std::cerr << "Error: Failed to get default encoder config: " << encoderConfig.error().message() << '\n';
            return encoderConfig.error();
        }
        encoderConfig.value().SetSize(args.inputWidth, args.inputHeight);
        args.Apply(encoderConfig.value());

        auto encoder = manager.CreateEncoder(encoderConfig.value());
        if (!encoder) {
            std::cerr << "Error: Failed to create pre-encode encoder: " << encoder.error().message() << '\n';
            return encoder.error();
        }

        decoderInputAccessUnits.reserve(inputFrames.size());
        for (const auto& nv12Frame : inputFrames) {
            const Nv12FrameView frameView{ nv12Frame.width, nv12Frame.height, nv12Frame.data };
            auto encodeRes = encoder.value().Encode(frameView, { .qp = args.qp });
            if (!encodeRes) {
                std::cerr << "Error: Failed to pre-encode frame: " << encodeRes.error().message() << '\n';
                return encodeRes.error();
            }
            const auto& bitStream = encodeRes.value().bitStream;
            MlvcAccessUnit accessUnit;
            accessUnit.data.assign(bitStream.begin(), bitStream.end());
            decoderInputAccessUnits.push_back(std::move(accessUnit));
        }
        std::cerr << "  " << decoderInputAccessUnits.size() << " access units pre-encoded.\n";
    }

    StatsDump dump;
    BenchmarkResults benchmarkResults;

    std::atomic<bool> stopFlag = false;
    std::vector<std::unique_ptr<IBenchmarkRunner>> runners;
    for (int i = 0; i < args.numEncoders; i++) {
        const std::string name = "encoder_" + std::to_string(i);
        auto runner =
            std::make_unique<EncoderBenchmarkRunner>(name, args, manager, inputFrames, dump, benchmarkResults, stopFlag);
        runners.push_back(std::move(runner));
    }
    for (int i = 0; i < args.numDecoders; i++) {
        const std::string name = "decoder_" + std::to_string(i);
        auto runner = std::make_unique<DecoderBenchmarkRunner>(name, args, manager, decoderInputAccessUnits, dump,
                                                               benchmarkResults, stopFlag);
        runners.push_back(std::move(runner));
    }

    for (auto& runner : runners) {
        if (auto ret = runner->Initialize(); !ret) {
            std::cerr << "Error: Failed to initialize runner: " << ret.error().message() << '\n';
            return ret.error();
        }
    }

    if (args.waitSeconds > 0) {
        std::cerr << "Waiting " << args.waitSeconds << " seconds before running benchmark...\n";
        std::this_thread::sleep_for(std::chrono::seconds(args.waitSeconds));
    }

    std::cerr << "Running benchmarks...\n";
    for (auto& runner : runners) {
        if (auto ret = runner->Run(); !ret) {
            std::cerr << "Error: Failed to run runner: " << ret.error().message() << '\n';
            stopFlag = true;
            for (auto& runnerToJoin : runners) {
                runnerToJoin->Join();
            }
            return ret.error();
        }
    }

    for (auto& runner : runners) {
        runner->Join();
        runner.reset();
    }

    dump.Save((outputDir / "stats.json").string());

    PrintBenchmarkSummary(benchmarkResults);

    if (args.waitSeconds > 0) {
        std::cerr << "Waiting " << args.waitSeconds << " seconds before exiting...\n";
        std::this_thread::sleep_for(std::chrono::seconds(args.waitSeconds));
    }
    return {};
}

expected<void> RunValidation(const ValidationArguments& args)
{
    PrintRuntimeInfo();

    auto managerResult = CreateManager(args);
    if (!managerResult) return managerResult.error();
    const auto& manager = managerResult.value();

    // Read dataset
    const auto dataset = Dataset::Load(args.datasetPath, args.scenariosList, args.numClipsLimit);
    if (!dataset) {
        std::cerr << "Error: Failed to read dataset: " << dataset.error().message() << '\n';
        return dataset.error();
    }

    // Print validation test configuration
    PrintDataset(dataset.value());

    const auto& encoderConfigOverrides = args;
    ValidationTestRunner testRunner(manager, args.mlvcVersion, dataset.value(), args.qpList, encoderConfigOverrides,
                                    args.excludeOverhead);
    if (auto ret = testRunner.Initialize(); !ret) {
        std::cerr << "Error: Failed to initialize validation runner: " << ret.error().message() << '\n';
        return ret.error();
    }
    PrintEncoderConfig(testRunner.GetEncoderConfig());

    std::cerr << "Validation: QP:";
    for (int qp : args.qpList)
        std::cerr << " " << qp;
    if (args.numClipsLimit > 0) std::cerr << ", limit: " << args.numClipsLimit;
    if (!args.anchorPath.empty()) std::cerr << ", anchor: " << args.anchorPath.filename().string();
    if (args.excludeOverhead) std::cerr << ", exclude-overhead";
    std::cerr << "\n";

    // Load anchor metrics if path is provided
    std::vector<ValidationClipResult> anchorMetrics;
    if (!args.anchorPath.empty()) {
        auto anchorResult = ReadAnchorMetrics(args.anchorPath, dataset.value().GetFps());
        if (!anchorResult) {
            std::cerr << "Error: Failed to read anchor metrics: " << anchorResult.error().message() << '\n';
            return anchorResult.error();
        }
        anchorMetrics = std::move(anchorResult.value());
    }

    // Run validation tests
    std::cerr << "Running validation tests...\n";
    const auto validationMetrics = testRunner.Run();
    if (!validationMetrics) {
        std::cerr << "Error: Validation test failed: " << validationMetrics.error().message() << '\n';
        return validationMetrics.error();
    }

    // Extract unique scenarios from dataset
    const auto& scenarios = dataset.value().GetScenarios();

    // Generate summaries for all scenarios
    std::map<std::string, ValidationTestSummary> summaries;
    for (const auto& scenario : scenarios) {
        auto summary = ComputeValidationSummary(validationMetrics.value(), anchorMetrics, scenario);
        if (!summary) {
            std::cerr << "Error: Failed to compute validation summary for scenario " << scenario << ": "
                      << summary.error().message() << '\n';
            return summary.error();
        }
        summaries.emplace(scenario, std::move(summary.value()));
    }

    PrintValidationResultsTable(summaries);
    return {};
}

expected<bool> RunInterop(const InteropArguments& args)
{
    PrintRuntimeInfo();
    auto managerResult = CreateManager(args);
    if (!managerResult) return managerResult.error();
    const auto& manager = managerResult.value();

    const auto datasetResult = Dataset::Load(args.datasetPath);
    if (!datasetResult) {
        std::cerr << "Error: Failed to read dataset: " << datasetResult.error().message() << '\n';
        return datasetResult.error();
    }
    const auto& dataset = datasetResult.value();

    // Initialize interop runner
    const auto& runQpList = args.createSnapshot ? args.snapshotQpList : args.qpList;
    auto runnerResult = InteropTestRunner::Create(manager, args.mlvcVersion, dataset, runQpList, args, args.snapshotsDir);
    if (!runnerResult) {
        std::cerr << "Error: Failed to initialize interop runner: " << runnerResult.error().message() << '\n';
        return runnerResult.error();
    }
    auto& runner = runnerResult.value();

    // Print configuration
    PrintDataset(dataset);

    if (args.createSnapshot) {
        // Snapshot mode: encode + save, then check for duplicates
        std::cerr << "Creating snapshot...\n";
        PrintEncoderConfig(runner.GetEncoderConfig());
        auto snapshot = runner.CreateSnapshot();
        if (!snapshot) {
            std::cerr << "Error: Failed to create snapshot: " << snapshot.error().message() << '\n';
            return snapshot.error();
        }
        PrintSnapshotResult(snapshot.value());
        std::cerr << "Saved to: " << (runner.GetSnapshotsDir() / snapshot.value().snapshotName).string() << "\n";

        const auto duplicates = runner.FindDuplicateSnapshots(snapshot.value());
        if (!duplicates.empty()) {
            std::cerr << "\n"
                      << ColorCode("\033[33m") << "Warning:" << ColorCode("\033[0m")
                      << " snapshot is bit-exact with existing reference(s):\n";
            for (const auto& name : duplicates)
                std::cerr << "  " << name << "\n";
            std::cerr << "Consider removing duplicate(s).\n";
        }
        return true;
    } else {

        // Interop test mode: decode reference bitstreams, evaluate
        const auto referenceSnapshots = runner.ListSnapshots();
        if (referenceSnapshots.empty()) {
            std::cerr << "\nNo reference snapshots found. Run with --snapshot first.\n";
            PrintPassFail(false, "no reference snapshots found.");
            return false;
        }
        std::cerr << "Running interop tests against " << referenceSnapshots.size() << " reference snapshot"
                  << (referenceSnapshots.size() != 1 ? "s" : "") << "...\n";

        bool allPassed = true;
        std::vector<InteropResult> interopResults;
        for (const auto& snapshotDir : referenceSnapshots) {
            auto interopResult = runner.RunInteropTest(snapshotDir);
            if (!interopResult) {
                std::cerr << "Error: Interop test failed for snapshot " << snapshotDir.string() << ": "
                          << interopResult.error().message() << '\n';
                return interopResult.error();
            }
            PrintInteropResult(interopResult.value());

            if (!interopResult.value().evaluationPassed) {
                allPassed = false;
            }
            interopResults.push_back(std::move(interopResult.value()));
        }

        PrintInteropSummaryTable(interopResults);
        PrintPassFail(allPassed, allPassed ? "all interop checks passed." : "one or more interop checks failed.");
        return allPassed;
    }
}

}  // namespace libmlvc

int main(int argc, char** argv)
{
    using namespace libmlvc;

    const std::string_view cmd{ argc >= 2 ? argv[1] : "" };
    if (cmd == "encode") {
        EncodeArguments args;
        if (auto ret = args.ParseCommandLine(argc - 2, argv + 2); !ret) {
            args.PrintHelp();
            return 1;
        }
        if (auto ret = RunEncode(args); !ret) return 1;
    } else if (cmd == "decode") {
        DecodeArguments args;
        if (auto ret = args.ParseCommandLine(argc - 2, argv + 2); !ret) {
            args.PrintHelp();
            return 1;
        }
        if (auto ret = RunDecode(args); !ret) return 1;
    } else if (cmd == "benchmark") {
        BenchmarkArguments args;
        if (auto ret = args.ParseCommandLine(argc - 2, argv + 2); !ret) {
            args.PrintHelp();
            return 1;
        }
        if (auto ret = RunBenchmark(args); !ret) return 1;
    } else if (cmd == "validate") {
        ValidationArguments args;
        if (auto ret = args.ParseCommandLine(argc - 2, argv + 2); !ret) {
            args.PrintHelp();
            return 1;
        }
        if (auto ret = RunValidation(args); !ret) return 1;
    } else if (cmd == "interop") {
        InteropArguments args;
        if (auto ret = args.ParseCommandLine(argc - 2, argv + 2); !ret) {
            args.PrintHelp();
            return 1;
        }
        auto ret = RunInterop(args);
        if (!ret) return 1;
        if (!ret.value()) return 2;
    } else if (cmd == "help") {
        PrintHelp();
    } else {
        std::cerr << "Unknown command: " << cmd << "\n";
        PrintHelp();
        return 1;
    }

    return 0;
}
