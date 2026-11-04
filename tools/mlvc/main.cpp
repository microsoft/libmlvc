// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "arguments.hpp"
#include "benchmark_runner.hpp"
#include "metrics_collector.hpp"
#include "reporting.hpp"

#include "libmlvc_support/dataset.hpp"
#include "libmlvc_support/interop.hpp"
#include "libmlvc_support/mlvc_io.hpp"
#include "libmlvc_support/validate.hpp"
#include "libmlvc_support/video_io.hpp"
#include <libmlvc/libmlvc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
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

expected<MlvcManager> CreateManager(const CommonArguments& args, InitializeProgressCallback progressCallback = nullptr)
{
    std::cerr << "Creating manager...\n";
    const ManagerParams managerParams{
        .computeUnit = args.computeUnit,
        .cacheDir = args.cacheDir,
        .enableModelCache = args.enableModelCache,
        .sessionPreloadMode = args.sessionPreloadMode,
        .winmlInitMode = args.winmlInitMode,
    };

    const std::filesystem::path modelBundlesDir(args.modelBundlesDir);
    const auto versionsToLoad = std::array{ args.mlvcVersion };
    auto manager =
        MlvcManager::CreateFromDirectory(managerParams, modelBundlesDir, versionsToLoad, {}, std::move(progressCallback));
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

expected<std::vector<Nv12Frame>> LoadInputFrames(const std::vector<InputClip>& inputs)
{
    constexpr size_t maxNumFramesPerInput = 128;
    std::vector<Nv12Frame> frames;
    for (const auto& input : inputs) {
        auto clipFrames = LoadNv12Frames(input.path, { .rawFrameWidth = input.width, .rawFrameHeight = input.height },
                                         maxNumFramesPerInput);
        if (!clipFrames) {
            std::cerr << "Error: Failed to read input " << input.path.string() << ": " << clipFrames.error().message()
                      << '\n';
            return clipFrames.error();
        }
        if (clipFrames->empty()) {
            std::cerr << "Error: Input " << input.path.string() << " contains no complete frames.\n";
            return make_error_code(Error::general_failure);
        }
        std::ranges::move(*clipFrames, std::back_inserter(frames));
    }
    return frames;
}

expected<std::vector<MlvcAccessUnit>> PreEncode(const MlvcManager& manager, const BenchmarkArguments& args,
                                                const std::vector<Nv12Frame>& frames)
{
    auto encoderConfig = manager.GetDefaultEncoderConfig(args.mlvcVersion);
    if (!encoderConfig) {
        std::cerr << "Error: Failed to get default encoder config: " << encoderConfig.error().message() << '\n';
        return encoderConfig.error();
    }
    encoderConfig->SetSize(frames.front().width, frames.front().height);
    args.Apply(*encoderConfig);

    auto encoder = manager.CreateEncoder(*encoderConfig);
    if (!encoder) {
        std::cerr << "Error: Failed to create pre-encode encoder: " << encoder.error().message() << '\n';
        return encoder.error();
    }

    std::vector<MlvcAccessUnit> accessUnits;
    accessUnits.reserve(frames.size());
    for (const auto& frame : frames) {
        encoderConfig->SetSize(frame.width, frame.height);
        if (auto ret = encoder->Configure(*encoderConfig); !ret) {
            std::cerr << "Error: Failed to configure pre-encode encoder: " << ret.error().message() << '\n';
            return ret.error();
        }
        auto encodeRes = encoder->Encode(frame.View(), { .qp = args.qp });
        if (!encodeRes) {
            std::cerr << "Error: Failed to pre-encode frame: " << encodeRes.error().message() << '\n';
            return encodeRes.error();
        }
        const auto& bitStream = encodeRes->bitStream;
        accessUnits.push_back({ .data = std::vector(bitStream.begin(), bitStream.end()) });
    }
    return accessUnits;
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
    // Profiling phases, in the order they start
    constexpr struct {
        const char* start = "start";
        const char* engine = "engine";
        const char* mlvc = "mlvc";
        const char* encode = "encode";
        const char* run = "run";
        const char* stop = "stop";
        const char* exit = "exit";
    } phases{};

    // With --profile, sample the process metrics in the background
    std::unique_ptr<MetricsCollector> collector;
    if (args.profile) {
        collector = std::make_unique<MetricsCollector>();
        if (auto ret = collector->Start(std::chrono::milliseconds(args.profileIntervalMs)); !ret) return ret.error();
    }
    MetricsCollector* const metrics = collector.get();
    // Profiling is best effort; the collector reports its own errors
    const auto record = [metrics](std::string phase, std::string event) {
        if (metrics) metrics->Record(std::move(phase), std::move(event));
    };
    const auto waitAfter = [metrics](int seconds, const char* phase, const std::string& what) {
        if (seconds <= 0) return;
        std::cerr << "Waiting " << seconds << " seconds after " << what << "...\n";
        if (metrics) metrics->RecordWait(phase, "Waiting after " + what);
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
    };

    record(phases.start, "Detecting platform");
    PrintRuntimeInfo();

    // Create the output directory and save the arguments in it
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

    // Load the input first, so that the runners start right after the waits. Encoders encode the input frames;
    // decoders decode them pre-encoded, unless given their own input.
    record(phases.start, "Loading input frames");
    const bool preEncode = args.numDecoders > 0 && args.decoderInputFile.empty();
    std::vector<Nv12Frame> inputFrames;
    if (args.numEncoders > 0 || preEncode) {
        std::cerr << "Reading benchmark input frames...\n";
        auto frames = LoadInputFrames(args.inputs);
        if (!frames) return frames.error();
        inputFrames = std::move(frames.value());
    }

    std::vector<MlvcAccessUnit> decoderInputAccessUnits;
    if (!args.decoderInputFile.empty()) {
        std::cerr << "Reading decoder input access units...\n";
        auto accessUnits = ReadMlvcAccessUnits(args.decoderInputFile);
        if (!accessUnits) {
            std::cerr << "Error: Failed to read decoder input: " << accessUnits.error().message() << '\n';
            return accessUnits.error();
        }
        if (accessUnits->empty()) {
            std::cerr << "Error: Decoder input contains no access units.\n";
            return make_error_code(Error::general_failure);
        }
        decoderInputAccessUnits = std::move(accessUnits.value());
    }

    // Start up the inference engine and load the model bundles
    record(phases.engine, "Creating manager");
    std::atomic<bool> mlvcStarted = false;
    auto managerResult = CreateManager(args, [&](const InitializeProgressEvent& event) {
        if (const auto* status = std::get_if<StatusMessage>(&event)) {
            record(mlvcStarted ? phases.mlvc : phases.engine, status->message);
        } else if (std::holds_alternative<InferenceBackendReady>(event)) {
            // Blocking here pauses the start-up, so that the inference engine can release its temporary memory first
            waitAfter(args.inferenceEngineWaitSeconds, phases.engine, "inference engine setup");
            // The next status message starts the MLVC phase
            mlvcStarted = true;
        }
    });
    if (!managerResult) return managerResult.error();
    std::optional<MlvcManager> manager = std::move(managerResult.value());
    waitAfter(args.waitSeconds, phases.mlvc, "start-up");

    // Encode the decoder input before the benchmark, and save it for later runs if requested
    if (preEncode) {
        record(phases.encode, "Pre-encoding decoder input");
        std::cerr << "Pre-encoding decoder input frames (qp=" << args.qp << ")...\n";
        auto accessUnits = PreEncode(*manager, args, inputFrames);
        if (!accessUnits) return accessUnits.error();
        decoderInputAccessUnits = std::move(accessUnits.value());
        std::cerr << "  " << decoderInputAccessUnits.size() << " access units pre-encoded.\n";
        if (!args.saveDecoderInputFile.empty()) {
            if (auto ret = WriteMlvcAccessUnits(args.saveDecoderInputFile, decoderInputAccessUnits); !ret) {
                std::cerr << "Error: Failed to save decoder input: " << ret.error().message() << '\n';
                return ret.error();
            }
            std::cerr << "  Saved to " << args.saveDecoderInputFile.string() << ".\n";
        }
        waitAfter(args.waitSeconds, phases.encode, "pre-encoding decoder input");
    }

    // Create and initialize one runner per stream
    const auto statsDump = args.saveStats ? std::make_unique<StatsDump>() : nullptr;
    BenchmarkResults benchmarkResults;

    record(phases.run, "Creating runners");
    std::atomic<bool> stopFlag = false;
    std::vector<std::unique_ptr<IBenchmarkRunner>> runners;
    for (int i = 0; i < args.numEncoders; i++) {
        const std::string name = "encoder_" + std::to_string(i);
        auto runner = std::make_unique<EncoderBenchmarkRunner>(name, args, *manager, inputFrames, statsDump.get(),
                                                               benchmarkResults, stopFlag);
        runners.push_back(std::move(runner));
    }
    for (int i = 0; i < args.numDecoders; i++) {
        const std::string name = "decoder_" + std::to_string(i);
        auto runner = std::make_unique<DecoderBenchmarkRunner>(name, args, *manager, decoderInputAccessUnits,
                                                               statsDump.get(), benchmarkResults, stopFlag);
        runners.push_back(std::move(runner));
    }

    for (auto& runner : runners) {
        if (auto ret = runner->Initialize(); !ret) {
            std::cerr << "Error: Failed to initialize runner: " << ret.error().message() << '\n';
            return ret.error();
        }
    }

    // Run all streams in parallel until they finish
    record(phases.run, "Running runners");
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
    }

    // Tear down step by step, waiting after each step so that the profile shows the memory it releases
    record(phases.stop, "Destroying runners");
    runners.clear();
    waitAfter(args.waitSeconds, phases.stop, "destroying runners");
    record(phases.exit, "Destroying manager");
    manager.reset();
    waitAfter(args.waitSeconds, phases.exit, "destroying manager");
    record(phases.exit, "Releasing input data");
    inputFrames = {};
    decoderInputAccessUnits = {};
    if (metrics) metrics->Stop();

    // Report the results
    if (statsDump) statsDump->Save((outputDir / "stats.json").string());

    PrintBenchmarkSummary(benchmarkResults);
    if (metrics) {
        PrintProfileSummary(metrics->AggregateByPhase());
        PrintProfileSamples(metrics->Samples(), args.profileVerbose);
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
    // --help or -h after the command lists the command's options instead of running it
    const bool help = std::ranges::any_of(argv + std::min(argc, 2), argv + argc,
                                          [](const std::string_view arg) { return arg == "--help" || arg == "-h"; });
    if (cmd == "encode") {
        EncodeArguments args;
        if (help || !args.ParseCommandLine(argc - 2, argv + 2)) {
            EncodeArguments{}.PrintHelp();
            return help ? 0 : 1;
        }
        if (auto ret = RunEncode(args); !ret) return 1;
    } else if (cmd == "decode") {
        DecodeArguments args;
        if (help || !args.ParseCommandLine(argc - 2, argv + 2)) {
            DecodeArguments{}.PrintHelp();
            return help ? 0 : 1;
        }
        if (auto ret = RunDecode(args); !ret) return 1;
    } else if (cmd == "benchmark") {
        BenchmarkArguments args;
        if (help || !args.ParseCommandLine(argc - 2, argv + 2)) {
            BenchmarkArguments{}.PrintHelp();
            return help ? 0 : 1;
        }
        if (auto ret = RunBenchmark(args); !ret) return 1;
    } else if (cmd == "validate") {
        ValidationArguments args;
        if (help || !args.ParseCommandLine(argc - 2, argv + 2)) {
            ValidationArguments{}.PrintHelp();
            return help ? 0 : 1;
        }
        if (auto ret = RunValidation(args); !ret) return 1;
    } else if (cmd == "interop") {
        InteropArguments args;
        if (help || !args.ParseCommandLine(argc - 2, argv + 2)) {
            InteropArguments{}.PrintHelp();
            return help ? 0 : 1;
        }
        auto ret = RunInterop(args);
        if (!ret) return 1;
        if (!ret.value()) return 2;
    } else if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        PrintHelp();
    } else {
        std::cerr << "Unknown command: " << cmd << "\n";
        PrintHelp();
        return 1;
    }

    return 0;
}
