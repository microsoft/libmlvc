// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/interop.hpp"
#include "libmlvc_support/hash.hpp"
#include "libmlvc_support/mlvc_io.hpp"
#include "libmlvc_support/serialization.hpp"
#include "libmlvc_support/test_data.hpp"
#include "libmlvc_support/video_io.hpp"

#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <set>
#include <sstream>

namespace libmlvc {

namespace {

std::string GetUtcNowString(const char* fmt)
{
    const auto now = std::chrono::system_clock::now();
    const auto timeT = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &timeT);
#else
    gmtime_r(&timeT, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, fmt);
    return ss.str();
}

expected<void> VerifyReferenceCompatibility(const std::string& currentDatasetName, const SnapshotResult& reference)
{
    if (currentDatasetName != reference.datasetName) {
        std::cerr << "Error: Dataset mismatch: current='" << currentDatasetName << "', reference='"
                  << reference.datasetName << "'\n";
        return make_error_code(Error::general_failure);
    }
    return {};
}

InteropThresholds ComputeInteropThresholds(int width, int height, const PlatformInfo& encPlatform,
                                           const ManagerInfo& encConfig, const PlatformInfo& decPlatform,
                                           const ManagerInfo& decConfig)
{
    double maxPsnrDrop = 0.0;

    // OS difference
    if (encPlatform.osName != decPlatform.osName) {
        maxPsnrDrop = std::max(maxPsnrDrop, 0.1);
    }

    // Vendor difference
    if (encPlatform.cpuVendor != decPlatform.cpuVendor) {
        maxPsnrDrop = std::max(maxPsnrDrop, 0.25);
    }

    // Chip difference
    if (encPlatform.cpuSeries != decPlatform.cpuSeries) {
        maxPsnrDrop = std::max(maxPsnrDrop, 0.2);
    }

    // Compute unit difference
    if (encConfig.computeUnit != decConfig.computeUnit) {
        maxPsnrDrop = std::max(maxPsnrDrop, 0.5);
    }

    // Allow extra PSNR drop if one side is using Apple NPU
    {
        auto isAppleNpu = [](const PlatformInfo& platformInfo, const ManagerInfo& managerInfo) {
            return platformInfo.cpuVendor == "Apple" && managerInfo.computeUnit == ComputeUnit::NPU;
        };
        const bool encIsAppleNpu = isAppleNpu(encPlatform, encConfig);
        const bool decIsAppleNpu = isAppleNpu(decPlatform, decConfig);
        if ((encIsAppleNpu || decIsAppleNpu) && !(encIsAppleNpu && decIsAppleNpu)) {
            maxPsnrDrop = std::max(maxPsnrDrop, 0.75);
        }
    }

    // EP version difference (e.g. QNN SDK / OpenVINO plugin update or ORT EP version change)
    if (!encConfig.windowsAppRuntimeEpVersion.empty() && !decConfig.windowsAppRuntimeEpVersion.empty()
        && encConfig.windowsAppRuntimeEpVersion != decConfig.windowsAppRuntimeEpVersion) {
        maxPsnrDrop = std::max(maxPsnrDrop, 0.15);
    }
    if (!encConfig.onnxRuntimeEpVersion.empty() && !decConfig.onnxRuntimeEpVersion.empty()
        && encConfig.onnxRuntimeEpVersion != decConfig.onnxRuntimeEpVersion) {
        maxPsnrDrop = std::max(maxPsnrDrop, 0.15);
    }

    // Driver version difference (same device, different driver can cause small numerical shifts)
    if (!encConfig.driverVersion.empty() && !decConfig.driverVersion.empty()
        && encConfig.driverVersion != decConfig.driverVersion) {
        maxPsnrDrop = std::max(maxPsnrDrop, 0.15);
    }

    // Resolution-dependent PSNR floors (dB)
    const auto computeMeanPsnrFloor = [](int width, int height) {
        const int shortSide = std::min(width, height);
        if (shortSide >= 540) return 43.5;
        if (shortSide >= 360) return 42.1;
        if (shortSide >= 240) return 40.7;
        return 39.7;
    };
    const auto computeMinPsnrFloor = [](int width, int height) {
        const int shortSide = std::min(width, height);
        if (shortSide >= 540) return 40.3;
        if (shortSide >= 360) return 38.9;
        if (shortSide >= 240) return 37.0;
        return 35.5;
    };

    return {
        .meanPsnrFloor = computeMeanPsnrFloor(width, height),
        .minPsnrFloor = computeMinPsnrFloor(width, height),
        .maxPsnrDrop = maxPsnrDrop,
        .requireBitExact = (maxPsnrDrop == 0.0),
    };
}

double ComputeHashMatchPct(const SnapshotResult& a, const SnapshotResult& b, const std::string SnapshotClipResult::* hashField)
{
    int total = 0;
    int matches = 0;
    for (const auto& clipA : a.clipResults) {
        if ((clipA.*hashField).empty()) continue;
        auto it = std::ranges::find_if(
            b.clipResults, [&](const auto& clip) { return clip.clipName == clipA.clipName && clip.qp == clipA.qp; });
        if (it == b.clipResults.end()) continue;
        if (((*it).*hashField).empty()) continue;

        if ((*it).*hashField == clipA.*hashField) matches++;
        total++;
    }

    if (total == 0) return 0.0;
    return 100.0 * matches / total;
}

const SnapshotClipResult* FindReferenceClip(const SnapshotResult& reference, const std::string& clipName, int qp)
{
    auto it = std::ranges::find_if(reference.clipResults,
                                   [&](const auto& clip) { return clip.clipName == clipName && clip.qp == qp; });
    if (it == reference.clipResults.end()) {
        std::cerr << "Error: Reference clip '" << clipName << "' QP=" << qp << " not found in snapshot\n";
        return nullptr;
    }
    return &*it;
}

}  // namespace

std::vector<int> SnapshotResult::GetQps() const
{
    std::set<int> qps;
    for (const auto& clip : clipResults) {
        qps.insert(clip.qp);
    }
    return { qps.begin(), qps.end() };
}

std::optional<AggregatedMetrics> SnapshotResult::AggregateMetricsAtQp(int qp) const
{
    auto filtered = clipResults | std::views::filter([qp](const auto& clip) { return clip.qp == qp; })
                    | std::views::transform(&SnapshotClipResult::metrics);
    std::vector<AggregatedMetrics> res(filtered.begin(), filtered.end());
    if (res.empty()) {
        return std::nullopt;
    }
    return AggregatedMetrics::Aggregate(res);
}

expected<InteropTestRunner> InteropTestRunner::Create(const MlvcManager& manager, const MlvcVersion mlvcVersion,
                                                      const Dataset& dataset, const std::vector<int>& qpList,
                                                      const EncoderConfigOverrides& configOverrides,
                                                      const std::filesystem::path& snapshotsDir)
{
    InteropTestRunner runner(manager, mlvcVersion, dataset, qpList);
    if (auto ret = runner.Initialize(configOverrides, snapshotsDir); !ret) {
        std::cerr << "Error: Failed to initialize interop test runner\n";
        return ret.error();
    }
    return runner;
}

InteropTestRunner::InteropTestRunner(const MlvcManager& manager, const MlvcVersion mlvcVersion, const Dataset& dataset,
                                     const std::vector<int>& qpList)
    : m_manager(manager), m_mlvcVersion(mlvcVersion), m_dataset(dataset), m_qpList(qpList)
{
}

expected<void> InteropTestRunner::Initialize(const EncoderConfigOverrides& configOverrides,
                                             const std::filesystem::path& snapshotsDir)
{
    if (m_dataset.Clips().empty()) {
        std::cerr << "Error: Dataset has no clips\n";
        return make_error_code(Error::general_failure);
    }
    const auto& firstClip = m_dataset.Clips().front();
    auto encoderConfig = m_manager.GetDefaultEncoderConfig(m_mlvcVersion);
    if (!encoderConfig) {
        std::cerr << "Error: Failed to get default encoder config: " << encoderConfig.error().message() << '\n';
        return encoderConfig.error();
    }
    m_encoderConfig = encoderConfig.value().SetSize(firstClip.width, firstClip.height);
    configOverrides.Apply(m_encoderConfig);

    // Cache snapshots dir
    m_snapshotsDir = snapshotsDir.empty()
                         ? GetTestDataDir() / "snapshots" / m_encoderConfig.mlvcVersion.ToString() / m_dataset.GetName()
                         : snapshotsDir;
    return {};
}

std::vector<std::filesystem::path> InteropTestRunner::ListSnapshots() const
{
    if (!std::filesystem::is_directory(m_snapshotsDir)) return {};

    std::vector<std::filesystem::path> res;
    for (const auto& entry : std::filesystem::directory_iterator(m_snapshotsDir)) {
        std::error_code ec;
        if (!entry.is_directory() || !std::filesystem::exists(entry.path() / "snapshot.json", ec)) {
            continue;
        }
        res.push_back(entry.path());
    }
    std::ranges::sort(res);
    return res;
}

std::vector<std::string> InteropTestRunner::FindDuplicateSnapshots(const SnapshotResult& snapshot) const
{
    std::vector<std::string> duplicates;
    for (const auto& dir : ListSnapshots()) {
        auto refResult = LoadSnapshot(dir);
        if (!refResult) {
            std::cerr << "Warning: Failed to load snapshot '" << dir.string() << "', skipping\n";
            continue;
        }
        const double bitstreamMatchPct =
            ComputeHashMatchPct(snapshot, refResult.value(), &SnapshotClipResult::bitstreamHash);
        const double reconMatchPct =
            ComputeHashMatchPct(snapshot, refResult.value(), &SnapshotClipResult::reconstructionHash);
        if (bitstreamMatchPct >= 100.0 && reconMatchPct >= 100.0 && refResult.value().snapshotName != snapshot.snapshotName) {
            duplicates.push_back(refResult.value().snapshotName);
        }
    }
    return duplicates;
}

expected<SnapshotResult> InteropTestRunner::CreateSnapshot() const
{
    const auto snapshotName = GenerateSnapshotName();
    const auto timestamp = GetUtcNowString("%Y-%m-%dT%H:%M:%SZ");
    const auto outputDir = m_snapshotsDir / snapshotName;

    auto clipRes = ProcessClips(outputDir);
    if (!clipRes) {
        std::cerr << "Error: Failed to process clips for snapshot\n";
        return clipRes.error();
    }

    auto res = SnapshotResult{
        .snapshotName = snapshotName,
        .timestamp = timestamp,
        .datasetName = m_dataset.GetName(),
        .buildInfo = GetBuildInfo(),
        .platformInfo = GetPlatformInfo(),
        .managerInfo = m_manager.GetInfo(),
        .encoderConfig = m_encoderConfig,
        .clipResults = std::move(clipRes.value()),
    };

    if (auto rc = SaveSnapshot(outputDir, res); !rc) {
        std::cerr << "Error: Failed to save snapshot to: " << outputDir.string() << '\n';
        return rc.error();
    }
    return res;
}

expected<InteropResult> InteropTestRunner::RunInteropTest(const std::filesystem::path& referenceDir) const
{
    const auto snapshotName = GenerateSnapshotName();
    const auto timestamp = GetUtcNowString("%Y-%m-%dT%H:%M:%SZ");

    // Load reference snapshot
    auto referenceRes = LoadSnapshot(referenceDir);
    if (!referenceRes) {
        std::cerr << "Error: Failed to load reference snapshot from: " << referenceDir.string() << '\n';
        return referenceRes.error();
    }
    auto reference = std::move(referenceRes.value());

    // Verify reference compatibility with current test configuration
    if (auto ret = VerifyReferenceCompatibility(m_dataset.GetName(), reference); !ret) {
        std::cerr << "Error: Reference snapshot is not compatible with current test configuration\n";
        return ret.error();
    }

    // Process all clips
    auto clipRes = ProcessClips({}, &reference, referenceDir);
    if (!clipRes) {
        std::cerr << "Error: Failed to process clips for interop test\n";
        return clipRes.error();
    }

    auto localDecode = SnapshotResult{
        .snapshotName = snapshotName,
        .timestamp = timestamp,
        .referenceName = reference.snapshotName,
        .datasetName = m_dataset.GetName(),
        .buildInfo = GetBuildInfo(),
        .platformInfo = GetPlatformInfo(),
        .managerInfo = m_manager.GetInfo(),
        .encoderConfig = m_encoderConfig,
        .clipResults = std::move(clipRes.value()),
    };

    // Interop metrics
    static constexpr int CHECK_QP = 0;
    const auto refMetrics = reference.AggregateMetricsAtQp(CHECK_QP);
    if (!refMetrics) {
        std::cerr << "Error: Reference snapshot has no QP=" << CHECK_QP << " data\n";
        return make_error_code(Error::general_failure);
    }
    const auto localMetrics = localDecode.AggregateMetricsAtQp(CHECK_QP);
    if (!localMetrics) {
        std::cerr << "Error: Local decode has no QP=" << CHECK_QP << " data\n";
        return make_error_code(Error::general_failure);
    }
    const double decoderBitExactPct = ComputeHashMatchPct(reference, localDecode, &SnapshotClipResult::reconstructionHash);
    constexpr double PSNR_DROP_ROUNDING_TOLERANCE = 1e-9;
    const double psnrDrop = refMetrics->psnr - localMetrics->psnr;

    const auto metrics = InteropMetrics{
        .meanPsnr = localMetrics->psnr,
        .minPsnr = localMetrics->psnrMin,
        .psnrDrop = std::abs(psnrDrop) < PSNR_DROP_ROUNDING_TOLERANCE ? 0.0 : psnrDrop,
        .decoderBitExactPct = decoderBitExactPct,
        .bitExact = (decoderBitExactPct >= 100.0),
    };

    // Determine thresholds and whether evaluation passed
    const auto thresholds =
        ComputeInteropThresholds(reference.encoderConfig.width, reference.encoderConfig.height, reference.platformInfo,
                                 reference.managerInfo, localDecode.platformInfo, localDecode.managerInfo);
    const bool evaluationPassed = thresholds.Passed(metrics);

    // Final interop result
    const auto res = InteropResult{
        .reference = std::move(reference),
        .localDecode = std::move(localDecode),
        .metrics = metrics,
        .thresholds = thresholds,
        .evaluationPassed = evaluationPassed,
    };

    return res;
}

std::string InteropTestRunner::GenerateSnapshotName() const
{
    const auto& platform = GetPlatformInfo();
    const auto& managerInfo = m_manager.GetInfo();

    static auto ToSnakeCase = [](const std::string& s) {
        std::string result;
        for (const char c : s) {
            result += (c == ' ') ? '_' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return result;
    };

    return GetUtcNowString("%Y%m%d-%H%M%S") + "-" + ToSnakeCase(platform.cpuVendor) + "-"
           + ToSnakeCase(platform.cpuSeries) + "-" + std::string(ComputeUnitToString(managerInfo.computeUnit));
}

expected<std::vector<SnapshotClipResult>> InteropTestRunner::ProcessClips(const std::filesystem::path& outputDir,
                                                                          const SnapshotResult* reference,
                                                                          const std::filesystem::path& referenceBaseDir) const
{
    std::vector<SnapshotClipResult> res;
    for (const auto& clip : m_dataset.Clips()) {
        auto frames =
            LoadNv12Frames(clip.path, { .rawFrameWidth = clip.width, .rawFrameHeight = clip.height }, clip.frames);
        if (!frames) {
            std::cerr << "Error: Failed to read clip frames for '" << clip.name << "': " << frames.error().message() << '\n';
            return frames.error();
        }

        for (const int qp : m_qpList) {
            std::filesystem::path inputBitstreamPath;
            if (reference) {
                auto* refClip = FindReferenceClip(*reference, clip.name, qp);
                if (!refClip) return make_error_code(Error::general_failure);
                inputBitstreamPath = referenceBaseDir / refClip->bitstreamPath;
            }

            auto clipResult = RunFrameLoop(clip, qp, frames.value(), m_dataset.GetFps(), outputDir, inputBitstreamPath);
            if (!clipResult) {
                std::cerr << "Error: Failed to process clip '" << clip.name << "' at QP=" << qp << ": "
                          << clipResult.error().message() << '\n';
                return clipResult.error();
            }

            res.push_back(std::move(clipResult.value()));
        }
    }
    return res;
}

expected<SnapshotClipResult> InteropTestRunner::RunFrameLoop(const ClipMetadata& clip, int qp,
                                                             const std::vector<Nv12Frame>& frames, double fps,
                                                             const std::filesystem::path& outputDir,
                                                             const std::filesystem::path& inputBitstreamPath) const
{
    constexpr auto kBitstreamFilename = "bitstream.mlvc";

    // Output directory
    std::filesystem::path relOutputDir;
    std::filesystem::path clipOutputDir;
    if (!outputDir.empty()) {
        const auto clipStem = std::filesystem::path(clip.name).stem().string();
        relOutputDir = std::filesystem::path(clip.scenario) / clipStem / ("qp" + std::to_string(qp));
        clipOutputDir = outputDir / relOutputDir;
        std::error_code ec;
        std::filesystem::create_directories(clipOutputDir, ec);
        if (ec) {
            std::cerr << "Error: Failed to create output directory: " << clipOutputDir.string() << '\n';
            return make_error_code(Error::io_error);
        }
    }

    // Create encoder or read input bitstream
    std::optional<MlvcEncoder> encoder;
    std::vector<MlvcAccessUnit> inputAccessUnits;
    if (inputBitstreamPath.empty()) {
        auto config = m_encoderConfig;
        config.SetSize(clip.width, clip.height);
        auto encoderResult = m_manager.CreateEncoder(config);
        if (!encoderResult) {
            std::cerr << "Error: Failed to create encoder: " << encoderResult.error().message() << '\n';
            return encoderResult.error();
        }
        encoder = std::move(encoderResult.value());
    } else {
        auto accessUnits = ReadMlvcAccessUnits(inputBitstreamPath);
        if (!accessUnits) {
            std::cerr << "Error: Failed to read bitstream: " << accessUnits.error().message() << '\n';
            return accessUnits.error();
        }
        if (accessUnits.value().size() != frames.size()) {
            std::cerr << "Error: Access unit count (" << accessUnits.value().size() << ") != frame count ("
                      << frames.size() << ")\n";
            return make_error_code(Error::general_failure);
        }
        inputAccessUnits = std::move(accessUnits.value());
    }

    auto decoderResult = m_manager.CreateDecoder();
    if (!decoderResult) {
        std::cerr << "Error: Failed to create decoder: " << decoderResult.error().message() << '\n';
        return decoderResult.error();
    }
    auto decoder = std::move(decoderResult.value());

    // Open output file and create hashers
    std::ofstream bitstreamOut;
    if (encoder && !clipOutputDir.empty()) {
        const auto outputBitstreamPath = clipOutputDir / kBitstreamFilename;
        bitstreamOut.open(outputBitstreamPath, std::ios::binary);
        if (!bitstreamOut.is_open()) {
            std::cerr << "Error: Failed to open bitstream file: " << outputBitstreamPath.string() << '\n';
            return make_error_code(Error::io_error);
        }
    }

    auto reconstructionHasher = Sha256::Create();
    if (!reconstructionHasher) {
        std::cerr << "Error: Failed to create reconstruction hasher: " << reconstructionHasher.error().message() << '\n';
        return reconstructionHasher.error();
    }
    auto bitstreamHasher = Sha256::Create();
    if (!bitstreamHasher) {
        std::cerr << "Error: Failed to create bitstream hasher: " << bitstreamHasher.error().message() << '\n';
        return bitstreamHasher.error();
    }

    // Frame-by-frame encode/decode and metric calculation
    std::vector<FrameMetrics> frameMetrics;
    for (size_t frameId = 0; frameId < frames.size(); frameId++) {
        const auto nv12View = frames[frameId].View();

        // Encode or read bitstream
        std::span<const std::byte> bitStream;
        if (encoder) {
            auto encodedFrame = encoder->Encode(nv12View, { .qp = qp });
            if (!encodedFrame) {
                std::cerr << "Error: Failed to encode frame " << frameId << ": " << encodedFrame.error().message() << '\n';
                return encodedFrame.error();
            }
            bitStream = encodedFrame.value().bitStream;

            // Write and hash bitstream incrementally
            if (bitstreamOut.is_open()) {
                bitstreamOut.write(reinterpret_cast<const char*>(bitStream.data()),
                                   static_cast<std::streamsize>(bitStream.size()));
                if (!bitstreamOut.good()) {
                    std::cerr << "Error: Failed to write bitstream at frame " << frameId << '\n';
                    return make_error_code(Error::io_error);
                }
            }
            if (auto ret = bitstreamHasher.value().Update(bitStream.data(), bitStream.size()); !ret) {
                std::cerr << "Error: Failed to update bitstream hash at frame " << frameId << ": "
                          << ret.error().message() << '\n';
                return ret.error();
            }
        } else {
            bitStream = inputAccessUnits[frameId].data;
        }

        auto decodedFrame = decoder.Decode(bitStream);
        if (!decodedFrame) {
            std::cerr << "Error: Failed to decode frame " << frameId << ": " << decodedFrame.error().message() << '\n';
            return decodedFrame.error();
        }
        const auto& recFrame = decodedFrame.value().frame;

        // Update reconstruction hash
        if (auto ret = reconstructionHasher.value().Update(recFrame.YPlane().data(), recFrame.YPlane().size()); !ret) {
            std::cerr << "Error: Failed to update reconstruction hash at frame " << frameId << ": "
                      << ret.error().message() << '\n';
            return ret.error();
        }
        if (auto ret = reconstructionHasher.value().Update(recFrame.UvPlane().data(), recFrame.UvPlane().size()); !ret) {
            std::cerr << "Error: Failed to update reconstruction hash at frame " << frameId << ": "
                      << ret.error().message() << '\n';
            return ret.error();
        }

        frameMetrics.push_back(CalculateFrameMetrics(nv12View, recFrame, bitStream.size(), fps));
    }

    // Close bitstream file and check for deferred write errors
    if (bitstreamOut.is_open()) {
        bitstreamOut.close();
        if (bitstreamOut.fail()) {
            std::cerr << "Error: Failed to close bitstream file\n";
            return make_error_code(Error::io_error);
        }
    }

    // Finalize hashes
    auto reconstructionHash = reconstructionHasher.value().Finalize();
    if (!reconstructionHash) {
        std::cerr << "Error: Failed to finalize reconstruction hash: " << reconstructionHash.error().message() << '\n';
        return reconstructionHash.error();
    }

    std::string bitstreamHash;
    if (encoder) {
        auto bitstreamHashResult = bitstreamHasher.value().Finalize();
        if (!bitstreamHashResult) {
            std::cerr << "Error: Failed to finalize bitstream hash: " << bitstreamHashResult.error().message() << '\n';
            return bitstreamHashResult.error();
        }
        bitstreamHash = std::move(bitstreamHashResult.value());
    }

    SnapshotClipResult res{
        .scenario = clip.scenario,
        .clipName = clip.name,
        .qp = qp,
        .bitstreamPath = relOutputDir.empty() ? std::string{} : (relOutputDir / kBitstreamFilename).generic_string(),
        .bitstreamHash = std::move(bitstreamHash),
        .reconstructionHash = std::move(reconstructionHash.value()),
        .encoderStats = encoder ? encoder->GetStats() : EncoderStats{},
        .decoderStats = decoder.GetStats(),
        .metrics = AggregatedMetrics::Aggregate(frameMetrics),
        .frameMetrics = std::move(frameMetrics),
    };
    return res;
}

}  // namespace libmlvc
