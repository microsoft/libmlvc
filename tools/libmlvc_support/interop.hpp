// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc_support/dataset.hpp"
#include "libmlvc_support/encoder_overrides.hpp"
#include "libmlvc_support/metrics.hpp"
#include "libmlvc_support/video_io.hpp"

#include <libmlvc/libmlvc.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace libmlvc {

struct SnapshotClipResult {
    std::string scenario{};
    std::string clipName{};
    int qp{};

    // Bitstream and reconstruction
    std::string bitstreamPath{};  // Relative to snapshot dir, empty if not saved
    std::string bitstreamHash{};
    std::string reconstructionHash{};

    // Stats
    EncoderStats encoderStats{};
    DecoderStats decoderStats{};

    // Metrics
    AggregatedMetrics metrics{};
    std::vector<FrameMetrics> frameMetrics{};
};

struct SnapshotResult {
    std::string snapshotName{};
    std::string timestamp{};  // ISO 8601 UTC creation time
    std::string referenceName{};
    std::string datasetName{};

    // Build and platform
    BuildInfo buildInfo{};
    PlatformInfo platformInfo{};
    ManagerInfo managerInfo{};
    EncoderConfig encoderConfig{};

    // Results
    std::vector<SnapshotClipResult> clipResults{};

    // Helper methods
    std::vector<int> GetQps() const;
    std::optional<AggregatedMetrics> AggregateMetricsAtQp(int qp) const;
};

struct InteropMetrics {
    double meanPsnr{};            // interop decode mean PSNR at QP=0 (dB)
    double minPsnr{};             // interop decode min per-frame PSNR at QP=0 (dB)
    double psnrDrop{};            // refMeanPsnr - meanPsnr at QP=0 (positive = degradation)
    double decoderBitExactPct{};  // percentage of bit-exact decoder matches
    bool bitExact{};              // true if decoder is 100% bit-exact
};

struct InteropThresholds {
    double meanPsnrFloor{};  // absolute minimum for mean PSNR at QP=0 (dB)
    double minPsnrFloor{};   // absolute minimum for per-frame PSNR at QP=0 (dB)
    double maxPsnrDrop{};    // maximum allowed PSNR drop at QP=0 (dB)
    bool requireBitExact{};  // if true, require decoder bit-exactness

    constexpr bool Passed(const InteropMetrics& m) const
    {
        return m.meanPsnr >= meanPsnrFloor && m.minPsnr >= minPsnrFloor && m.psnrDrop <= maxPsnrDrop
               && (!requireBitExact || m.bitExact);
    }
};

struct InteropResult {
    SnapshotResult reference{};    // original snapshot metadata (from reference JSON)
    SnapshotResult localDecode{};  // interop decode of reference bitstreams
    InteropMetrics metrics{};
    InteropThresholds thresholds{};
    bool evaluationPassed{ false };
};

class InteropTestRunner {
public:
    static expected<InteropTestRunner> Create(const MlvcManager& manager, const MlvcVersion mlvcVersion,
                                              const Dataset& dataset, const std::vector<int>& qpList,
                                              const EncoderConfigOverrides& configOverrides = {},
                                              const std::filesystem::path& snapshotsDir = {});

    std::vector<std::filesystem::path> ListSnapshots() const;
    std::vector<std::string> FindDuplicateSnapshots(const SnapshotResult& snapshot) const;
    expected<SnapshotResult> CreateSnapshot() const;
    expected<InteropResult> RunInteropTest(const std::filesystem::path& referenceDir) const;
    const EncoderConfig& GetEncoderConfig() const { return m_encoderConfig; }
    const std::filesystem::path& GetSnapshotsDir() const { return m_snapshotsDir; }

private:
    InteropTestRunner(const MlvcManager& manager, const MlvcVersion mlvcVersion, const Dataset& dataset,
                      const std::vector<int>& qpList);
    expected<void> Initialize(const EncoderConfigOverrides& configOverrides, const std::filesystem::path& snapshotsDir);

    std::string GenerateSnapshotName() const;
    expected<std::vector<SnapshotClipResult>> ProcessClips(const std::filesystem::path& outputDir,
                                                           const SnapshotResult* reference = nullptr,
                                                           const std::filesystem::path& referenceBaseDir = {}) const;
    expected<SnapshotClipResult> RunFrameLoop(const ClipMetadata& clip, int qp, const std::vector<Nv12Frame>& frames,
                                              double fps, const std::filesystem::path& outputDir,
                                              const std::filesystem::path& inputBitstreamPath = {}) const;

    const MlvcManager& m_manager;
    const MlvcVersion m_mlvcVersion;
    const Dataset& m_dataset;
    const std::vector<int> m_qpList;
    EncoderConfig m_encoderConfig{};
    std::filesystem::path m_snapshotsDir{};
};

}  // namespace libmlvc
