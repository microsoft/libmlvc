// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc_support/dataset.hpp"
#include "libmlvc_support/encoder_overrides.hpp"
#include "libmlvc_support/metrics.hpp"

#include <libmlvc/libmlvc.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace libmlvc {

struct ValidationClipResult {
    std::string scenario{};
    std::string name{};
    int qp{};
    std::vector<FrameMetrics> frameMetrics{};
    AggregatedMetrics metrics{};

    std::string ClipKey() const { return scenario + "/" + name; }
};

struct ValidationTestSummary {
    size_t numCommonClips{};
    // Test ranges
    double testPsnrMin{};
    double testPsnrMax{};
    double testKbpsMin{};
    double testKbpsMax{};
    // Anchor ranges
    double anchorPsnrMin{};
    double anchorPsnrMax{};
    double anchorKbpsMin{};
    double anchorKbpsMax{};
    // BD-rates
    double bdRate{};
    double bdRateY{};
    double bdRateU{};
    double bdRateV{};
};

expected<std::vector<ValidationClipResult>> ReadAnchorMetrics(const std::filesystem::path& anchorPath, double fps);

expected<ValidationTestSummary> ComputeValidationSummary(const std::vector<ValidationClipResult>& testMetrics,
                                                         const std::vector<ValidationClipResult>& anchorMetrics,
                                                         const std::string& scenario = {});

class ValidationTestRunner {
public:
    ValidationTestRunner(const MlvcManager& manager, MlvcVersion mlvcVersion, const Dataset& dataset,
                         const std::vector<int>& qpList, const EncoderConfigOverrides& configOverrides,
                         bool excludeOverhead = false);
    expected<void> Initialize();
    expected<std::vector<ValidationClipResult>> Run();
    const EncoderConfig& GetEncoderConfig() const { return m_encoderConfig; }

private:
    const MlvcManager& m_manager;
    const MlvcVersion m_mlvcVersion;
    const Dataset& m_dataset;
    const std::vector<int> m_qpList;
    const EncoderConfigOverrides m_configOverrides;
    const bool m_excludeOverhead;
    EncoderConfig m_encoderConfig{};
};

}  // namespace libmlvc
