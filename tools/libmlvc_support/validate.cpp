// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/validate.hpp"
#include "libmlvc_support/video_io.hpp"

#include <libmlvc/error_codes.hpp>

#include <boost/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <ranges>
#include <set>

namespace libmlvc {

namespace {

std::set<std::string> FindCommonClips(const std::vector<ValidationClipResult>& metrics1,
                                      const std::vector<ValidationClipResult>& metrics2)
{
    auto ComposeClipKeySet = [](const std::vector<ValidationClipResult>& metrics) {
        std::set<std::string> keys;
        for (const auto& m : metrics) {
            keys.emplace(m.ClipKey());
        }
        return keys;
    };

    const auto clips1 = ComposeClipKeySet(metrics1);
    const auto clips2 = ComposeClipKeySet(metrics2);
    std::set<std::string> res;
    std::ranges::set_intersection(clips1, clips2, std::inserter(res, res.begin()));
    return res;
}

std::vector<ValidationClipResult> FilterClipMetrics(const std::vector<ValidationClipResult>& metrics,
                                                    const std::string& scenario, const std::set<std::string>& clips)
{
    std::vector<ValidationClipResult> res;
    for (const auto& m : metrics) {
        if (!scenario.empty() && m.scenario != scenario) continue;
        if (!clips.contains(m.ClipKey())) continue;
        res.push_back(m);
    }
    return res;
}

std::map<int, AggregatedMetrics> AggregateMetricsByQp(const std::vector<ValidationClipResult>& metrics)
{
    std::map<int, std::vector<AggregatedMetrics>> groups;
    for (const auto& m : metrics) {
        groups[m.qp].push_back(m.metrics);
    }

    std::map<int, AggregatedMetrics> res;
    for (const auto& [qp, group] : groups) {
        res[qp] = AggregatedMetrics::Aggregate(group);
    }
    return res;
}

}  // namespace

expected<std::vector<ValidationClipResult>> ReadAnchorMetrics(const std::filesystem::path& anchorPath, double fps)
{
    // Read the JSON file
    std::ifstream file(anchorPath);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open anchor file: " << anchorPath.string() << '\n';
        return make_error_code(Error::io_error);
    }

    std::string jsonContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) {
        std::cerr << "Error: Failed to read anchor file: " << anchorPath.string() << '\n';
        return make_error_code(Error::io_error);
    }

    // Parse JSON
    std::error_code ec;
    boost::json::value root = boost::json::parse(jsonContent, ec);
    if (ec) {
        std::cerr << "Error: Failed to parse anchor JSON: " << ec.message() << '\n';
        return make_error_code(Error::json_parse_error);
    }

    const auto* rootObj = root.if_object();
    if (!rootObj) {
        std::cerr << "Error: Anchor JSON root is not an object\n";
        return make_error_code(Error::json_parse_error);
    }

    // Helper to extract a numeric value as double (handles both int64 and double in JSON)
    auto GetDouble = [](const boost::json::value& v) -> std::optional<double> {
        if (const auto* d = v.if_double()) return *d;
        if (const auto* i = v.if_int64()) return static_cast<double>(*i);
        return std::nullopt;
    };

    std::vector<ValidationClipResult> res;

    // Iterate scenarios
    for (const auto& [scenarioName, scenarioValue] : *rootObj) {
        const auto* scenarioObj = scenarioValue.if_object();
        if (!scenarioObj) {
            continue;
        }

        // Iterate clips
        for (const auto& [clipName, clipValue] : *scenarioObj) {
            const auto* clipObj = clipValue.if_object();
            if (!clipObj) {
                continue;
            }

            // Iterate QP entries
            for (const auto& [qpKey, qpValue] : *clipObj) {
                const auto* qpObj = qpValue.if_object();
                if (!qpObj) {
                    continue;
                }

                // Extract required fields
                const auto* bppPtr = qpObj->if_contains("ave_all_frame_bpp");
                const auto* psnrPtr = qpObj->if_contains("ave_all_frame_psnr");
                const auto* psnrYPtr = qpObj->if_contains("ave_all_frame_psnr_y");
                const auto* psnrUPtr = qpObj->if_contains("ave_all_frame_psnr_u");
                const auto* psnrVPtr = qpObj->if_contains("ave_all_frame_psnr_v");
                const auto* pixelNumPtr = qpObj->if_contains("frame_pixel_num");
                const auto* qpPtr = qpObj->if_contains("p_frame_q_index");
                const auto* iFrameNumPtr = qpObj->if_contains("i_frame_num");
                const auto* pFrameNumPtr = qpObj->if_contains("p_frame_num");

                if (!bppPtr || !psnrPtr || !psnrYPtr || !psnrUPtr || !psnrVPtr || !pixelNumPtr || !qpPtr
                    || !iFrameNumPtr || !pFrameNumPtr) {
                    std::cerr << "Error: Missing required fields in anchor entry for clip '" << std::string(clipName)
                              << "'\n";
                    return make_error_code(Error::json_parse_error);
                }

                const auto bppVal = GetDouble(*bppPtr);
                const auto psnrVal = GetDouble(*psnrPtr);
                const auto psnrYVal = GetDouble(*psnrYPtr);
                const auto psnrUVal = GetDouble(*psnrUPtr);
                const auto psnrVVal = GetDouble(*psnrVPtr);
                const auto pixelNumVal = GetDouble(*pixelNumPtr);
                const auto* qpVal = qpPtr->if_int64();
                const auto* iFrameNumVal = iFrameNumPtr->if_int64();
                const auto* pFrameNumVal = pFrameNumPtr->if_int64();

                if (!bppVal || !psnrVal || !psnrYVal || !psnrUVal || !psnrVVal || !pixelNumVal || !qpVal
                    || !iFrameNumVal || !pFrameNumVal) {
                    std::cerr << "Error: Invalid field types in anchor entry for clip '" << std::string(clipName) << "'\n";
                    return make_error_code(Error::json_parse_error);
                }

                const double bpp = *bppVal;
                const double psnr = *psnrVal;
                const double psnrY = *psnrYVal;
                const double psnrU = *psnrUVal;
                const double psnrV = *psnrVVal;
                const double pixelNum = *pixelNumVal;
                const int qp = static_cast<int>(*qpVal);
                const int frameCount = static_cast<int>(*iFrameNumVal + *pFrameNumVal);
                const double kbps = bpp * pixelNum * fps / 1024.0;

                res.push_back(ValidationClipResult{
                    .scenario = std::string(scenarioName),
                    .name = std::string(clipName),
                    .qp = qp,
                    .frameMetrics = {},
                    .metrics =
                        AggregatedMetrics{
                            .psnr = psnr,
                            .psnrY = psnrY,
                            .psnrU = psnrU,
                            .psnrV = psnrV,
                            .bpp = bpp,
                            .kbps = kbps,
                            .psnrMin = psnr,
                            .psnrMax = psnr,
                            .count = frameCount,
                        },
                });
            }
        }
    }

    return res;
}

expected<ValidationTestSummary> ComputeValidationSummary(const std::vector<ValidationClipResult>& testMetrics,
                                                         const std::vector<ValidationClipResult>& anchorMetrics,
                                                         const std::string& scenario)
{
    const bool hasAnchor = !anchorMetrics.empty();

    // Determine clips to use
    const auto commonClips = hasAnchor ? FindCommonClips(anchorMetrics, testMetrics) : [&] {
        std::set<std::string> clips;
        for (const auto& m : testMetrics)
            clips.insert(m.ClipKey());
        return clips;
    }();

    if (hasAnchor && commonClips.empty()) {
        std::cerr << "Error: No common clips found between anchor and test metrics\n";
        return make_error_code(Error::invalid_argument);
    }

    // Helper to extract metric vectors from aggregated data
    auto ExtractMetrics = [](const auto& agg) {
        struct {
            std::vector<double> psnr, psnrY, psnrU, psnrV, kbps;
        } m;
        for (const auto& v : agg | std::views::values) {
            m.psnr.push_back(v.psnr);
            m.psnrY.push_back(v.psnrY);
            m.psnrU.push_back(v.psnrU);
            m.psnrV.push_back(v.psnrV);
            m.kbps.push_back(v.kbps);
        }
        return m;
    };

    // Filter and aggregate test metrics
    const auto filteredTest = FilterClipMetrics(testMetrics, scenario, commonClips);
    const auto testAgg = AggregateMetricsByQp(filteredTest);
    if (testAgg.empty()) return ValidationTestSummary{};

    auto test = ExtractMetrics(testAgg);
    auto [testPsnrMin, testPsnrMax] = std::ranges::minmax(test.psnr);
    auto [testKbpsMin, testKbpsMax] = std::ranges::minmax(test.kbps);

    ValidationTestSummary result{
        .numCommonClips = filteredTest.size() / testAgg.size(),
        .testPsnrMin = testPsnrMin,
        .testPsnrMax = testPsnrMax,
        .testKbpsMin = testKbpsMin,
        .testKbpsMax = testKbpsMax,
        .anchorPsnrMin = std::nan(""),
        .anchorPsnrMax = std::nan(""),
        .anchorKbpsMin = std::nan(""),
        .anchorKbpsMax = std::nan(""),
        .bdRate = std::nan(""),
        .bdRateY = std::nan(""),
        .bdRateU = std::nan(""),
        .bdRateV = std::nan(""),
    };

    if (!hasAnchor) {
        return result;
    }

    // Filter and aggregate anchor metrics
    auto anchor = ExtractMetrics(AggregateMetricsByQp(FilterClipMetrics(anchorMetrics, scenario, commonClips)));

    auto [anchorPsnrMin, anchorPsnrMax] = std::ranges::minmax(anchor.psnr);
    auto [anchorKbpsMin, anchorKbpsMax] = std::ranges::minmax(anchor.kbps);
    result.anchorPsnrMin = anchorPsnrMin;
    result.anchorPsnrMax = anchorPsnrMax;
    result.anchorKbpsMin = anchorKbpsMin;
    result.anchorKbpsMax = anchorKbpsMax;

    // Calculate BD-rates
    auto bdRate = CalculateBdRate(anchor.kbps, anchor.psnr, test.kbps, test.psnr);
    if (!bdRate) return bdRate.error();
    auto bdRateY = CalculateBdRate(anchor.kbps, anchor.psnrY, test.kbps, test.psnrY);
    if (!bdRateY) return bdRateY.error();
    auto bdRateU = CalculateBdRate(anchor.kbps, anchor.psnrU, test.kbps, test.psnrU);
    if (!bdRateU) return bdRateU.error();
    auto bdRateV = CalculateBdRate(anchor.kbps, anchor.psnrV, test.kbps, test.psnrV);
    if (!bdRateV) return bdRateV.error();

    result.bdRate = *bdRate;
    result.bdRateY = *bdRateY;
    result.bdRateU = *bdRateU;
    result.bdRateV = *bdRateV;

    return result;
}

ValidationTestRunner::ValidationTestRunner(const MlvcManager& manager, const MlvcVersion mlvcVersion,
                                           const Dataset& dataset, const std::vector<int>& qpList,
                                           const EncoderConfigOverrides& configOverrides, const bool excludeOverhead)
    : m_manager(manager)
    , m_mlvcVersion(mlvcVersion)
    , m_dataset(dataset)
    , m_qpList(qpList)
    , m_configOverrides(configOverrides)
    , m_excludeOverhead(excludeOverhead)
{
}

expected<void> ValidationTestRunner::Initialize()
{
    auto encoderConfig = m_manager.GetDefaultEncoderConfig(m_mlvcVersion);
    if (!encoderConfig) {
        std::cerr << "Error: Failed to get default encoder config: " << encoderConfig.error().message() << '\n';
        return encoderConfig.error();
    }
    const auto& clip = m_dataset.Clips().front();
    m_encoderConfig = encoderConfig.value().SetSize(clip.width, clip.height);
    m_configOverrides.Apply(m_encoderConfig);
    return {};
}

expected<std::vector<ValidationClipResult>> ValidationTestRunner::Run()
{
    // Run tests
    std::vector<ValidationClipResult> res;
    for (const auto& clip : m_dataset.Clips()) {
        // Read test clip
        auto frames =
            LoadNv12Frames(clip.path, { .rawFrameWidth = clip.width, .rawFrameHeight = clip.height }, clip.frames);
        if (!frames) {
            std::cerr << "Error: Failed to read test clip frames: " << frames.error().message() << '\n';
            return frames.error();
        }

        for (const auto qp : m_qpList) {
            // Create encoder
            auto config = m_encoderConfig;
            config.SetSize(clip.width, clip.height);
            auto encoder = m_manager.CreateEncoder(config);
            if (!encoder) {
                std::cerr << "Error: Failed to create encoder: " << encoder.error().message() << '\n';
                return encoder.error();
            }

            // Create decoder
            auto decoder = m_manager.CreateDecoder();
            if (!decoder) {
                std::cerr << "Error: Failed to create decoder: " << decoder.error().message() << '\n';
                return decoder.error();
            }

            // Frame loop
            std::vector<FrameMetrics> frameMetrics;
            for (size_t i = 0; i < frames.value().size(); i++) {
                // Encode
                const auto& frame = frames.value()[i];
                const Nv12FrameView inputFrame{ frame.width, frame.height, frame.data };
                auto encodedFrame = encoder.value().Encode(inputFrame, { .qp = qp });
                if (!encodedFrame) {
                    std::cerr << "Error: Failed to encode frame: " << encodedFrame.error().message() << '\n';
                    return encodedFrame.error();
                }
                const auto& encodedBitstream = encodedFrame.value().bitStream;

                // Decode
                auto decodedFrame = decoder.value().Decode(encodedBitstream);
                if (!decodedFrame) {
                    std::cerr << "Error: Failed to decode frame: " << decodedFrame.error().message() << '\n';
                    return decodedFrame.error();
                }
                const auto& recFrame = decodedFrame.value().frame;

                // Calculate metrics
                const std::size_t bytes = m_excludeOverhead ? encodedFrame.value().payloadBytes : encodedBitstream.size();
                const auto m = CalculateFrameMetrics(inputFrame, recFrame, bytes, m_dataset.GetFps());
                frameMetrics.push_back(std::move(m));
            }

            AggregatedMetrics aggregated = AggregatedMetrics::Aggregate(frameMetrics);
            res.push_back(ValidationClipResult{
                .scenario = clip.scenario,
                .name = clip.name,
                .qp = qp,
                .frameMetrics = std::move(frameMetrics),
                .metrics = std::move(aggregated),
            });
        }
    }

    return res;
}

}  // namespace libmlvc
