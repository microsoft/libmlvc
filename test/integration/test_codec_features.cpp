// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/codec.hpp"
#include "libmlvc/codec/core/core.hpp"
#include "libmlvc/codec/core/reference_manager.hpp"
#include "libmlvc/codec/manager.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc_support/test_data.hpp"
#include "libmlvc_support/video_io.hpp"
#include "support/recovery_controller.hpp"
#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <vector>

using namespace libmlvc;

namespace {

Nv12Frame CropNv12Frame(const Nv12Frame& src, const int dstWidth, const int dstHeight)
{
    assert(dstWidth > 0 && dstHeight > 0);
    assert(dstWidth % 2 == 0 && dstHeight % 2 == 0);
    assert(dstWidth <= src.width && dstHeight <= src.height);
    assert(src.data.size() == static_cast<size_t>(src.width) * src.height * 3 / 2);

    const int offsetX = ((src.width - dstWidth) / 2) & ~1;
    const int offsetY = ((src.height - dstHeight) / 2) & ~1;

    const size_t dstYSize = static_cast<size_t>(dstWidth) * dstHeight;
    const size_t dstUvSize = static_cast<size_t>(dstWidth) * dstHeight / 2;

    Nv12Frame dst{ dstWidth, dstHeight, std::vector<std::byte>(dstYSize + dstUvSize) };

    const auto* srcY = src.data.data();
    const auto* srcUv = src.data.data() + static_cast<size_t>(src.width) * src.height;

    for (int y = 0; y < dstHeight; y++) {
        std::memcpy(&dst.data[y * dstWidth], &srcY[(y + offsetY) * src.width + offsetX], dstWidth);
    }

    const int uvOffsetX = offsetX;
    const int uvOffsetY = offsetY / 2;
    for (int y = 0; y < dstHeight / 2; y++) {
        std::memcpy(&dst.data[dstYSize + y * dstWidth], &srcUv[(y + uvOffsetY) * src.width + uvOffsetX], dstWidth);
    }

    return dst;
}

Nv12Frame ResizeNv12Frame(const Nv12Frame& src, const int dstWidth, const int dstHeight)
{
    assert(dstWidth > 0 && dstHeight > 0);
    assert(dstWidth % 2 == 0 && dstHeight % 2 == 0);
    assert(src.width > 0 && src.height > 0);
    assert(src.data.size() == static_cast<size_t>(src.width) * src.height * 3 / 2);

    const size_t dstYSize = static_cast<size_t>(dstWidth) * dstHeight;
    const size_t dstUvSize = static_cast<size_t>(dstWidth) * dstHeight / 2;

    Nv12Frame dst{ dstWidth, dstHeight, std::vector<std::byte>(dstYSize + dstUvSize) };

    const auto* srcY = src.data.data();
    const auto* srcUv = src.data.data() + static_cast<size_t>(src.width) * src.height;
    auto* dstY = dst.data.data();
    auto* dstUv = dst.data.data() + dstYSize;

    for (int y = 0; y < dstHeight; y++) {
        const int srcRow = y * src.height / dstHeight;
        for (int x = 0; x < dstWidth; x++) {
            const int srcCol = x * src.width / dstWidth;
            dstY[y * dstWidth + x] = srcY[srcRow * src.width + srcCol];
        }
    }

    for (int y = 0; y < dstHeight / 2; y++) {
        const int srcRow = y * (src.height / 2) / (dstHeight / 2);
        for (int x = 0; x < dstWidth / 2; x++) {
            const int srcCol = x * (src.width / 2) / (dstWidth / 2);
            dstUv[y * dstWidth + x * 2] = srcUv[srcRow * src.width + srcCol * 2];
            dstUv[y * dstWidth + x * 2 + 1] = srcUv[srcRow * src.width + srcCol * 2 + 1];
        }
    }

    return dst;
}

Nv12Frame FitNv12Frame(const Nv12Frame& src, const int dstWidth, const int dstHeight)
{
    assert(dstWidth > 0 && dstHeight > 0);
    assert(dstWidth % 2 == 0 && dstHeight % 2 == 0);
    assert(src.width > 0 && src.height > 0);
    assert(src.data.size() == static_cast<size_t>(src.width) * src.height * 3 / 2);

    if (src.width == dstWidth && src.height == dstHeight) {
        return src;
    }

    const double scale = std::max(static_cast<double>(dstWidth) / src.width, static_cast<double>(dstHeight) / src.height);
    const int midWidth = (static_cast<int>(src.width * scale + 0.5) + 1) & ~1;
    const int midHeight = (static_cast<int>(src.height * scale + 0.5) + 1) & ~1;

    auto mid = (midWidth != src.width || midHeight != src.height) ? ResizeNv12Frame(src, midWidth, midHeight) : src;
    if (mid.width != dstWidth || mid.height != dstHeight) {
        return CropNv12Frame(mid, dstWidth, dstHeight);
    }
    return mid;
}

// -----------------------------------------------------------------------------
// Test parameters
// -----------------------------------------------------------------------------

constexpr auto DEFAULT_CLIP_FILENAME = "VCD_s1_0380a3_640x360_30fps.nv12.gz";
constexpr int DEFAULT_WIDTH = 640;
constexpr int DEFAULT_HEIGHT = 360;

struct CodecFeatureTestParams {
    struct FrameOverride {
        // Config changes (any set field triggers encoder reconfiguration + IDR)
        std::optional<int> width;
        std::optional<int> height;
        std::optional<int> numTemporalLayers;
        // Per-frame encode parameters (sticky - persists until next override)
        std::optional<QpParams> qp;
        // LTR actions
        std::optional<int> markLtrSlot;
        std::optional<int> useLtrSlot;
    };

    // Input clip
    std::string clipFilename = DEFAULT_CLIP_FILENAME;
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int numFrames = 128;

    // Encoder
    int numTemporalLayers = 1;
    int iframePeriod = 0;

    // Encoder (LTR)
    LtrMode ltrMode = LtrMode::INTERNAL;
    int ltrStartIdx = 0;
    int ltrPeriod = 0;
    int ltrNumSlots = 4;
    int ltrRecoveryPeriod = 0;

    // Encoder (QP)
    QpParams qp{ DEFAULT_QP };

    // Decoder
    int maxDecoderTemporalLayers = MAX_TEMPORAL_LAYERS;

    // Per-frame events
    std::set<int> lostFrameIds;
    std::map<int, FrameOverride> frameOverrides;
};

// -----------------------------------------------------------------------------
// CodecFeatureTestRunner
// -----------------------------------------------------------------------------

class CodecFeatureTestRunner {
public:
    CodecFeatureTestRunner(const CodecFeatureTestParams& params, const MlvcVersion& mlvcVersion,
                           const ManagerParams& managerParams, const std::filesystem::path& testDataDir)
        : m_params{ params }
        , m_mlvcVersion{ mlvcVersion }
        , m_managerParams{ managerParams }
        , m_testDataDir{ testDataDir }
        , m_recoveryController{ m_params.lostFrameIds,
                                /*idrOnly=*/m_params.ltrPeriod == 0 && m_params.ltrMode == LtrMode::INTERNAL }
        , m_qp{ m_params.qp }
    {
    }

    void Initialize()
    {
        auto videoPath = m_testDataDir / "clips" / m_params.clipFilename;
        MLVC_LOG_INFO("Reading clip: %s", videoPath.string().c_str());
        auto clipFrames = LoadNv12Frames(
            videoPath, { .rawFrameWidth = m_params.width, .rawFrameHeight = m_params.height }, m_params.numFrames);
        ASSERT_TRUE(clipFrames) << "Failed to read clip: " << videoPath.string();
        ASSERT_GE(std::ssize(clipFrames.value()), m_params.numFrames)
            << "Clip has fewer frames (" << clipFrames.value().size() << ") than requested (" << m_params.numFrames << ")";
        m_clipFrames = std::move(clipFrames.value());

        m_manager = std::make_shared<MlvcManagerImpl>(m_managerParams);
        ASSERT_TRUE(m_manager->Initialize({}, std::array{ m_mlvcVersion })) << "Failed to initialize manager";

        auto encoderConfig = m_manager->GetDefaultEncoderConfig(m_mlvcVersion);
        ASSERT_TRUE(encoderConfig) << "Failed to get default encoder config: " << encoderConfig.error().message();
        m_config = encoderConfig.value().SetSize(m_params.width, m_params.height);
        m_config.numTemporalLayers = m_params.numTemporalLayers;
        m_config.iframePeriod = m_params.iframePeriod;
        m_config.ltrMode = m_params.ltrMode;
        m_config.ltrStartIdx = m_params.ltrStartIdx;
        m_config.ltrPeriod = m_params.ltrPeriod;
        m_config.ltrNumSlots = m_params.ltrNumSlots;
        m_config.ltrRecoveryPeriod = m_params.ltrRecoveryPeriod;
        m_ltr = std::make_unique<LtrManager>(m_config);

        auto encoder = m_manager->CreateEncoder(m_config);
        ASSERT_TRUE(encoder) << "Failed to create encoder: " << encoder.error().message();
        m_encoder = std::move(encoder.value());

        auto decoder = m_manager->CreateDecoder();
        ASSERT_TRUE(decoder) << "Failed to create decoder: " << decoder.error().message();
        m_decoder = std::move(decoder.value());
    }

    void Run()
    {
        int decodedFrameCount = 0;
        int lostFrameCount = 0;
        int filteredFrameCount = 0;

        for (int frameId = 0; frameId < m_params.numFrames; frameId++) {
            MLVC_LOG_DEBUG("CodecFeatureTest: frame %d", frameId);

            // Apply overrides for this frame
            const auto& frameOverride = GetFrameOverride(frameId);
            const bool configChangeForcesIdr = ApplyFrameOverrides(frameOverride, frameId);
            if (frameOverride.markLtrSlot) {
                EXPECT_TRUE(m_encoder->MarkNextFrameAsLtr(*frameOverride.markLtrSlot))
                    << "Failed to mark LTR at frame " << frameId << " slot " << *frameOverride.markLtrSlot;
                m_ltr->MarkNextFrameAsLtr(*frameOverride.markLtrSlot);
            }

            // Determine recovery / LTR usage (frame overrides take priority over recovery controller)
            const auto recoveryInfo = m_recoveryController.ComputeRecovery(frameId);
            const auto useLtrSlotIdx = frameOverride.useLtrSlot ? frameOverride.useLtrSlot : recoveryInfo.useLtrSlotIdx;

            // Compute expected state (before encode, to ensure expectations are independent of actual output)
            const auto expected = ComputeExpectedState(recoveryInfo.idr, useLtrSlotIdx, configChangeForcesIdr);

            // Encode, parse, and validate
            const auto inputFrame = FitNv12Frame(m_clipFrames[frameId], m_config.width, m_config.height);
            const EncodeParams encodeParams{
                .qp = m_qp,
                .forceIdr = recoveryInfo.idr,
                .useLtrSlotIdx = useLtrSlotIdx,
            };

            // GetNextFrameInfo must predict what the next Encode produces. Sample before encoding.
            auto previewInfo = m_encoder->GetNextFrameInfo(encodeParams);
            ASSERT_TRUE(previewInfo)
                << "GetNextFrameInfo failed at frame " << frameId << ": " << previewInfo.error().message();

            auto encodedFrame = m_encoder->Encode(inputFrame.View(), encodeParams);
            ASSERT_TRUE(encodedFrame) << "Failed to encode frame " << frameId << ": " << encodedFrame.error().message();

            ASSERT_NO_FATAL_FAILURE(ValidateGetNextFrameInfoParity(previewInfo.value(), encodedFrame.value().info, frameId));

            auto frameData = m_parser.Parse(encodedFrame.value().bitStream);
            ASSERT_TRUE(frameData) << "Failed to parse frame " << frameId << ": " << frameData.error().message();

            m_recoveryController.Update(frameId, frameData.value().ltrSlots);
            ASSERT_NO_FATAL_FAILURE(ValidateEncoderOutput(frameData.value(), expected, frameId));
            m_predictionChainLength = encodedFrame.value().info.predictionChainLength;

            // Decode if frame reaches the decoder (not lost and not filtered by TL)
            const bool isLost = m_params.lostFrameIds.contains(frameId);
            const bool isFiltered = frameData.value().temporalId >= m_params.maxDecoderTemporalLayers;
            if (isLost) {
                lostFrameCount++;
            } else if (isFiltered) {
                filteredFrameCount++;
            } else {
                auto decodedFrame = m_decoder->Decode(encodedFrame.value().bitStream);
                ASSERT_TRUE(decodedFrame)
                    << "Failed to decode frame " << frameId << ": " << decodedFrame.error().message();
                decodedFrameCount++;

                ASSERT_NO_FATAL_FAILURE(ValidateDecoderOutput(decodedFrame.value(), expected, frameId));
                ASSERT_NO_FATAL_FAILURE(ValidateReferenceDrift(m_encoder->GetCore(), m_decoder->GetCore(), frameId));
            }

            m_gopIdx++;
            m_frameIdx++;
        }

        MLVC_LOG_INFO("CodecFeatureTest complete: %d encoded, %d decoded, %d lost, %d filtered", m_params.numFrames,
                      decodedFrameCount, lostFrameCount, filteredFrameCount);
        EXPECT_EQ(decodedFrameCount + lostFrameCount + filteredFrameCount, m_params.numFrames);
    }

private:
    static constexpr CodecFeatureTestParams::FrameOverride NO_OVERRIDE{};

    const CodecFeatureTestParams::FrameOverride& GetFrameOverride(const int frameId) const
    {
        auto it = m_params.frameOverrides.find(frameId);
        return (it != m_params.frameOverrides.end()) ? it->second : NO_OVERRIDE;
    }

    bool ApplyFrameOverrides(const CodecFeatureTestParams::FrameOverride& frameOverride, const int frameId)
    {
        if (frameOverride.qp) {
            m_qp = *frameOverride.qp;
        }

        // Config changes (only flag as changed if the value actually differs)
        bool configChangeForcesIdr = false;
        bool anyConfigChanged = false;
        if (frameOverride.width && *frameOverride.width != m_config.width) {
            m_config.width = *frameOverride.width;
            configChangeForcesIdr = true;
            anyConfigChanged = true;
        }
        if (frameOverride.height && *frameOverride.height != m_config.height) {
            m_config.height = *frameOverride.height;
            configChangeForcesIdr = true;
            anyConfigChanged = true;
        }
        if (frameOverride.numTemporalLayers && *frameOverride.numTemporalLayers != m_config.numTemporalLayers) {
            m_config.numTemporalLayers = *frameOverride.numTemporalLayers;
            anyConfigChanged = true;
        }

        if (anyConfigChanged) {
            EXPECT_TRUE(m_encoder->Configure(m_config)) << "Failed to reconfigure encoder at frame " << frameId;
            m_gopIdx = 0;  // Mirror encoder: config change resets gopIdx (libmlvc_gop_manager.cpp:99)
        }

        return configChangeForcesIdr;
    }

    struct FrameExpectations {
        int width{};
        int height{};
        FrameType frameType{};
        bool featureResetFlag{};
        int temporalId{};
        int qp{};
        int frameIdx{};
        std::optional<int> refFrameIdx;  // nullopt for IDR frames
        std::array<LtrSlotInfo, MAX_LTR_SLOTS> ltrSlots{};
    };

    FrameExpectations ComputeExpectedState(const bool recoveryIdr, const std::optional<int>& useLtrSlotIdx,
                                           const bool configChangeForcesIdr)
    {
        const int naturalTemporalId = m_gopIdx % m_config.numTemporalLayers;
        auto recoveryResult = m_ltr->ComputeRecoveryFrameIdx(useLtrSlotIdx, naturalTemporalId);

        // Chain cap (mirrors ComputeMaxPredictionChainLength in codec_core.cpp)
        int maxChain = 0;
        if (m_config.iframePeriod > 0 && m_config.ltrMode == LtrMode::INTERNAL && m_config.ltrRecoveryPeriod > 0) {
            const int threshold = m_config.ltrPeriod * m_config.ltrRecoveryPeriod;
            maxChain = threshold > 0 ? m_config.ltrStartIdx + m_config.iframePeriod / threshold + threshold - 2
                                     : m_config.iframePeriod;
        }
        const bool chainCapExceeded = maxChain > 0 && m_predictionChainLength > maxChain;

        // IDR decision (mirrors DecideFrameType condition order)
        bool idr = false;
        if (!recoveryResult) {
            idr = true;
        } else if (recoveryIdr) {
            idr = true;
        } else if (configChangeForcesIdr) {
            idr = true;
        } else if (m_frameIdx == 0) {
            idr = true;
        } else if (m_config.iframePeriod > 0 && (m_gopIdx >= m_config.iframePeriod || m_frameIdx >= m_config.iframePeriod)) {
            idr = true;
        } else if (chainCapExceeded) {
            idr = true;
        } else if (m_frameIdx >= (1 << FRAME_IDX_BITS)) {
            idr = true;
        }

        const bool isLtrRecovery = !idr && recoveryResult && recoveryResult->has_value();

        // Index resets
        if (idr) {
            m_ltr->Reset();
            m_gopIdx = 0;
            m_frameIdx = 0;
        } else if (isLtrRecovery) {
            m_gopIdx = 0;
        }

        // Temporal ID and QP (test-specific GOP pattern)
        const int temporalId = (idr || isLtrRecovery) ? 0 : naturalTemporalId;
        const int qp = m_qp.qps[temporalId];

        // Reference frame index
        std::optional<int> refFrameIdx;
        if (!idr && isLtrRecovery) {
            refFrameIdx = recoveryResult->value();
        } else if (!idr) {
            // tId=0 walks history for last base-layer frame (mirrors GopManager::ComputeFrameInfo).
            // tId=1 always references the immediately preceding frame (which is always tId=0).
            refFrameIdx = (temporalId == 0) ? m_lastBaseLayerFrameIdx.value_or(m_frameIdx - 1) : m_frameIdx - 1;
        }

        // LTR frame marking (mirrors encoder's UpdateFrameMarking, must be after temporal ID computation).
        m_ltr->UpdateFrameMarking(m_frameIdx, temporalId, isLtrRecovery, useLtrSlotIdx);

        if (temporalId == 0) {
            m_lastBaseLayerFrameIdx = m_frameIdx;
        }

        return {
            .width = m_config.width,
            .height = m_config.height,
            .frameType = idr ? FrameType::I_FRAME : (isLtrRecovery ? FrameType::LTR_RECOVERY : FrameType::P_FRAME),
            .featureResetFlag = false,
            .temporalId = temporalId,
            .qp = qp,
            .frameIdx = m_frameIdx,
            .refFrameIdx = refFrameIdx,
            .ltrSlots = m_ltr->GetLtrSlots(),
        };
    }

    static void ValidateGetNextFrameInfoParity(const FrameInfo& preview, const FrameInfo& encoded, const int frameId)
    {
        EXPECT_EQ(preview.frameType, encoded.frameType) << "GetNextFrameInfo frameType mismatch at frame " << frameId;
        EXPECT_EQ(preview.frameIdx, encoded.frameIdx) << "GetNextFrameInfo frameIdx mismatch at frame " << frameId;
        EXPECT_EQ(preview.refFrameIdx, encoded.refFrameIdx) << "GetNextFrameInfo refFrameIdx mismatch at frame " << frameId;
        EXPECT_EQ(preview.temporalId, encoded.temporalId) << "GetNextFrameInfo temporalId mismatch at frame " << frameId;
        EXPECT_EQ(preview.predictionChainLength, encoded.predictionChainLength)
            << "GetNextFrameInfo predictionChainLength mismatch at frame " << frameId;
    }

    void ValidateEncoderOutput(const FrameData& frameData, const FrameExpectations& expected, const int frameId) const
    {
        EXPECT_EQ(frameData.DisplayWidth(), expected.width) << "Wrong display width at frame " << frameId;
        EXPECT_EQ(frameData.DisplayHeight(), expected.height) << "Wrong display height at frame " << frameId;
        EXPECT_EQ(frameData.frameType == FrameType::I_FRAME, expected.frameType == FrameType::I_FRAME)
            << "Wrong frame type at frame " << frameId;
        EXPECT_EQ(frameData.featureResetFlag, expected.featureResetFlag) << "Wrong feature_reset_flag at frame " << frameId;
        EXPECT_EQ(frameData.maxTemporalLayers, MAX_TEMPORAL_LAYERS) << "Wrong max_temporal_layers at frame " << frameId;
        EXPECT_EQ(frameData.temporalId, expected.temporalId) << "Wrong temporal_id at frame " << frameId;
        EXPECT_LT(frameData.temporalId, m_config.numTemporalLayers)
            << "temporalId must be < numTemporalLayers at frame " << frameId;
        EXPECT_EQ(frameData.qp, expected.qp) << "Wrong QP at frame " << frameId;
        EXPECT_EQ(frameData.curFrameIdx, expected.frameIdx) << "Wrong cur_frame_idx at frame " << frameId;
        if (expected.refFrameIdx) {
            EXPECT_EQ(frameData.refFrameIdx, *expected.refFrameIdx) << "Wrong ref_frame_idx at frame " << frameId;
        }
        EXPECT_EQ(frameData.ltrSlots, expected.ltrSlots) << "Wrong LTR slots at frame " << frameId;
    }

    void ValidateDecoderOutput(const DecodedFrame& decoded, const FrameExpectations& expected, const int frameId) const
    {
        EXPECT_EQ(decoded.frame.Width(), expected.width) << "Wrong decoded width at frame " << frameId;
        EXPECT_EQ(decoded.frame.Height(), expected.height) << "Wrong decoded height at frame " << frameId;
        EXPECT_EQ(decoded.info.frameType == FrameType::I_FRAME, expected.frameType == FrameType::I_FRAME)
            << "Wrong decoded IDR flag at frame " << frameId;
        EXPECT_EQ(decoded.info.temporalId, expected.temporalId) << "Wrong decoded temporal_id at frame " << frameId;
    }

    static void ValidateReferenceDrift(const IMlvcEncoderCore& encoderCore, const IMlvcDecoderCore& decoderCore,
                                       const int frameId)
    {
        const auto encoderType = encoderCore.GetEncoderInterfaceType();
        const auto decoderType = decoderCore.GetDecoderInterfaceType();

        if (encoderType == EncoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P
            && decoderType == DecoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P) {
            ASSERT_NO_FATAL_FAILURE(ValidateReferenceDrift(static_cast<const MlvcEncoderCore&>(encoderCore),
                                                           static_cast<const MlvcDecoderCore&>(decoderCore), frameId));
        } else {
            FAIL() << "Unknown encoder/decoder core type: encoder=" << static_cast<int>(encoderType)
                   << " decoder=" << static_cast<int>(decoderType);
        }
    }

    static void ValidateReferenceDrift(const MlvcEncoderCore& encoderCore, const MlvcDecoderCore& decoderCore,
                                       const int frameId)
    {
        const auto encoderFeature = encoderCore.GetModelOutputs().feature.Data();
        const auto decoderFeature = decoderCore.GetModelOutputs().feature.Data();
        ASSERT_EQ(encoderFeature.size(), decoderFeature.size()) << "Feature size mismatch at frame " << frameId;
        EXPECT_TRUE(std::ranges::equal(encoderFeature, decoderFeature)) << "Reference buffer drift at frame " << frameId;
    }

    const CodecFeatureTestParams m_params;
    const MlvcVersion m_mlvcVersion;
    const ManagerParams m_managerParams;
    const std::filesystem::path m_testDataDir;

    std::vector<Nv12Frame> m_clipFrames;
    std::shared_ptr<MlvcManagerImpl> m_manager;
    std::unique_ptr<MlvcEncoderImpl> m_encoder;
    std::unique_ptr<MlvcDecoderImpl> m_decoder;
    MlvcParser m_parser;
    RecoveryController m_recoveryController;
    EncoderConfig m_config{};
    std::unique_ptr<LtrManager> m_ltr;
    int m_gopIdx{};
    int m_frameIdx{};
    int m_predictionChainLength{};
    std::optional<int> m_lastBaseLayerFrameIdx;
    QpParams m_qp{};
};

}  // namespace

// -----------------------------------------------------------------------------
// Test fixture
// -----------------------------------------------------------------------------

class CodecFeatureTests : public ::testing::Test {
protected:
    CodecFeatureTests()
    {
        // To speed up tests, disable session caching
        m_managerParams.computeUnit = GetTestConfig().computeUnit;
        m_managerParams.enableSessionCaching = false;
        m_managerParams.winmlInitMode = GetTestConfig().winmlInitMode;
    }

    void RunTest(const CodecFeatureTestParams& params)
    {
        CodecFeatureTestRunner runner(params, m_mlvcVersion, m_managerParams, m_testDataDir);
        ASSERT_NO_FATAL_FAILURE(runner.Initialize());
        ASSERT_NO_FATAL_FAILURE(runner.Run());
    }

    void SetupEncoder(MlvcManager& manager, std::optional<MlvcEncoder>& encoder, int width = DEFAULT_WIDTH,
                      int height = DEFAULT_HEIGHT, LtrMode ltrMode = LtrMode::INTERNAL)
    {
        auto encoderConfig = manager.GetDefaultEncoderConfig(m_mlvcVersion);
        ASSERT_TRUE(encoderConfig) << "Failed to get default encoder config: " << encoderConfig.error().message();
        encoderConfig.value().SetSize(width, height);
        encoderConfig.value().ltrMode = ltrMode;
        auto enc = manager.CreateEncoder(encoderConfig.value());
        ASSERT_TRUE(enc) << "Failed to create encoder: " << enc.error().message();
        encoder = std::move(enc.value());
    }

    expected<MlvcManager> CreateManager()
    {
        return MlvcManager::CreateFromDirectory(m_managerParams, {}, std::array{ m_mlvcVersion });
    }

    static Nv12Frame MakeDummyFrame(int width = DEFAULT_WIDTH, int height = DEFAULT_HEIGHT)
    {
        EXPECT_GT(width, 0);
        EXPECT_GT(height, 0);
        EXPECT_EQ(width % 2, 0) << "NV12 requires even width";
        EXPECT_EQ(height % 2, 0) << "NV12 requires even height";
        return { width, height, std::vector<std::byte>(width * height * 3 / 2, std::byte{ 128 }) };
    }

    std::filesystem::path m_testDataDir = GetTestDataDir();
    MlvcVersion m_mlvcVersion = GetTestConfig().mlvcVersion;
    ManagerParams m_managerParams;
};

// -----------------------------------------------------------------------------
// Basic tests (multi-resolution)
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, Basic_960x540)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_960x540_30fps.nv12.gz",
        .width = 960,
        .height = 540,
        .numFrames = 32,
    }));
}

TEST_F(CodecFeatureTests, Basic_640x360)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_640x360_30fps.nv12.gz",
        .width = 640,
        .height = 360,
        .numFrames = 128,
    }));
}

TEST_F(CodecFeatureTests, Basic_320x180)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_320x180_30fps.nv12.gz",
        .width = 320,
        .height = 180,
        .numFrames = 32,
    }));
}

TEST_F(CodecFeatureTests, Basic_320x240)
{
    // 4:3 aspect ratio, resized from 640x360 clip
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_640x360_30fps.nv12.gz",
        .width = 640,
        .height = 360,
        .numFrames = 32,
        .frameOverrides = { { 0, { .width = 320, .height = 240 } } },
    }));
}

TEST_F(CodecFeatureTests, Basic_240x180)
{
    // 4:3 aspect ratio, resized from 320x180 clip
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_320x180_30fps.nv12.gz",
        .width = 320,
        .height = 180,
        .numFrames = 32,
        .frameOverrides = { { 0, { .width = 240, .height = 180 } } },
    }));
}

TEST_F(CodecFeatureTests, Basic_360x360)
{
    // 1:1 aspect ratio, resized from 640x360 clip
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_640x360_30fps.nv12.gz",
        .width = 640,
        .height = 360,
        .numFrames = 32,
        .frameOverrides = { { 0, { .width = 360, .height = 360 } } },
    }));
}

TEST_F(CodecFeatureTests, Basic_360x640)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s4_021e30_360x640_30fps.nv12.gz",
        .width = 360,
        .height = 640,
        .numFrames = 32,
    }));
}

// -----------------------------------------------------------------------------
// Resolution and aspect ratio changes
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, ResolutionChange)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_640x360_30fps.nv12.gz",
        .width = 640,
        .height = 360,
        .numFrames = 64,
        .frameOverrides = {
            { 15, { .width = 320, .height = 180 } },
            { 30, { .width = 480, .height = 270 } },
        },
    }));
}

TEST_F(CodecFeatureTests, AspectRatioChange)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_640x360_30fps.nv12.gz",
        .width = 640,
        .height = 360,
        .numFrames = 64,
        .frameOverrides = {
            { 15, { .width = 360, .height = 360 } },   // 16:9 → 1:1
            { 30, { .width = 480, .height = 360 } },   // 1:1 → 4:3
        },
    }));
}

TEST_F(CodecFeatureTests, TemporalLayers_ResolutionChange)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .clipFilename = "VCD_s1_0380a3_640x360_30fps.nv12.gz",
        .width = 640,
        .height = 360,
        .numFrames = 64,
        .numTemporalLayers = 2,
        .frameOverrides = {
            { 15, { .width = 320, .height = 180 } },
            { 30, { .width = 480, .height = 270 } },
        },
    }));
}

// -----------------------------------------------------------------------------
// Temporal layers and QP
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, TemporalLayers_2Layer)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 32, .numTemporalLayers = 2 }));
}

TEST_F(CodecFeatureTests, TemporalLayers_Decode_TL0_Only)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 32, .numTemporalLayers = 2, .maxDecoderTemporalLayers = 1 }));
}

TEST_F(CodecFeatureTests, TemporalLayerChange)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .numTemporalLayers = 1,
        .frameOverrides = {
            { 20, { .numTemporalLayers = 2 } },
            { 40, { .numTemporalLayers = 1 } },
        },
    }));
}

TEST_F(CodecFeatureTests, TemporalLayerChange_Oscillation)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .numTemporalLayers = 2,
        .ltrPeriod = 0,
        .lostFrameIds = { 15, 30 },
        .frameOverrides = {
            { 3, { .numTemporalLayers = 1 } },
            { 4, { .numTemporalLayers = 2 } },
            { 6, { .numTemporalLayers = 1 } },
            { 26, { .numTemporalLayers = 2 } },
            { 35, { .numTemporalLayers = 1 } },
            { 41, { .numTemporalLayers = 2 } },
            { 42, { .numTemporalLayers = 1 } },
            { 50, { .numTemporalLayers = 2 } },
        },
    }));
}

TEST_F(CodecFeatureTests, QpEnhancementLayer)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 32,
        .numTemporalLayers = 2,
        .qp = QpParams(DEFAULT_QP, 10),
    }));
}

TEST_F(CodecFeatureTests, QpChange)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .qp = 20,
        .frameOverrides = {
            { 20, { .qp = 40 } },
            { 40, { .qp = 10 } },
        },
    }));
}

// -----------------------------------------------------------------------------
// I-frame period
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, IframePeriod)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 64, .iframePeriod = 16 }));
}

// -----------------------------------------------------------------------------
// IDR recovery
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, IdrRecovery)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 64, .ltrPeriod = 0, .lostFrameIds = { 40 } }));
}

TEST_F(CodecFeatureTests, IdrRecovery_BurstLoss)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 64, .ltrPeriod = 0, .lostFrameIds = { 40, 41, 42 } }));
}

TEST_F(CodecFeatureTests, TemporalLayers_IdrRecovery)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .numTemporalLayers = 2,
        .ltrPeriod = 0,
        .lostFrameIds = { 40 },
    }));
}

// -----------------------------------------------------------------------------
// LTR recovery
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, LtrRecovery)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 64, .ltrPeriod = 32, .lostFrameIds = { 40 } }));
}

TEST_F(CodecFeatureTests, LtrRecovery_BurstLoss)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 64, .ltrPeriod = 32, .lostFrameIds = { 40, 41, 42 } }));
}

TEST_F(CodecFeatureTests, TemporalLayers_LtrRecovery)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .numTemporalLayers = 2,
        .ltrPeriod = 32,
        .lostFrameIds = { 40 },
    }));
}

TEST_F(CodecFeatureTests, LtrRecovery_WithStartIdx)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .ltrStartIdx = 8,
        .ltrPeriod = 32,
        .lostFrameIds = { 40 },
    }));
}

TEST_F(CodecFeatureTests, LtrRecovery_IframePeriodFromFrameIdx)
{
    // LTR recovery doesn't reset frameIdx. With iframePeriod=48, after LTR recovery at frame 21,
    // frameIdx continues growing until frameIdx >= 48 triggers IDR even though gopIdx is small.
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .iframePeriod = 48,
        .ltrPeriod = 16,
        .lostFrameIds = { 20 },
    }));
}

// -----------------------------------------------------------------------------
// Proactive LTR recovery
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, ProactiveLtr)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 128, .ltrPeriod = 16, .ltrRecoveryPeriod = 2 }));
}

TEST_F(CodecFeatureTests, TemporalLayers_ProactiveLtr)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 128,
        .numTemporalLayers = 2,
        .ltrPeriod = 16,
        .ltrRecoveryPeriod = 2,
    }));
}

TEST_F(CodecFeatureTests, ProactiveLtr_WithStartIdx)
{
    // With ltrStartIdx > 0, the IDR frame (frameIdx=0) is excluded from proactive recovery.
    // The first proactive-eligible mark happens at ltrStartIdx.
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 128,
        .ltrStartIdx = 8,
        .ltrPeriod = 16,
        .ltrRecoveryPeriod = 2,
    }));
}

TEST_F(CodecFeatureTests, ProactiveLtr_FrequentLoss)
{
    // Heavy frame loss (every 2nd frame starting at frame 8) with proactive LTR recovery.
    // Useful for inspecting prediction chain length growth under loss conditions.
    // Uses same ltrPeriod/ltrRecoveryPeriod as ProactiveLtr for comparable chain length analysis.
    std::set<int> lostFrames;
    for (int i = 8; i < 128; i += 2) {
        lostFrames.insert(i);
    }
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 128,
        .ltrPeriod = 16,
        .ltrRecoveryPeriod = 2,
        .lostFrameIds = lostFrames,
    }));
}

TEST_F(CodecFeatureTests, ProactiveLtr_ChainLengthCap)
{
    // Under heavy loss, internal proactive/external-requested LTR recoveries can grow chain length; the cap forces IDR.
    std::set<int> lostFrames;
    for (int i = 8; i < 256; i += 2) {
        lostFrames.insert(i);
    }
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 256,
        .numTemporalLayers = 1,
        .iframePeriod = 256,
        .ltrStartIdx = 8,
        .ltrPeriod = 16,
        .ltrRecoveryPeriod = 4,
        .lostFrameIds = lostFrames,
    }));
}

TEST_F(CodecFeatureTests, ProactiveLtr_ChainLengthCap_2Layer)
{
    // Same as ProactiveLtr_ChainLengthCap, but with 2TL enabled.
    std::set<int> lostFrames;
    for (int i = 8; i < 256; i += 2) {
        lostFrames.insert(i);
    }
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 256,
        .numTemporalLayers = 2,
        .iframePeriod = 256,
        .ltrStartIdx = 8,
        .ltrPeriod = 16,
        .ltrRecoveryPeriod = 4,
        .lostFrameIds = lostFrames,
    }));
}

// -----------------------------------------------------------------------------
// LTR configuration
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, ExternalLtrMode)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .ltrMode = LtrMode::EXTERNAL,
        .frameOverrides = {
            { 10, { .markLtrSlot = 1 } },
            { 20, { .markLtrSlot = 2 } },
            { 30, { .useLtrSlot = 1 } },
        },
    }));
}

TEST_F(CodecFeatureTests, ExternalLtrMode_TemporalLayers)
{
    // Mark LTR on TL1 frame (frame 11, gopIdx=11 is TL1 with 2TL) - mark deferred to next TL0 (frame 12)
    ASSERT_NO_FATAL_FAILURE(RunTest({
        .numFrames = 64,
        .numTemporalLayers = 2,
        .ltrMode = LtrMode::EXTERNAL,
        .frameOverrides = {
            { 10, { .markLtrSlot = 1 } },  // TL0 - marks immediately
            { 11, { .markLtrSlot = 2 } },  // TL1 - deferred to frame 12 (next TL0)
            { 30, { .useLtrSlot = 1 } },
        },
    }));
}

TEST_F(CodecFeatureTests, SingleLtrSlot)
{
    // Single slot overwritten on every periodic mark; recovery always references the most recent mark
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 64, .ltrPeriod = 16, .ltrNumSlots = 1, .lostFrameIds = { 40 } }));
}

TEST_F(CodecFeatureTests, MultipleLtrSlots)
{
    ASSERT_NO_FATAL_FAILURE(RunTest({ .numFrames = 128, .ltrPeriod = 16, .ltrNumSlots = 2 }));
}

// -----------------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, Decoder_MissingIdr)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder));

    auto decoder = manager->CreateDecoder();
    ASSERT_TRUE(decoder);

    auto frame = MakeDummyFrame();

    // Copy bitstream - encoder reuses internal buffer on next Encode()
    auto idrResult = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(idrResult);
    std::vector<std::byte> idrBitstream(idrResult.value().bitStream.begin(), idrResult.value().bitStream.end());

    auto pFrame = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(pFrame);

    // Feed only P-frame to fresh decoder -> should be a recoverable error
    auto result = decoder.value().Decode(pFrame.value().bitStream);
    EXPECT_FALSE(result) << "P-frame without prior IDR should fail";
    if (!result) {
        EXPECT_TRUE(IsRecoverableDecoderError(result.error()));
    }

    // Verify a fresh decoder can decode the IDR (decoder factory remains usable)
    auto decoder2 = manager->CreateDecoder();
    ASSERT_TRUE(decoder2);
    auto recoveryResult = decoder2.value().Decode(idrBitstream);
    EXPECT_TRUE(recoveryResult) << "Fresh decoder should decode IDR: " << recoveryResult.error().message();
}

TEST_F(CodecFeatureTests, Decoder_VersionMismatch)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder));

    auto decoder = manager->CreateDecoder();
    ASSERT_TRUE(decoder);

    auto frame = MakeDummyFrame();

    // Encode a valid frame, then corrupt the MLVC version in the bitstream
    auto encodedFrame = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(encodedFrame);

    // Corrupt mlvc_version_major in the SPS NALU
    std::vector<std::byte> corrupted(encodedFrame.value().bitStream.begin(), encodedFrame.value().bitStream.end());
    constexpr size_t spsVersionMajorOffset = 6;  // start code (4) + NALU header (2)
    ASSERT_GT(corrupted.size(), spsVersionMajorOffset);
    corrupted[spsVersionMajorOffset] = std::byte{ 255 };

    auto result = decoder.value().Decode(corrupted);
    EXPECT_FALSE(result) << "Corrupted MLVC version should fail";
    if (!result) {
        EXPECT_FALSE(IsRecoverableDecoderError(result.error()));
    }

    // Manager should still be able to create working decoders
    auto decoder2 = manager->CreateDecoder();
    ASSERT_TRUE(decoder2);
    auto validResult = decoder2.value().Decode(encodedFrame.value().bitStream);
    EXPECT_TRUE(validResult) << "Fresh decoder should work: " << validResult.error().message();
}

TEST_F(CodecFeatureTests, Decoder_EmptyBitstream)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder));

    auto decoder = manager->CreateDecoder();
    ASSERT_TRUE(decoder);

    std::vector<std::byte> empty;
    auto result = decoder.value().Decode(empty);
    EXPECT_FALSE(result) << "Empty bitstream should fail";
    if (!result) {
        EXPECT_TRUE(IsPartialAccessUnitError(result.error()));
        EXPECT_TRUE(IsRecoverableDecoderError(result.error()));
    }

    // Decoder should remain usable
    auto frame = MakeDummyFrame();
    auto encodedFrame = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(encodedFrame);
    auto recoveryResult = decoder.value().Decode(encodedFrame.value().bitStream);
    EXPECT_TRUE(recoveryResult) << "Decoder should recover: " << recoveryResult.error().message();
}

TEST_F(CodecFeatureTests, Decoder_TruncatedBitstream)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder));

    auto decoder = manager->CreateDecoder();
    ASSERT_TRUE(decoder);

    auto frame = MakeDummyFrame();

    auto encodedFrame = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(encodedFrame);

    const auto& bs = encodedFrame.value().bitStream;
    std::vector<std::byte> truncated(bs.begin(), bs.begin() + bs.size() / 2);
    auto result = decoder.value().Decode(truncated);
    EXPECT_FALSE(result) << "Truncated bitstream should fail";
    if (!result) {
        EXPECT_TRUE(IsRecoverableDecoderError(result.error()));
    }

    // Decoder should remain usable
    auto recoveryResult = decoder.value().Decode(encodedFrame.value().bitStream);
    EXPECT_TRUE(recoveryResult) << "Decoder should recover: " << recoveryResult.error().message();
}

TEST_F(CodecFeatureTests, Decoder_CorruptedPayload)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder));

    auto decoder = manager->CreateDecoder();
    ASSERT_TRUE(decoder);

    auto frame = MakeDummyFrame();

    auto encodedFrame = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(encodedFrame);

    // Corrupt the last few payload bytes (flip bits)
    std::vector<std::byte> corrupted(encodedFrame.value().bitStream.begin(), encodedFrame.value().bitStream.end());
    constexpr size_t numBytesToCorrupt = 10;
    ASSERT_GT(corrupted.size(), numBytesToCorrupt);
    for (size_t i = corrupted.size() - numBytesToCorrupt; i < corrupted.size(); i++) {
        corrupted[i] ^= std::byte{ 0xFF };
    }
    auto result = decoder.value().Decode(corrupted);
    EXPECT_FALSE(result) << "Corrupted payload should fail";
    if (!result) {
        EXPECT_TRUE(IsRecoverableDecoderError(result.error()));
    }

    // Decoder should remain usable after IDR
    auto freshIdr = encoder->Encode(frame.View(), { .qp = DEFAULT_QP, .forceIdr = true });
    ASSERT_TRUE(freshIdr);
    auto recoveryResult = decoder.value().Decode(freshIdr.value().bitStream);
    EXPECT_TRUE(recoveryResult) << "Decoder should recover after IDR: " << recoveryResult.error().message();
}

TEST_F(CodecFeatureTests, Decoder_RecoveryAfterError)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder));

    auto frame = MakeDummyFrame();

    // Copy bitstreams - encoder reuses internal buffer on next Encode()
    auto idrResult = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(idrResult);
    std::vector<std::byte> idrBitstream(idrResult.value().bitStream.begin(), idrResult.value().bitStream.end());

    auto pResult = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(pResult);
    std::vector<std::byte> pBitstream(pResult.value().bitStream.begin(), pResult.value().bitStream.end());

    // Decode IDR successfully
    auto decoder = manager->CreateDecoder();
    ASSERT_TRUE(decoder);
    auto result1 = decoder.value().Decode(idrBitstream);
    ASSERT_TRUE(result1);

    // Inject truncated frame to cause error
    std::vector<std::byte> truncated(pBitstream.begin(), pBitstream.begin() + pBitstream.size() / 2);
    auto errorResult = decoder.value().Decode(truncated);
    EXPECT_FALSE(errorResult) << "Truncated P-frame should fail";

    // Same decoder recovers with a new IDR
    auto freshIdr = encoder->Encode(frame.View(), { .qp = DEFAULT_QP, .forceIdr = true });
    ASSERT_TRUE(freshIdr);
    auto recoveryResult = decoder.value().Decode(freshIdr.value().bitStream);
    EXPECT_TRUE(recoveryResult) << "Decoder should recover with IDR: " << recoveryResult.error().message();

    // Continue decoding normally
    auto pFrame2 = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(pFrame2);
    auto normalResult = decoder.value().Decode(pFrame2.value().bitStream);
    EXPECT_TRUE(normalResult) << "Normal decode should work: " << normalResult.error().message();
}

TEST_F(CodecFeatureTests, Encoder_InvalidResolution)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder));

    for (const auto& frame : { MakeDummyFrame(320, 180), MakeDummyFrame(DEFAULT_WIDTH, 358), MakeDummyFrame(2000, 2000) }) {
        auto result = encoder->Encode(frame.View(), {});
        ASSERT_FALSE(result) << "Mismatched frame dimensions should be rejected";
        EXPECT_EQ(result.error(), make_error_code(Error::invalid_argument));
    }
    EXPECT_EQ(encoder->GetStats().numFramesAttempted, 0);

    // Encoder should remain usable
    auto validFrame = MakeDummyFrame();
    auto validResult = encoder->Encode(validFrame.View(), {});
    EXPECT_TRUE(validResult) << "Encoder should remain usable: " << validResult.error().message();
}

TEST_F(CodecFeatureTests, Encoder_InvalidQp)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder));

    auto frame = MakeDummyFrame();

    auto resultLow = encoder->Encode(frame.View(), { .qp = -1 });
    EXPECT_FALSE(resultLow) << "QP -1 should be rejected";

    auto resultHigh = encoder->Encode(frame.View(), { .qp = 52 });
    EXPECT_FALSE(resultHigh) << "QP 52 should be rejected";

    // Valid QP should still work
    auto validResult = encoder->Encode(frame.View(), {});
    EXPECT_TRUE(validResult) << "Encoder should remain usable: " << validResult.error().message();
}

TEST_F(CodecFeatureTests, Encoder_InvalidLtrSlot)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    std::optional<MlvcEncoder> encoder;
    ASSERT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder, DEFAULT_WIDTH, DEFAULT_HEIGHT, LtrMode::EXTERNAL));

    auto decoder = manager->CreateDecoder();
    ASSERT_TRUE(decoder);

    auto frame = MakeDummyFrame();

    auto idrResult = encoder->Encode(frame.View(), {});
    ASSERT_TRUE(idrResult);
    ASSERT_TRUE(decoder.value().Decode(idrResult.value().bitStream));

    // Try using LTR slot that hasn't been marked - encoder ignores stale hint
    auto pResult = encoder->Encode(frame.View(), { .qp = DEFAULT_QP, .useLtrSlotIdx = 5 });
    ASSERT_TRUE(pResult) << "Encoder should handle stale LTR hint: " << pResult.error().message();
    auto decodedFallback = decoder.value().Decode(pResult.value().bitStream);
    ASSERT_TRUE(decodedFallback) << "Decoder should decode frame: " << decodedFallback.error().message();
    EXPECT_NE(decodedFallback.value().info.frameType, FrameType::I_FRAME)
        << "Empty LTR slot hint should be ignored (not trigger IDR)";

    // Encoder should remain usable
    auto normalResult = encoder->Encode(frame.View(), {});
    EXPECT_TRUE(normalResult) << "Encoder should remain usable: " << normalResult.error().message();
}

// -----------------------------------------------------------------------------
// Frame input tests
// -----------------------------------------------------------------------------

TEST_F(CodecFeatureTests, Encoder_NonContiguousFrame)
{
    auto manager = CreateManager();
    ASSERT_TRUE(manager);

    auto encodeDummyFrame = [&](int width, int height, int stride, int chromaOffset = 0,
                                int bufferExtraBytes = 0) -> std::vector<std::byte> {
        std::optional<MlvcEncoder> encoder;
        EXPECT_NO_FATAL_FAILURE(SetupEncoder(*manager, encoder, width, height));
        if (!encoder) return {};

        // Create a dummy NV12 frame with given stride
        std::vector<std::byte> frameBuffer(stride * height * 3 / 2 + chromaOffset + bufferExtraBytes, std::byte{ 0 });
        std::span<std::byte> yPlane{ frameBuffer.data(), static_cast<size_t>(stride) * height };
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                yPlane[y * stride + x] = std::byte{ 128 };
            }
        }
        std::span<std::byte> uvPlane{ frameBuffer.data() + stride * height + chromaOffset,
                                      static_cast<size_t>(stride) * height / 2 };
        for (int y = 0; y < height / 2; y++) {
            for (int x = 0; x < width / 2; x++) {
                uvPlane[y * stride + x * 2] = std::byte{ 64 };
                uvPlane[y * stride + x * 2 + 1] = std::byte{ 192 };
            }
        }

        const Nv12FrameView frame{ width, height, stride, yPlane, uvPlane };
        auto encodedFrame = encoder->Encode(frame, {});
        if (!encodedFrame) {
            ADD_FAILURE() << "Failed to encode frame: " << encodedFrame.error().message();
            return {};
        }
        return { encodedFrame.value().bitStream.begin(), encodedFrame.value().bitStream.end() };
    };

    // Base case: contiguous frame
    auto contiguous = encodeDummyFrame(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_WIDTH);
    ASSERT_FALSE(contiguous.empty());

    // Stride 672
    ASSERT_EQ(encodeDummyFrame(DEFAULT_WIDTH, DEFAULT_HEIGHT, 672), contiguous);

    // Stride 720
    ASSERT_EQ(encodeDummyFrame(DEFAULT_WIDTH, DEFAULT_HEIGHT, 720), contiguous);

    // Stride 720 with chroma offset 128
    ASSERT_EQ(encodeDummyFrame(DEFAULT_WIDTH, DEFAULT_HEIGHT, 720, 128), contiguous);

    // Stride 720 with chroma offset 128 and extra bytes at the end
    ASSERT_EQ(encodeDummyFrame(DEFAULT_WIDTH, DEFAULT_HEIGHT, 720, 128, 256), contiguous);
}
