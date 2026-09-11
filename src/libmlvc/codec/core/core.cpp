// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/core.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"

#include <algorithm>

namespace libmlvc {

MlvcCodecCoreCommon::MlvcCodecCoreCommon(const MlvcVersion mlvcVersion, const int displayWidth, const int displayHeight,
                                         const std::shared_ptr<const ModelMetadata>& modelMetadata,
                                         std::unique_ptr<IScaleDecoder> scaleDecoder)
    : m_mlvcVersion{ mlvcVersion }
    , m_displayWidth{ displayWidth }
    , m_displayHeight{ displayHeight }
    , m_modelMetadata{ modelMetadata }
    , m_scaleDecoder{ std::move(scaleDecoder) }
{
}

expected<void> MlvcCodecCoreCommon::Initialize()
{
    if (m_displayWidth % 2 != 0 || m_displayHeight % 2 != 0 || m_displayWidth <= 0 || m_displayHeight <= 0) {
        MLVC_LOG_ERROR("Image dimensions must be even and positive: %d x %d", m_displayWidth, m_displayHeight);
        return make_error_code(Error::invalid_argument);
    }
    if (!m_modelMetadata) {
        MLVC_LOG_ERROR("Model metadata is null");
        return make_error_code(Error::invalid_argument);
    }
    return {};
}

expected<int> MlvcCodecCoreCommon::MapQpToQIndex(const int qp) const
{
    if (qp < MIN_QP || qp > MAX_QP) {
        MLVC_LOG_ERROR("QP value out of range: %d (expected %d - %d)", qp, MIN_QP, MAX_QP);
        return make_error_code(Error::invalid_argument);
    }

    int qIndex;
    if (!m_modelMetadata->qpMapping.has_value()) {
        qIndex = qp;
    } else {
        if (m_modelMetadata->qpMapping->size() != (MAX_QP - MIN_QP + 1)) {
            MLVC_LOG_ERROR("Invalid qpMapping size %zu", m_modelMetadata->qpMapping->size());
            return make_error_code(Error::invalid_argument);
        }
        qIndex = m_modelMetadata->qpMapping.value()[qp];
    }

    // Check qIndex range
    if (qIndex < 0 || qIndex > m_modelMetadata->qpNum) {
        MLVC_LOG_ERROR("qIndex value out of range: %d (expected 0 - %d)", qIndex, m_modelMetadata->qpNum);
        return make_error_code(Error::invalid_argument);
    }

    return qIndex;
}

expected<int> MlvcCodecCoreCommon::GetQIndexShifted(const int qIndex, const int curFrameIdx) const
{
    const auto& frameIndexMap = m_modelMetadata->frameIndexMap;
    const auto& qpShift = m_modelMetadata->qpShift;
    if (frameIndexMap.empty() || qpShift.empty()) {
        return qIndex;
    }

    const int faIdx = frameIndexMap[(curFrameIdx + 1) % frameIndexMap.size()];
    if (faIdx < 0 || faIdx >= static_cast<int>(qpShift.size())) {
        MLVC_LOG_ERROR("Invalid fa_idx=%d for qp_shift size %zu", faIdx, qpShift.size());
        return make_error_code(Error::invalid_argument);
    }

    return qIndex + qpShift[faIdx];
}

MlvcEncoderCore::MlvcEncoderCore(const MlvcVersion mlvcVersion, const EncoderConfig& config,
                                 const std::shared_ptr<const ModelMetadata>& modelMetadata,
                                 std::unique_ptr<IScaleDecoder> scaleDecoder,
                                 const std::shared_ptr<const GaussianCoderPmf>& gaussianPmf,
                                 const std::shared_ptr<const BitEstimatorPmf>& bitEstimatorPmf,
                                 const InferenceSessionPtr& session)
    : MlvcCodecCoreCommon{ mlvcVersion, config.width, config.height, modelMetadata, std::move(scaleDecoder) }
    , m_config{ config }
    , m_model{ session }
    , m_ref{ session }
    , m_ltr{ m_config }
    , m_gop{ m_config }
    , m_inputs{ session }
    , m_outputs{ session }
    , m_gaussianEncoder{ gaussianPmf }
    , m_bitEstimatorZ{ bitEstimatorPmf }
{
}

MlvcEncoderCore::~MlvcEncoderCore()
{
    MLVC_LOG_INFO("~Encoder core: preprocess: %s, inference: %s, scale_decoder: %s, entropy_coding: %s (mean/std)",
                  TimerStats2StrShort(m_stats.preprocess, true).c_str(),
                  TimerStats2StrShort(m_stats.inference, true).c_str(),
                  TimerStats2StrShort(m_stats.scaleDecoder, true).c_str(),
                  TimerStats2StrShort(m_stats.entropyCoding, true).c_str());
}

expected<void> MlvcEncoderCore::Initialize()
{
    if (auto ret = MlvcCodecCoreCommon::Initialize(); !ret) return ret.error();
    if (auto ret = m_model.Initialize(); !ret) return ret.error();
    if (auto ret = m_ref.Initialize(); !ret) return ret.error();
    if (auto ret = m_inputs.Initialize(); !ret) return ret.error();
    if (auto ret = m_outputs.Initialize(); !ret) return ret.error();
    if (auto ret = m_gaussianEncoder.Initialize(); !ret) return ret.error();
    if (auto ret = m_bitEstimatorZ.Initialize(); !ret) return ret.error();

    if (auto err = m_stream.Initialize(rans::RansVariant::RansByte, m_buffer); err) {
        MLVC_LOG_ERROR("Failed to initialize RANS encoder stream: %s", err.message().c_str());
        return make_error_code(Error::entropy_coder_init_error);
    }
    m_initialized = true;
    return {};
}

expected<void> MlvcEncoderCore::Configure(const EncoderConfig& config)
{
    if (config == m_config) {
        return {};  // No change
    }

    if (config.width != m_displayWidth || config.height != m_displayHeight) {
        MLVC_LOG_ERROR("Cannot change frame dimensions: %dx%d -> %dx%d", m_displayWidth, m_displayHeight, config.width,
                       config.height);
        return make_error_code(Error::invalid_argument);
    }

    m_config = config;
    return {};
}

expected<void> MlvcEncoderCore::MarkNextFrameAsLtr(const int ltrSlotIdx)
{
    if (!m_initialized) {
        MLVC_LOG_ERROR("Encoder not initialized");
        return make_error_code(Error::not_initialized);
    }

    if (auto ret = m_ltr.MarkNextFrameAsLtr(ltrSlotIdx); !ret) {
        return ret.error();
    }
    return {};
}

expected<FrameInfo> MlvcEncoderCore::GetNextFrameInfo(const EncoderConfig& config, const EncodeParams& params) const
{
    if (!m_initialized) {
        MLVC_LOG_ERROR("Encoder not initialized");
        return make_error_code(Error::not_initialized);
    }

    const int nextTemporalId = m_gop.PeekNextTemporalId(config);
    const auto recoveryFrameIdx = m_ltr.ComputeRecoveryFrameIdx(params.useLtrSlotIdx, nextTemporalId);
    return m_gop.ComputeFrameInfo(config, params, recoveryFrameIdx);
}

expected<EncoderCoreResult> MlvcEncoderCore::Encode(const Nv12FrameView& frame, const EncodeParams& params)
{
    // Validate arguments
    if (!m_initialized) {
        MLVC_LOG_ERROR("Encoder not initialized");
        return make_error_code(Error::not_initialized);
    }

    if (frame.Width() != m_displayWidth || frame.Height() != m_displayHeight) {
        MLVC_LOG_ERROR("Invalid input frame size: %dx%d (expected %dx%d)", frame.Width(), frame.Height(),
                       m_displayWidth, m_displayHeight);
        return make_error_code(Error::invalid_argument);
    }

    const int nextTemporalId = m_gop.PeekNextTemporalId(m_config);
    const auto recoveryFrameIdx = m_ltr.ComputeRecoveryFrameIdx(params.useLtrSlotIdx, nextTemporalId);
    const auto frameInfo = m_gop.ComputeFrameInfo(m_config, params, recoveryFrameIdx);
    if (frameInfo.frameType == FrameType::I_FRAME) {
        m_ref.Reset();
        m_ltr.Reset();
        m_gop.Reset();
    }
    const auto frameIdx = frameInfo.frameIdx;
    const auto refFrameIdx = frameInfo.refFrameIdx;
    const auto temporalId = frameInfo.temporalId;
    const int qpFrame = params.qp.qps[temporalId];

    // Qp mapping / shifting
    const auto qIndex = MapQpToQIndex(qpFrame);
    if (!qIndex) return qIndex.error();
    const auto qIndexShifted = GetQIndexShifted(qIndex.value(), frameIdx);
    if (!qIndexShifted) return qIndexShifted.error();

    // Reconfigure input and output
    {
        const auto refIdx = frameIdx == refFrameIdx ? std::nullopt : std::make_optional(refFrameIdx);
        auto slot = m_ref.ResolveInputSlot(refIdx);
        if (!slot) return slot.error();
        if (auto ret = m_inputs.BindReference(slot.value()->GetData()); !ret) return ret.error();
    }

    {
        auto slot = m_ref.AcquireOutputSlot(frameIdx, temporalId);
        if (!slot) return slot.error();
        if (auto ret = m_outputs.BindReference(slot.value()->GetData()); !ret) return ret.error();
    }

    // Prepare input tensor
    InputTransformer::Result inputTransformRes{};
    {
        ScopedOpTimer opTimer{ m_stats.preprocess };
        inputTransformRes =
            m_inputTransformer.Transform(frame, m_modelMetadata->modelWidth, m_modelMetadata->modelHeight, m_inputs.x);
    }

    // Inference
    {
        ScopedOpTimer opTimer{ m_stats.inference };
        m_inputs.qIndexShifted(0) = qIndexShifted.value();
        if (auto ret = m_model.Run(m_inputs, m_outputs); !ret) return ret.error();
    }

    m_zRawInt32.Create(m_outputs.zRaw.Shape());
    m_yRaw0Int32.Create(m_outputs.yRaw0.Shape());
    m_yRaw1Int32.Create(m_outputs.yRaw1.Shape());
    {
        Float16ToInt32(m_outputs.zRaw, m_zRawInt32);
        Float16ToInt32(m_outputs.yRaw0, m_yRaw0Int32);
        Float16ToInt32(m_outputs.yRaw1, m_yRaw1Int32);
    }

    // Scale decoder
    {
        ScopedOpTimer opTimer{ m_stats.scaleDecoder };
        if (auto ret = m_scaleDecoder->ExtractScales(m_zRawInt32); !ret) return ret.error();
    }
    const ScaleValues& scales = m_scaleDecoder->GetScales();

    // Entropy coding
    {
        ScopedOpTimer opTimer{ m_stats.entropyCoding };
        // Ensure stream is empty before encoding
        (void)m_stream.Flush();
        if (auto ret = m_gaussianEncoder.Encode(m_stream, m_yRaw1Int32, scales.scales1); !ret) return ret.error();
        if (auto ret = m_gaussianEncoder.Encode(m_stream, m_yRaw0Int32, scales.scales0); !ret) return ret.error();
        if (auto ret = m_bitEstimatorZ.Encode(m_stream, m_zRawInt32, qIndex.value()); !ret) return ret.error();
    }

    // Update
    const bool isRecovery = frameInfo.frameType == FrameType::LTR_RECOVERY;
    m_ltr.UpdateFrameMarking(frameIdx, temporalId, isRecovery, params.useLtrSlotIdx);
    m_ref.PruneUnreferencedSlots(m_ltr.GetLtrSlots());
    m_gop.Update(m_config, frameInfo);

    // Compose output
    return EncoderCoreResult{
        .info = frameInfo,
        .data =
            FrameData{
                .mlvcVersion = m_mlvcVersion,
                .modelWidth = m_modelMetadata->modelWidth,
                .modelHeight = m_modelMetadata->modelHeight,
                .cropOffsets = inputTransformRes.cropOffsets,
                .transposeFlag = inputTransformRes.transposeFlag,
                .frameIdxBits = FRAME_IDX_BITS,
                .temporalId = temporalId,
                .maxTemporalLayers = MAX_TEMPORAL_LAYERS,
                .frameType = frameInfo.frameType,
                .qp = qpFrame,
                .featureResetFlag = false,
                .curFrameIdx = frameIdx,
                .refFrameIdx = refFrameIdx,
                .ltrSlots = m_ltr.GetLtrSlots(),
                .payload = m_stream.Flush(),
            },
    };
}

MlvcDecoderCore::MlvcDecoderCore(const MlvcVersion mlvcVersion, const int displayWidth, const int displayHeight,
                                 const std::shared_ptr<const ModelMetadata>& modelMetadata,
                                 std::unique_ptr<IScaleDecoder> scaleDecoder,
                                 const std::shared_ptr<const GaussianCoderPmf>& gaussianPmf,
                                 const std::shared_ptr<const BitEstimatorPmf>& bitEstimatorPmf,
                                 const InferenceSessionPtr& session)
    : MlvcCodecCoreCommon{ mlvcVersion, displayWidth, displayHeight, modelMetadata, std::move(scaleDecoder) }
    , m_model{ session }
    , m_ref{ session }
    , m_gop{}
    , m_inputs{ session }
    , m_outputs{ session }
    , m_gaussianDecoder{ gaussianPmf }
    , m_bitEstimatorZ{ bitEstimatorPmf }
{
}

MlvcDecoderCore::~MlvcDecoderCore()
{
    MLVC_LOG_INFO("~Decoder core: entropy_coding: %s, scale_decoder: %s, inference: %s, postprocess: %s (mean/std)",
                  TimerStats2StrShort(m_stats.entropyCoding, true).c_str(),
                  TimerStats2StrShort(m_stats.scaleDecoder, true).c_str(),
                  TimerStats2StrShort(m_stats.inference, true).c_str(),
                  TimerStats2StrShort(m_stats.postprocess, true).c_str());
}

expected<void> MlvcDecoderCore::Initialize()
{
    if (auto ret = MlvcCodecCoreCommon::Initialize(); !ret) return ret.error();
    if (auto ret = m_model.Initialize(); !ret) return ret.error();
    if (auto ret = m_ref.Initialize(); !ret) return ret.error();
    if (auto ret = m_inputs.Initialize(); !ret) return ret.error();
    if (auto ret = m_outputs.Initialize(); !ret) return ret.error();
    if (auto res = m_outputTransformer.Initialize(); !res) return res.error();
    if (auto err = m_stream.Initialize(rans::RansVariant::RansByte); err) {
        MLVC_LOG_ERROR("Failed to initialize RANS decoder stream: %s", err.message().c_str());
        return make_error_code(Error::entropy_coder_init_error);
    }
    if (auto ret = m_gaussianDecoder.Initialize(); !ret) return ret.error();
    if (auto ret = m_bitEstimatorZ.Initialize(); !ret) return ret.error();
    m_initialized = true;
    return {};
}

expected<DecodedFrame> MlvcDecoderCore::Decode(const FrameData& frameData)
{
    if (!m_initialized) {
        MLVC_LOG_ERROR("Decoder not initialized");
        return make_error_code(Error::not_initialized);
    }

    const auto frameInfo = m_gop.ComputeFrameInfo(frameData);
    if (frameData.frameType == FrameType::I_FRAME) {
        m_ref.Reset();
        m_gop.Reset();
    }
    const auto qIndex = MapQpToQIndex(frameData.qp);
    if (!qIndex) return qIndex.error();
    const auto qIndexShifted = GetQIndexShifted(qIndex.value(), frameData.curFrameIdx);
    if (!qIndexShifted) return qIndexShifted.error();

    // Entropy decoding
    {
        double entropyCodingMs = 0.0;
        // Decode z_raw
        {
            ManualOpTimer timer;
            m_stream.Open(frameData.payload);
            m_zRawInt32.Create(m_inputs.zRaw.Shape());
            if (auto ret = m_bitEstimatorZ.Decode(m_stream, qIndex.value(), m_zRawInt32); !ret) {
                MLVC_LOG_INFO("Failed to decode zRaw: %s", ret.error().message().c_str());
                return ret.error();
            }
            entropyCodingMs += timer.Stop();
        }

        // Scale decoder
        {
            ScopedOpTimer opTimer{ m_stats.scaleDecoder };
            if (auto ret = m_scaleDecoder->ExtractScales(m_zRawInt32); !ret) {
                MLVC_LOG_INFO("Failed to extract scales: %s", ret.error().message().c_str());
                return ret.error();
            }
        }

        // Decode y_raw0 and y_raw1
        {
            ManualOpTimer timer;
            m_yRaw0Int32.Create(m_inputs.yRaw0.Shape());
            m_yRaw1Int32.Create(m_inputs.yRaw1.Shape());
            const auto& scales = m_scaleDecoder->GetScales();
            if (auto ret = m_gaussianDecoder.Decode(m_stream, scales.scales0, m_yRaw0Int32); !ret) {
                MLVC_LOG_INFO("Failed to decode yRaw0: %s", ret.error().message().c_str());
                return ret.error();
            }
            if (auto ret = m_gaussianDecoder.Decode(m_stream, scales.scales1, m_yRaw1Int32); !ret) {
                MLVC_LOG_INFO("Failed to decode yRaw1: %s", ret.error().message().c_str());
                return ret.error();
            }
            entropyCodingMs += timer.Stop();
        }
        m_stats.entropyCoding.Append(entropyCodingMs);
    }

    // Configure input and output
    {
        const auto refIdx =
            frameData.curFrameIdx == frameData.refFrameIdx ? std::nullopt : std::make_optional(frameData.refFrameIdx);
        auto slot = m_ref.ResolveInputSlot(refIdx);
        if (!slot) return slot.error();
        if (auto ret = m_inputs.BindReference(slot.value()->GetData()); !ret) return ret.error();
    }

    {
        auto slot = m_ref.AcquireOutputSlot(frameData.curFrameIdx, frameData.temporalId);
        if (!slot) return slot.error();
        if (auto ret = m_outputs.BindReference(slot.value()->GetData()); !ret) return ret.error();
    }

    {
        Int32ToFloat16(m_zRawInt32, m_inputs.zRaw);
        Int32ToFloat16(m_yRaw0Int32, m_inputs.yRaw0);
        Int32ToFloat16(m_yRaw1Int32, m_inputs.yRaw1);
    }

    // Inference
    {
        ScopedOpTimer opTimer{ m_stats.inference };
        m_inputs.qIndexShifted(0) = qIndexShifted.value();
        if (auto ret = m_model.Run(m_inputs, m_outputs); !ret) return ret.error();
    }

    // Update
    m_ref.PruneUnreferencedSlots(frameData.ltrSlots);
    m_gop.Update(frameInfo);

    // Prepare output
    {
        ScopedOpTimer opTimer{ m_stats.postprocess };
        auto reconFrame =
            m_outputTransformer.Transform(m_outputs.xHat, frameData.DisplayWidth(), frameData.DisplayHeight(),
                                          frameData.cropOffsets, frameData.transposeFlag);
        if (!reconFrame) return reconFrame.error();

        return DecodedFrame{
            .info = std::move(frameInfo),
            .qp = frameData.qp,
            .frame = reconFrame.value(),
        };
    }
}

}  // namespace libmlvc
