// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/codec.hpp"
#include "libmlvc/codec/core/core.hpp"
#include "libmlvc/codec/manager.hpp"
#include "libmlvc/common/utils.hpp"

#include <sstream>

namespace libmlvc {

namespace {

std::string LtrSlots2Str(const std::array<LtrSlotInfo, MAX_LTR_SLOTS>& slots)
{
    int numSlots = 0;
    for (int i = 0; i < MAX_LTR_SLOTS; ++i) {
        if (slots[i].HasValue()) numSlots = i + 1;
    }

    std::ostringstream stream;
    for (int i = 0; i < numSlots; ++i) {
        stream << (slots[i].HasValue() ? slots[i].frameIdx : -1);
        if (i + 1 < numSlots) stream << ", ";
    }
    return stream.str();
}

expected<std::unique_ptr<IScaleDecoder>> BuildScaleDecoder(const ModelMetadata& metadata, const int scaleLevels,
                                                           const ScaleDecoderData& scaleDecoderData)
{
    const int scaleMaxIdx = scaleLevels - 1;
    return CreateScaleDecoder(metadata.scaleDecoderType, scaleDecoderData, metadata.modelWidth, metadata.modelHeight,
                              metadata.latentChannels, metadata.downsampleLatent, metadata.downsampleHyperprior,
                              metadata.yScaleRepeat, scaleMaxIdx);
}

void AggregateCoreStats(const MlvcEncoderCore& core, EncoderStats& stats)
{
    const auto& s = core.GetStats();
    stats.preprocess.Append(s.preprocess.LatestMs());
    stats.inference.Append(s.inference.LatestMs());
    stats.scaleDecoder.Append(s.scaleDecoder.LatestMs());
    stats.entropyCoding.Append(s.entropyCoding.LatestMs());
}

void AggregateCoreStats(const IMlvcEncoderCore& core, EncoderStats& stats)
{
    switch (core.GetEncoderInterfaceType()) {
    case EncoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P:
        AggregateCoreStats(static_cast<const MlvcEncoderCore&>(core), stats);
        break;
    }
}

void AggregateCoreStats(const MlvcDecoderCore& core, DecoderStats& stats)
{
    const auto& s = core.GetStats();
    stats.entropyCoding.Append(s.entropyCoding.LatestMs());
    stats.scaleDecoder.Append(s.scaleDecoder.LatestMs());
    stats.inference.Append(s.inference.LatestMs());
    stats.postprocess.Append(s.postprocess.LatestMs());
}

void AggregateCoreStats(const IMlvcDecoderCore& core, DecoderStats& stats)
{
    switch (core.GetDecoderInterfaceType()) {
    case DecoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P:
        AggregateCoreStats(static_cast<const MlvcDecoderCore&>(core), stats);
        break;
    }
}

void AppendIfUpdated(const OpTimerStats& src, OpTimerStats& dst, int& prevCount)
{
    if (src.Count() > prevCount) {
        dst.Append(src.LatestMs());
        prevCount = src.Count();
    }
}

template <typename T, std::size_t N>
std::array<T, N> CalcDelta(const std::array<T, N>& lhs, const std::array<T, N>& rhs)
{
    std::array<T, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

double CalcKbps(std::size_t totalBytes, int frameCount, double fps = 30.0)
{
    return frameCount > 0 ? static_cast<double>(totalBytes) * 8.0 * fps / (frameCount * 1024.0) : 0.0;
}

double Ratio(std::size_t part, std::size_t total)
{
    return total > 0 ? static_cast<double>(part) / static_cast<double>(total) : 0.0;
}

double CalcFps(const OpTimerStats& frameInterval)
{
    const double avgMs = frameInterval.AverageMs();
    return avgMs > 0.0 ? 1000.0 / avgMs : 0.0;
}

}  // namespace

void EncoderStatsLogger::Update(const EncoderStats& stats)
{
    AppendIfUpdated(stats.reconfigure, m_stats.reconfigure, m_prevTimerCounts.reconfigure);
    AppendIfUpdated(stats.preprocess, m_stats.preprocess, m_prevTimerCounts.preprocess);
    AppendIfUpdated(stats.inference, m_stats.inference, m_prevTimerCounts.inference);
    AppendIfUpdated(stats.scaleDecoder, m_stats.scaleDecoder, m_prevTimerCounts.scaleDecoder);
    AppendIfUpdated(stats.entropyCoding, m_stats.entropyCoding, m_prevTimerCounts.entropyCoding);
    AppendIfUpdated(stats.total, m_stats.total, m_prevTimerCounts.total);
    AppendIfUpdated(stats.frameInterval, m_stats.frameInterval, m_prevTimerCounts.frameInterval);

    if (++m_frameCount >= m_logInterval) {
        const auto deltaFrames = CalcDelta(stats.framesPerLayer, m_counterBaseline.framesPerLayer);
        const auto deltaBytes = CalcDelta(stats.bytesPerLayer, m_counterBaseline.bytesPerLayer);
        const auto totalFrames = deltaFrames[0] + deltaFrames[1];
        const auto totalBytes = deltaBytes[0] + deltaBytes[1];
        const double fps = CalcFps(m_stats.frameInterval);
        MLVC_LOG_INFO(
            "[%s] Encoder frames: %d/%d (%d/%d), fps: %.1f, kbps: %.1f (%.2f/%.2f), idr: %d/%d (forced/total), "
            "ltr_recoveries: %d/%d (forced/total)",
            m_tag.c_str(), stats.numFramesEncoded - m_counterBaseline.numFramesEncoded,
            stats.numFramesAttempted - m_counterBaseline.numFramesAttempted, deltaFrames[0], deltaFrames[1], fps,
            CalcKbps(totalBytes, totalFrames, fps), Ratio(deltaBytes[0], totalBytes), Ratio(deltaBytes[1], totalBytes),
            stats.numIdrsForced - m_counterBaseline.numIdrsForced, stats.numIdrsTotal - m_counterBaseline.numIdrsTotal,
            stats.numLtrRecoveriesForced - m_counterBaseline.numLtrRecoveriesForced,
            stats.numLtrRecoveriesTotal - m_counterBaseline.numLtrRecoveriesTotal);
        MLVC_LOG_INFO(
            "[%s] Encoder total: %s, reconfigure: %s, preprocess: %s, inference: %s, scale_decoder: %s, "
            "entropy_coding: %s, interval: %s (mean/std/min/max)",
            m_tag.c_str(), TimerStats2Str(m_stats.total).c_str(),
            TimerStats2StrShort(m_stats.reconfigure, true).c_str(), TimerStats2StrShort(m_stats.preprocess).c_str(),
            TimerStats2StrShort(m_stats.inference).c_str(), TimerStats2StrShort(m_stats.scaleDecoder).c_str(),
            TimerStats2StrShort(m_stats.entropyCoding).c_str(), TimerStats2StrShort(m_stats.frameInterval).c_str());
        m_counterBaseline = stats;
        m_stats = {};
        m_frameCount = 0;
    }
}

void DecoderStatsLogger::Update(const DecoderStats& stats)
{
    AppendIfUpdated(stats.reconfigure, m_stats.reconfigure, m_prevTimerCounts.reconfigure);
    AppendIfUpdated(stats.entropyCoding, m_stats.entropyCoding, m_prevTimerCounts.entropyCoding);
    AppendIfUpdated(stats.scaleDecoder, m_stats.scaleDecoder, m_prevTimerCounts.scaleDecoder);
    AppendIfUpdated(stats.inference, m_stats.inference, m_prevTimerCounts.inference);
    AppendIfUpdated(stats.postprocess, m_stats.postprocess, m_prevTimerCounts.postprocess);
    AppendIfUpdated(stats.total, m_stats.total, m_prevTimerCounts.total);
    AppendIfUpdated(stats.frameInterval, m_stats.frameInterval, m_prevTimerCounts.frameInterval);

    if (++m_frameCount >= m_logInterval) {
        const auto deltaFrames = CalcDelta(stats.framesPerLayer, m_counterBaseline.framesPerLayer);
        const auto deltaBytes = CalcDelta(stats.bytesPerLayer, m_counterBaseline.bytesPerLayer);
        const auto totalFrames = deltaFrames[0] + deltaFrames[1];
        const auto totalBytes = deltaBytes[0] + deltaBytes[1];
        const double fps = CalcFps(m_stats.frameInterval);
        MLVC_LOG_INFO(
            "[%s] Decoder calls: %d, frames: %d/%d (%d/%d), fps: %.1f, kbps: %.1f (%.2f/%.2f), idr: %d, "
            "ltr_recoveries: %d",
            m_tag.c_str(), stats.numDecodeCalls - m_counterBaseline.numDecodeCalls,
            stats.numFramesDecoded - m_counterBaseline.numFramesDecoded,
            stats.numFramesAttempted - m_counterBaseline.numFramesAttempted, deltaFrames[0], deltaFrames[1], fps,
            CalcKbps(totalBytes, totalFrames, fps), Ratio(deltaBytes[0], totalBytes), Ratio(deltaBytes[1], totalBytes),
            stats.numIdrs - m_counterBaseline.numIdrs, stats.numLtrRecoveries - m_counterBaseline.numLtrRecoveries);
        MLVC_LOG_INFO(
            "[%s] Decoder total: %s, reconfigure: %s, entropy_coding: %s, scale_decoder: %s, inference: %s, "
            "postprocess: %s, interval: %s (mean/std/min/max)",
            m_tag.c_str(), TimerStats2Str(m_stats.total).c_str(),
            TimerStats2StrShort(m_stats.reconfigure, true).c_str(), TimerStats2StrShort(m_stats.entropyCoding).c_str(),
            TimerStats2StrShort(m_stats.scaleDecoder).c_str(), TimerStats2StrShort(m_stats.inference).c_str(),
            TimerStats2StrShort(m_stats.postprocess).c_str(), TimerStats2StrShort(m_stats.frameInterval).c_str());
        m_counterBaseline = stats;
        m_stats = {};
        m_frameCount = 0;
    }
}

MlvcEncoderImpl::~MlvcEncoderImpl()
{
    m_encoderCore.reset();
    const auto totalFrames = m_stats.framesPerLayer[0] + m_stats.framesPerLayer[1];
    const auto totalBytes = m_stats.bytesPerLayer[0] + m_stats.bytesPerLayer[1];
    const double fps = CalcFps(m_stats.frameInterval);
    MLVC_LOG_INFO(
        "[%s] ~Encoder frames: %d/%d (%d/%d), fps: %.1f, kbps: %.1f (%.2f/%.2f), idr: %d/%d (forced/total), "
        "ltr_recoveries: %d/%d (forced/total)",
        m_tag.c_str(), m_stats.numFramesEncoded, m_stats.numFramesAttempted, m_stats.framesPerLayer[0],
        m_stats.framesPerLayer[1], fps, CalcKbps(totalBytes, totalFrames, fps),
        Ratio(m_stats.bytesPerLayer[0], totalBytes), Ratio(m_stats.bytesPerLayer[1], totalBytes), m_stats.numIdrsForced,
        m_stats.numIdrsTotal, m_stats.numLtrRecoveriesForced, m_stats.numLtrRecoveriesTotal);
    MLVC_LOG_INFO(
        "[%s] ~Encoder total: %s, reconfigure: %s, preprocess: %s, inference: %s, scale_decoder: %s, "
        "entropy_coding: %s, interval: %s (mean/std/min/max)",
        m_tag.c_str(), TimerStats2Str(m_stats.total).c_str(), TimerStats2StrShort(m_stats.reconfigure, true).c_str(),
        TimerStats2StrShort(m_stats.preprocess).c_str(), TimerStats2StrShort(m_stats.inference).c_str(),
        TimerStats2StrShort(m_stats.scaleDecoder).c_str(), TimerStats2StrShort(m_stats.entropyCoding).c_str(),
        TimerStats2StrShort(m_stats.frameInterval).c_str());
}

expected<void> MlvcEncoderImpl::Configure(const EncoderConfig& config)
{
    if (auto ret = ValidateEncoderConfig(config); !ret) {
        return ret;
    }
    if (config == m_config) {
        return {};  // No change
    }
    m_config = config;
    m_configChanged = true;
    m_markLtrSlotIdx.reset();  // Reset any pending LTR marking
    return {};
}

expected<void> MlvcEncoderImpl::MarkNextFrameAsLtr(const int ltrSlotIdx)
{
    m_markLtrSlotIdx = ltrSlotIdx;
    return {};
}

expected<FrameInfo> MlvcEncoderImpl::GetNextFrameInfo(const EncodeParams& params) const
{
    if (auto ret = ValidateEncodeParams(params); !ret) return ret.error();
    if (IsCoreRebuildRequired()) {
        return FrameInfo{ .frameType = FrameType::I_FRAME };
    }
    return m_encoderCore->GetNextFrameInfo(m_config, params);
}

expected<EncodedFrame> MlvcEncoderImpl::Encode(const Nv12FrameView& frame, const EncodeParams& params)
{
    if (auto ret = ValidateInputFrame(frame); !ret) return ret.error();
    if (auto ret = ValidateEncodeParams(params); !ret) return ret.error();

    ++m_stats.numFramesAttempted;
    ManualOpTimer timer{ m_stats.total };

    // Build encode context string for logging
    char encodeCtx[128];
    int encodeCtxLen = std::snprintf(encodeCtx, sizeof(encodeCtx), "size=%dx%d, stride=%d, qp=%d/%d", frame.Width(),
                                     frame.Height(), frame.Stride(), params.qp.qps[0], params.qp.qps[1]);
    if (params.forceIdr) {
        std::snprintf(encodeCtx + encodeCtxLen, sizeof(encodeCtx) - encodeCtxLen, ", force_idr");
    } else if (params.useLtrSlotIdx.has_value()) {
        std::snprintf(encodeCtx + encodeCtxLen, sizeof(encodeCtx) - encodeCtxLen, ", force_recovery, ltr_slot=%d",
                      params.useLtrSlotIdx.value());
    }
    MLVC_LOG_DEBUG("[%s] Encode: %s", m_tag.c_str(), encodeCtx);

    // Reconfigure encoder if needed
    if (auto ret = ConfigureEncoder(); !ret) {
        MLVC_LOG_ERROR("[%s] Failed to reconfigure encoder (%s): %s", m_tag.c_str(), encodeCtx,
                       ret.error().message().c_str());
        return ret.error();
    }

    // Mark next frame as LTR if requested
    if (m_markLtrSlotIdx.has_value()) {
        if (auto ret = m_encoderCore->MarkNextFrameAsLtr(m_markLtrSlotIdx.value()); !ret) {
            MLVC_LOG_ERROR("[%s] Failed to mark next frame as LTR (%s): %s", m_tag.c_str(), encodeCtx,
                           ret.error().message().c_str());
            return ret.error();
        }
        m_markLtrSlotIdx.reset();
    }

    // Encode frame
    const auto encoderCoreResult = m_encoderCore->Encode(frame, params);
    if (!encoderCoreResult) {
        MLVC_LOG_ERROR("[%s] Failed to encode frame (%s): %s", m_tag.c_str(), encodeCtx,
                       encoderCoreResult.error().message().c_str());
        return encoderCoreResult.error();
    }
    const auto& frameInfo = encoderCoreResult->info;
    const auto& frameData = encoderCoreResult->data;

    // Encode bitstream
    auto bitstream = m_bitStreamEncoder.Encode(frameData);
    if (!bitstream) {
        MLVC_LOG_ERROR("[%s] Failed to encode bitstream (%s): %s", m_tag.c_str(), encodeCtx,
                       bitstream.error().message().c_str());
        return bitstream.error();
    }

    // Update stats
    AggregateCoreStats(*m_encoderCore, m_stats);
    m_stats.framesPerLayer[frameData.temporalId]++;
    m_stats.bytesPerLayer[frameData.temporalId] += bitstream.value().size();
    const char* frameTypeStr = "p";
    if (frameData.frameType == FrameType::I_FRAME) {
        ++m_stats.numIdrsTotal;
        if (params.forceIdr) {
            ++m_stats.numIdrsForced;
            frameTypeStr = "idr";
        } else {
            frameTypeStr = "i";
        }
    } else if (frameData.frameType == FrameType::LTR_RECOVERY) {
        ++m_stats.numLtrRecoveriesTotal;
        if (params.useLtrSlotIdx.has_value()) {
            ++m_stats.numLtrRecoveriesForced;
            frameTypeStr = "recovery";
        } else {
            frameTypeStr = "proactive";
        }
    }

    // Log encode result
    MLVC_LOG_INFO("[%s] Encoded: cur=%d, ref=%d, tid=%d, type=%s, qp=%d, size=%dx%d, ltr=[%s], bytes=%zu/%zu, chain=%d",
                  m_tag.c_str(), frameData.curFrameIdx, frameData.refFrameIdx, frameData.temporalId, frameTypeStr,
                  frameData.qp, frameData.DisplayWidth(), frameData.DisplayHeight(),
                  LtrSlots2Str(frameData.ltrSlots).c_str(), bitstream.value().size(), frameData.payload.size(),
                  frameInfo.predictionChainLength);

    // Finalize stats and return
    timer.Stop();
    m_frameIntervalTimer.Tick();
    ++m_stats.numFramesEncoded;
    m_statsLogger.Update(m_stats);
    return EncodedFrame{
        .info = frameInfo,
        .payloadBytes = frameData.payload.size(),
        .bitStream = bitstream.value(),
    };
}

MlvcEncoderImpl::MlvcEncoderImpl(const std::string& tag, const std::shared_ptr<const MlvcManagerImpl>& manager)
    : m_tag(tag), m_manager{ manager }, m_bitStreamEncoder{}
{
}

expected<void> MlvcEncoderImpl::Initialize(const EncoderConfig& config)
{
    if (auto ret = ValidateEncoderConfig(config); !ret) return ret;
    if (auto ret = m_bitStreamEncoder.Initialize(); !ret) return ret.error();
    m_config = config;
    m_configChanged = true;
    return {};
}

expected<void> MlvcEncoderImpl::ValidateEncoderConfig(const EncoderConfig& config) const
{
    if (config.width <= 0 || config.width % 2 != 0 || config.height <= 0 || config.height % 2 != 0) {
        MLVC_LOG_ERROR("[%s] Invalid frame dimensions: %dx%d (must be positive and even)", m_tag.c_str(), config.width,
                       config.height);
        return make_error_code(Error::invalid_argument);
    }

    if (config.iframePeriod < 0) {
        MLVC_LOG_ERROR("[%s] Invalid iframePeriod: %d (must be >= 0)", m_tag.c_str(), config.iframePeriod);
        return make_error_code(Error::invalid_argument);
    }

    if (config.numTemporalLayers < 1 || config.numTemporalLayers > MAX_TEMPORAL_LAYERS) {
        MLVC_LOG_ERROR("[%s] Invalid numTemporalLayers: %d (must be 1-%d)", m_tag.c_str(), config.numTemporalLayers,
                       MAX_TEMPORAL_LAYERS);
        return make_error_code(Error::invalid_argument);
    }

    if (config.ltrNumSlots < 0 || config.ltrNumSlots > MAX_LTR_SLOTS) {
        MLVC_LOG_ERROR("[%s] Invalid ltrNumSlots: %d (must be 0-%d)", m_tag.c_str(), config.ltrNumSlots, MAX_LTR_SLOTS);
        return make_error_code(Error::invalid_argument);
    }

    if (config.ltrPeriod < 0) {
        MLVC_LOG_ERROR("[%s] Invalid ltrPeriod: %d (must be >= 0)", m_tag.c_str(), config.ltrPeriod);
        return make_error_code(Error::invalid_argument);
    }

    if (config.ltrStartIdx < 0) {
        MLVC_LOG_ERROR("[%s] Invalid ltrStartIdx: %d (must be >= 0)", m_tag.c_str(), config.ltrStartIdx);
        return make_error_code(Error::invalid_argument);
    }

    if (config.ltrRecoveryPeriod < 0) {
        MLVC_LOG_ERROR("[%s] Invalid ltrRecoveryPeriod: %d (must be >= 0)", m_tag.c_str(), config.ltrRecoveryPeriod);
        return make_error_code(Error::invalid_argument);
    }

    if (config.ltrMode == LtrMode::INTERNAL && config.ltrPeriod > 0 && config.ltrNumSlots <= 0) {
        MLVC_LOG_ERROR("[%s] ltrNumSlots must be > 0 when ltrMode is INTERNAL and ltrPeriod > 0", m_tag.c_str());
        return make_error_code(Error::invalid_argument);
    }

    if (config.ltrMode == LtrMode::INTERNAL && config.ltrRecoveryPeriod > 0 && config.ltrPeriod > 0) {
        if (config.ltrStartIdx >= config.ltrPeriod) {
            MLVC_LOG_ERROR("[%s] ltrStartIdx (%d) must be < ltrPeriod (%d) when proactive recovery is enabled",
                           m_tag.c_str(), config.ltrStartIdx, config.ltrPeriod);
            return make_error_code(Error::invalid_argument);
        }
        if (config.ltrNumSlots < 2) {
            MLVC_LOG_ERROR("[%s] ltrNumSlots (%d) must be >= 2 when proactive recovery is enabled", m_tag.c_str(),
                           config.ltrNumSlots);
            return make_error_code(Error::invalid_argument);
        }
    }

    return {};
}

expected<void> MlvcEncoderImpl::ValidateInputFrame(const Nv12FrameView& frame) const
{
    if (frame.Stride() < frame.Width()) {
        MLVC_LOG_ERROR("Invalid input frame stride: %d (expected >= %d)", frame.Stride(), frame.Width());
        return make_error_code(Error::invalid_argument);
    }

    if (frame.YPlane().size() != static_cast<size_t>(frame.Stride() * frame.Height())) {
        MLVC_LOG_ERROR("Invalid Y plane size: %zu (expected %d)", frame.YPlane().size(),
                       static_cast<size_t>(frame.Stride() * frame.Height()));
        return make_error_code(Error::invalid_argument);
    }

    if (frame.UvPlane().size() != static_cast<size_t>(frame.Stride() * frame.Height() / 2)) {
        MLVC_LOG_ERROR("Invalid UV plane size: %zu (expected %d)", frame.UvPlane().size(),
                       static_cast<size_t>(frame.Stride() * frame.Height() / 2));
        return make_error_code(Error::invalid_argument);
    }
    return {};
}

expected<void> MlvcEncoderImpl::ValidateEncodeParams(const EncodeParams& params) const
{
    for (int i = 0; i < MAX_TEMPORAL_LAYERS; ++i) {
        if (params.qp.qps[i] < MIN_QP || params.qp.qps[i] > MAX_QP) {
            MLVC_LOG_ERROR("Invalid qp for layer %d: %d", i, params.qp.qps[i]);
            return make_error_code(Error::invalid_argument);
        }
    }

    if (params.useLtrSlotIdx.has_value()
        && (params.useLtrSlotIdx.value() < 0 || params.useLtrSlotIdx.value() >= MAX_LTR_SLOTS)) {
        MLVC_LOG_ERROR("Invalid LTR slot index to use: %d", params.useLtrSlotIdx.value());
        return make_error_code(Error::invalid_argument);
    }

    return {};
}

expected<void> MlvcEncoderImpl::ConfigureEncoder()
{
    auto const LogConfig = [this](const char* msg, const EncoderConfig& config) {
        MLVC_LOG_INFO(
            "[%s] %s: version=%s, size=%dx%d, iframe_period=%d, num_temporal_layers=%d, "
            "ltr_mode=%s, ltr_start_idx=%d, ltr_period=%d, ltr_num_slots=%d, ltr_recovery_period=%d",
            m_tag.c_str(), msg, config.mlvcVersion.ToString().c_str(), config.width, config.height, config.iframePeriod,
            config.numTemporalLayers, LtrModeToString(config.ltrMode), config.ltrStartIdx, config.ltrPeriod,
            config.ltrNumSlots, config.ltrRecoveryPeriod);
    };

    if (IsCoreRebuildRequired()) {
        ManualOpTimer timer(m_stats.reconfigure);
        LogConfig("Configuring encoder", m_config);

        const auto modelId = m_manager->GetOptimalModelId(m_config.mlvcVersion, m_config.width, m_config.height);
        if (!modelId) {
            return modelId.error();
        }

        const auto modelManifest = m_manager->GetModelManifest(m_config.mlvcVersion, modelId.value());
        if (!modelManifest) {
            return modelManifest.error();
        }

        const auto modelMetadata = m_manager->GetModelMetadata(m_config.mlvcVersion, modelId.value());
        if (!modelMetadata) {
            return modelMetadata.error();
        }

        const auto gaussianPmf = m_manager->GetGaussianPmf(m_config.mlvcVersion, modelId.value());
        if (!gaussianPmf) {
            return gaussianPmf.error();
        }

        const auto bitEstimatorPmf = m_manager->GetBitEstimatorPmf(m_config.mlvcVersion, modelId.value());
        if (!bitEstimatorPmf) {
            return bitEstimatorPmf.error();
        }

        const auto scaleDecoderData = m_manager->GetScaleDecoderData(m_config.mlvcVersion, modelId.value());
        if (!scaleDecoderData) {
            return scaleDecoderData.error();
        }

        auto scaleDecoder =
            BuildScaleDecoder(*modelMetadata.value(), gaussianPmf.value()->scaleLevels, *scaleDecoderData.value());
        if (!scaleDecoder) {
            return scaleDecoder.error();
        }

        std::unique_ptr<IMlvcEncoderCore> encoderCore;
        if (modelManifest->encoderInterfaceType == EncoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P) {
            const auto session =
                m_manager->GetInferenceSession(m_config.mlvcVersion, modelId.value(), ModelPartId::ENCODER);
            if (!session) {
                return session.error();
            }
            encoderCore.reset(new MlvcEncoderCore(m_config.mlvcVersion, m_config, modelMetadata.value(),
                                                  std::move(scaleDecoder.value()), gaussianPmf.value(),
                                                  bitEstimatorPmf.value(), session.value()));
        } else {
            MLVC_LOG_ERROR("[%s] Unsupported encoder interface type: %s", m_tag.c_str(),
                           EncoderInterfaceTypeToString(modelManifest->encoderInterfaceType));
            return make_error_code(Error::invalid_argument);
        }

        if (auto ret = encoderCore->Initialize(); !ret) {
            return ret.error();
        }

        m_encoderCore = std::move(encoderCore);
        const auto durationMs = timer.Stop();
        MLVC_LOG_INFO("[%s] Configured encoder in %.2f ms", m_tag.c_str(), durationMs);
    } else if (m_configChanged) {
        LogConfig("Reconfiguring encoder", m_config);
        if (auto ret = m_encoderCore->Configure(m_config); !ret) {
            return ret.error();
        }
    }

    m_configChanged = false;
    return {};
}

bool MlvcEncoderImpl::IsCoreRebuildRequired() const
{
    return !m_encoderCore || m_encoderCore->GetMlvcVersion() != m_config.mlvcVersion
           || m_encoderCore->GetDisplayWidth() != m_config.width || m_encoderCore->GetDisplayHeight() != m_config.height;
}

MlvcDecoderImpl::~MlvcDecoderImpl()
{
    m_decoderCore.reset();
    const auto totalFrames = m_stats.framesPerLayer[0] + m_stats.framesPerLayer[1];
    const auto totalBytes = m_stats.bytesPerLayer[0] + m_stats.bytesPerLayer[1];
    const double fps = CalcFps(m_stats.frameInterval);
    MLVC_LOG_INFO(
        "[%s] ~Decoder calls: %d, frames: %d/%d (%d/%d), fps: %.1f, kbps: %.1f (%.2f/%.2f), idr: %d, ltr_recoveries: "
        "%d",
        m_tag.c_str(), m_stats.numDecodeCalls, m_stats.numFramesDecoded, m_stats.numFramesAttempted,
        m_stats.framesPerLayer[0], m_stats.framesPerLayer[1], fps, CalcKbps(totalBytes, totalFrames, fps),
        Ratio(m_stats.bytesPerLayer[0], totalBytes), Ratio(m_stats.bytesPerLayer[1], totalBytes), m_stats.numIdrs,
        m_stats.numLtrRecoveries);
    MLVC_LOG_INFO(
        "[%s] ~Decoder total: %s, reconfigure: %s, entropy_coding: %s, scale_decoder: %s, inference: %s, "
        "postprocess: %s, interval: %s (mean/std/min/max)",
        m_tag.c_str(), TimerStats2Str(m_stats.total).c_str(), TimerStats2StrShort(m_stats.reconfigure, true).c_str(),
        TimerStats2StrShort(m_stats.entropyCoding).c_str(), TimerStats2StrShort(m_stats.scaleDecoder).c_str(),
        TimerStats2StrShort(m_stats.inference).c_str(), TimerStats2StrShort(m_stats.postprocess).c_str(),
        TimerStats2StrShort(m_stats.frameInterval).c_str());
}

expected<DecodedFrame> MlvcDecoderImpl::Decode(std::span<const std::byte> bytes)
{
    ++m_stats.numDecodeCalls;
    ManualOpTimer timer{ m_stats.total };

    // Decode bitstream
    expected<FrameData> frameData = m_bitStreamDecoder.Decode(bytes);
    if (!frameData) {
        if (!IsPartialAccessUnitError(frameData.error())) {
            MLVC_LOG_ERROR("[%s] Failed to decode bitstream (bytes=%zu): %s", m_tag.c_str(), bytes.size(),
                           frameData.error().message().c_str());
        }
        return frameData.error();
    }
    ++m_stats.numFramesAttempted;

    // Build decode context string for logging
    const char* decodeTypeStr = frameData->frameType == FrameType::I_FRAME        ? "i"
                                : frameData->frameType == FrameType::LTR_RECOVERY ? "recovery"
                                                                                  : "p";
    char decodeCtx[128];
    std::snprintf(decodeCtx, sizeof(decodeCtx), "cur=%d, ref=%d, tid=%d, type=%s, qp=%d", frameData->curFrameIdx,
                  frameData->refFrameIdx, frameData->temporalId, decodeTypeStr, frameData->qp);

    // Check version availability
    if (!m_manager->IsVersionAvailable(frameData->mlvcVersion)) {
        MLVC_LOG_ERROR("[%s] MLVC version %s is not available (%s)", m_tag.c_str(),
                       frameData->mlvcVersion.ToString().c_str(), decodeCtx);
        return make_error_code(Error::general_failure);
    }

    // Reconfigure decoder if needed
    if (auto ret = ConfigureDecoder(frameData->mlvcVersion, frameData->DisplayWidth(), frameData->DisplayHeight(),
                                    frameData->modelWidth, frameData->modelHeight);
        !ret) {
        MLVC_LOG_ERROR("[%s] Failed to reconfigure decoder (%s): %s", m_tag.c_str(), decodeCtx,
                       ret.error().message().c_str());
        return ret.error();
    }

    // Decode the frame
    const auto decodedFrame = m_decoderCore->Decode(frameData.value());
    if (!decodedFrame) {
        MLVC_LOG_ERROR("[%s] Failed to decode frame (%s): %s", m_tag.c_str(), decodeCtx,
                       decodedFrame.error().message().c_str());
        return decodedFrame.error();
    }

    // Log decode result
    MLVC_LOG_INFO("[%s] Decoded: %s, size=%dx%d, ltr=[%s], bytes=%zu/%zu, chain=%d", m_tag.c_str(), decodeCtx,
                  frameData->DisplayWidth(), frameData->DisplayHeight(), LtrSlots2Str(frameData->ltrSlots).c_str(),
                  bytes.size(), frameData->payload.size(), decodedFrame->info.predictionChainLength);

    // Update stats
    AggregateCoreStats(*m_decoderCore, m_stats);
    m_stats.framesPerLayer[frameData->temporalId]++;
    m_stats.bytesPerLayer[frameData->temporalId] += bytes.size();
    if (frameData->frameType == FrameType::I_FRAME) ++m_stats.numIdrs;
    if (frameData->frameType == FrameType::LTR_RECOVERY) ++m_stats.numLtrRecoveries;

    // Finalize stats and return
    timer.Stop();
    m_frameIntervalTimer.Tick();
    ++m_stats.numFramesDecoded;
    m_statsLogger.Update(m_stats);
    return decodedFrame;
}

MlvcDecoderImpl::MlvcDecoderImpl(const std::string& tag, const std::shared_ptr<const MlvcManagerImpl>& manager)
    : m_tag(tag), m_manager{ manager }, m_bitStreamDecoder{}
{
}

expected<void> MlvcDecoderImpl::Initialize()
{
    if (auto ret = m_bitStreamDecoder.Initialize(); !ret) return ret.error();
    return {};
}

expected<void> MlvcDecoderImpl::ConfigureDecoder(const MlvcVersion mlvcVersion, const int displayWidth,
                                                 const int displayHeight, const int modelWidth, const int modelHeight)
{
    if (m_decoderCore && m_decoderCore->GetMlvcVersion() == mlvcVersion && m_decoderCore->GetModelWidth() == modelWidth
        && m_decoderCore->GetModelHeight() == modelHeight && m_decoderCore->GetDisplayWidth() == displayWidth
        && m_decoderCore->GetDisplayHeight() == displayHeight) {
        return {};  // No change
    }

    ManualOpTimer timer(m_stats.reconfigure);
    const std::string modelId = std::to_string(modelWidth) + "x" + std::to_string(modelHeight);
    MLVC_LOG_INFO("[%s] Configuring decoder: version=%s, model_id=%s, display=%dx%d", m_tag.c_str(),
                  mlvcVersion.ToString().c_str(), modelId.c_str(), displayWidth, displayHeight);

    const auto modelManifest = m_manager->GetModelManifest(mlvcVersion, modelId);
    if (!modelManifest) {
        return modelManifest.error();
    }

    const auto modelMetadata = m_manager->GetModelMetadata(mlvcVersion, modelId);
    if (!modelMetadata) {
        return modelMetadata.error();
    }

    const auto gaussianPmf = m_manager->GetGaussianPmf(mlvcVersion, modelId);
    if (!gaussianPmf) {
        return gaussianPmf.error();
    }

    const auto bitEstimatorPmf = m_manager->GetBitEstimatorPmf(mlvcVersion, modelId);
    if (!bitEstimatorPmf) {
        return bitEstimatorPmf.error();
    }

    const auto scaleDecoderData = m_manager->GetScaleDecoderData(mlvcVersion, modelId);
    if (!scaleDecoderData) {
        return scaleDecoderData.error();
    }

    auto scaleDecoder =
        BuildScaleDecoder(*modelMetadata.value(), gaussianPmf.value()->scaleLevels, *scaleDecoderData.value());
    if (!scaleDecoder) {
        return scaleDecoder.error();
    }

    std::unique_ptr<IMlvcDecoderCore> decoderCore;
    if (modelManifest->decoderInterfaceType == DecoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P) {
        const auto session = m_manager->GetInferenceSession(mlvcVersion, modelId, ModelPartId::DECODER);
        if (!session) {
            return session.error();
        }
        decoderCore.reset(new MlvcDecoderCore(mlvcVersion, displayWidth, displayHeight, modelMetadata.value(),
                                              std::move(scaleDecoder.value()), gaussianPmf.value(),
                                              bitEstimatorPmf.value(), session.value()));
    } else {
        MLVC_LOG_ERROR("[%s] Unsupported decoder interface type: %s", m_tag.c_str(),
                       DecoderInterfaceTypeToString(modelManifest->decoderInterfaceType));
        return make_error_code(Error::invalid_argument);
    }

    if (auto ret = decoderCore->Initialize(); !ret) {
        MLVC_LOG_ERROR("[%s] Failed to initialize decoder core: %s", m_tag.c_str(), ret.error().message().c_str());
        return ret.error();
    }
    m_decoderCore = std::move(decoderCore);
    const auto durationMs = timer.Stop();
    MLVC_LOG_INFO("[%s] Configured decoder in %.2f ms", m_tag.c_str(), durationMs);
    return {};
}

}  // namespace libmlvc
