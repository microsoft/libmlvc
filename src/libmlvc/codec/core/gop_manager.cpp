// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/gop_manager.hpp"
#include "libmlvc/common/logging.hpp"

#include <libmlvc/error_codes.hpp>

#include <limits>

namespace libmlvc {

namespace {

int GetTemporalId(const int gopIdx, const int numTemporalLayers)
{
    if (numTemporalLayers <= 1) {
        return 0;
    } else if (numTemporalLayers == 2) {
        return gopIdx % 2;
    }
    static_assert(MAX_TEMPORAL_LAYERS <= 2, "GetTemporalId needs to be updated to support more temporal layers");
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// GopManager
// ---------------------------------------------------------------------------

void GopManager::Reset()
{
    m_frameInfoHistory.clear();
}

int GopManager::PeekNextTemporalId(const EncoderConfig& config) const
{
    const int gopIdx =
        m_frameInfoHistory.empty() || m_config != config ? 0 : (m_frameInfoHistory.rbegin()->second.gopIdx + 1);
    return GetTemporalId(gopIdx, config.numTemporalLayers);
}

EncoderFrameInfo GopManager::ComputeFrameInfo(const EncoderConfig& config, const EncodeParams& params,
                                              const expected<std::optional<int>>& recoveryFrameIdx) const
{
    // Called by encoder
    auto ComputeMaxPredictionChainLength = [&config]() -> int {
        // Max prediction chain before forcing IDR, derived from proactive LTR recovery threshold.
        // E.g. iframePeriod=64, ltrStartIdx=2, ltrPeriod=4, recoveryPeriod=2 → 2 + 8 + 8 - 2 = 16
        if (config.iframePeriod <= 0 || config.ltrMode != LtrMode::INTERNAL || config.ltrRecoveryPeriod <= 0) return 0;
        const int recoveryThreshold = config.ltrPeriod * config.ltrRecoveryPeriod;
        return recoveryThreshold > 0 ? config.ltrStartIdx + config.iframePeriod / recoveryThreshold + recoveryThreshold - 2
                                     : config.iframePeriod;
    };

    auto MakeIframeInfo = []() {
        return EncoderFrameInfo{
            FrameInfo{
                .frameType = FrameType::I_FRAME,
            },
            /*gopIdx=*/0,
        };
    };

    //  Force IDR for recovery errors or if explicitly requested
    if (!recoveryFrameIdx && recoveryFrameIdx.error() == make_error_code(Error::reference_error)) {
        MLVC_LOG_WARN("ComputeFrameInfo: Failed to compute LTR recovery due to reference error, forcing IDR");
        return MakeIframeInfo();
    } else if (!recoveryFrameIdx) {
        MLVC_LOG_ERROR("ComputeFrameInfo: Failed to compute LTR recovery, forcing IDR: %s",
                       recoveryFrameIdx.error().message().c_str());
        return MakeIframeInfo();
    } else if (params.forceIdr) {
        MLVC_LOG_DEBUG("ComputeFrameInfo: External IDR request received");
        return MakeIframeInfo();
    } else if (ConfigChangeForcesIdr(config)) {
        MLVC_LOG_DEBUG("ComputeFrameInfo: IDR triggered by config change");
        return MakeIframeInfo();
    }

    // Calculate next P-frame info
    const int frameIdx = m_frameInfoHistory.empty() ? 0 : m_frameInfoHistory.rbegin()->second.frameIdx + 1;
    const int gopIdx = m_frameInfoHistory.empty() || config != m_config || recoveryFrameIdx.value().has_value()
                           ? 0
                           : (m_frameInfoHistory.rbegin()->second.gopIdx + 1);
    const int temporalId = GetTemporalId(gopIdx, config.numTemporalLayers);
    const int refFrameIdx = recoveryFrameIdx.value().value_or(GetReferenceFrameIdx(frameIdx, temporalId));
    const int predictionChainLength = GetPredictionChainLength(refFrameIdx) + 1;

    // Periodic I-frame
    if (config.iframePeriod > 0 && (gopIdx >= config.iframePeriod || frameIdx >= config.iframePeriod)) {
        MLVC_LOG_DEBUG("ComputeFrameInfo: Periodic I-frame (frameIdx=%d, gopIdx=%d, period=%d)", frameIdx, gopIdx,
                       config.iframePeriod);
        return MakeIframeInfo();
    }

    // Frame index overflow check
    if (frameIdx >= (1 << FRAME_IDX_BITS)) {
        MLVC_LOG_WARN("ComputeFrameInfo: Frame index overflow (frameIdx=%d), forcing IDR", frameIdx);
        return MakeIframeInfo();
    }

    // Force IDR if prediction chain is too long
    const int maxPredictionChainLength = ComputeMaxPredictionChainLength();
    if (maxPredictionChainLength > 0 && predictionChainLength > maxPredictionChainLength) {
        MLVC_LOG_WARN("ComputeFrameInfo: Prediction chain reached max %d > %d for frame %d (ref: %d), forcing IDR",
                      predictionChainLength, maxPredictionChainLength, frameIdx, refFrameIdx);
        return MakeIframeInfo();
    }

    auto MakeFrameInfo = [&](FrameType frameType) {
        return EncoderFrameInfo{
            FrameInfo{
                .frameType = frameType,
                .frameIdx = frameIdx,
                .refFrameIdx = refFrameIdx,
                .temporalId = temporalId,
                .predictionChainLength = predictionChainLength,
            },
            gopIdx,
        };
    };

    // LTR recovery frame
    if (recoveryFrameIdx.value().has_value()) {
        MLVC_LOG_DEBUG("ComputeFrameInfo: Using LTR recovery frame index %d", recoveryFrameIdx.value().value());
        return MakeFrameInfo(FrameType::LTR_RECOVERY);
    }

    // Default I/P frame
    if (frameIdx == 0) {
        MLVC_LOG_DEBUG("ComputeFrameInfo: First frame in GOP, using I-frame");
        return MakeIframeInfo();
    }
    return MakeFrameInfo(FrameType::P_FRAME);
}

void GopManager::Update(const EncoderConfig& config, const EncoderFrameInfo& frameInfo)
{
    m_config = config;
    if (frameInfo.frameType == FrameType::I_FRAME) {
        m_frameInfoHistory.clear();
    }
    m_frameInfoHistory[frameInfo.frameIdx] = frameInfo;
}

int GopManager::GetPredictionChainLength(const int frameIdx) const
{
    if (frameIdx < 0) {
        return 0;
    }

    if (!m_frameInfoHistory.contains(frameIdx)) {
        return std::numeric_limits<int>::max() - 1;
    }

    return m_frameInfoHistory.at(frameIdx).predictionChainLength;
}

int GopManager::GetReferenceFrameIdx(const int frameIdx, const int temporalId) const
{
    if (temporalId == 0) {
        // Walk history backward for last tId=0 reference (survives TL-only config change)
        for (auto it = m_frameInfoHistory.rbegin(); it != m_frameInfoHistory.rend(); ++it) {
            if (it->second.temporalId == 0) {
                return it->second.frameIdx;
            }
        }
        return frameIdx - 1;
    }
    // temporalId == 1: predecessor is always tId=0
    return frameIdx - 1;
}

bool GopManager::ConfigChangeForcesIdr(const EncoderConfig& newConfig) const
{
    if (m_config == newConfig) {
        return false;
    }

    // Changes in temporal layer count alone do not force IDR
    {
        EncoderConfig tempConfig = m_config;
        tempConfig.numTemporalLayers = newConfig.numTemporalLayers;
        if (tempConfig == newConfig) {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// GopTracker
// ---------------------------------------------------------------------------

void GopTracker::Reset()
{
    m_frameInfoHistory.clear();
}

FrameInfo GopTracker::ComputeFrameInfo(const FrameData& frameData) const
{
    const int predictionChainLength =
        (frameData.frameType == FrameType::I_FRAME) ? 0 : GetPredictionChainLength(frameData.refFrameIdx) + 1;

    return FrameInfo{
        .frameType = frameData.frameType,
        .frameIdx = frameData.curFrameIdx,
        .refFrameIdx = frameData.refFrameIdx,
        .temporalId = frameData.temporalId,
        .predictionChainLength = predictionChainLength,
    };
}

void GopTracker::Update(const FrameInfo& frameInfo)
{
    if (frameInfo.frameType == FrameType::I_FRAME) {
        m_frameInfoHistory.clear();
    }
    m_frameInfoHistory[frameInfo.frameIdx] = frameInfo;
}

int GopTracker::GetPredictionChainLength(const int frameIdx) const
{
    if (frameIdx < 0) {
        return 0;
    }

    if (!m_frameInfoHistory.contains(frameIdx)) {
        return std::numeric_limits<int>::max() - 1;
    }

    return m_frameInfoHistory.at(frameIdx).predictionChainLength;
}

}  // namespace libmlvc
