// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <map>
#include <optional>

namespace libmlvc {

constexpr int FRAME_IDX_BITS = 10;

struct EncoderFrameInfo : public FrameInfo {
    int gopIdx{};
};

class GopManager {
public:
    explicit GopManager(const EncoderConfig& config) : m_config{ config } {}

    void Reset();
    int PeekNextTemporalId(const EncoderConfig& config) const;
    EncoderFrameInfo ComputeFrameInfo(const EncoderConfig& config, const EncodeParams& params,
                                      const expected<std::optional<int>>& recoveryFrameIdx) const;
    void Update(const EncoderConfig& config, const EncoderFrameInfo& frameInfo);

private:
    EncoderConfig m_config{};
    std::map<int, EncoderFrameInfo> m_frameInfoHistory;

    int GetPredictionChainLength(const int frameIdx) const;
    int GetReferenceFrameIdx(const int frameIdx, const int temporalId) const;
    bool ConfigChangeForcesIdr(const EncoderConfig& newConfig) const;
};

class GopTracker {
public:
    void Reset();
    FrameInfo ComputeFrameInfo(const FrameData& frameData) const;
    void Update(const FrameInfo& frameInfo);

private:
    std::map<int, FrameInfo> m_frameInfoHistory;

    int GetPredictionChainLength(const int frameIdx) const;
};

}  // namespace libmlvc
