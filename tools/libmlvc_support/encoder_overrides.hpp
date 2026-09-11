// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/types.hpp>

#include <optional>

namespace libmlvc {

struct EncoderConfigOverrides {
    std::optional<int> iframePeriod;
    std::optional<int> numTemporalLayers;
    std::optional<LtrMode> ltrMode;
    std::optional<int> ltrStartIdx;
    std::optional<int> ltrPeriod;
    std::optional<int> ltrNumSlots;
    std::optional<int> ltrRecoveryPeriod;

    void Apply(EncoderConfig& config) const
    {
        if (iframePeriod.has_value()) {
            config.iframePeriod = *iframePeriod;
        }
        if (numTemporalLayers.has_value()) {
            config.numTemporalLayers = *numTemporalLayers;
        }
        if (ltrMode.has_value()) {
            config.ltrMode = *ltrMode;
        }
        if (ltrStartIdx.has_value()) {
            config.ltrStartIdx = *ltrStartIdx;
        }
        if (ltrPeriod.has_value()) {
            config.ltrPeriod = *ltrPeriod;
        }
        if (ltrNumSlots.has_value()) {
            config.ltrNumSlots = *ltrNumSlots;
        }
        if (ltrRecoveryPeriod.has_value()) {
            config.ltrRecoveryPeriod = *ltrRecoveryPeriod;
        }
    }
};

}  // namespace libmlvc
