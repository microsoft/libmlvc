// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/logging.hpp"

#include <libmlvc/types.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <set>

namespace libmlvc {

class RecoveryController {
public:
    struct RecoveryInfo {
        bool idr{};
        std::optional<int> useLtrSlotIdx{};
    };

    RecoveryController(const std::set<int>& lostFrameIds, bool idrOnly = false)
        : m_lostFrameIds{ lostFrameIds }, m_idrOnly{ idrOnly }
    {
    }

    RecoveryInfo ComputeRecovery(const int frameId)
    {
        auto isFrameLost = [this](const int id) { return m_lostFrameIds.contains(id); };
        if (!isFrameLost(frameId - 1)) {
            return {};
        }

        // Find LTR slot with latest frame index (skip if idrOnly)
        std::optional<int> latestLtrSlotIdx;
        if (!m_idrOnly) {
            for (std::size_t slotIdx = 0; slotIdx < m_slots.size(); slotIdx++) {
                if (!m_slots[slotIdx].HasValue()) {
                    continue;
                }
                if (!latestLtrSlotIdx || m_slots[slotIdx].frameIdx > m_slots[latestLtrSlotIdx.value()].frameIdx) {
                    latestLtrSlotIdx = static_cast<int>(slotIdx);
                }
            }
        }

        if (latestLtrSlotIdx.has_value()) {
            MLVC_LOG_INFO("Frame lost, encoding %d from LTR %d frame", frameId, latestLtrSlotIdx.value());
            return RecoveryInfo{ false, latestLtrSlotIdx };
        }
        MLVC_LOG_INFO("Frame lost, encoding frame %d as IDR frame", frameId);
        return RecoveryInfo{ true, std::nullopt };
    }
    void Update(const int frameId, const std::array<LtrSlotInfo, MAX_LTR_SLOTS>& ltrSlots)
    {
        if (!m_lostFrameIds.contains(frameId)) {
            m_slots = ltrSlots;
        }
    }

private:
    const std::set<int> m_lostFrameIds;
    const bool m_idrOnly{};
    std::array<LtrSlotInfo, MAX_LTR_SLOTS> m_slots{};
};

}  // namespace libmlvc
