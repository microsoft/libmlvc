// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/codec/core/reference_data.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/inference/interface.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace libmlvc {

class ReferenceManager {
public:
    class Slot {
    public:
        static expected<std::unique_ptr<Slot>> Create(const InferenceSessionPtr& session);
        Slot(const InferenceSessionPtr& session);
        void Clear();
        void Set(const int frameIdx, const int temporalId);
        bool IsEmpty() const { return m_empty; }
        int GetFrameIdx() const { return m_frameIdx; }
        const ReferenceData& GetData() const { return m_data; }

    private:
        expected<void> Initialize();

        bool m_empty{ true };
        int m_frameIdx{};
        int m_temporalId{};
        ReferenceData m_data;
    };

    ReferenceManager(const InferenceSessionPtr& session);
    expected<void> Initialize();
    expected<void> Reset();
    expected<const Slot*> ResolveInputSlot(const std::optional<int> refFrameIdx) const;
    expected<const Slot*> AcquireOutputSlot(const int frameIdx, const int temporalId);
    void PruneUnreferencedSlots(const std::array<LtrSlotInfo, MAX_LTR_SLOTS>& ltrSlots);

private:
    const InferenceSessionPtr m_session;
    std::vector<std::unique_ptr<Slot>> m_slots;
    std::array<int, MAX_TEMPORAL_LAYERS> m_lastFrameIdx{};
};

class LtrSlots {
public:
    LtrSlots(const int numSlots);
    void Reset();
    void InvalidateFramesAfter(const int frameIdx);
    void Mark(const int slotIdx, const int frameIdx, const bool isRoot);
    int Acquire();
    std::optional<int> FindLatestFrameIdx(const bool rootOnly, const std::optional<int> lastValidFrameIdx = {}) const;
    const std::array<LtrSlotInfo, MAX_LTR_SLOTS>& GetSlots() const { return m_slots; }

private:
    int m_numSlots;
    std::array<LtrSlotInfo, MAX_LTR_SLOTS> m_slots{};
    std::array<bool, MAX_LTR_SLOTS> m_isRoot{};

    void Invalidate(const int slotIdx);
};

class LtrBranch {
public:
    LtrBranch(const EncoderConfig& config, LtrSlots& slots);
    void Reset();
    void InvalidateFramesAfter(const int frameIdx);
    expected<std::optional<int>> ComputeProactiveRecoveryFrameIdx(const std::optional<int> lastValidFrameIdx = {},
                                                                  const bool isBaseLayerFrame = true) const;
    void UpdateFrameMarking(const int frameIdx, const bool isRecovery);
    int GetBranchId() const { return m_branchId; }
    int GetBranchDepth() const { return m_branchDepth; }
    std::optional<int> GetRootFrameIdx() const { return m_rootFrameIdx; }

private:
    const EncoderConfig& m_config;
    LtrSlots& m_slots;
    int m_branchId{};
    std::optional<int> m_rootFrameIdx{};
    int m_branchDepth{};
    int m_ltrMarkingCounter{};

    bool NeedsProactiveRecovery(std::optional<int> lastValidFrameIdx = {}, bool isBaseLayerFrame = true) const;
    void StartNewBranch();
};

class LtrManager {
public:
    LtrManager(const EncoderConfig& config);
    void Reset();
    expected<void> MarkNextFrameAsLtr(const int ltrSlotIdx);
    expected<std::optional<int>> ComputeRecoveryFrameIdx(const std::optional<int> useSlotIdx, const int temporalId) const;
    void UpdateFrameMarking(const int frameIdx, const int temporalId, const bool isRecovery,
                            const std::optional<int> useSlotIdx = {});
    const std::array<LtrSlotInfo, MAX_LTR_SLOTS>& GetLtrSlots() const { return m_slots.GetSlots(); }
    const LtrSlots& GetSlots() const { return m_slots; }
    const LtrBranch& GetBranch() const { return m_branch; }

private:
    const EncoderConfig& m_config;
    LtrSlots m_slots;
    LtrBranch m_branch;
    std::optional<int> m_targetLtrSlotToMark{};

    void InvalidateFramesAfter(const int frameIdx);
};

}  // namespace libmlvc
