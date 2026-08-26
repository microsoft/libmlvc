// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/reference_manager.hpp"
#include "libmlvc/common/logging.hpp"

#include <algorithm>
#include <limits>

namespace libmlvc {

namespace {

constexpr int IDR_REF_IDX = -99;  // Sentinel frame index for the IDR (Instantaneous Decoder Refresh) reference slot
constexpr int INVALID_REF_IDX = -100;  // Sentinel frame index indicating an empty/cleared slot

}  // namespace

expected<std::unique_ptr<ReferenceManager::Slot>> ReferenceManager::Slot::Create(const InferenceSessionPtr& session)
{
    auto res = std::make_unique<Slot>(session);
    if (auto ret = res->Initialize(); !ret) return ret.error();
    return res;
}

void ReferenceManager::Slot::Clear()
{
    m_empty = true;
    m_frameIdx = INVALID_REF_IDX;
    m_temporalId = 0;
}

void ReferenceManager::Slot::Set(const int frameIdx, const int temporalId)
{
    m_empty = false;
    m_frameIdx = frameIdx;
    m_temporalId = temporalId;
}

ReferenceManager::Slot::Slot(const InferenceSessionPtr& session) : m_data(session) {}

expected<void> ReferenceManager::Slot::Initialize()
{
    if (auto ret = m_data.Initialize(); !ret) return ret.error();
    return {};
}

ReferenceManager::ReferenceManager(const InferenceSessionPtr& session) : m_session{ session }
{
    std::fill(m_lastFrameIdx.begin(), m_lastFrameIdx.end(), -1);
}

expected<void> ReferenceManager::Initialize()
{
    return Reset();
}

expected<void> ReferenceManager::Reset()
{
    std::fill(m_lastFrameIdx.begin(), m_lastFrameIdx.end(), -1);
    for (auto& slot : m_slots) {
        slot->Clear();
    }

    // Ensure at least 3 slots (IDR + 2) are preallocated
    while (m_slots.size() < 3) {
        auto slot = Slot::Create(m_session);
        if (!slot) return slot.error();
        m_slots.push_back(std::move(slot.value()));
    }

    // Prepare IDR slot (slot 0)
    {
        auto& idrSlot = m_slots.front();
        idrSlot->Set(IDR_REF_IDX, 0);  // IDR slot has a special sentinel frame index
    }
    return {};
}

expected<const ReferenceManager::Slot*> ReferenceManager::ResolveInputSlot(const std::optional<int> refFrameIdx) const
{
    const auto refIdx = refFrameIdx.value_or(IDR_REF_IDX);
    for (const auto& slot : m_slots) {
        if (!slot->IsEmpty() && slot->GetFrameIdx() == refIdx) {
            return slot.get();
        }
    }
    MLVC_LOG_ERROR("Reference not found for %d", refIdx);
    return make_error_code(Error::reference_error);
}

expected<const ReferenceManager::Slot*> ReferenceManager::AcquireOutputSlot(const int frameIdx, const int temporalId)
{
    if (temporalId < 0 || temporalId >= static_cast<int>(m_lastFrameIdx.size())) {
        MLVC_LOG_ERROR("Invalid temporalId: %d", temporalId);
        return make_error_code(Error::invalid_argument);
    }

    auto AcquireEmptySlot = [&]() -> expected<int> {
        for (int i = 0; i < static_cast<int>(m_slots.size()); i++) {
            if (m_slots[i]->IsEmpty()) {
                return i;
            }
        }
        MLVC_LOG_DEBUG("Allocating reference slot, new number of slots: %zu", m_slots.size() + 1);
        auto slot = Slot::Create(m_session);
        if (!slot) {
            MLVC_LOG_ERROR("Failed to create reference slot: %s", slot.error().message().c_str());
            return slot.error();
        }
        m_slots.push_back(std::move(slot.value()));
        return static_cast<int>(m_slots.size()) - 1;
    };

    const auto slotIdx = AcquireEmptySlot();
    if (!slotIdx) return slotIdx.error();

    Slot& slot = *m_slots[slotIdx.value()];
    slot.Set(frameIdx, temporalId);

    // Update last used frameIdx for this temporalId
    m_lastFrameIdx[temporalId] = frameIdx;

    return &slot;
}

void ReferenceManager::PruneUnreferencedSlots(const std::array<LtrSlotInfo, MAX_LTR_SLOTS>& ltrSlots)
{
    // Never prune IDR slot (slot 0)
    for (std::size_t i = 1; i < m_slots.size(); i++) {
        auto& slot = *m_slots[i];
        if (slot.IsEmpty()) continue;
        bool keep = false;
        for (const auto& ltrSlot : ltrSlots) {
            if (ltrSlot.HasValue() && ltrSlot.frameIdx == slot.GetFrameIdx()) {
                keep = true;
                break;
            }
        }
        keep |= std::find(m_lastFrameIdx.begin(), m_lastFrameIdx.end(), slot.GetFrameIdx()) != m_lastFrameIdx.end();
        if (!keep) {
            slot.Clear();
        }
    }
}

// ---------------------------------------------------------------------------
// LtrSlots
// ---------------------------------------------------------------------------

LtrSlots::LtrSlots(const int numSlots)
{
    if (numSlots < 0) {
        MLVC_LOG_WARN("Number of LTR slots cannot be negative, capping to zero: %d", numSlots);
    } else if (numSlots > MAX_LTR_SLOTS) {
        MLVC_LOG_WARN("Number of LTR slots cannot exceed %d, capping to maximum", MAX_LTR_SLOTS);
    }
    m_numSlots = std::clamp(numSlots, 0, MAX_LTR_SLOTS);
}

void LtrSlots::Reset()
{
    m_slots.fill(LtrSlotInfo{});
    m_isRoot.fill(false);
}

void LtrSlots::Invalidate(const int slotIdx)
{
    m_slots[slotIdx] = LtrSlotInfo{};
    m_isRoot[slotIdx] = false;
}

void LtrSlots::InvalidateFramesAfter(const int frameIdx)
{
    for (int i = 0; i < static_cast<int>(m_slots.size()); i++) {
        if (m_slots[i].HasValue() && m_slots[i].frameIdx > frameIdx) {
            Invalidate(i);
        }
    }
}

void LtrSlots::Mark(const int slotIdx, const int frameIdx, const bool isRoot)
{
    m_slots[slotIdx] = LtrSlotInfo{ frameIdx };
    m_isRoot[slotIdx] = isRoot;
}

int LtrSlots::Acquire()
{
    MLVC_ASSERT(m_numSlots > 0);

    // 1. First free slot
    for (int i = 0; i < m_numSlots; i++) {
        if (!m_slots[i].HasValue()) {
            return i;
        }
    }

    // Count root slots to decide protection policy
    int rootCount = 0;
    for (int i = 0; i < m_numSlots; i++) {
        if (m_isRoot[i]) {
            rootCount++;
        }
    }

    // 2. Oldest non-root slot (roots are protected when rootCount <= 2)
    if (rootCount <= 2) {
        int oldestSlotIdx = -1;
        int oldestFrameIdx = std::numeric_limits<int>::max();
        for (int i = 0; i < m_numSlots; i++) {
            if (!m_isRoot[i] && m_slots[i].frameIdx < oldestFrameIdx) {
                oldestFrameIdx = m_slots[i].frameIdx;
                oldestSlotIdx = i;
            }
        }
        if (oldestSlotIdx >= 0) {
            Invalidate(oldestSlotIdx);
            return oldestSlotIdx;
        }
    }

    // 3. Oldest slot (fallback: excess roots, all slots are roots, or numSlots <= 2)
    {
        int oldestSlotIdx = -1;
        int oldestFrameIdx = std::numeric_limits<int>::max();
        for (int i = 0; i < m_numSlots; i++) {
            if (m_slots[i].HasValue() && m_slots[i].frameIdx < oldestFrameIdx) {
                oldestFrameIdx = m_slots[i].frameIdx;
                oldestSlotIdx = i;
            }
        }
        if (oldestSlotIdx >= 0) {
            Invalidate(oldestSlotIdx);
            return oldestSlotIdx;
        }
    }

    // 4. Should never reach here
    {
        MLVC_LOG_ERROR("No LTR slots available");
        MLVC_ASSERT(false);
        Invalidate(0);
        return 0;
    }
}

std::optional<int> LtrSlots::FindLatestFrameIdx(const bool rootOnly, const std::optional<int> lastValidFrameIdx) const
{
    int latestFrameIdx = -1;
    bool found = false;
    for (int i = 0; i < static_cast<int>(m_slots.size()); i++) {
        if (!m_slots[i].HasValue()) continue;
        if (rootOnly && !m_isRoot[i]) continue;
        if (lastValidFrameIdx.has_value() && m_slots[i].frameIdx > *lastValidFrameIdx) continue;
        if (!found || m_slots[i].frameIdx > latestFrameIdx) {
            latestFrameIdx = m_slots[i].frameIdx;
            found = true;
        }
    }
    return found ? std::optional<int>(latestFrameIdx) : std::nullopt;
}

// ---------------------------------------------------------------------------
// LtrBranch
// ---------------------------------------------------------------------------

LtrBranch::LtrBranch(const EncoderConfig& config, LtrSlots& slots) : m_config{ config }, m_slots{ slots } {}

void LtrBranch::Reset()
{
    m_branchId = 0;
    m_rootFrameIdx = std::nullopt;
    m_branchDepth = 0;
    m_ltrMarkingCounter = 0;
}

void LtrBranch::InvalidateFramesAfter(const int frameIdx)
{
    if (m_rootFrameIdx.has_value() && *m_rootFrameIdx > frameIdx) {
        m_rootFrameIdx = std::nullopt;
    }
}

bool LtrBranch::NeedsProactiveRecovery(const std::optional<int> lastValidFrameIdx, const bool isBaseLayerFrame) const
{
    // Simulate InvalidateFramesAfter on m_rootFrameIdx without mutating it.
    const bool rootSurvives =
        m_rootFrameIdx.has_value() && (!lastValidFrameIdx.has_value() || *m_rootFrameIdx <= *lastValidFrameIdx);

    const int recoveryThreshold = m_config.ltrPeriod * m_config.ltrRecoveryPeriod;
    if (recoveryThreshold <= 0) return false;
    if (rootSurvives && m_branchDepth >= recoveryThreshold && isBaseLayerFrame) return true;
    if (m_branchId > 0 && !rootSurvives) return true;
    return false;
}

void LtrBranch::StartNewBranch()
{
    m_branchId++;
    m_branchDepth = 0;
    m_rootFrameIdx = std::nullopt;
    m_ltrMarkingCounter = 0;
}

expected<std::optional<int>> LtrBranch::ComputeProactiveRecoveryFrameIdx(const std::optional<int> lastValidFrameIdx,
                                                                         const bool isBaseLayerFrame) const
{
    if (!NeedsProactiveRecovery(lastValidFrameIdx, isBaseLayerFrame)) return std::nullopt;

    auto rootFrameIdx = m_slots.FindLatestFrameIdx(true, lastValidFrameIdx);
    if (!rootFrameIdx) {
        MLVC_LOG_ERROR("No root LTR available for proactive recovery, falling back to IDR");
        return make_error_code(Error::reference_error);
    }
    return *rootFrameIdx;
}

void LtrBranch::UpdateFrameMarking(const int frameIdx, const bool isRecovery)
{
    if (isRecovery && NeedsProactiveRecovery()) StartNewBranch();

    const int recoveryThreshold = m_config.ltrPeriod * m_config.ltrRecoveryPeriod;

    bool markAsRoot = false;
    bool markAsInBranch = false;
    if (recoveryThreshold > 0 && !m_rootFrameIdx.has_value() && frameIdx >= m_config.ltrStartIdx) {
        markAsRoot = true;
    } else if (isRecovery || frameIdx == 0 || (m_config.ltrPeriod > 0 && m_ltrMarkingCounter >= m_config.ltrPeriod)) {
        markAsInBranch = true;
    }

    if (markAsRoot || markAsInBranch) {
        const auto slotIdx = m_slots.Acquire();
        m_slots.Mark(slotIdx, frameIdx, markAsRoot);
        if (markAsRoot) {
            m_rootFrameIdx = frameIdx;
        }
        m_ltrMarkingCounter = 0;
        MLVC_LOG_DEBUG("LTR marked: frameIdx=%d, slot=%d, isRoot=%d, isRecovery=%d, branchId=%d, branchDepth=%d",
                       frameIdx, slotIdx, markAsRoot, isRecovery, m_branchId, m_branchDepth);
    }

    m_branchDepth++;
    m_ltrMarkingCounter++;
}

// ---------------------------------------------------------------------------
// LtrManager
// ---------------------------------------------------------------------------

LtrManager::LtrManager(const EncoderConfig& config)
    : m_config{ config }, m_slots{ config.ltrNumSlots }, m_branch{ config, m_slots }
{
}

void LtrManager::Reset()
{
    // Called on IDR frames, should not reset m_targetLtrSlotToMark set by MarkNextFrameAsLtr()
    m_slots.Reset();
    m_branch.Reset();
}

expected<void> LtrManager::MarkNextFrameAsLtr(const int ltrSlotIdx)
{
    if (m_config.ltrMode != LtrMode::EXTERNAL) {
        MLVC_LOG_DEBUG("Cannot mark LTR frame when LTR mode is not EXTERNAL, ignoring request");
        return {};
    }

    if (ltrSlotIdx < 0 || ltrSlotIdx >= static_cast<int>(m_slots.GetSlots().size())) {
        MLVC_LOG_ERROR("Invalid LTR slot index: %d", ltrSlotIdx);
        return make_error_code(Error::invalid_argument);
    }
    m_targetLtrSlotToMark = ltrSlotIdx;
    return {};
}

void LtrManager::InvalidateFramesAfter(const int frameIdx)
{
    m_slots.InvalidateFramesAfter(frameIdx);
    m_branch.InvalidateFramesAfter(frameIdx);
}

expected<std::optional<int>> LtrManager::ComputeRecoveryFrameIdx(const std::optional<int> useSlotIdx, const int temporalId) const
{
    // LTR disabled and no external hint -> no recovery
    if (m_config.ltrPeriod == 0 && !useSlotIdx.has_value()) return std::nullopt;

    bool hintProvided = useSlotIdx.has_value();
    std::optional<int> lastValidFrameIdx;

    // Step 2: Process external hint
    if (useSlotIdx.has_value()) {
        const int slotIdx = *useSlotIdx;
        if (slotIdx < 0 || slotIdx >= static_cast<int>(m_slots.GetSlots().size())) {
            MLVC_LOG_ERROR("LTR slot index must be in [0, %d], got %d", static_cast<int>(m_slots.GetSlots().size()) - 1,
                           slotIdx);
            return make_error_code(Error::invalid_argument);
        }
        if (!m_slots.GetSlots()[slotIdx].HasValue()) {
            MLVC_LOG_WARN("LTR slot %d is empty (stale hint), ignoring", slotIdx);
            hintProvided = false;
        } else {
            lastValidFrameIdx = m_slots.GetSlots()[slotIdx].frameIdx;
        }
    }

    // Step 3: Proactive recovery (internal mode only)
    if (m_config.ltrMode == LtrMode::INTERNAL) {
        const bool isBaseLayerFrame = (temporalId == 0);
        auto proactive = m_branch.ComputeProactiveRecoveryFrameIdx(lastValidFrameIdx, isBaseLayerFrame);
        if (!proactive) return proactive.error();
        if (proactive->has_value()) return *proactive;
    }

    // Step 4: Hint-based in-branch recovery
    if (hintProvided) {
        const int slotIdx = *useSlotIdx;
        if (m_slots.GetSlots()[slotIdx].HasValue()) {
            return m_slots.GetSlots()[slotIdx].frameIdx;
        }
        auto latestFrameIdx = m_slots.FindLatestFrameIdx(false, lastValidFrameIdx);
        if (latestFrameIdx.has_value()) return *latestFrameIdx;
        MLVC_LOG_WARN("No surviving LTR slot after invalidation, falling back to IDR");
        return make_error_code(Error::reference_error);
    }

    // Step 5: No recovery
    return std::nullopt;
}

void LtrManager::UpdateFrameMarking(const int frameIdx, const int temporalId, const bool isRecovery,
                                    const std::optional<int> useSlotIdx)
{
    if (isRecovery && useSlotIdx.has_value()) {
        const int slotIdx = *useSlotIdx;
        MLVC_ASSERT(slotIdx >= 0 && slotIdx < static_cast<int>(m_slots.GetSlots().size()));
        if (m_slots.GetSlots()[slotIdx].HasValue()) {
            InvalidateFramesAfter(m_slots.GetSlots()[slotIdx].frameIdx);
        }
    }

    if (temporalId != 0) return;

    // External mode: preserve backward compat
    if (m_config.ltrMode == LtrMode::EXTERNAL) {
        if (frameIdx == 0) {
            m_slots.Mark(0, frameIdx, false);
            m_targetLtrSlotToMark = std::nullopt;
        } else if (m_targetLtrSlotToMark.has_value()) {
            m_slots.Mark(*m_targetLtrSlotToMark, frameIdx, false);
            m_targetLtrSlotToMark = std::nullopt;
        }
        return;
    }

    // Internal mode
    if (m_config.ltrPeriod == 0) return;
    m_branch.UpdateFrameMarking(frameIdx, isRecovery);
}

}  // namespace libmlvc
