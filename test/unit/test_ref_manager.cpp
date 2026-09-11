// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/reference_manager.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>

using namespace libmlvc;

namespace {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

class FakeInferenceTensor : public IInferenceTensor {
public:
    explicit FakeInferenceTensor(std::string_view name) : m_name(name) {}

    expected<std::string_view> Name() const override { return m_name; }
    expected<TensorDataType> DataType() const override { return TensorDataType::FP16; }
    expected<std::span<const int>> Shape() const override { return m_shape; }
    expected<std::span<const int>> Strides() const override { return m_strides; }
    expected<std::span<std::byte>> Data() override { return m_data; }

private:
    std::string m_name;
    std::vector<int> m_shape = { 1, 64, 8, 8 };
    std::vector<int> m_strides = { 4096, 64, 8, 1 };
    std::vector<std::byte> m_data{ 8192, std::byte{ 0 } };
};

class FakeInferenceSession : public IInferenceSession {
public:
    expected<std::span<const std::string>> GetInputNames() override { return m_inputNames; }
    expected<std::span<const std::string>> GetOutputNames() override { return m_outputNames; }
    expected<InferenceTensorPtr> CreateTensor(TensorIoType, std::string_view name, bool) override
    {
        return std::make_shared<FakeInferenceTensor>(name);
    }
    expected<void> Run(const std::vector<InferenceTensorPtr>&, const std::vector<InferenceTensorPtr>&) override
    {
        return {};
    }

private:
    std::vector<std::string> m_inputNames = { "ref_feature" };
    std::vector<std::string> m_outputNames = { "ref_feature" };
};

int CountUsedSlots(const std::array<LtrSlotInfo, MAX_LTR_SLOTS>& slots)
{
    return static_cast<int>(std::count_if(slots.begin(), slots.end(), [](const auto& slot) { return slot.HasValue(); }));
}

}  // namespace

// -----------------------------------------------------------------------------
// ReferenceManager Tests
// -----------------------------------------------------------------------------

class UnitTestReferenceManager : public ::testing::Test {
protected:
    InferenceSessionPtr m_session;

    void SetUp() override { m_session = std::make_shared<FakeInferenceSession>(); }
};

TEST_F(UnitTestReferenceManager, InitializeAndReset)
{
    ReferenceManager refManager(m_session);
    ASSERT_TRUE(refManager.Initialize());
    ASSERT_TRUE(refManager.Reset());
}

TEST_F(UnitTestReferenceManager, ResolveInputSlot_IdrAndRegularReference)
{
    ReferenceManager refManager(m_session);
    ASSERT_TRUE(refManager.Initialize());

    // IDR slot accessible via nullopt
    auto idrSlot = refManager.ResolveInputSlot(std::nullopt);
    ASSERT_TRUE(idrSlot);
    EXPECT_FALSE(idrSlot.value()->IsEmpty());

    // Non-existent reference fails
    EXPECT_FALSE(refManager.ResolveInputSlot(999));

    // Create and retrieve a reference
    ASSERT_TRUE(refManager.AcquireOutputSlot(5, 0));
    auto slot = refManager.ResolveInputSlot(5);
    ASSERT_TRUE(slot);
    EXPECT_EQ(slot.value()->GetFrameIdx(), 5);
}

TEST_F(UnitTestReferenceManager, PruneUnreferencedSlots_KeepsLtrAndLastFrames)
{
    ReferenceManager refManager(m_session);
    ASSERT_TRUE(refManager.Initialize());

    // Create slots for frames 1, 2, 3
    ASSERT_TRUE(refManager.AcquireOutputSlot(1, 0));
    ASSERT_TRUE(refManager.AcquireOutputSlot(2, 0));
    ASSERT_TRUE(refManager.AcquireOutputSlot(3, 0));

    // LTR slots only contain frame 2
    std::array<LtrSlotInfo, MAX_LTR_SLOTS> ltrSlots{};
    ltrSlots[0] = LtrSlotInfo{ 2 };
    refManager.PruneUnreferencedSlots(ltrSlots);

    // IDR always kept, frame 2 kept (LTR), frame 3 kept (last frame)
    EXPECT_TRUE(refManager.ResolveInputSlot(std::nullopt));
    EXPECT_TRUE(refManager.ResolveInputSlot(2));
    EXPECT_TRUE(refManager.ResolveInputSlot(3));
    // Frame 1 pruned
    EXPECT_FALSE(refManager.ResolveInputSlot(1));
}

// -----------------------------------------------------------------------------
// LtrSlots Tests
// -----------------------------------------------------------------------------

TEST(UnitTestLtrSlots, Reset_ClearsAllSlots)
{
    LtrSlots slots(4);
    slots.Mark(0, 10, false);
    slots.Mark(1, 20, true);
    EXPECT_EQ(CountUsedSlots(slots.GetSlots()), 2);

    slots.Reset();
    EXPECT_EQ(CountUsedSlots(slots.GetSlots()), 0);
    EXPECT_FALSE(slots.FindLatestFrameIdx(true).has_value());
}

TEST(UnitTestLtrSlots, Acquire_FreeSlotFirst)
{
    LtrSlots slots(4);

    // All slots free, should return 0
    EXPECT_EQ(slots.Acquire(), 0);

    slots.Mark(0, 10, false);
    EXPECT_EQ(slots.Acquire(), 1);
}

TEST(UnitTestLtrSlots, Acquire_OldestNonProtected)
{
    LtrSlots slots(3);
    slots.Mark(0, 10, true);   // root, protected
    slots.Mark(1, 20, false);  // non-protected
    slots.Mark(2, 30, false);  // non-protected

    EXPECT_EQ(slots.Acquire(), 1);  // oldest non-protected (frameIdx=20)
}

TEST(UnitTestLtrSlots, Acquire_FallbackToOldestProtected)
{
    LtrSlots slots(2);
    slots.Mark(0, 10, true);  // root, protected
    slots.Mark(1, 20, true);  // root, protected

    EXPECT_EQ(slots.Acquire(), 0);  // oldest protected (frameIdx=10)
}

TEST(UnitTestLtrSlots, InvalidateFramesAfter)
{
    LtrSlots slots(4);
    slots.Mark(0, 5, false);
    slots.Mark(1, 10, true);
    slots.Mark(2, 15, false);
    slots.Mark(3, 20, true);

    slots.InvalidateFramesAfter(10);

    EXPECT_TRUE(slots.GetSlots()[0].HasValue());   // frameIdx=5 <= 10, kept
    EXPECT_TRUE(slots.GetSlots()[1].HasValue());   // frameIdx=10 <= 10, kept
    EXPECT_FALSE(slots.GetSlots()[2].HasValue());  // frameIdx=15 > 10, invalidated
    EXPECT_FALSE(slots.GetSlots()[3].HasValue());  // frameIdx=20 > 10, invalidated

    // Root 20 should be removed from root list
    EXPECT_EQ(slots.FindLatestFrameIdx(true), 10);
}

TEST(UnitTestLtrSlots, Mark_RootProtection)
{
    LtrSlots slots(4);
    slots.Mark(0, 10, true);
    slots.Mark(1, 20, true);

    EXPECT_EQ(slots.FindLatestFrameIdx(true), 20);

    // Third root: Mark() just sets the flag (no cap)
    slots.Mark(2, 30, true);
    // All 3 roots remain: isRoot = [T, T, T, F]

    // Acquire() sees rootCount=3 > 2, falls through to oldest-any
    slots.Mark(3, 40, false);
    EXPECT_EQ(slots.Acquire(), 0);  // slot 0 (frameIdx=10) is oldest overall
}

TEST(UnitTestLtrSlots, Mark_OverwriteRemovesOldRoot)
{
    LtrSlots slots(2);
    slots.Mark(0, 10, true);  // root
    slots.Mark(1, 20, false);

    EXPECT_EQ(slots.FindLatestFrameIdx(true), 10);

    // Overwrite slot 0 with non-root
    slots.Mark(0, 30, false);
    EXPECT_FALSE(slots.FindLatestFrameIdx(true).has_value());  // root 10 removed
}

TEST(UnitTestLtrSlots, FindLatestFrameIdx)
{
    LtrSlots slots(4);

    // Empty slots
    EXPECT_FALSE(slots.FindLatestFrameIdx(true).has_value());
    EXPECT_FALSE(slots.FindLatestFrameIdx(false).has_value());

    // With mixed root and non-root slots
    slots.Mark(0, 10, false);
    slots.Mark(1, 20, true);
    slots.Mark(2, 30, false);
    EXPECT_EQ(slots.FindLatestFrameIdx(true), 20);
    EXPECT_EQ(slots.FindLatestFrameIdx(false), 30);
}

TEST(UnitTestLtrSlots, FindLatestFrameIdx_LastValid)
{
    // Shared setup: slots 10/20/30/40 with roots at 20 and 30.
    LtrSlots slots(4);
    slots.Mark(0, 10, false);
    slots.Mark(1, 20, true);
    slots.Mark(2, 30, true);
    slots.Mark(3, 40, false);

    {
        // Pins `>` (not `>=`): a slot at exactly the bound is kept.
        SCOPED_TRACE("boundary is inclusive");
        EXPECT_EQ(slots.FindLatestFrameIdx(false, 20), 20);
        EXPECT_EQ(slots.FindLatestFrameIdx(false, 19), 10);
    }
    {
        // rootOnly and lastValidFrameIdx must compose.
        SCOPED_TRACE("bound can exclude root while non-root survives");
        EXPECT_EQ(slots.FindLatestFrameIdx(false, 15), 10);
        EXPECT_FALSE(slots.FindLatestFrameIdx(true, 15).has_value());
    }
    {
        // Keystone: bounded query must equal the result of mutating
        // InvalidateFramesAfter on an identical second LtrSlots.
        SCOPED_TRACE("matches InvalidateFramesAfter semantics");
        LtrSlots invalidated(4);
        invalidated.Mark(0, 10, false);
        invalidated.Mark(1, 20, true);
        invalidated.Mark(2, 30, true);
        invalidated.Mark(3, 40, false);
        invalidated.InvalidateFramesAfter(25);

        EXPECT_EQ(slots.FindLatestFrameIdx(false, 25), invalidated.FindLatestFrameIdx(false));
        EXPECT_EQ(slots.FindLatestFrameIdx(true, 25), invalidated.FindLatestFrameIdx(true));
    }
}

// -----------------------------------------------------------------------------
// LtrBranch Tests
// -----------------------------------------------------------------------------

class UnitTestLtrBranch : public ::testing::Test {
protected:
    EncoderConfig m_config;
    std::unique_ptr<LtrSlots> m_slots;
    std::unique_ptr<LtrBranch> m_branch;

    void SetUp() override
    {
        m_config = EncoderConfig{
            .numTemporalLayers = 1,
            .ltrMode = LtrMode::INTERNAL,
            .ltrStartIdx = 2,
            .ltrPeriod = 4,
            .ltrNumSlots = 4,
            .ltrRecoveryPeriod = 2,
        };
        m_slots = std::make_unique<LtrSlots>(m_config.ltrNumSlots);
        m_branch = std::make_unique<LtrBranch>(m_config, *m_slots);
    }
};

TEST_F(UnitTestLtrBranch, UpdateFrameMarking_RootMarking)
{
    // Frame 0: no root mark (frameIdx < ltrStartIdx)
    m_branch->UpdateFrameMarking(0, false);
    EXPECT_FALSE(m_branch->GetRootFrameIdx().has_value());
    EXPECT_EQ(m_slots->GetSlots()[0], LtrSlotInfo{ 0 });  // in-branch (frameIdx==0)

    // Frame 1: no mark
    m_branch->UpdateFrameMarking(1, false);
    EXPECT_FALSE(m_branch->GetRootFrameIdx().has_value());

    // Frame 2: root mark (frameIdx >= ltrStartIdx, no root yet)
    m_branch->UpdateFrameMarking(2, false);
    EXPECT_EQ(m_branch->GetRootFrameIdx(), 2);
    EXPECT_EQ(m_slots->GetSlots()[1], LtrSlotInfo{ 2 });
}

TEST_F(UnitTestLtrBranch, UpdateFrameMarking_PeriodicInBranch)
{
    // Frames 0-2: root at 2
    for (int i = 0; i <= 2; i++) {
        m_branch->UpdateFrameMarking(i, false);
    }
    EXPECT_EQ(m_branch->GetRootFrameIdx(), 2);

    // Frames 3-5: counter grows
    for (int i = 3; i <= 5; i++) {
        m_branch->UpdateFrameMarking(i, false);
    }

    // Frame 6: counter=4 >= ltrPeriod=4 -> in-branch mark
    m_branch->UpdateFrameMarking(6, false);
    EXPECT_EQ(m_slots->GetSlots()[2], LtrSlotInfo{ 6 });
}

TEST_F(UnitTestLtrBranch, UpdateFrameMarking_RecoveryMarksInBranch)
{
    // Setup: root at 2
    for (int i = 0; i <= 2; i++) {
        m_branch->UpdateFrameMarking(i, false);
    }

    // Recovery frame: marked as in-branch LTR (not root, since root exists)
    m_branch->UpdateFrameMarking(3, true);
    EXPECT_EQ(m_branch->GetRootFrameIdx(), 2);  // root unchanged
    EXPECT_EQ(m_slots->GetSlots()[2], LtrSlotInfo{ 3 });
}

TEST_F(UnitTestLtrBranch, ComputeProactiveRecovery_BranchFull)
{
    // Process 8 frames (recoveryThreshold = 4*2 = 8)
    for (int i = 0; i < 8; i++) {
        m_branch->UpdateFrameMarking(i, false);
    }
    EXPECT_EQ(m_branch->GetBranchDepth(), 8);
    EXPECT_EQ(m_branch->GetRootFrameIdx(), 2);

    // Proactive recovery should trigger
    auto result = m_branch->ComputeProactiveRecoveryFrameIdx();
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->has_value());
    EXPECT_EQ(**result, 2);  // recovers from root
    m_branch->UpdateFrameMarking(8, true);

    // New branch started
    EXPECT_EQ(m_branch->GetBranchId(), 1);
    EXPECT_EQ(m_branch->GetBranchDepth(), 1);
    EXPECT_EQ(m_branch->GetRootFrameIdx(), 8);
}

TEST_F(UnitTestLtrBranch, ComputeProactiveRecovery_RootLost)
{
    // Branch 0: establish root at frame 2
    for (int i = 0; i < 8; i++) {
        m_branch->UpdateFrameMarking(i, false);
    }

    // Proactive -> branch 1
    auto result1 = m_branch->ComputeProactiveRecoveryFrameIdx();
    ASSERT_TRUE(result1);
    ASSERT_TRUE(result1->has_value());

    // Mark root for branch 1
    m_branch->UpdateFrameMarking(8, true);
    EXPECT_EQ(m_branch->GetRootFrameIdx(), 8);

    // Invalidate root (simulating loss) - must invalidate both slots and branch
    m_slots->InvalidateFramesAfter(7);
    m_branch->InvalidateFramesAfter(7);
    EXPECT_FALSE(m_branch->GetRootFrameIdx().has_value());

    // Root lost, branchId > 0 -> needs proactive recovery
    auto result2 = m_branch->ComputeProactiveRecoveryFrameIdx();
    ASSERT_TRUE(result2);
    ASSERT_TRUE(result2->has_value());
    EXPECT_EQ(**result2, 2);  // falls back to previous root
}

TEST_F(UnitTestLtrBranch, ComputeProactiveRecovery_Disabled)
{
    m_config.ltrRecoveryPeriod = 0;
    m_slots = std::make_unique<LtrSlots>(m_config.ltrNumSlots);
    m_branch = std::make_unique<LtrBranch>(m_config, *m_slots);

    for (int i = 0; i < 16; i++) {
        m_branch->UpdateFrameMarking(i, false);
    }

    auto result = m_branch->ComputeProactiveRecoveryFrameIdx();
    ASSERT_TRUE(result);
    EXPECT_FALSE(result->has_value());
}

TEST_F(UnitTestLtrBranch, ComputeProactiveRecovery_NoRootAvailable)
{
    // Process enough frames for branch full
    for (int i = 0; i < 8; i++) {
        m_branch->UpdateFrameMarking(i, false);
    }

    // Proactive -> new branch
    auto result1 = m_branch->ComputeProactiveRecoveryFrameIdx();
    ASSERT_TRUE(result1);

    // Mark new root at frame 8
    m_branch->UpdateFrameMarking(8, true);

    // Invalidate ALL roots (both frame 2 and frame 8)
    m_slots->InvalidateFramesAfter(1);
    m_branch->InvalidateFramesAfter(1);

    // No roots available -> error (IDR fallback)
    auto result2 = m_branch->ComputeProactiveRecoveryFrameIdx();
    EXPECT_FALSE(result2);
    EXPECT_EQ(result2.error(), make_error_code(Error::reference_error));
}

// -----------------------------------------------------------------------------
// LtrManager Tests
// -----------------------------------------------------------------------------

class UnitTestLtrManager : public ::testing::Test {
protected:
    EncoderConfig m_config;

    void SetUp() override
    {
        m_config = EncoderConfig{
            .width = 640,
            .height = 360,
            .numTemporalLayers = 1,
            .ltrMode = LtrMode::INTERNAL,
            .ltrStartIdx = 0,
            .ltrPeriod = 4,
            .ltrNumSlots = 4,
            .ltrRecoveryPeriod = 0,
        };
    }
};

TEST_F(UnitTestLtrManager, Reset_ClearsAllSlots)
{
    LtrManager ltr(m_config);

    // All LTR slots should be empty after construction
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 0);

    // Mark an LTR frame
    ltr.Reset();
    ltr.UpdateFrameMarking(0, 0, false);
    ASSERT_TRUE(ltr.GetLtrSlots()[0].HasValue());

    // Reset and verify all slots are cleared
    ltr.Reset();
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 0);
}

TEST_F(UnitTestLtrManager, Reset_IdrRootWhenStartIdxZero)
{
    // IDR reset flow: when ltrStartIdx==0 and recoveryThreshold>0,
    // frame 0 should be marked as root (step 1), not in-branch.
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    ltr.UpdateFrameMarking(0, 0, false);

    // Frame 0 should be root: branch has no root, frameIdx(0) >= ltrStartIdx(0), threshold > 0
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 0);
    EXPECT_EQ(ltr.GetSlots().FindLatestFrameIdx(true), 0);
}

TEST_F(UnitTestLtrManager, Reset_IdrInBranchWhenStartIdxNonZero)
{
    // IDR reset flow: when ltrStartIdx>0, frame 0 is in-branch (step 2, frameIdx==0).
    m_config.ltrStartIdx = 2;
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    ltr.UpdateFrameMarking(0, 0, false);

    // Frame 0 is in-branch, not root (0 < ltrStartIdx=2)
    EXPECT_FALSE(ltr.GetBranch().GetRootFrameIdx().has_value());
    EXPECT_FALSE(ltr.GetSlots().FindLatestFrameIdx(true).has_value());
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });  // marked as in-branch
}

TEST_F(UnitTestLtrManager, Reset_IdrInBranchWhenNoProactive)
{
    // IDR reset flow: when ltrStartIdx==0 and recoveryThreshold==0,
    // frame 0 is in-branch (step 1 guard fails, step 2 catches frameIdx==0).
    LtrManager ltr(m_config);
    ltr.Reset();

    ltr.UpdateFrameMarking(0, 0, false);

    // Frame 0 is in-branch, not root (recoveryThreshold==0 skips step 1)
    EXPECT_FALSE(ltr.GetBranch().GetRootFrameIdx().has_value());
    EXPECT_FALSE(ltr.GetSlots().FindLatestFrameIdx(true).has_value());
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
}

TEST_F(UnitTestLtrManager, Reset_PreservesPendingExternalMark)
{
    // Reset should not clear m_targetLtrSlotToMark set by MarkNextFrameAsLtr.
    // IDR frame marks slot 0 and consumes the pending mark.
    m_config.ltrMode = LtrMode::EXTERNAL;
    LtrManager ltr(m_config);

    ASSERT_TRUE(ltr.MarkNextFrameAsLtr(2));
    ltr.Reset();

    // IDR frame (frame 0) marks slot 0, ignoring and consuming the external mark
    ltr.UpdateFrameMarking(0, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
    EXPECT_FALSE(ltr.GetLtrSlots()[2].HasValue());

    // Pending mark was consumed, so frame 1 has no pending mark
    ltr.UpdateFrameMarking(1, 0, false);
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 1);
}

TEST_F(UnitTestLtrManager, ExternalMode_MarkValidation)
{
    // In internal mode, mark requests are ignored (not an error)
    LtrManager ltrInternal(m_config);
    ASSERT_TRUE(ltrInternal.MarkNextFrameAsLtr(0));

    // In external mode, invalid slot indices fail
    m_config.ltrMode = LtrMode::EXTERNAL;
    LtrManager ltrExternal(m_config);
    EXPECT_FALSE(ltrExternal.MarkNextFrameAsLtr(-1));
    EXPECT_FALSE(ltrExternal.MarkNextFrameAsLtr(MAX_LTR_SLOTS));
    EXPECT_EQ(CountUsedSlots(ltrExternal.GetLtrSlots()), 0);
}

TEST_F(UnitTestLtrManager, ExternalMode_MarkAndProcess)
{
    m_config.ltrMode = LtrMode::EXTERNAL;
    LtrManager ltr(m_config);

    // IDR frame always marks slot 0
    ltr.UpdateFrameMarking(0, 0, false);
    EXPECT_TRUE(ltr.GetLtrSlots()[0].HasValue());
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });

    ASSERT_TRUE(ltr.MarkNextFrameAsLtr(2));
    ltr.UpdateFrameMarking(1, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 1 });

    ltr.UpdateFrameMarking(2, 0, false);
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 2);  // Slot 0 (IDR) + Slot 2
}

TEST_F(UnitTestLtrManager, ExternalMode_MultipleMarkCallsLastOneWins_NonIdr)
{
    m_config.ltrMode = LtrMode::EXTERNAL;
    LtrManager ltr(m_config);

    // IDR frame marks slot 0
    ltr.UpdateFrameMarking(0, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });

    // Multiple mark calls - last one wins
    ASSERT_TRUE(ltr.MarkNextFrameAsLtr(1));
    ASSERT_TRUE(ltr.MarkNextFrameAsLtr(2));
    ASSERT_TRUE(ltr.MarkNextFrameAsLtr(3));
    ltr.UpdateFrameMarking(1, 0, false);

    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });  // Original IDR mark preserved
    EXPECT_FALSE(ltr.GetLtrSlots()[1].HasValue());
    EXPECT_FALSE(ltr.GetLtrSlots()[2].HasValue());
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 1 });  // Last mark wins
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 2);    // Slot 0 (IDR) + Slot 3
}

TEST_F(UnitTestLtrManager, ExternalMode_IdrAlwaysMarksSlot0)
{
    m_config.ltrMode = LtrMode::EXTERNAL;
    LtrManager ltr(m_config);

    // External mark request for slot 2 before IDR frame
    ASSERT_TRUE(ltr.MarkNextFrameAsLtr(2));

    // IDR frame (frame 0) always marks slot 0, ignoring external mark
    ltr.UpdateFrameMarking(0, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
    EXPECT_FALSE(ltr.GetLtrSlots()[2].HasValue());
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 1);
}

TEST_F(UnitTestLtrManager, ExternalMode_SlotReuse)
{
    m_config.ltrMode = LtrMode::EXTERNAL;
    LtrManager ltr(m_config);

    // IDR frame marks slot 0
    ltr.UpdateFrameMarking(0, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });

    // Reuse slot 0 with explicit mark
    ASSERT_TRUE(ltr.MarkNextFrameAsLtr(0));
    ltr.UpdateFrameMarking(10, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 10 });

    auto recovery = ltr.ComputeRecoveryFrameIdx(0, 0);
    ASSERT_TRUE(recovery);
    EXPECT_EQ(*recovery, 10);
}

TEST_F(UnitTestLtrManager, ExternalMode_NonBaseLayerDeferred)
{
    m_config.ltrMode = LtrMode::EXTERNAL;
    m_config.numTemporalLayers = 2;
    LtrManager ltr(m_config);

    // Frame 0: IDR base layer - always marks slot 0
    ltr.UpdateFrameMarking(0, 0, false);
    EXPECT_TRUE(ltr.GetLtrSlots()[0].HasValue());
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 1);

    // Frame 1: mark LTR on non-base layer
    // Non-base layer frame (layer 1) should not be marked - mark is deferred
    ASSERT_TRUE(ltr.MarkNextFrameAsLtr(1));
    ltr.UpdateFrameMarking(1, 1, false);
    EXPECT_FALSE(ltr.GetLtrSlots()[1].HasValue());

    // Frame 2: Next base layer frame (layer 0) should receive the deferred mark
    ltr.UpdateFrameMarking(2, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 2 });

    // Subsequent frames without a new mark request should not affect the slots
    ltr.UpdateFrameMarking(3, 1, false);
    ltr.UpdateFrameMarking(4, 0, false);
    ltr.UpdateFrameMarking(5, 1, false);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 2 });
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 2);
}

TEST_F(UnitTestLtrManager, InternalMode_MarksLtrPeriodically)
{
    LtrManager ltr(m_config);

    // 1st GOP
    ltr.Reset();
    for (int i = 0; i < 16; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 4 });
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 8 });
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 12 });
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 4);
}

TEST_F(UnitTestLtrManager, InternalMode_NonZeroStartIdx)
{
    m_config.ltrStartIdx = 4;
    m_config.ltrPeriod = 6;
    LtrManager ltr(m_config);
    ltr.Reset();

    // Frame 0: in-branch LTR (frameIdx==0)
    ltr.UpdateFrameMarking(0, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 1);

    // Frames 1-3 should not add new marks
    for (int i = 1; i < 4; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 1);

    // Frame 4 (ltrStartIdx) - counter=4 < ltrPeriod=6, no mark yet
    ltr.UpdateFrameMarking(4, 0, false);
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 1);

    // Frame 6: counter=6 >= ltrPeriod=6 -> in-branch mark
    ltr.UpdateFrameMarking(5, 0, false);
    ltr.UpdateFrameMarking(6, 0, false);
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 6 });
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 2);
}

TEST_F(UnitTestLtrManager, InternalMode_LtrDisabled)
{
    m_config.ltrPeriod = 0;
    LtrManager ltr(m_config);
    ltr.Reset();

    // When ltrPeriod is 0, no frames should be marked
    ltr.UpdateFrameMarking(0, 0, false);
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 0);

    ltr.UpdateFrameMarking(1, 0, false);
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 0);
}

TEST_F(UnitTestLtrManager, InternalMode_LtrNumSlotsLimitsUsedSlots)
{
    m_config.ltrNumSlots = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    for (int i = 0; i < 16; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }

    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 2);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 8 });
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 12 });
}

TEST_F(UnitTestLtrManager, ComputeRecoveryFrameIdx_NoRequest)
{
    LtrManager ltr(m_config);
    ltr.Reset();

    auto recoveryFrameIdx = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
    ASSERT_TRUE(recoveryFrameIdx);
    EXPECT_FALSE(recoveryFrameIdx->has_value());
}

TEST_F(UnitTestLtrManager, ComputeRecoveryFrameIdx_ValidSlot)
{
    LtrManager ltr(m_config);
    ltr.Reset();

    for (int i = 0; i < 8; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }

    auto recoveryFrameIdx = ltr.ComputeRecoveryFrameIdx(1, 0);
    ASSERT_TRUE(recoveryFrameIdx);
    EXPECT_EQ(recoveryFrameIdx.value(), 4);
}

TEST_F(UnitTestLtrManager, ComputeRecoveryFrameIdx_InvalidSlotIndexFails)
{
    LtrManager ltr(m_config);
    ltr.Reset();

    // Invalid slot idx returns invalid_argument error
    auto result1 = ltr.ComputeRecoveryFrameIdx(-1, 0);
    EXPECT_FALSE(result1);
    EXPECT_EQ(result1.error(), make_error_code(Error::invalid_argument));

    auto result2 = ltr.ComputeRecoveryFrameIdx(MAX_LTR_SLOTS, 0);
    EXPECT_FALSE(result2);
    EXPECT_EQ(result2.error(), make_error_code(Error::invalid_argument));
}

TEST_F(UnitTestLtrManager, ComputeRecoveryFrameIdx_EmptySlotStaleHint)
{
    LtrManager ltr(m_config);
    ltr.Reset();

    // Empty slot -> stale hint warning, continue as no hint -> nullopt
    auto recoveryFrameIdx = ltr.ComputeRecoveryFrameIdx(0, 0);
    ASSERT_TRUE(recoveryFrameIdx);
    EXPECT_FALSE(recoveryFrameIdx->has_value());
}

TEST_F(UnitTestLtrManager, ComputeRecoveryFrameIdx_OverwrittenSlot)
{
    m_config.ltrNumSlots = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    for (int i = 0; i < 16; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }

    auto recoveryFrameIdx = ltr.ComputeRecoveryFrameIdx(0, 0);

    ASSERT_TRUE(recoveryFrameIdx);
    EXPECT_EQ(recoveryFrameIdx.value(), 8);
}

TEST_F(UnitTestLtrManager, ComputeRecoveryFrameIdx_LtrDisabled)
{
    m_config.ltrPeriod = 0;
    LtrManager ltr(m_config);
    ltr.Reset();

    auto recoveryFrameIdx = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
    ASSERT_TRUE(recoveryFrameIdx);
    EXPECT_FALSE(recoveryFrameIdx->has_value());
}

TEST_F(UnitTestLtrManager, ProactiveRecovery_TriggersAtThreshold)
{
    for (const int ltrRecoveryPeriod : { 2, 4, 8 }) {
        SCOPED_TRACE(testing::Message() << "ltrRecoveryPeriod=" << ltrRecoveryPeriod);

        m_config.ltrStartIdx = 2;
        m_config.ltrPeriod = 4;
        m_config.ltrRecoveryPeriod = ltrRecoveryPeriod;
        LtrManager ltr(m_config);
        ltr.Reset();

        const int recoveryThreshold = m_config.ltrPeriod * m_config.ltrRecoveryPeriod;
        for (int frameIdx = 0; frameIdx < recoveryThreshold; frameIdx++) {
            auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
            ASSERT_TRUE(recovery) << "ComputeRecoveryFrameIdx failed at frame " << frameIdx;
            EXPECT_FALSE(recovery->has_value()) << "Unexpected proactive recovery at frame " << frameIdx;
            ltr.UpdateFrameMarking(frameIdx, 0, false);
        }

        auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
        ASSERT_TRUE(recovery) << "ComputeRecoveryFrameIdx failed at threshold frame " << recoveryThreshold;
        ASSERT_TRUE(recovery->has_value()) << "Expected proactive recovery at frame " << recoveryThreshold;
        EXPECT_EQ(**recovery, 2) << "Unexpected proactive recovery root at frame " << recoveryThreshold;
    }
}

TEST_F(UnitTestLtrManager, ProactiveRecovery_DefersToBaseLayerFrame)
{
    // Branch depth counts only base-layer frames, so with 2 temporal layers the recovery threshold is
    // crossed on an enhancement frame. Recovery must defer to the next base-layer frame, not fire early.
    m_config.numTemporalLayers = 2;
    m_config.ltrStartIdx = 2;
    m_config.ltrPeriod = 4;
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    // A branch spans recoveryThreshold base-layer frames, i.e. that many * numTemporalLayers frame
    // indices. Proactive recovery fires at each branch boundary, recovering from the previous branch's
    // root: ltrStartIdx for branch 0, the prior boundary thereafter.
    const int branchSpan = m_config.ltrPeriod * m_config.ltrRecoveryPeriod * m_config.numTemporalLayers;
    const int numBranches = 4;

    auto expectedProactiveRecovery = [&](const int frameIdx) -> std::optional<int> {
        if (frameIdx == 0 || frameIdx % branchSpan != 0) return std::nullopt;
        return frameIdx == branchSpan ? m_config.ltrStartIdx : frameIdx - branchSpan;
    };

    for (int frameIdx = 0; frameIdx < numBranches * branchSpan; frameIdx++) {
        const int temporalId = frameIdx % m_config.numTemporalLayers;
        auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, temporalId);
        ASSERT_TRUE(recovery) << "ComputeRecoveryFrameIdx failed at frame " << frameIdx;

        if (const auto expectedRoot = expectedProactiveRecovery(frameIdx)) {
            ASSERT_TRUE(recovery->has_value()) << "Expected proactive recovery at base-layer frame " << frameIdx;
            EXPECT_EQ(temporalId, 0) << "Proactive recovery must land on a base-layer frame";
            EXPECT_EQ(**recovery, *expectedRoot) << "Wrong recovery root at frame " << frameIdx;
        } else {
            EXPECT_FALSE(recovery->has_value()) << "Unexpected proactive recovery at frame " << frameIdx;
        }

        ltr.UpdateFrameMarking(frameIdx, temporalId, recovery->has_value());
    }

    EXPECT_EQ(ltr.GetBranch().GetBranchId(), numBranches - 1);
}

TEST_F(UnitTestLtrManager, ProactiveRecovery_DisabledWhenPeriodZero)
{
    m_config.ltrRecoveryPeriod = 0;
    LtrManager ltr(m_config);
    ltr.Reset();

    for (int i = 0; i < 16; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }

    auto recoveryFrameIdx = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
    ASSERT_TRUE(recoveryFrameIdx);
    EXPECT_FALSE(recoveryFrameIdx->has_value());
}

TEST_F(UnitTestLtrManager, ProactiveRecovery_RootProtectedDuringRotation)
{
    m_config.ltrNumSlots = 2;
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    // Process 8 frames then trigger proactive
    for (int i = 0; i < 8; i++) {
        auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
        ASSERT_TRUE(recovery);
        ltr.UpdateFrameMarking(i, 0, recovery->has_value());
    }

    // Proactive at frame 8
    auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
    ASSERT_TRUE(recovery);
    ASSERT_TRUE(recovery->has_value());
    EXPECT_EQ(**recovery, 0);  // root of branch 0

    // Mark frame 8 as new root
    ltr.UpdateFrameMarking(8, 0, true);

    // Both roots should be in slots
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 2);
}

TEST_F(UnitTestLtrManager, ProactiveRecovery_TakesPriorityOverHint)
{
    // When both proactive recovery triggers AND an external hint is provided,
    // proactive should take priority (design step 3 before step 4).
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    // Process 8 frames to fill branch (recoveryThreshold=8)
    for (int i = 0; i < 8; i++) {
        auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
        ASSERT_TRUE(recovery);
        ltr.UpdateFrameMarking(i, 0, recovery->has_value());
    }

    // At frame 8: proactive is ready (depth=8). Also provide a hint to slot 1 (frameIdx=4).
    // Proactive should win, recovering from root (frame 0), not from hint (frame 4).
    auto recovery = ltr.ComputeRecoveryFrameIdx(1, 0);
    ASSERT_TRUE(recovery);
    ASSERT_TRUE(recovery->has_value());
    EXPECT_EQ(**recovery, 0);  // proactive from root, not hint slot 1 (frameIdx=4)
}

TEST_F(UnitTestLtrManager, ProactiveRecovery_RootLostTriggersAtManagerLevel)
{
    // Root invalidated via external hint, next ComputeRecoveryFrameIdx triggers proactive.
    m_config.ltrStartIdx = 2;
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    // Branch 0: frames 0-7, root at P2
    for (int i = 0; i < 8; i++) {
        auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
        ASSERT_TRUE(recovery);
        ltr.UpdateFrameMarking(i, 0, recovery->has_value());
    }

    // Proactive -> branch 1, recover from P2
    auto recovery1 = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
    ASSERT_TRUE(recovery1);
    ASSERT_TRUE(recovery1->has_value());
    EXPECT_EQ(**recovery1, 2);

    // Mark P8 as root of branch 1
    ltr.UpdateFrameMarking(8, 0, true);
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 8);

    // External hint invalidates P8's root: hint to slot holding P2 (frameIdx=2)
    // InvalidateFramesAfter(2) clears slots > 2 (P6, P8, P12) and branch root (P8)
    auto recovery2 = ltr.ComputeRecoveryFrameIdx(1, 0);
    ASSERT_TRUE(recovery2);
    ASSERT_TRUE(recovery2->has_value());
    // Root P8 was invalidated -> branchId>0 && !root -> proactive triggers
    // Finds P2 as latest root -> recovers from P2
    EXPECT_EQ(**recovery2, 2);
}

TEST_F(UnitTestLtrManager, ProactiveRecovery_AllRootsLost_ErrorPropagates)
{
    // All roots invalidated at manager level -> ComputeRecoveryFrameIdx returns error (IDR fallback).
    m_config.ltrStartIdx = 2;
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    // Branch 0: frames 0-7, root at P2
    for (int i = 0; i < 8; i++) {
        auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
        ASSERT_TRUE(recovery);
        ltr.UpdateFrameMarking(i, 0, recovery->has_value());
    }

    // Proactive -> branch 1, recover from P2
    auto recovery1 = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
    ASSERT_TRUE(recovery1);
    ASSERT_TRUE(recovery1->has_value());
    ltr.UpdateFrameMarking(8, 0, true);

    // External hint invalidates everything > 0 (clears P2 root slot, P6, P8 root)
    // Only slot with frameIdx=0 (I0) survives, but it's not a root
    auto recovery2 = ltr.ComputeRecoveryFrameIdx(0, 0);
    // Hint slot 0 (I0, frameIdx=0) is valid -> invalidate > 0 clears P2, P6, P8
    // Proactive: branchId=2 (StartNewBranch from step 3), no root -> needs proactive
    // FindLatestFrameIdx(rootOnly=true) -> nullopt (all roots cleared)
    // -> error (reference_error)
    EXPECT_FALSE(recovery2);
    EXPECT_EQ(recovery2.error(), make_error_code(Error::reference_error));
}

TEST_F(UnitTestLtrManager, UpdateFrameMarking_StaleHintWithProactive_DoesNotAssert)
{
    // Regression: when the caller's queued useLtrSlotIdx points to a slot that
    // is empty by the time UpdateFrameMarking runs, the commit-phase prologue
    // must not assert. ComputeRecoveryFrameIdx internally clears its own hint,
    // but the encoder still passes the raw useLtrSlotIdx into the commit call.
    m_config.ltrStartIdx = 0;
    m_config.ltrRecoveryPeriod = 2;  // recoveryThreshold = ltrPeriod * ltrRecoveryPeriod = 8
    LtrManager ltr(m_config);
    ltr.Reset();

    // Fill branch 0 so NeedsProactiveRecovery() becomes true (depth >= 8).
    for (int i = 0; i <= 7; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }
    EXPECT_EQ(ltr.GetBranch().GetBranchDepth(), 8);
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 0);
    ASSERT_FALSE(ltr.GetLtrSlots()[3].HasValue());  // empty -> stale hint

    // Query returns the proactive recovery frame; the empty hint is ignored.
    auto recovery = ltr.ComputeRecoveryFrameIdx(3, 0);
    ASSERT_TRUE(recovery);
    ASSERT_TRUE(recovery->has_value());
    EXPECT_EQ(**recovery, 0);

    // Commit must not assert and must advance the branch.
    ltr.UpdateFrameMarking(8, 0, true, 3);
    EXPECT_EQ(ltr.GetBranch().GetBranchId(), 1);
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 8);
    EXPECT_EQ(ltr.GetBranch().GetBranchDepth(), 1);
}

TEST_F(UnitTestLtrManager, ExternalRecovery_InvalidatesNewerSlots)
{
    LtrManager ltr(m_config);
    ltr.Reset();

    // Mark frames 0-12: slots = [0, 4, 8, 12]
    for (int i = 0; i <= 12; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 4);

    // External recovery to slot 2 (frameIdx=8): pure query returns recovery frameIdx
    auto recovery = ltr.ComputeRecoveryFrameIdx(2, 0);
    ASSERT_TRUE(recovery);
    EXPECT_EQ(*recovery, 8);

    // Commit the recovery: processing an LTR_RECOVERY frame with the hint slot
    // triggers InvalidateFramesAfter(8), clearing frame 12 from slot 3.
    // The recovery frame itself (13) is then marked into the now-free slot 3.
    ltr.UpdateFrameMarking(13, 0, true, 2);

    // Slots <= 8 are preserved
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 4 });
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 8 });
    // Frame 12 was invalidated; slot 3 now holds the recovery frame 13
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 13 });
}

TEST_F(UnitTestLtrManager, ExternalRecovery_NoNewerSlots)
{
    LtrManager ltr(m_config);
    ltr.Reset();

    // Mark frames 0-4: slots = [0, 4]
    for (int i = 0; i <= 4; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }

    // External recovery to slot 1 (frameIdx=4): nothing newer to clear
    auto recovery = ltr.ComputeRecoveryFrameIdx(1, 0);
    ASSERT_TRUE(recovery);
    EXPECT_EQ(*recovery, 4);

    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 4 });
    EXPECT_EQ(CountUsedSlots(ltr.GetLtrSlots()), 2);
}

TEST_F(UnitTestLtrManager, ExternalRecovery_HintSlotInvalidated_FallsBackToLatest)
{
    LtrManager ltr(m_config);
    ltr.Reset();

    // Mark frames 0-12: slots = [0, 4, 8, 12]
    for (int i = 0; i <= 12; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }

    // Hint points to slot 0 (frameIdx=0): invalidate everything > 0
    // Slot 0 itself is preserved, slots 1,2,3 (frameIdx 4,8,12 > 0) are cleared
    // But hint slot (0) is still valid -> returns frameIdx=0
    auto recovery = ltr.ComputeRecoveryFrameIdx(0, 0);
    ASSERT_TRUE(recovery);
    EXPECT_EQ(*recovery, 0);
}

TEST_F(UnitTestLtrManager, ExternalRecovery_AllSlotsLost)
{
    LtrManager ltr(m_config);
    ltr.Reset();

    ltr.UpdateFrameMarking(0, 0, false);

    // Manually invalidate slot 0
    ltr.Reset();

    // Now all slots empty. External hint to slot 0 -> stale hint -> no recovery
    auto recovery = ltr.ComputeRecoveryFrameIdx(0, 0);
    ASSERT_TRUE(recovery);
    EXPECT_FALSE(recovery->has_value());  // stale hint treated as no hint
}

TEST_F(UnitTestLtrManager, ExternalRecovery_RecoveryFrameLost_ReRecoversFromSameLtr)
{
    // Edge case 5: recovery frame is lost, next hint points back to same pre-loss reference.
    // InvalidateFramesAfter clears the lost recovery frame's slot, encoder re-recovers.
    LtrManager ltr(m_config);
    ltr.Reset();

    // Mark frames 0-8: slots = [0, 4, 8]
    for (int i = 0; i <= 8; i++) {
        ltr.UpdateFrameMarking(i, 0, false);
    }

    // First recovery from slot 2 (frameIdx=8), invalidates slots > 8
    auto recovery1 = ltr.ComputeRecoveryFrameIdx(2, 0);
    ASSERT_TRUE(recovery1);
    EXPECT_EQ(*recovery1, 8);

    // Mark recovery frame 11 as in-branch LTR
    ltr.UpdateFrameMarking(11, 0, true, 2);

    // Recovery frame 11 is lost! Next hint points back to slot 2 (frameIdx=8) again.
    // InvalidateFramesAfter(8) should clear the slot holding frame 11.
    auto recovery2 = ltr.ComputeRecoveryFrameIdx(2, 0);
    ASSERT_TRUE(recovery2);
    EXPECT_EQ(*recovery2, 8);  // re-recovers from same LTR
}

TEST_F(UnitTestLtrManager, FullIllustration_8Branches)
{
    m_config.ltrStartIdx = 2;
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    const int totalFrames = 64;

    for (int frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
        // Compute recovery
        auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
        ASSERT_TRUE(recovery) << "ComputeRecoveryFrameIdx failed at frame " << frameIdx;

        bool isRecovery = recovery->has_value();

        // Process marking
        ltr.UpdateFrameMarking(frameIdx, 0, isRecovery);

        // Verify proactive recovery triggers at branch boundaries
        if (frameIdx == 8) {
            ASSERT_TRUE(isRecovery) << "Expected proactive recovery at frame 8";
            EXPECT_EQ(**recovery, 2) << "Frame 8 should recover from root P2";
        } else if (frameIdx == 16) {
            ASSERT_TRUE(isRecovery) << "Expected proactive recovery at frame 16";
            EXPECT_EQ(**recovery, 8) << "Frame 16 should recover from root P8";
        } else if (frameIdx == 24) {
            ASSERT_TRUE(isRecovery) << "Expected proactive recovery at frame 24";
            EXPECT_EQ(**recovery, 16) << "Frame 24 should recover from root P16";
        } else if (frameIdx == 32) {
            ASSERT_TRUE(isRecovery) << "Expected proactive recovery at frame 32";
            EXPECT_EQ(**recovery, 24) << "Frame 32 should recover from root P24";
        } else if (frameIdx == 40) {
            ASSERT_TRUE(isRecovery) << "Expected proactive recovery at frame 40";
            EXPECT_EQ(**recovery, 32) << "Frame 40 should recover from root P32";
        } else if (frameIdx == 48) {
            ASSERT_TRUE(isRecovery) << "Expected proactive recovery at frame 48";
            EXPECT_EQ(**recovery, 40) << "Frame 48 should recover from root P40";
        } else if (frameIdx == 56) {
            ASSERT_TRUE(isRecovery) << "Expected proactive recovery at frame 56";
            EXPECT_EQ(**recovery, 48) << "Frame 56 should recover from root P48";
        } else {
            EXPECT_FALSE(isRecovery) << "Unexpected recovery at frame " << frameIdx;
        }
    }

    // Verify branch state
    EXPECT_EQ(ltr.GetBranch().GetBranchId(), 7);

    // Verify final slot state (after branch 7: P56-P63)
    // Slots should contain P56(root), P52(kept), P48(root), P60(in-branch)
    // roots=[P48, P56]
    const auto& slots = ltr.GetLtrSlots();
    EXPECT_EQ(slots[0], LtrSlotInfo{ 56 });
    EXPECT_EQ(slots[1], LtrSlotInfo{ 52 });
    EXPECT_EQ(slots[2], LtrSlotInfo{ 48 });
    EXPECT_EQ(slots[3], LtrSlotInfo{ 60 });

    // Verify root protection: the two most recent roots are P48 and P56
    EXPECT_EQ(ltr.GetSlots().FindLatestFrameIdx(true), 56);
}

TEST_F(UnitTestLtrManager, FullIllustration_SlotTraceVerification)
{
    m_config.ltrStartIdx = 2;
    m_config.ltrRecoveryPeriod = 2;
    LtrManager ltr(m_config);
    ltr.Reset();

    auto ProcessBranch = [&](const int startFrame, const int endFrame) {
        for (int i = startFrame; i <= endFrame; i++) {
            auto recovery = ltr.ComputeRecoveryFrameIdx(std::nullopt, 0);
            ASSERT_TRUE(recovery) << "Frame " << i;
            ltr.UpdateFrameMarking(i, 0, recovery->has_value());
        }
    };

    // Branch 0 (I0-P7)
    ProcessBranch(0, 7);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 0 });  // I0 (in-branch)
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 2 });  // P2 (root)
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 6 });  // P6 (in-branch)
    EXPECT_FALSE(ltr.GetLtrSlots()[3].HasValue());      // empty
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 2);

    // Branch 1 (P8-P15)
    ProcessBranch(8, 15);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 12 });  // P12 (in-branch, evicts I0)
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 2 });   // P2 (root)
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 6 });   // P6 (in-branch)
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 8 });   // P8 (root)
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 8);

    // Branch 2 (P16-P23)
    ProcessBranch(16, 23);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 12 });  // P12 (kept)
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 20 });  // P20 (in-branch, evicts P2)
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 16 });  // P16 (root, evicts P6)
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 8 });   // P8 (root)
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 16);

    // Branch 3 (P24-P31)
    ProcessBranch(24, 31);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 24 });  // P24 (root, evicts P12)
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 20 });  // P20 (kept)
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 16 });  // P16 (root)
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 28 });  // P28 (in-branch, evicts P8)
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 24);

    // Branch 4 (P32-P39)
    ProcessBranch(32, 39);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 24 });  // P24 (root)
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 32 });  // P32 (root, evicts P20)
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 36 });  // P36 (in-branch, evicts P16)
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 28 });  // P28 (kept)
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 32);

    // Branch 5 (P40-P47)
    ProcessBranch(40, 47);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 44 });  // P44 (in-branch, evicts P24)
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 32 });  // P32 (root)
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 36 });  // P36 (kept)
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 40 });  // P40 (root, evicts P28)
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 40);

    // Branch 6 (P48-P55)
    ProcessBranch(48, 55);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 44 });  // P44 (kept)
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 52 });  // P52 (in-branch, evicts P32)
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 48 });  // P48 (root, evicts P36)
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 40 });  // P40 (root)
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 48);

    // Branch 7 (P56-P63)
    ProcessBranch(56, 63);
    EXPECT_EQ(ltr.GetLtrSlots()[0], LtrSlotInfo{ 56 });  // P56 (root, evicts P44)
    EXPECT_EQ(ltr.GetLtrSlots()[1], LtrSlotInfo{ 52 });  // P52 (kept)
    EXPECT_EQ(ltr.GetLtrSlots()[2], LtrSlotInfo{ 48 });  // P48 (root)
    EXPECT_EQ(ltr.GetLtrSlots()[3], LtrSlotInfo{ 60 });  // P60 (in-branch, evicts P40)
    EXPECT_EQ(ltr.GetBranch().GetRootFrameIdx(), 56);
    EXPECT_EQ(ltr.GetBranch().GetBranchId(), 7);
}
