// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/gop_manager.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>
#include <libmlvc/unexpected.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <optional>

using namespace libmlvc;

namespace {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

expected<std::optional<int>> NoRecovery()
{
    return std::optional<int>{};
}

expected<std::optional<int>> Recover(int frameIdx)
{
    return std::optional<int>{ frameIdx };
}

expected<std::optional<int>> RecoveryError(Error e)
{
    return make_unexpected(make_error_code(e));
}

EncoderFrameInfo MakeFrameInfo(FrameType frameType, int frameIdx, int refFrameIdx = 0, int temporalId = 0,
                               int gopIdx = 0, int predictionChainLength = 0)
{
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
}

void ExpectFreshIdr(const EncoderFrameInfo& info)
{
    EXPECT_EQ(info.frameType, FrameType::I_FRAME);
    EXPECT_EQ(info.frameIdx, 0);
    EXPECT_EQ(info.refFrameIdx, 0);
    EXPECT_EQ(info.temporalId, 0);
    EXPECT_EQ(info.gopIdx, 0);
    EXPECT_EQ(info.predictionChainLength, 0);
}

}  // namespace

// -----------------------------------------------------------------------------
// GopManager Tests
// -----------------------------------------------------------------------------

TEST(UnitTestGopManager, FirstFrameIsIdr)
{
    const EncoderConfig config{};
    GopManager gop{ config };
    ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery()));
}

TEST(UnitTestGopManager, TemporalLayerAndRefFrameAssignment)
{
    {
        SCOPED_TRACE("TL=1");
        const EncoderConfig config{};
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));
        const std::array<int, 3> expectedChains = { 1, 2, 3 };
        for (int i = 0; i < 3; ++i) {
            const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery());
            EXPECT_EQ(info.frameType, FrameType::P_FRAME);
            EXPECT_EQ(info.frameIdx, i + 1);
            EXPECT_EQ(info.refFrameIdx, i);
            EXPECT_EQ(info.temporalId, 0);
            EXPECT_EQ(info.gopIdx, i + 1);
            EXPECT_EQ(info.predictionChainLength, expectedChains[i]);
            gop.Update(config, info);
        }
    }

    // TL0 refs idx-2 (previous TL0), TL1 refs idx-1; chains diverge from TL=1 because TL0 sub-chain is shorter.
    {
        SCOPED_TRACE("TL=2");
        const EncoderConfig config{ .numTemporalLayers = 2 };
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));

        struct Expected {
            int frameIdx;
            int refFrameIdx;
            int temporalId;
            int gopIdx;
            int chain;
        };
        const std::array<Expected, 5> table = {
            Expected{ 1, 0, 1, 1, 1 }, Expected{ 2, 0, 0, 2, 1 }, Expected{ 3, 2, 1, 3, 2 },
            Expected{ 4, 2, 0, 4, 2 }, Expected{ 5, 4, 1, 5, 3 },
        };
        for (const auto& e : table) {
            const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery());
            EXPECT_EQ(info.frameType, FrameType::P_FRAME);
            EXPECT_EQ(info.frameIdx, e.frameIdx);
            EXPECT_EQ(info.refFrameIdx, e.refFrameIdx);
            EXPECT_EQ(info.temporalId, e.temporalId);
            EXPECT_EQ(info.gopIdx, e.gopIdx);
            EXPECT_EQ(info.predictionChainLength, e.chain);
            gop.Update(config, info);
        }
    }
}

TEST(UnitTestGopManager, PeekNextTemporalIdMatchesNaturalCadence)
{
    {
        SCOPED_TRACE("empty history");
        const EncoderConfig config{};
        GopManager gop{ config };
        // Differing call-time configs intentionally exercise the empty-history layer-count-agnostic contract.
        EXPECT_EQ(gop.PeekNextTemporalId(EncoderConfig{ .numTemporalLayers = 1 }), 0);
        EXPECT_EQ(gop.PeekNextTemporalId(EncoderConfig{ .numTemporalLayers = 2 }), 0);
    }

    {
        SCOPED_TRACE("TL=1");
        const EncoderConfig config{ .numTemporalLayers = 1 };
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));
        for (int gopIdx = 1; gopIdx <= 4; ++gopIdx) {
            EXPECT_EQ(gop.PeekNextTemporalId(config), 0);
            gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, gopIdx, gopIdx - 1, 0, gopIdx, gopIdx));
        }
    }

    // Proactive-LTR gate relies on peek matching the temporalId ComputeFrameInfo will assign.
    {
        SCOPED_TRACE("TL=2 mirrors ComputeFrameInfo");
        const EncoderConfig config{ .numTemporalLayers = 2 };
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));
        EXPECT_EQ(gop.PeekNextTemporalId(config), 1);
        for (int i = 0; i < 6; ++i) {
            const int peeked = gop.PeekNextTemporalId(config);
            const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery());
            EXPECT_EQ(peeked, info.temporalId) << "peek must match natural temporalId at frameIdx " << info.frameIdx;
            gop.Update(config, info);
        }
    }

    // Any config diff makes peek behave as if gopIdx=0: same config -> 1 (natural next), changed -> 0.
    {
        SCOPED_TRACE("config change resets peek cadence");
        const EncoderConfig configA{ .numTemporalLayers = 2 };
        GopManager gop{ configA };
        gop.Update(configA, MakeFrameInfo(FrameType::I_FRAME, 0));
        gop.Update(configA, MakeFrameInfo(FrameType::P_FRAME, 1, 0, 1, 1, 1));
        gop.Update(configA, MakeFrameInfo(FrameType::P_FRAME, 2, 0, 0, 2, 1));
        EXPECT_EQ(gop.PeekNextTemporalId(configA), 1);
        EncoderConfig configB = configA;
        configB.iframePeriod = 64;
        EXPECT_EQ(gop.PeekNextTemporalId(configB), 0);
    }
}

TEST(UnitTestGopManager, ForcedAndPeriodicIdrTriggers)
{
    {
        SCOPED_TRACE("reference_error");
        const EncoderConfig config{};
        GopManager gop{ config };
        ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{}, RecoveryError(Error::reference_error)));
    }
    {
        SCOPED_TRACE("general_failure");
        const EncoderConfig config{};
        GopManager gop{ config };
        ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{}, RecoveryError(Error::general_failure)));
    }
    {
        SCOPED_TRACE("forceIdr alone");
        const EncoderConfig config{};
        GopManager gop{ config };
        ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{ .forceIdr = true }, NoRecovery()));
    }
    // forceIdr is checked before the LTR_RECOVERY return path -> Recover(5) does not win.
    {
        SCOPED_TRACE("forceIdr beats Recover(5)");
        const EncoderConfig config{};
        GopManager gop{ config };
        ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{ .forceIdr = true }, Recover(5)));
    }

    {
        SCOPED_TRACE("periodic via gopIdx");
        const EncoderConfig config{ .iframePeriod = 4 };
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));
        for (int i = 0; i < 3; ++i) {
            const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery());
            ASSERT_EQ(info.frameType, FrameType::P_FRAME);
            gop.Update(config, info);
        }
        ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery()));
    }

    // After LTR_RECOVERY resets gopIdx, frameIdx and gopIdx diverge; only the frameIdx arm can trigger IDR.
    {
        SCOPED_TRACE("periodic via frameIdx after recovery");
        const EncoderConfig config{ .iframePeriod = 5 };
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));
        for (int i = 0; i < 3; ++i) {
            const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery());
            ASSERT_EQ(info.frameType, FrameType::P_FRAME);
            gop.Update(config, info);
        }
        const auto recovery = gop.ComputeFrameInfo(config, EncodeParams{}, Recover(2));
        ASSERT_EQ(recovery.frameType, FrameType::LTR_RECOVERY);
        ASSERT_EQ(recovery.frameIdx, 4);
        ASSERT_EQ(recovery.gopIdx, 0);
        gop.Update(config, recovery);
        ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery()));
    }

    {
        SCOPED_TRACE("frameIdx overflow");
        const EncoderConfig config{};
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, (1 << FRAME_IDX_BITS) - 1, 0, 0, 0, 1));
        ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery()));
    }
}

TEST(UnitTestGopManager, LtrRecoveryFrame)
{
    {
        SCOPED_TRACE("non-empty history");
        const EncoderConfig config{};
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/1, /*ref=*/0, /*temporalId=*/0,
                                         /*gopIdx=*/1, /*chain=*/1));
        const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, Recover(1));
        EXPECT_EQ(info.frameType, FrameType::LTR_RECOVERY);
        EXPECT_EQ(info.frameIdx, 2);
        EXPECT_EQ(info.refFrameIdx, 1);
        EXPECT_EQ(info.gopIdx, 0);
        EXPECT_EQ(info.predictionChainLength, 2);
    }
    // LTR_RECOVERY returns before the gopIdx==0 I-frame fallback; unknown ref -> chain saturates to int::max.
    {
        SCOPED_TRACE("empty history beats I-frame fallback");
        const EncoderConfig config{};
        GopManager gop{ config };
        const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, Recover(5));
        EXPECT_EQ(info.frameType, FrameType::LTR_RECOVERY);
        EXPECT_EQ(info.frameIdx, 0);
        EXPECT_EQ(info.refFrameIdx, 5);
        EXPECT_EQ(info.gopIdx, 0);
        EXPECT_EQ(info.predictionChainLength, std::numeric_limits<int>::max());
    }
}

TEST(UnitTestGopManager, PredictionChainCap_AllBranches)
{
    // Cap formula: ltrStartIdx + iframePeriod/(ltrPeriod*ltrRecoveryPeriod) + (ltrPeriod*ltrRecoveryPeriod) - 2
    // -> 2 + 64/8 + 8 - 2 = 16.
    const EncoderConfig ladder{
        .iframePeriod = 64,
        .ltrStartIdx = 2,
        .ltrPeriod = 4,
        .ltrRecoveryPeriod = 2,
    };

    // Cap check sits before the LTR_RECOVERY return -> chain==cap+1 is an IDR (cap is a `>` backstop).
    {
        SCOPED_TRACE("ladder: cap pre-empts LTR_RECOVERY at boundary");
        GopManager gop{ ladder };
        gop.Update(ladder, MakeFrameInfo(FrameType::P_FRAME, 16, 0, 0, 0, 16));
        ExpectFreshIdr(gop.ComputeFrameInfo(ladder, EncodeParams{}, Recover(16)));
    }
    // Boundary is `>`: chain==cap must stay LTR_RECOVERY.
    {
        SCOPED_TRACE("ladder: chain == cap stays LTR_RECOVERY");
        GopManager gop{ ladder };
        gop.Update(ladder, MakeFrameInfo(FrameType::P_FRAME, 15, 0, 0, 0, 15));
        const auto info = gop.ComputeFrameInfo(ladder, EncodeParams{}, Recover(15));
        EXPECT_EQ(info.frameType, FrameType::LTR_RECOVERY);
        EXPECT_EQ(info.predictionChainLength, 16);
    }

    // Each disabling condition individually short-circuits ComputeMaxPredictionChainLength to 0.
    auto seedChain15 = [](const EncoderConfig& cfg) {
        GopManager gop{ cfg };
        gop.Update(cfg, MakeFrameInfo(FrameType::P_FRAME, 15, 0, 0, 0, 15));
        return gop;
    };
    {
        SCOPED_TRACE("no cap: ltrMode=EXTERNAL");
        EncoderConfig c = ladder;
        c.ltrMode = LtrMode::EXTERNAL;
        GopManager gop = seedChain15(c);
        EXPECT_EQ(gop.ComputeFrameInfo(c, EncodeParams{}, Recover(15)).frameType, FrameType::LTR_RECOVERY);
    }
    {
        SCOPED_TRACE("no cap: ltrRecoveryPeriod=0");
        EncoderConfig c = ladder;
        c.ltrRecoveryPeriod = 0;
        GopManager gop = seedChain15(c);
        EXPECT_EQ(gop.ComputeFrameInfo(c, EncodeParams{}, Recover(15)).frameType, FrameType::LTR_RECOVERY);
    }
    {
        SCOPED_TRACE("no cap: iframePeriod=0");
        EncoderConfig c = ladder;
        c.iframePeriod = 0;
        GopManager gop = seedChain15(c);
        EXPECT_EQ(gop.ComputeFrameInfo(c, EncodeParams{}, Recover(15)).frameType, FrameType::LTR_RECOVERY);
    }

    // ltrPeriod=0 -> recoveryThreshold=0 -> cap=iframePeriod (else arm). Values chosen so frameIdx=127<128
    // (periodic-via-frameIdx does not pre-empt) but chain=129>cap=128 fires the cap branch in isolation.
    {
        SCOPED_TRACE("fallback: recoveryThreshold==0, cap=iframePeriod");
        const EncoderConfig fallback{
            .iframePeriod = 128,
            .ltrPeriod = 0,
            .ltrRecoveryPeriod = 1,
        };
        GopManager gop{ fallback };
        gop.Update(fallback, MakeFrameInfo(FrameType::P_FRAME, 126, 0, 0, 0, 128));
        ExpectFreshIdr(gop.ComputeFrameInfo(fallback, EncodeParams{}, Recover(126)));
    }
}

TEST(UnitTestGopManager, ConfigChangeIdrBehavior)
{
    // ConfigChangeForcesIdr: TL-only diffs are soft (no IDR), other diffs are hard. Recover(1) makes the
    // two outcomes distinguishable: soft -> LTR_RECOVERY, hard -> fresh IDR (pre-empts the recovery return).
    const EncoderConfig seed{ .numTemporalLayers = 1 };
    auto seedOnePFrame = [&seed] {
        GopManager gop{ seed };
        gop.Update(seed, MakeFrameInfo(FrameType::I_FRAME, 0));
        gop.Update(seed, MakeFrameInfo(FrameType::P_FRAME, 1, 0, 0, 1, 1));
        return gop;
    };

    // gopIdx=0 in the LTR_RECOVERY output also proves the `config != m_config` reset in ComputeFrameInfo fires.
    {
        SCOPED_TRACE("TL-only change + recovery -> LTR_RECOVERY");
        GopManager gop = seedOnePFrame();
        EncoderConfig c = seed;
        c.numTemporalLayers = 2;
        const auto info = gop.ComputeFrameInfo(c, EncodeParams{}, Recover(1));
        EXPECT_EQ(info.frameType, FrameType::LTR_RECOVERY);
        EXPECT_EQ(info.refFrameIdx, 1);
        EXPECT_EQ(info.gopIdx, 0);
    }

    {
        SCOPED_TRACE("non-TL change + recovery -> fresh IDR (pre-empts LTR_RECOVERY)");
        GopManager gop = seedOnePFrame();
        EncoderConfig c = seed;
        c.iframePeriod = 64;
        ExpectFreshIdr(gop.ComputeFrameInfo(c, EncodeParams{}, Recover(1)));
    }

    // Production scenario from the TL-only Configure fix: history ends on tId=1, reconfigure 2TL->1TL,
    // next P-frame must skip past frame 3 (tId=1) and reference frame 2 (last tId=0). Without recovery
    // and without IDR, this is the only path that exercises GetReferenceFrameIdx's walk-back loop.
    {
        SCOPED_TRACE("TL=2 -> TL=1 reconfigure with last frame tId=1 -> P-frame refs last tId=0");
        const EncoderConfig tl2{ .numTemporalLayers = 2 };
        GopManager gop{ tl2 };
        gop.Update(tl2, MakeFrameInfo(FrameType::I_FRAME, 0));
        gop.Update(tl2, MakeFrameInfo(FrameType::P_FRAME, 1, 0, 1, 1, 1));
        gop.Update(tl2, MakeFrameInfo(FrameType::P_FRAME, 2, 0, 0, 2, 1));
        gop.Update(tl2, MakeFrameInfo(FrameType::P_FRAME, 3, 2, 1, 3, 2));

        const EncoderConfig tl1{ .numTemporalLayers = 1 };
        const auto info = gop.ComputeFrameInfo(tl1, EncodeParams{}, NoRecovery());
        EXPECT_EQ(info.frameType, FrameType::P_FRAME);
        EXPECT_EQ(info.frameIdx, 4);
        EXPECT_EQ(info.gopIdx, 0);
        EXPECT_EQ(info.temporalId, 0);
        EXPECT_EQ(info.refFrameIdx, 2);
        EXPECT_EQ(info.predictionChainLength, 2);
    }
}

TEST(UnitTestGopManager, ResetClearsAllHistory)
{
    const EncoderConfig config{};
    GopManager gop{ config };
    gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));
    gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, 1, 0, 0, 1, 1));
    gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, 2, 1, 0, 2, 2));

    gop.Reset();

    ExpectFreshIdr(gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery()));
}

TEST(UnitTestGopManager, UpdateHistoryManagement)
{
    {
        SCOPED_TRACE("I_FRAME clears history");
        const EncoderConfig config{};
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, 0, 0, 0, 0, 0));
        gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, 1, 0, 0, 1, 1));
        gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, 2, 1, 0, 2, 2));
        gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));

        const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery());
        EXPECT_EQ(info.frameType, FrameType::P_FRAME);
        EXPECT_EQ(info.frameIdx, 1);
        EXPECT_EQ(info.gopIdx, 1);
        EXPECT_EQ(info.refFrameIdx, 0);
        EXPECT_EQ(info.predictionChainLength, 1);
    }

    // chain=3 (ref-stored 2 + 1) proves frame 2 survived the LTR_RECOVERY Update.
    {
        SCOPED_TRACE("LTR_RECOVERY does NOT clear history");
        const EncoderConfig config{};
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::I_FRAME, 0));
        gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, 1, 0, 0, 1, 1));
        gop.Update(config, MakeFrameInfo(FrameType::LTR_RECOVERY, 2, 1, 0, 0, 2));

        const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, NoRecovery());
        EXPECT_EQ(info.frameType, FrameType::P_FRAME);
        EXPECT_EQ(info.frameIdx, 3);
        EXPECT_EQ(info.predictionChainLength, 3);
    }

    // ltrMode=EXTERNAL disables the chain cap so the inflated stored chain does not flip recovery to IDR.
    {
        SCOPED_TRACE("Update overwrites at same frameIdx");
        const EncoderConfig config{ .ltrMode = LtrMode::EXTERNAL };
        GopManager gop{ config };
        gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, 5, 0, 0, 0, 1));
        gop.Update(config, MakeFrameInfo(FrameType::P_FRAME, 5, 0, 0, 0, 99));
        const auto info = gop.ComputeFrameInfo(config, EncodeParams{}, Recover(5));
        EXPECT_EQ(info.frameType, FrameType::LTR_RECOVERY);
        EXPECT_EQ(info.predictionChainLength, 100);
    }
}

// -----------------------------------------------------------------------------
// GopTracker Tests
// -----------------------------------------------------------------------------

TEST(UnitTestGopTracker, ComputeFrameInfo_AllCases)
{
    auto MakeData = [](FrameType frameType, int curFrameIdx, int refFrameIdx = 0, int temporalId = 0) {
        FrameData fd{};
        fd.frameType = frameType;
        fd.curFrameIdx = curFrameIdx;
        fd.refFrameIdx = refFrameIdx;
        fd.temporalId = temporalId;
        return fd;
    };

    auto SeedTwoP = [](GopTracker& gop) {
        gop.Update(MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/1, /*ref=*/0, /*temporalId=*/0, /*gopIdx=*/1,
                                 /*chain=*/1));
        gop.Update(MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/2, /*ref=*/1, /*temporalId=*/0, /*gopIdx=*/2,
                                 /*chain=*/2));
    };

    {
        SCOPED_TRACE("I_FRAME (chain=0, fields propagate)");
        GopTracker gop;
        SeedTwoP(gop);
        const auto info = gop.ComputeFrameInfo(MakeData(FrameType::I_FRAME, /*curFrameIdx=*/10, /*ref=*/42,
                                                        /*temporalId=*/1));
        EXPECT_EQ(info.frameType, FrameType::I_FRAME);
        EXPECT_EQ(info.frameIdx, 10);
        EXPECT_EQ(info.refFrameIdx, 42);
        EXPECT_EQ(info.temporalId, 1);
        EXPECT_EQ(info.predictionChainLength, 0);
    }

    {
        SCOPED_TRACE("LTR_RECOVERY (chain from ref)");
        GopTracker gop;
        SeedTwoP(gop);
        const auto info = gop.ComputeFrameInfo(MakeData(FrameType::LTR_RECOVERY, /*curFrameIdx=*/11, /*ref=*/2));
        EXPECT_EQ(info.frameType, FrameType::LTR_RECOVERY);
        EXPECT_EQ(info.predictionChainLength, 3);
    }

    {
        SCOPED_TRACE("P_FRAME non-empty");
        GopTracker gop;
        SeedTwoP(gop);
        const auto info = gop.ComputeFrameInfo(MakeData(FrameType::P_FRAME, /*curFrameIdx=*/3, /*ref=*/2));
        EXPECT_EQ(info.frameType, FrameType::P_FRAME);
        EXPECT_EQ(info.predictionChainLength, 3);
    }

    // Unknown ref -> !contains branch returns int::max - 1; chain saturates to int::max (no overflow UB).
    {
        SCOPED_TRACE("P_FRAME empty history (chain=max)");
        GopTracker gop;
        const auto info = gop.ComputeFrameInfo(MakeData(FrameType::P_FRAME, /*curFrameIdx=*/5, /*ref=*/4));
        EXPECT_EQ(info.predictionChainLength, std::numeric_limits<int>::max());
    }

    // Negative ref -> GetPredictionChainLength's <0 branch returns 0; chain=1.
    {
        SCOPED_TRACE("P_FRAME negative ref (<0 branch)");
        GopTracker gop;
        const auto info = gop.ComputeFrameInfo(MakeData(FrameType::P_FRAME, /*curFrameIdx=*/0, /*ref=*/-1));
        EXPECT_EQ(info.predictionChainLength, 1);
    }

    // Stored chain at int::max-1 -> (max-1)+1 = max; saturating boundary, no signed-overflow UB.
    {
        SCOPED_TRACE("P_FRAME stored max-1 boundary");
        GopTracker gop;
        gop.Update(MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/1, /*ref=*/0, /*temporalId=*/0, /*gopIdx=*/1,
                                 /*chain=*/std::numeric_limits<int>::max() - 1));
        const auto info = gop.ComputeFrameInfo(MakeData(FrameType::P_FRAME, /*curFrameIdx=*/2, /*ref=*/1));
        EXPECT_EQ(info.predictionChainLength, std::numeric_limits<int>::max());
    }
}

TEST(UnitTestGopTracker, HistoryManagement)
{
    auto ProbeChain = [](const GopTracker& gop, int ref) {
        FrameData fd{};
        fd.frameType = FrameType::P_FRAME;
        fd.curFrameIdx = ref + 1;
        fd.refFrameIdx = ref;
        return gop.ComputeFrameInfo(fd).predictionChainLength;
    };

    // Cleared history -> ref unknown -> chain saturates to int::max.
    {
        SCOPED_TRACE("Reset clears history");
        GopTracker gop;
        gop.Update(MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/1, /*ref=*/0, /*temporalId=*/0, /*gopIdx=*/1,
                                 /*chain=*/5));
        gop.Reset();
        EXPECT_EQ(ProbeChain(gop, /*ref=*/1), std::numeric_limits<int>::max());
    }

    {
        SCOPED_TRACE("I_FRAME Update clears history");
        GopTracker gop;
        gop.Update(MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/1, /*ref=*/0, /*temporalId=*/0, /*gopIdx=*/1,
                                 /*chain=*/5));
        gop.Update(MakeFrameInfo(FrameType::I_FRAME, /*frameIdx=*/0));
        EXPECT_EQ(ProbeChain(gop, /*ref=*/1), std::numeric_limits<int>::max());
    }

    // chain=6 (ref-stored 5 + 1) proves frame 1 survived the LTR_RECOVERY Update.
    {
        SCOPED_TRACE("LTR_RECOVERY does NOT clear history");
        GopTracker gop;
        gop.Update(MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/1, /*ref=*/0, /*temporalId=*/0, /*gopIdx=*/1,
                                 /*chain=*/5));
        gop.Update(MakeFrameInfo(FrameType::LTR_RECOVERY, /*frameIdx=*/2, /*ref=*/1, /*temporalId=*/0,
                                 /*gopIdx=*/0, /*chain=*/6));
        EXPECT_EQ(ProbeChain(gop, /*ref=*/1), 6);
    }

    {
        SCOPED_TRACE("Update overwrites at same frameIdx");
        GopTracker gop;
        gop.Update(MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/5, /*ref=*/0, /*temporalId=*/0, /*gopIdx=*/0,
                                 /*chain=*/1));
        gop.Update(MakeFrameInfo(FrameType::P_FRAME, /*frameIdx=*/5, /*ref=*/0, /*temporalId=*/0, /*gopIdx=*/0,
                                 /*chain=*/99));
        EXPECT_EQ(ProbeChain(gop, /*ref=*/5), 100);
    }
}
