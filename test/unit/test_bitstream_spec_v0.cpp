// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/bitstream/coder.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace libmlvc;

namespace {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

std::vector<std::byte> Bytes(std::initializer_list<uint8_t> vals)
{
    std::vector<std::byte> r;
    r.reserve(vals.size());
    for (auto v : vals)
        r.push_back(std::byte{ v });
    return r;
}

struct TestFrame : FrameData {
    static TestFrame IFrame(std::span<const std::byte> payload, int width = 320, int height = 180, int qp = 26,
                            int frameIdxBits = 10, int maxTemporalLayers = 1, MlvcVersion version = { 0, 1 })
    {
        TestFrame f;
        f.mlvcVersion = version;
        f.modelWidth = width;
        f.modelHeight = height;
        f.frameIdxBits = frameIdxBits;
        f.maxTemporalLayers = maxTemporalLayers;
        f.frameType = FrameType::I_FRAME;
        f.qp = qp;
        f.payload = payload;
        return f;
    }

    TestFrame& AsPFrame(int frameIdx, int refFrameIdx = 0, int temporalId = 0, bool featureReset = false)
    {
        this->frameType = FrameType::P_FRAME;
        this->temporalId = temporalId;
        this->curFrameIdx = frameIdx;
        this->refFrameIdx = refFrameIdx;
        this->featureResetFlag = featureReset;
        return *this;
    }

    TestFrame& WithCrop(CropOffsets crop, bool transpose = false)
    {
        this->cropOffsets = crop;
        this->transposeFlag = transpose;
        return *this;
    }

    TestFrame& WithLtr(std::initializer_list<std::pair<int, int>> slots)
    {
        for (auto [idx, frameIdx] : slots)
            this->ltrSlots[idx] = LtrSlotInfo{ frameIdx };
        return *this;
    }
};

// -----------------------------------------------------------------------------
// Test data for v0.x frozen spec
// -----------------------------------------------------------------------------

struct TestEntry {
    std::string name;
    std::vector<TestFrame> frames;
    std::vector<std::vector<std::byte>> expectedBitstreams;
};

const auto PAYLOAD = Bytes({ 0xCA, 0xFE });
const auto EMUL_PAYLOAD = Bytes({ 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xBB });
const auto EMUL_PAYLOAD_02_03 = Bytes({ 0xDD, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03, 0xEE });

// clang-format off
const auto TEST_DATA = std::vector<TestEntry>{
    // Single I-frame: SPS field variations
    {
        .name = "BasicIFrame",
        .frames = { TestFrame::IFrame(PAYLOAD) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "MinorVersionZero",
        .frames = { TestFrame::IFrame(PAYLOAD, 320, 180, 26, 10, 1, {0, 0}) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x00,0x85,0x00,0x2D,0x01,0xC0,  // SPS (minor=0)
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "SmallResolution",
        .frames = { TestFrame::IFrame(PAYLOAD, 64, 64) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x81,0x00,0x10,0x01,0xC0,  // SPS (32x32 div2)
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "LargeResolution",
        .frames = { TestFrame::IFrame(PAYLOAD, 1920, 1080) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x9E,0x01,0x0E,0x01,0xC0,  // SPS (960x540 div2)
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "MaxResolution",
        .frames = { TestFrame::IFrame(PAYLOAD, 8190, 8190) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0xFF,0xFF,0xFF,0x81,0xC0,  // SPS (4095x4095 div2)
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "CropOnly",
        .frames = { TestFrame::IFrame(PAYLOAD).WithCrop({.left = 10, .right = 20}) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x26,0x17,0x87,  // SPS (crop_flag=1)
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "TransposeOnly",
        .frames = { TestFrame::IFrame(PAYLOAD).WithCrop({}, true) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x41,0xC0,  // SPS (transpose=1)
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "CropAndTranspose",
        .frames = { TestFrame::IFrame(PAYLOAD).WithCrop({.left = 20, .right = 40, .top = 10, .bottom = 30}, true) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x62,0xC2,0xA6,0x08,0x07,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },

    // Single I-frame: PPS and frame header variations
    {
        .name = "QpZero",
        .frames = { TestFrame::IFrame(PAYLOAD, 320, 180, 0) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xC1,0xAC,  // PPS: se(-26)
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "QpMax",
        .frames = { TestFrame::IFrame(PAYLOAD, 320, 180, 51) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xC1,0x94,  // PPS: se(25)
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }) },
    },
    {
        .name = "LtrSlotSingle",
        .frames = { TestFrame::IFrame(PAYLOAD).WithLtr({{0, 5}}) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC4,0x02,0x80,0xCA,0xFE,0x80,  // IDR (1 LTR slot)
        }) },
    },
    {
        .name = "LtrSlots",
        .frames = { TestFrame::IFrame(PAYLOAD).WithLtr({{0, 10}, {2, 20}}) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xCC,0x05,0x40,0xA0,0xCA,0xFE,0x80,  // IDR (LTR)
        }) },
    },
    {
        .name = "LtrSlotsAll8",
        .frames = { TestFrame::IFrame(PAYLOAD).WithLtr({{0,0},{1,10},{2,20},{3,30},{4,40},{5,50},{6,60},{7,70}}) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xE0,0x00,0x00,0xA0,0x28,0x07,0x81,0x40,0x32,0x07,0x81,0x18,  // IDR (8 LTR)
            0xCA,0xFE,0x80,
        }) },
    },

    // Single I-frame: emulation prevention
    {
        .name = "EmulationPrevention",
        .frames = { TestFrame::IFrame(EMUL_PAYLOAD) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,  // IDR hdr
            0xAA,0x00,0x00,0x03,0x00,0x00,0x03,0x00,0x01,0xBB,0x80,  // IDR payload
        }) },
    },
    {
        .name = "EmulationPrevention0203",
        .frames = { TestFrame::IFrame(EMUL_PAYLOAD_02_03) },
        .expectedBitstreams = { Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,  // IDR hdr
            0xDD,0x00,0x00,0x03,0x02,0x00,0x00,0x03,0x03,0xEE,0x80,  // IDR payload (EPB on 0x02 and 0x03)
        }) },
    },

    // I+P sequences
    {
        .name = "PFrame",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD).AsPFrame(1) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x01,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R
            }),
        },
    },
    {
        .name = "TsaFrame",
        .frames = { TestFrame::IFrame(PAYLOAD, 320, 180, 26, 10, 2), TestFrame::IFrame(PAYLOAD, 320, 180, 26, 10, 2).AsPFrame(1, 0, 1) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x05,0xC0,  // SPS (2 TL)
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x06,0x02,0xC0,0x01,0x00,0x00,0xCA,0xFE,0x80,  // TSA_R
            }),
        },
    },
    {
        .name = "FeatureReset",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD).AsPFrame(5, 4, 0, true) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x05,0x01,0x20,0xCA,0xFE,0x80,  // TRAIL_R
            }),
        },
    },
    {
        .name = "PFrameQpDelta",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD, 320, 180, 30).AsPFrame(1) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0x88,0x00,0x04,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R (se(4) qpDelta)
            }),
        },
    },
    {
        .name = "NegativeQpDelta",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD, 320, 180, 20).AsPFrame(1) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0x8D,0x00,0x04,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R (se(-6) qpDelta)
            }),
        },
    },
    {
        .name = "PFrameWithLtr",
        .frames = { TestFrame::IFrame(PAYLOAD).WithLtr({{0, 10}}), TestFrame::IFrame(PAYLOAD).AsPFrame(1).WithLtr({{0, 15}}) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC4,0x05,0x00,0xCA,0xFE,0x80,  // IDR (LTR slot 0=10)
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC4,0x07,0x80,0x20,0x00,0xCA,0xFE,0x80,  // TRAIL_R (LTR slot 0=15)
            }),
        },
    },
    {
        .name = "LtrSlotsWithGap",
        .frames = { TestFrame::IFrame(PAYLOAD).WithLtr({{0, 10}, {2, 20}}),
                    TestFrame::IFrame(PAYLOAD).AsPFrame(1).WithLtr({{0, 15}, {2, 25}}) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xCC,0x05,0x40,0xA0,0xCA,0xFE,0x80,  // IDR (LTR slots 0,2)
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xCC,0x07,0xC0,0xC8,0x02,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R (LTR gap)
            }),
        },
    },
    {
        .name = "LargeFrameIdx",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD).AsPFrame(1023, 1022) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC3,0xFF,0xFF,0x80,0xCA,0xFE,0x80,  // TRAIL_R (max u(10) indices)
            }),
        },
    },
    {
        .name = "FrameIdxBits8",
        .frames = { TestFrame::IFrame(PAYLOAD, 320, 180, 26, 8), TestFrame::IFrame(PAYLOAD, 320, 180, 26, 8).AsPFrame(1) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x03,  // SPS (frameIdxBitsMinus8=0)
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x04,0x00,0xCA,0xFE,0x80,  // TRAIL_R (u(8) indices)
            }),
        },
    },
    {
        .name = "FrameIdxBits16",
        .frames = { TestFrame::IFrame(PAYLOAD, 320, 180, 26, 16), TestFrame::IFrame(PAYLOAD, 320, 180, 26, 16).AsPFrame(1) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x00,0x4C,  // SPS (ue(8) frameIdxBitsMinus8)
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x00,0x04,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R (u(16) indices)
            }),
        },
    },

    // Multi-frame sequences
    {
        .name = "ThreeFrameSequence",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD).AsPFrame(1, 0), TestFrame::IFrame(PAYLOAD).AsPFrame(2, 1) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x01,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R (idx=1, ref=0)
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x02,0x00,0x40,0xCA,0xFE,0x80,  // TRAIL_R (idx=2, ref=1)
            }),
        },
    },
    {
        .name = "TwoIFrames",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
        },
    },
    {
        .name = "IdrQpDelta",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD, 320, 180, 30) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (initQpMinus26=0)
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR (qpDelta=0)
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS (same)
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (same, not updated)
                0x00,0x00,0x00,0x01,0x28,0x01,0x88,0x00,0xCA,0xFE,0x80,  // IDR (qpDelta=4)
            }),
        },
    },
    {
        .name = "SpsIdCycle",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD, 640, 360) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS (id=0)
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (id=0)
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x42,0x80,0x16,0x80,0x70,  // SPS (id=1, 640x360)
                0x00,0x00,0x00,0x01,0x44,0x01,0x4B,  // PPS (id=1, spsId=1)
                0x00,0x00,0x00,0x01,0x28,0x01,0x50,0xCA,0xFE,0x80,  // IDR (ppsId=1)
            }),
        },
    },
    {
        .name = "SpsIdWrap",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD, 640, 360),
                    TestFrame::IFrame(PAYLOAD, 1920, 1080), TestFrame::IFrame(PAYLOAD, 64, 64) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS (id=0)
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (id=0)
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x42,0x80,0x16,0x80,0x70,  // SPS (id=1)
                0x00,0x00,0x00,0x01,0x44,0x01,0x4B,  // PPS (id=1)
                0x00,0x00,0x00,0x01,0x28,0x01,0x50,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x67,0x80,0x43,0x80,0x70,  // SPS (id=2)
                0x00,0x00,0x00,0x01,0x44,0x01,0x6F,  // PPS (id=2)
                0x00,0x00,0x00,0x01,0x28,0x01,0x70,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x81,0x00,0x10,0x01,0xC0,  // SPS (id=0, wrapped)
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (id=0, wrapped)
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
        },
    },
    {
        .name = "IdrPIdrP",
        .frames = { TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD).AsPFrame(1), TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD).AsPFrame(1) },
        .expectedBitstreams = {
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x01,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS (same id=0)
                0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (same id=0)
                0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            }),
            Bytes({
                0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x01,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R
            }),
        },
    },
};
// clang-format on

// -----------------------------------------------------------------------------
// Error test data: input bytes → expected error (or nullopt for success)
// -----------------------------------------------------------------------------

struct ErrorTestEntry {
    std::string name;
    std::vector<std::byte> input;
    std::error_code expectedError;  // default = expect success
};

// clang-format off
const auto ERROR_TEST_DATA = std::vector<ErrorTestEntry>{
    // Truncated/degenerate input
    {   // Start code followed by a single byte — NALU header truncated
        .name = "OnlyStartCode",
        .input = Bytes({0x00,0x00,0x00,0x01,0x28}),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },
    {
        .name = "EmptyInput",
        .input = {},
        .expectedError = make_error_code(Error::bit_stream_partial_access_unit_error),
    },
    {
        .name = "TruncatedSps",
        .input = Bytes({0x00,0x00,0x00,0x01,0x42,0x01}),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },
    {
        .name = "TruncatedPps",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,  // PPS (header only, no body)
        }),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },
    {
        .name = "TruncatedFrame",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,  // IDR (header only, no frame body)
        }),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },
    {   // IDR with frame header but no payload bytes or stop bit
        .name = "EmptyPayloadFrame",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,  // IDR (frame header only, no payload/stop)
        }),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },
    {   // SPS with stop bit = 0 (last byte 0x80 instead of 0xC0)
        .name = "MissingStopBit",
        .input = Bytes({0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0x80}),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },
    {   // SPS with non-zero padding after stop bit (last byte 0xC1 instead of 0xC0)
        .name = "NonZeroPadding",
        .input = Bytes({0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC1}),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },

    // Incomplete access units
    {
        .name = "IncompleteAccessUnit",
        .input = Bytes({0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0}),
        .expectedError = make_error_code(Error::bit_stream_partial_access_unit_error),
    },
    {
        .name = "SpsAndPpsOnly",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (no frame follows)
        }),
        .expectedError = make_error_code(Error::bit_stream_partial_access_unit_error),
    },

    // NALU header errors
    {
        .name = "ForbiddenBit",
        .input = Bytes({0x00,0x00,0x00,0x01,0xC2,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0}),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },

    // SPS/PPS reference errors
    {
        .name = "PpsWithoutSps",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (no prior SPS)
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }),
        .expectedError = make_error_code(Error::bit_stream_missing_sps_error),
    },
    {   // PPS.spsId=1 but SPS.spsId=0
        .name = "FrameWithMismatchedSpsId",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS (spsId=0)
            0x00,0x00,0x00,0x01,0x44,0x01,0xAC,  // PPS (ppsId=0, spsId=1, se(0))
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }),
        .expectedError = make_error_code(Error::bit_stream_missing_sps_error),
    },
    {   // IDR.ppsId=1 but active PPS.ppsId=0
        .name = "FrameWithWrongPpsId",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS (ppsId=0)
            0x00,0x00,0x00,0x01,0x28,0x01,0x50,0xCA,0xFE,0x80,  // IDR (ppsId=1)
        }),
        .expectedError = make_error_code(Error::bit_stream_missing_pps_error),
    },
    {
        .name = "PFrameWithoutIdr",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x02,0x01,0xC0,0x01,0x00,0x00,0xCA,0xFE,0x80,  // TRAIL_R (no prior SPS/PPS)
        }),
        .expectedError = make_error_code(Error::bit_stream_missing_pps_error),
    },

    // Frame header field errors
    {   // numLtrSlots=9 exceeds MAX_LTR_SLOTS(8)
        .name = "TooManyLtrSlots",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xE4,0xCA,0xFE,0x80,  // IDR (numLtrSlots=9)
        }),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },

    // Structural errors
    {
        .name = "ExtraNaluAfterFrame",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // extra SPS
        }),
        .expectedError = make_error_code(Error::bit_stream_unexpected_error),
    },

    // Robustness (expected success)
    {   // 0x7E = 0_111111_0... (forbidden=0, naluType=63) → unknown type skipped
        .name = "UnknownNaluType",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x7E,0x01,0x80,  // unknown type=63
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }),
        .expectedError = {},  // success
    },
    {   // Two unknown NALUs between SPS and PPS — both skipped gracefully
        .name = "MultipleUnknownNalus",
        .input = Bytes({
            0x00,0x00,0x00,0x01,0x42,0x01,0x00,0x01,0x85,0x00,0x2D,0x01,0xC0,  // SPS
            0x00,0x00,0x00,0x01,0x7E,0x01,0x80,  // unknown type=63
            0x00,0x00,0x00,0x01,0x7E,0x01,0x80,  // unknown type=63 (2nd)
            0x00,0x00,0x00,0x01,0x44,0x01,0xF0,  // PPS
            0x00,0x00,0x00,0x01,0x28,0x01,0xC0,0xCA,0xFE,0x80,  // IDR
        }),
        .expectedError = {},  // success
    },
};
// clang-format on

}  // namespace

// -----------------------------------------------------------------------------
// Test fixture
// -----------------------------------------------------------------------------

class UnitTestBitstreamSpecV0 : public ::testing::Test {
protected:
    static void ExpectFrameDataEqual(const FrameData& a, const FrameData& e)
    {
        ASSERT_EQ(a.frameType, e.frameType);
        EXPECT_EQ(a.mlvcVersion.major, e.mlvcVersion.major);
        EXPECT_EQ(a.mlvcVersion.minor, e.mlvcVersion.minor);
        EXPECT_EQ(a.modelWidth, e.modelWidth);
        EXPECT_EQ(a.modelHeight, e.modelHeight);
        EXPECT_EQ(a.cropOffsets.left, e.cropOffsets.left);
        EXPECT_EQ(a.cropOffsets.right, e.cropOffsets.right);
        EXPECT_EQ(a.cropOffsets.top, e.cropOffsets.top);
        EXPECT_EQ(a.cropOffsets.bottom, e.cropOffsets.bottom);
        EXPECT_EQ(a.transposeFlag, e.transposeFlag);
        EXPECT_EQ(a.frameIdxBits, e.frameIdxBits);
        EXPECT_EQ(a.temporalId, e.temporalId);
        EXPECT_EQ(a.maxTemporalLayers, e.maxTemporalLayers);
        EXPECT_EQ(a.qp, e.qp);
        EXPECT_EQ(a.featureResetFlag, e.featureResetFlag);
        EXPECT_EQ(a.curFrameIdx, e.curFrameIdx);
        EXPECT_EQ(a.refFrameIdx, e.refFrameIdx);
        for (int i = 0; i < MAX_LTR_SLOTS; i++)
            EXPECT_EQ(a.ltrSlots[i], e.ltrSlots[i]) << "ltrSlot[" << i << "]";
        EXPECT_EQ(std::vector<std::byte>(a.payload.begin(), a.payload.end()),
                  std::vector<std::byte>(e.payload.begin(), e.payload.end()));
    }

    void ExpectEncoderMatch(const TestEntry& entry)
    {
        ASSERT_EQ(entry.frames.size(), entry.expectedBitstreams.size());
        BitstreamEncoder encoder;
        ASSERT_TRUE(encoder.Initialize());
        for (size_t i = 0; i < entry.frames.size(); i++) {
            auto r = encoder.Encode(entry.frames[i]);
            ASSERT_TRUE(r) << "frame[" << i << "]: " << r.error().message();
            std::vector<std::byte> actual(r.value().begin(), r.value().end());
            EXPECT_EQ(actual, entry.expectedBitstreams[i]) << "frame[" << i << "]";
        }
    }

    void ExpectDecoderMatch(const TestEntry& entry)
    {
        ASSERT_EQ(entry.frames.size(), entry.expectedBitstreams.size());
        BitstreamDecoder decoder;
        ASSERT_TRUE(decoder.Initialize());
        std::vector<std::vector<std::byte>> payloads;
        for (size_t i = 0; i < entry.expectedBitstreams.size(); i++) {
            auto r = decoder.Decode(entry.expectedBitstreams[i]);
            ASSERT_TRUE(r) << "frame[" << i << "]: " << r.error().message();
            payloads.emplace_back(r.value().payload.begin(), r.value().payload.end());
            auto decoded = r.value();
            decoded.payload = payloads.back();
            ExpectFrameDataEqual(decoded, entry.frames[i]);
        }
    }

    void ExpectRoundTrip(const std::vector<FrameData>& frames)
    {
        BitstreamEncoder encoder;
        ASSERT_TRUE(encoder.Initialize());
        BitstreamDecoder decoder;
        ASSERT_TRUE(decoder.Initialize());
        for (size_t i = 0; i < frames.size(); i++) {
            auto enc = encoder.Encode(frames[i]);
            ASSERT_TRUE(enc) << "encode[" << i << "]: " << enc.error().message();
            std::vector<std::byte> bs(enc.value().begin(), enc.value().end());
            auto dec = decoder.Decode(bs);
            ASSERT_TRUE(dec) << "decode[" << i << "]: " << dec.error().message();
            std::vector<std::byte> payload(dec.value().payload.begin(), dec.value().payload.end());
            auto decoded = dec.value();
            decoded.payload = payload;
            ExpectFrameDataEqual(decoded, frames[i]);
        }
    }

    void TestDecoderError(const ErrorTestEntry& entry)
    {
        BitstreamDecoder decoder;
        ASSERT_TRUE(decoder.Initialize());
        auto r = decoder.Decode(entry.input);
        if (entry.expectedError) {
            EXPECT_FALSE(r);
            EXPECT_EQ(r.error(), entry.expectedError);
        } else {
            EXPECT_TRUE(r) << r.error().message();
        }
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

TEST_F(UnitTestBitstreamSpecV0, Encoder)
{
    for (const auto& entry : TEST_DATA) {
        SCOPED_TRACE(entry.name);
        ExpectEncoderMatch(entry);
    }
}

TEST_F(UnitTestBitstreamSpecV0, Decoder)
{
    for (const auto& entry : TEST_DATA) {
        SCOPED_TRACE(entry.name);
        ExpectDecoderMatch(entry);
    }
}

TEST_F(UnitTestBitstreamSpecV0, ErrorHandling)
{
    for (const auto& entry : ERROR_TEST_DATA) {
        SCOPED_TRACE(entry.name);
        TestDecoderError(entry);
    }
}

TEST_F(UnitTestBitstreamSpecV0, RoundTrip)
{
    // Returns n evenly-spaced even values in [2, 8190].
    auto evenSizes = [](int n) {
        std::vector<int> v(n);
        for (int i = 0; i < n; i++) {
            v[i] = 2 + i * 8188 / (n - 1);
            v[i] -= v[i] % 2;
        }
        return v;
    };

    const auto W120 = evenSizes(120);
    const auto WH100 = evenSizes(100);
    const auto WH50 = evenSizes(50);
    const auto WH10 = evenSizes(10);

    // I-frame: single-dimension sweeps

    {
        SCOPED_TRACE("AllWidths");
        for (int w = 2; w <= 8190; w += 2)
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, w, 180) });
    }
    {
        SCOPED_TRACE("AllHeights");
        for (int h = 2; h <= 8190; h += 2)
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, h) });
    }
    {
        SCOPED_TRACE("AllQPs");
        for (int qp = 0; qp <= 51; qp++)
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, 180, qp) });
    }
    {
        SCOPED_TRACE("AllMinorVersions");
        for (int minor = 0; minor <= 255; minor++)
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, 180, 26, 10, 1, { 0, minor }) });
    }
    {
        SCOPED_TRACE("AllBitsxTL");
        for (int bits = 8; bits <= 16; bits++)
            for (int tl = 1; tl <= 2; tl++)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, 180, 26, bits, tl) });
    }

    // I-frame: multi-dimension sweeps

    {
        SCOPED_TRACE("QPxMinor");
        for (int qp = 0; qp <= 51; qp++)
            for (int minor = 0; minor <= 255; minor++)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, 180, qp, 10, 1, { 0, minor }) });
    }
    {
        SCOPED_TRACE("WxH");
        for (int w : W120)
            for (int h : W120)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, w, h) });
    }
    {
        SCOPED_TRACE("WxQP");
        for (int w : WH100)
            for (int qp = 0; qp <= 51; qp++)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, w, 180, qp) });
    }
    {
        SCOPED_TRACE("HxQP");
        for (int h : WH100)
            for (int qp = 0; qp <= 51; qp++)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, h, qp) });
    }
    {
        SCOPED_TRACE("QPxBitsxTL");
        for (int qp = 0; qp <= 51; qp++)
            for (int bits = 8; bits <= 16; bits++)
                for (int tl = 1; tl <= 2; tl++)
                    ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, 180, qp, bits, tl) });
    }
    {
        SCOPED_TRACE("WxMinor");
        for (int w : WH50)
            for (int minor = 0; minor <= 255; minor++)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, w, 180, 26, 10, 1, { 0, minor }) });
    }
    {
        SCOPED_TRACE("HxMinor");
        for (int h : WH50)
            for (int minor = 0; minor <= 255; minor++)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, h, 26, 10, 1, { 0, minor }) });
    }
    {
        SCOPED_TRACE("WxHxQP");
        for (int w : WH10)
            for (int h : WH10)
                for (int qp = 0; qp <= 51; qp++)
                    ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, w, h, qp) });
    }

    // Crop sweeps

    {
        SCOPED_TRACE("CropSingleDim");
        for (int v = 0; v <= 318; v += 2) {
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithCrop({ .left = v }) });
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithCrop({ .right = v }) });
            if (v <= 178) {
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithCrop({ .top = v }) });
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithCrop({ .bottom = v }) });
            }
        }
    }
    {
        SCOPED_TRACE("CropLxR");
        for (int left = 0; left <= 318; left += 8)
            for (int right = 0; left + right <= 318; right += 8)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithCrop({ .left = left, .right = right }) });
    }
    {
        SCOPED_TRACE("CropTxB");
        for (int top = 0; top <= 178; top += 4)
            for (int bottom = 0; top + bottom <= 178; bottom += 4)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithCrop({ .top = top, .bottom = bottom }) });
    }
    {
        SCOPED_TRACE("CropTranspose");
        for (int left = 0; left < 320; left += 16)
            for (int top = 0; top < 180; top += 8)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithCrop({ .left = left, .top = top }, true) });
    }

    // LTR sweeps

    {
        SCOPED_TRACE("LtrSlot0");
        for (int fi = 0; fi < 1024; fi++)
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithLtr({ { 0, fi } }) });
    }
    {
        SCOPED_TRACE("LtrMulti");
        for (int n = 1; n <= 8; n++)
            for (int base = 0; base < 25; base++) {
                TestFrame frame = TestFrame::IFrame(PAYLOAD);
                for (int s = 0; s < n; s++)
                    frame.ltrSlots[s] = LtrSlotInfo{ base * 8 + s };
                ExpectRoundTrip({ frame });
            }
    }

    // P-frame sweeps

    {
        SCOPED_TRACE("PFrameQP");
        for (int qp = 0; qp <= 51; qp++)
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD), TestFrame::IFrame(PAYLOAD, 320, 180, qp).AsPFrame(1) });
    }
    {
        SCOPED_TRACE("PFrameIdx");
        for (int bits = 8; bits <= 16; bits++) {
            int maxIdx = (1 << bits) - 1;
            int step = std::max(1, maxIdx / 1023);
            for (int fi = 1; fi <= maxIdx; fi += step)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, 180, 26, bits),
                                  TestFrame::IFrame(PAYLOAD, 320, 180, 26, bits).AsPFrame(fi, fi - 1) });
        }
    }
    {
        SCOPED_TRACE("PFrameFeatTidQP");
        for (bool fr : { false, true })
            for (int tid : { 0, 1 }) {
                int tl = (tid > 0) ? 2 : 1;
                for (int qp = 0; qp <= 51; qp++)
                    ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, 320, 180, 26, 10, tl),
                                      TestFrame::IFrame(PAYLOAD, 320, 180, qp, 10, tl).AsPFrame(1, 0, tid, fr) });
            }
    }
    {
        SCOPED_TRACE("PFrameLtr");
        for (int fi = 0; fi < 256; fi++)
            ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD).WithLtr({ { 0, fi } }),
                              TestFrame::IFrame(PAYLOAD).AsPFrame(1).WithLtr({ { 0, fi + 1 } }) });
    }
    {
        SCOPED_TRACE("PFrameQPxRes");
        const std::pair<int, int> res[] = {
            { 2, 2 },       { 64, 64 },     { 320, 180 }, { 640, 360 }, { 960, 540 },
            { 1920, 1080 }, { 8190, 8190 }, { 320, 640 }, { 360, 360 }, { 100, 100 },
        };
        for (auto [w, h] : res)
            for (int qp = 0; qp <= 51; qp++)
                ExpectRoundTrip({ TestFrame::IFrame(PAYLOAD, w, h), TestFrame::IFrame(PAYLOAD, w, h, qp).AsPFrame(1) });
    }
}
