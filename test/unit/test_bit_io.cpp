// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/bitstream/bit_io.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace libmlvc;

//------------------------------------------------------------------------------
// UnitTestBitWriter
//------------------------------------------------------------------------------

TEST(UnitTestBitWriter, InitialState)
{
    BitWriter writer;
    EXPECT_TRUE(writer.Empty());
    EXPECT_TRUE(writer.GetBytes().empty());
}

TEST(UnitTestBitWriter, Reset)
{
    BitWriter writer;
    writer.WriteUint(0xFF, 8);
    writer.Reset();
    EXPECT_TRUE(writer.Empty());
}

TEST(UnitTestBitWriter, WriteBits)
{
    BitWriter writer;
    for (bool bit : { true, false, true, true, false, false, true, false }) {
        writer.WriteBit(bit);
    }
    EXPECT_EQ(writer.GetBytes()[0], std::byte{ 0b10110010 });
}

TEST(UnitTestBitWriter, WriteUint)
{
    BitWriter writer;
    writer.WriteUint(0b101, 3);
    writer.WriteUint(0xAB, 8);
    writer.WriteUint(0x1234, 16);
    writer.WriteUint(0x7ABCDEF0, 31);

    BitReader reader(writer.GetBytes());
    EXPECT_EQ(reader.ReadUint(3).value(), 0b101);
    EXPECT_EQ(reader.ReadUint(8).value(), 0xAB);
    EXPECT_EQ(reader.ReadUint(16).value(), 0x1234);
    EXPECT_EQ(reader.ReadUint(31).value(), 0x7ABCDEF0);
}

TEST(UnitTestBitWriter, WriteUintCrossByteBoundary)
{
    BitWriter writer;
    writer.WriteUint(0b111100001111, 12);

    auto bytes = writer.GetBytes();
    ASSERT_EQ(bytes.size(), 2);
    EXPECT_EQ(bytes[0], std::byte{ 0b11110000 });
    EXPECT_EQ(bytes[1], std::byte{ 0b11110000 });
}

TEST(UnitTestBitWriter, WriteBytes)
{
    std::vector<std::byte> const payload = { std::byte{ 0xDE }, std::byte{ 0xAD }, std::byte{ 0xBE }, std::byte{ 0xEF } };

    BitWriter writer;
    writer.WriteUint(0b111, 3);
    writer.WriteBytes(payload);

    auto bytes = writer.GetBytes();
    EXPECT_EQ(bytes[0], std::byte{ 0b11100000 });
    EXPECT_EQ(bytes[1], std::byte{ 0xDE });
    EXPECT_EQ(bytes[4], std::byte{ 0xEF });
}

TEST(UnitTestBitWriter, WriteBytesAligned)
{
    std::vector<std::byte> const payload = { std::byte{ 0xCA }, std::byte{ 0xFE } };

    BitWriter writer;
    writer.WriteBytes(payload);

    auto bytes = writer.GetBytes();
    ASSERT_EQ(bytes.size(), 2);
    EXPECT_EQ(bytes[0], std::byte{ 0xCA });
    EXPECT_EQ(bytes[1], std::byte{ 0xFE });
}

TEST(UnitTestBitWriter, WriteBytesEmpty)
{
    std::vector<std::byte> empty;

    BitWriter writer;
    writer.WriteUint(0xFF, 8);
    writer.WriteBytes(empty);

    EXPECT_EQ(writer.GetBytes().size(), 1);
}

TEST(UnitTestBitWriter, BytePadding)
{
    for (int bits = 1; bits <= 7; bits++) {
        BitWriter writer;
        writer.WriteUint(0, bits);
        writer.WriteBytePadding();
        EXPECT_EQ(writer.GetBytes().size(), 1);
    }

    // Padding when already aligned adds nothing
    BitWriter writer;
    writer.WriteUint(0xFF, 8);
    writer.WriteBytePadding();
    EXPECT_EQ(writer.GetBytes().size(), 1);
}

TEST(UnitTestBitWriter, StopBit)
{
    BitWriter writer;
    writer.WriteUint(0b1010, 4);
    writer.WriteStopBit();
    EXPECT_EQ(writer.GetBytes()[0], std::byte{ 0b10101000 });
}

TEST(UnitTestBitWriter, WriteUeKnownValues)
{
    std::byte const expected[] = {
        std::byte{ 0x80 }, std::byte{ 0x40 }, std::byte{ 0x60 }, std::byte{ 0x20 },
        std::byte{ 0x28 }, std::byte{ 0x30 }, std::byte{ 0x38 }, std::byte{ 0x10 },
        std::byte{ 0x12 }, std::byte{ 0x14 }, std::byte{ 0x16 }, std::byte{ 0x18 },
    };

    for (int i = 0; i < 12; i++) {
        BitWriter writer;
        writer.WriteUe(i);
        EXPECT_EQ(writer.GetBytes()[0], expected[i]) << "UE(" << i << ")";
    }
}

TEST(UnitTestBitWriter, WriteSeKnownValues)
{
    std::byte const expected[] = {
        std::byte{ 0x1E }, std::byte{ 0x1A }, std::byte{ 0x16 }, std::byte{ 0x12 }, std::byte{ 0x38 },
        std::byte{ 0x28 }, std::byte{ 0x60 }, std::byte{ 0x80 }, std::byte{ 0x40 }, std::byte{ 0x20 },
        std::byte{ 0x30 }, std::byte{ 0x10 }, std::byte{ 0x14 }, std::byte{ 0x18 }, std::byte{ 0x1C },
    };

    for (int i = 0; i < 15; i++) {
        BitWriter writer;
        writer.WriteSe(i - 7);
        EXPECT_EQ(writer.GetBytes()[0], expected[i]) << "SE(" << (i - 7) << ")";
    }
}

//------------------------------------------------------------------------------
// UnitTestBitReader
//------------------------------------------------------------------------------

TEST(UnitTestBitReader, EmptyBuffer)
{
    std::vector<std::byte> empty;
    BitReader reader(empty);

    EXPECT_FALSE(reader.ReadBit());
    EXPECT_FALSE(reader.ReadUint(1));
    EXPECT_FALSE(reader.ReadUe());
}

TEST(UnitTestBitReader, ReadBits)
{
    std::vector<std::byte> bytes = { std::byte{ 0b10110010 } };
    BitReader reader(bytes);

    bool expected[] = { true, false, true, true, false, false, true, false };
    for (bool exp : expected) {
        EXPECT_EQ(reader.ReadBit().value(), exp);
    }
}

TEST(UnitTestBitReader, OutOfBounds)
{
    std::vector<std::byte> bytes = { std::byte{ 0xFF } };
    BitReader reader(bytes);

    for (int i = 0; i < 8; i++) {
        EXPECT_TRUE(reader.ReadBit());
    }
    EXPECT_FALSE(reader.ReadBit());
}

TEST(UnitTestBitReader, ReadUintOutOfBounds)
{
    std::vector<std::byte> bytes = { std::byte{ 0xFF } };
    BitReader reader(bytes);
    EXPECT_FALSE(reader.ReadUint(16));
}

TEST(UnitTestBitReader, ReadUintCrossBoundary)
{
    std::vector<std::byte> bytes = { std::byte{ 0b11110000 }, std::byte{ 0b11110000 } };
    BitReader reader(bytes);
    EXPECT_EQ(reader.ReadUint(12).value(), 0b111100001111);
}

TEST(UnitTestBitReader, BytePadding)
{
    BitWriter writer;
    writer.WriteUint(0b101, 3);
    writer.WriteBytePadding();

    BitReader reader(writer.GetBytes());
    EXPECT_EQ(reader.ReadUint(3).value(), 0b101);
    EXPECT_TRUE(reader.ReadBytePadding());
}

TEST(UnitTestBitReader, BytePaddingWhenAligned)
{
    std::vector<std::byte> bytes = { std::byte{ 0xFF } };
    BitReader reader(bytes);
    (void)reader.ReadUint(8);
    EXPECT_TRUE(reader.ReadBytePadding());
}

TEST(UnitTestBitReader, BytePaddingFailsOnNonZero)
{
    std::vector<std::byte> bytes = { std::byte{ 0b10000001 } };
    BitReader reader(bytes);
    (void)reader.ReadUint(1);
    EXPECT_FALSE(reader.ReadBytePadding());
}

TEST(UnitTestBitReader, StopBit)
{
    BitWriter writer;
    writer.WriteUint(0b1010, 4);
    writer.WriteStopBit();

    BitReader reader(writer.GetBytes());
    EXPECT_EQ(reader.ReadUint(4).value(), 0b1010);
    EXPECT_TRUE(reader.ReadStopBit());
}

TEST(UnitTestBitReader, StopBitFailsOnInvalidPadding)
{
    std::vector<std::byte> bytes = { std::byte{ 0b01110000 } };
    BitReader reader(bytes);
    (void)reader.ReadUint(1);
    EXPECT_FALSE(reader.ReadStopBit());
}

TEST(UnitTestBitReader, StopBitFailsWhenNotSet)
{
    std::vector<std::byte> bytes = { std::byte{ 0b00000000 } };
    BitReader reader(bytes);
    EXPECT_FALSE(reader.ReadStopBit());
}

TEST(UnitTestBitReader, ReadBytes)
{
    std::vector<std::byte> const payload = { std::byte{ 0xCA }, std::byte{ 0xFE }, std::byte{ 0xBA }, std::byte{ 0xBE } };

    BitWriter writer;
    writer.WriteUint(0b1010, 4);
    writer.WriteStopBit();
    writer.WriteBytes(payload);
    writer.WriteStopBit();

    BitReader reader(writer.GetBytes());
    EXPECT_EQ(reader.ReadUint(4).value(), 0b1010);
    EXPECT_TRUE(reader.ReadStopBit());

    auto bytes = reader.ReadBytes();
    ASSERT_TRUE(bytes);
    EXPECT_EQ(bytes->size(), payload.size());
}

TEST(UnitTestBitReader, ReadUeRoundTrip)
{
    for (int v : { 0, 1, 7, 8, 15, 16, 127, 128, 255, 256, 1023, 1024 }) {
        BitWriter writer;
        writer.WriteUe(v);
        BitReader reader(writer.GetBytes());
        EXPECT_EQ(reader.ReadUe().value(), v);
    }
}

TEST(UnitTestBitReader, ReadSeRoundTrip)
{
    for (int v : { 0, 1, -1, 2, -2, 127, -127, 128, -128, 1000, -1000 }) {
        BitWriter writer;
        writer.WriteSe(v);
        BitReader reader(writer.GetBytes());
        EXPECT_EQ(reader.ReadSe().value(), v);
    }
}

TEST(UnitTestBitReader, ReadBytesFailsAtEnd)
{
    BitWriter writer;
    writer.WriteStopBit();

    BitReader reader(writer.GetBytes());
    EXPECT_TRUE(reader.ReadStopBit());
    EXPECT_FALSE(reader.ReadBytes());
}

//------------------------------------------------------------------------------
// UnitTestBitstreamIntegration
//------------------------------------------------------------------------------

TEST(UnitTestBitstreamIntegration, MixedTypes)
{
    BitWriter writer;
    writer.WriteUint(42, 7);
    writer.WriteBit(true);
    writer.WriteUe(100);
    writer.WriteSe(-50);
    writer.WriteBit(false);
    writer.WriteUint(7, 4);
    writer.WriteStopBit();

    BitReader reader(writer.GetBytes());
    EXPECT_EQ(reader.ReadUint(7).value(), 42);
    EXPECT_EQ(reader.ReadBit().value(), true);
    EXPECT_EQ(reader.ReadUe().value(), 100);
    EXPECT_EQ(reader.ReadSe().value(), -50);
    EXPECT_EQ(reader.ReadBit().value(), false);
    EXPECT_EQ(reader.ReadUint(4).value(), 7);
    EXPECT_TRUE(reader.ReadStopBit());
}

TEST(UnitTestBitstreamIntegration, LargeData)
{
    BitWriter writer;
    for (int i = 0; i < 100; i++) {
        writer.WriteUint(i % 256, 8);
        writer.WriteUe(i);
        writer.WriteSe(i - 50);
    }
    writer.WriteStopBit();

    BitReader reader(writer.GetBytes());
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(reader.ReadUint(8).value(), i % 256);
        EXPECT_EQ(reader.ReadUe().value(), i);
        EXPECT_EQ(reader.ReadSe().value(), i - 50);
    }
    EXPECT_TRUE(reader.ReadStopBit());
}

//------------------------------------------------------------------------------
// UnitTestEmulationPrevention
//------------------------------------------------------------------------------

TEST(UnitTestEmulationPrevention, AddsPreventionByte)
{
    struct TestCase {
        std::vector<std::byte> input;
        std::byte expectedThird;
    };

    TestCase cases[] = {
        { { std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x00 } }, std::byte{ 0x00 } },
        { { std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x01 } }, std::byte{ 0x01 } },
        { { std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x02 } }, std::byte{ 0x02 } },
        { { std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x03 } }, std::byte{ 0x03 } },
    };

    for (auto const& tc : cases) {
        std::vector<std::byte> output;
        AddEmulationPreventionBytes(tc.input, output);

        ASSERT_EQ(output.size(), 4);
        EXPECT_EQ(output[2], std::byte{ 0x03 });
        EXPECT_EQ(output[3], tc.expectedThird);
    }
}

TEST(UnitTestEmulationPrevention, NoPreventionNeeded)
{
    std::vector<std::byte> const input = { std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x04 } };
    std::vector<std::byte> output;
    AddEmulationPreventionBytes(input, output);
    EXPECT_EQ(output.size(), 3);
}

TEST(UnitTestEmulationPrevention, PassthroughNonPattern)
{
    std::vector<std::byte> const input = { std::byte{ 0xDE }, std::byte{ 0xAD }, std::byte{ 0xBE }, std::byte{ 0xEF } };
    std::vector<std::byte> output;
    AddEmulationPreventionBytes(input, output);
    EXPECT_EQ(output, input);
}

TEST(UnitTestEmulationPrevention, RemovePreventionByte)
{
    struct TestCase {
        std::vector<std::byte> input;
        std::byte expectedLast;
    };

    TestCase cases[] = {
        { { std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x03 }, std::byte{ 0x00 } }, std::byte{ 0x00 } },
        { { std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x03 }, std::byte{ 0x01 } }, std::byte{ 0x01 } },
    };

    for (auto const& tc : cases) {
        std::vector<std::byte> output;
        RemoveEmulationPreventionBytes(tc.input, output);

        ASSERT_EQ(output.size(), 3);
        EXPECT_EQ(output[2], tc.expectedLast);
    }
}

TEST(UnitTestEmulationPrevention, RoundTrip)
{
    std::vector<std::byte> const original = { std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x00 },
                                              std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x01 },
                                              std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x02 },
                                              std::byte{ 0x00 }, std::byte{ 0x00 }, std::byte{ 0x03 } };

    std::vector<std::byte> encoded, decoded;
    AddEmulationPreventionBytes(original, encoded);
    RemoveEmulationPreventionBytes(encoded, decoded);

    EXPECT_EQ(decoded, original);
}

TEST(UnitTestEmulationPrevention, EmptyInput)
{
    std::vector<std::byte> empty, output;

    AddEmulationPreventionBytes(empty, output);
    EXPECT_TRUE(output.empty());

    RemoveEmulationPreventionBytes(empty, output);
    EXPECT_TRUE(output.empty());
}
