// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"

#include <libmlvc/expected.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace libmlvc {

void AddEmulationPreventionBytes(std::span<const std::byte> input, std::vector<std::byte>& output);
void RemoveEmulationPreventionBytes(std::span<const std::byte> input, std::vector<std::byte>& output);

class BitWriter {
public:
    BitWriter(size_t reservedSize = 64) { m_buffer.reserve(reservedSize); }

    void Reset()
    {
        m_buffer.clear();
        m_pos = 0;
    }

    bool Empty() const { return m_pos == 0; }

    void WriteBit(const bool value)
    {
        const size_t bytePos = m_pos / CHAR_BIT;
        while (bytePos >= m_buffer.size()) {
            m_buffer.push_back(std::byte{ 0 });
        }
        const std::byte tmp = std::byte(0x01) << (CHAR_BIT - 1 - m_pos % CHAR_BIT);
        if (value) {
            m_buffer[bytePos] |= tmp;
        } else {
            m_buffer[bytePos] &= ~tmp;
        }
        m_pos++;
    }

    void WriteUint(const int value, const int numBits)
    {
        MLVC_ASSERT(value >= 0);
        MLVC_ASSERT(numBits <= sizeof(int) * CHAR_BIT - 1);
        for (int i = numBits - 1; i >= 0; --i) {
            WriteBit((value >> i) & 0x01);
        }
    }

    void WriteUe(const int value)
    {
        // Write unsigned Exponential Golomb code
        MLVC_ASSERT(value >= 0);
        MLVC_ASSERT(value < std::numeric_limits<int>::max());

        int numBits = 0;
        int tmp = value + 1;
        while (tmp > 0) {
            tmp >>= 1;
            numBits++;
        }
        WriteUint(0, numBits - 1);
        WriteUint(value + 1, numBits);
    }

    void WriteSe(const int value)
    {
        if (value <= 0) {
            MLVC_ASSERT(value > std::numeric_limits<int>::min() / 2);
            WriteUe(-2 * value);
        } else {
            MLVC_ASSERT(value <= std::numeric_limits<int>::max() / 2);
            WriteUe(2 * value - 1);
        }
    }

    void WriteBytePadding()
    {
        const int padding = CHAR_BIT - m_pos % CHAR_BIT;
        if (padding != CHAR_BIT) {
            WriteUint(0, padding);
        }
    }

    void WriteStopBit()
    {
        WriteBit(true);
        WriteBytePadding();
    }

    void WriteBytes(std::span<const std::byte> bytes)
    {
        WriteBytePadding();
        const size_t bytePos = m_pos / CHAR_BIT;
        m_buffer.resize(bytePos + bytes.size());
        std::copy(bytes.begin(), bytes.end(), m_buffer.begin() + bytePos);
        m_pos += bytes.size() * CHAR_BIT;
    }

    std::span<const std::byte> GetBytes() const { return std::span<const std::byte>(m_buffer); }

private:
    std::vector<std::byte> m_buffer;
    size_t m_pos{ 0 };
};

class BitReader {
public:
    BitReader(std::span<const std::byte> buffer) : m_buffer(buffer) {}

    expected<bool> ReadBit()
    {
        if (m_pos >= m_buffer.size() * CHAR_BIT) {
            MLVC_LOG_ERROR("ReadBit: out of bounds");
            return make_error_code(Error::bit_stream_unexpected_error);
        }
        const size_t bytePos = m_pos / CHAR_BIT;
        const std::byte tmp = std::byte(0x01) << (CHAR_BIT - 1 - m_pos % CHAR_BIT);
        const bool value = (m_buffer[bytePos] & tmp) != std::byte{ 0 };
        m_pos++;
        return value;
    }

    expected<int> ReadUint(const int numBits)
    {
        MLVC_ASSERT(numBits <= CHAR_BIT * sizeof(int));
        int value = 0;
        for (int i = 0; i < numBits; ++i) {
            auto bit = ReadBit();
            if (!bit) return bit.error();
            value = (value << 1) | (bit.value() ? 1 : 0);
        }
        return value;
    }

    expected<int> ReadUe()
    {
        // Read unsigned Exponential Golomb code
        int numBits = 1;
        while (true) {
            auto bit = ReadBit();
            if (!bit) return bit.error();
            if (bit.value()) {
                m_pos--;
                break;
            }
            numBits++;
            if (numBits >= sizeof(int) * CHAR_BIT) {
                MLVC_LOG_ERROR("ReadUe: out of bounds");
                return make_error_code(Error::bit_stream_unexpected_error);
            }
        }

        auto res = ReadUint(numBits);
        if (!res) return res.error();
        return res.value() - 1;
    }

    expected<int> ReadSe()
    {
        auto ue = ReadUe();
        if (!ue) return ue.error();
        return ue.value() % 2 == 0 ? -ue.value() / 2 : (ue.value() + 1) / 2;
    }

    expected<void> ReadBytePadding()
    {
        const size_t padding = CHAR_BIT - m_pos % CHAR_BIT;
        if (padding != CHAR_BIT) {
            auto ret = ReadUint(static_cast<int>(padding));
            if (!ret) return ret.error();
            if (ret.value() != 0) {
                MLVC_LOG_ERROR("ReadBytePadding: padding is not zero");
                return make_error_code(Error::bit_stream_unexpected_error);
            }
        }
        return {};
    }

    expected<void> ReadStopBit()
    {
        auto ret = ReadBit();
        if (!ret) return ret.error();
        if (!ret.value()) {
            MLVC_LOG_ERROR("ReadStopBit: stop bit is not set");
            return make_error_code(Error::bit_stream_unexpected_error);
        }
        return ReadBytePadding();
    }

    expected<std::span<const std::byte>> ReadBytes()
    {
        if (auto ret = ReadBytePadding(); !ret) return ret.error();
        if (m_pos >= m_buffer.size() * CHAR_BIT) {
            MLVC_LOG_ERROR("ReadBytes: out of bounds");
            return make_error_code(Error::bit_stream_unexpected_error);
        }

        // -1 for the stop bit
        const size_t numBytes = (m_buffer.size() * CHAR_BIT - m_pos - 1) / CHAR_BIT;
        auto res = m_buffer.subspan(m_pos / CHAR_BIT, numBytes);
        m_pos += numBytes * CHAR_BIT;
        return res;
    }

private:
    std::span<const std::byte> m_buffer;
    size_t m_pos{ 0 };
};

}  // namespace libmlvc
