// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/bitstream/coder.hpp"
#include "libmlvc/common/utils.hpp"

#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <memory>
#include <optional>
#include <span>

namespace libmlvc {

// Forward declarations
class MlvcManagerImpl;
class IMlvcEncoderCore;
class IMlvcDecoderCore;

class EncoderStatsLogger {
public:
    EncoderStatsLogger(const std::string& tag) : m_tag(tag) {}
    void Update(const EncoderStats& stats);

private:
    const std::string& m_tag;
    const int m_logInterval{ 128 };
    struct PrevTimerCounts {
        int reconfigure{};
        int preprocess{};
        int inference{};
        int scaleDecoder{};
        int entropyCoding{};
        int total{};
        int frameInterval{};
    };
    EncoderStats m_stats{};
    int m_frameCount{};
    PrevTimerCounts m_prevTimerCounts{};
    EncoderStats m_counterBaseline{};
};

class DecoderStatsLogger {
public:
    DecoderStatsLogger(const std::string& tag) : m_tag(tag) {}
    void Update(const DecoderStats& stats);

private:
    const std::string& m_tag;
    const int m_logInterval{ 128 };
    struct PrevTimerCounts {
        int reconfigure{};
        int entropyCoding{};
        int scaleDecoder{};
        int inference{};
        int postprocess{};
        int total{};
        int frameInterval{};
    };
    DecoderStats m_stats{};
    int m_frameCount{};
    PrevTimerCounts m_prevTimerCounts{};
    DecoderStats m_counterBaseline{};
};

class MlvcEncoderImpl {
public:
    ~MlvcEncoderImpl();
    expected<void> Configure(const EncoderConfig& config);
    expected<void> MarkNextFrameAsLtr(const int ltrSlotIdx);
    expected<FrameInfo> GetNextFrameInfo(const EncodeParams& params) const;
    expected<EncodedFrame> Encode(const Nv12FrameView& frame, const EncodeParams& params);
    EncoderConfig GetConfig() const { return m_config; }
    const IMlvcEncoderCore& GetCore() const { return *m_encoderCore; }
    const EncoderStats& GetStats() const { return m_stats; }

protected:
    friend class MlvcManagerImpl;

    std::string m_tag;
    std::shared_ptr<const MlvcManagerImpl> m_manager;
    EncoderConfig m_config{ .mlvcVersion = { -1, -1 } };
    bool m_configChanged{ true };
    std::optional<int> m_markLtrSlotIdx{};
    std::unique_ptr<IMlvcEncoderCore> m_encoderCore;
    BitstreamEncoder m_bitStreamEncoder;
    EncoderStats m_stats{};
    IntervalOpTimer m_frameIntervalTimer{ m_stats.frameInterval };
    EncoderStatsLogger m_statsLogger{ m_tag };

    MlvcEncoderImpl(const std::string& tag, const std::shared_ptr<const MlvcManagerImpl>& manager);
    expected<void> Initialize(const EncoderConfig& config);
    expected<void> ValidateEncoderConfig(const EncoderConfig& config) const;
    expected<void> ValidateInputFrame(const Nv12FrameView& frame) const;
    expected<void> ValidateEncodeParams(const EncodeParams& params) const;
    bool IsCoreRebuildRequired() const;
    expected<void> ConfigureEncoder();
};

class MlvcDecoderImpl {
public:
    ~MlvcDecoderImpl();
    expected<DecodedFrame> Decode(std::span<const std::byte> bytes);
    const IMlvcDecoderCore& GetCore() const { return *m_decoderCore; }
    const DecoderStats& GetStats() const { return m_stats; }

protected:
    friend class MlvcManagerImpl;

    std::string m_tag;
    std::shared_ptr<const MlvcManagerImpl> m_manager;
    BitstreamDecoder m_bitStreamDecoder;
    std::unique_ptr<IMlvcDecoderCore> m_decoderCore;
    DecoderStats m_stats{};
    IntervalOpTimer m_frameIntervalTimer{ m_stats.frameInterval };
    DecoderStatsLogger m_statsLogger{ m_tag };

    MlvcDecoderImpl(const std::string& tag, const std::shared_ptr<const MlvcManagerImpl>& manager);
    expected<void> Initialize();
    expected<void> ConfigureDecoder(const MlvcVersion mlvcVersion, const int displayWidth, const int displayHeight,
                                    const int modelWidth, const int modelHeight);
};

}  // namespace libmlvc
