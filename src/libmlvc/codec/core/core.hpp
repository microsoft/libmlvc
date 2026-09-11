// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/codec/core/frame_processing.hpp"
#include "libmlvc/codec/core/gop_manager.hpp"
#include "libmlvc/codec/core/inference_model.hpp"
#include "libmlvc/codec/core/reference_manager.hpp"
#include "libmlvc/codec/core/scale_decoder.hpp"
#include "libmlvc/codec/core/transforms.hpp"
#include "libmlvc/common/tensor.hpp"
#include "libmlvc/common/utils.hpp"
#include "libmlvc/entropy/latent_coder.hpp"
#include "libmlvc/schema/bundle.hpp"

#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace libmlvc {

struct EncoderCoreResult {
    FrameInfo info;
    FrameData data;
};

class IMlvcCodecCoreCommon {
public:
    virtual ~IMlvcCodecCoreCommon() = default;
    virtual MlvcVersion GetMlvcVersion() const = 0;
    virtual int GetDisplayWidth() const = 0;
    virtual int GetDisplayHeight() const = 0;
    virtual int GetModelWidth() const = 0;
    virtual int GetModelHeight() const = 0;
};

class IMlvcEncoderCore : public virtual IMlvcCodecCoreCommon {
public:
    virtual ~IMlvcEncoderCore() = default;
    virtual expected<void> Initialize() = 0;
    virtual expected<void> Configure(const EncoderConfig& config) = 0;
    virtual expected<void> MarkNextFrameAsLtr(const int ltrSlotIdx) = 0;
    virtual expected<FrameInfo> GetNextFrameInfo(const EncoderConfig& config, const EncodeParams& params) const = 0;
    virtual expected<EncoderCoreResult> Encode(const Nv12FrameView& frame, const EncodeParams& params) = 0;
    virtual EncoderInterfaceType GetEncoderInterfaceType() const = 0;
};

class IMlvcDecoderCore : public virtual IMlvcCodecCoreCommon {
public:
    virtual ~IMlvcDecoderCore() = default;
    virtual expected<void> Initialize() = 0;
    virtual expected<DecodedFrame> Decode(const FrameData& frameData) = 0;
    virtual DecoderInterfaceType GetDecoderInterfaceType() const = 0;
};

class MlvcCodecCoreCommon : public virtual IMlvcCodecCoreCommon {
public:
    MlvcCodecCoreCommon(const MlvcVersion mlvcVersion, const int displayWidth, const int displayHeight,
                        const std::shared_ptr<const ModelMetadata>& modelMetadata,
                        std::unique_ptr<IScaleDecoder> scaleDecoder);
    expected<void> Initialize();
    MlvcVersion GetMlvcVersion() const override { return m_mlvcVersion; }
    int GetDisplayWidth() const override { return m_displayWidth; }
    int GetDisplayHeight() const override { return m_displayHeight; }
    int GetModelWidth() const override { return m_modelMetadata->modelWidth; }
    int GetModelHeight() const override { return m_modelMetadata->modelHeight; }
    const ScaleValues& GetScales() const { return m_scaleDecoder->GetScales(); }

protected:
    bool m_initialized{ false };
    const MlvcVersion m_mlvcVersion{};
    const int m_displayWidth{};
    const int m_displayHeight{};
    const std::shared_ptr<const ModelMetadata> m_modelMetadata{};
    const std::unique_ptr<IScaleDecoder> m_scaleDecoder{};
    Tensor<int32_t, 3> m_yRaw0Int32{};
    Tensor<int32_t, 3> m_yRaw1Int32{};
    Tensor<int32_t, 3> m_zRawInt32{};

    expected<int> MapQpToQIndex(const int qp) const;
    expected<int> GetQIndexShifted(const int qIndex, const int curFrameIdx) const;
};

class MlvcEncoderCore : public MlvcCodecCoreCommon, public IMlvcEncoderCore {
public:
    struct Stats {
        OpTimerStats preprocess;
        OpTimerStats inference;
        OpTimerStats scaleDecoder;
        OpTimerStats entropyCoding;
    };

    MlvcEncoderCore(const MlvcVersion mlvcVersion, const EncoderConfig& config,
                    const std::shared_ptr<const ModelMetadata>& modelMetadata, std::unique_ptr<IScaleDecoder> scaleDecoder,
                    const std::shared_ptr<const GaussianCoderPmf>& gaussianPmf,
                    const std::shared_ptr<const BitEstimatorPmf>& bitEstimatorPmf, const InferenceSessionPtr& session);
    ~MlvcEncoderCore();
    expected<void> Initialize() override;
    expected<void> Configure(const EncoderConfig& config) override;
    expected<void> MarkNextFrameAsLtr(const int ltrSlotIdx) override;
    expected<FrameInfo> GetNextFrameInfo(const EncoderConfig& config, const EncodeParams& params) const override;
    expected<EncoderCoreResult> Encode(const Nv12FrameView& frame, const EncodeParams& params) override;
    EncoderInterfaceType GetEncoderInterfaceType() const override
    {
        return EncoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P;
    }
    const MlvcEncoderModel::Inputs& GetModelInputs() const { return m_inputs; }
    const MlvcEncoderModel::Outputs& GetModelOutputs() const { return m_outputs; }
    const Stats& GetStats() const { return m_stats; }

protected:
    EncoderConfig m_config;
    MlvcEncoderModel m_model;
    ReferenceManager m_ref;
    LtrManager m_ltr;
    GopManager m_gop;
    MlvcEncoderModel::Inputs m_inputs;
    MlvcEncoderModel::Outputs m_outputs;
    InputTransformer m_inputTransformer{};
    rans::HeapResizableBuffer m_buffer{};
    rans::RansEncoderStream m_stream{};
    GaussianEncoder m_gaussianEncoder;
    BitEstimatorEncoder m_bitEstimatorZ;
    Stats m_stats{};
};

class MlvcDecoderCore : public MlvcCodecCoreCommon, public IMlvcDecoderCore {
public:
    struct Stats {
        OpTimerStats entropyCoding;
        OpTimerStats scaleDecoder;
        OpTimerStats inference;
        OpTimerStats postprocess;
    };

    MlvcDecoderCore(const MlvcVersion mlvcVersion, const int displayWidth, const int displayHeight,
                    const std::shared_ptr<const ModelMetadata>& modelMetadata, std::unique_ptr<IScaleDecoder> scaleDecoder,
                    const std::shared_ptr<const GaussianCoderPmf>& gaussianPmf,
                    const std::shared_ptr<const BitEstimatorPmf>& bitEstimatorPmf, const InferenceSessionPtr& session);
    ~MlvcDecoderCore();
    expected<void> Initialize() override;
    expected<DecodedFrame> Decode(const FrameData& frameData) override;
    DecoderInterfaceType GetDecoderInterfaceType() const override
    {
        return DecoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P;
    }

    const MlvcDecoderModel::Inputs& GetModelInputs() const { return m_inputs; }
    const MlvcDecoderModel::Outputs& GetModelOutputs() const { return m_outputs; }
    const Stats& GetStats() const { return m_stats; }

protected:
    MlvcDecoderModel m_model;
    ReferenceManager m_ref;
    GopTracker m_gop;
    MlvcDecoderModel::Inputs m_inputs;
    MlvcDecoderModel::Outputs m_outputs;
    rans::RansDecoderStream m_stream{};
    GaussianDecoder m_gaussianDecoder;
    BitEstimatorDecoder m_bitEstimatorZ;
    OutputTransformer<mlvc_f16_t> m_outputTransformer{};
    Stats m_stats{};
};

}  // namespace libmlvc
