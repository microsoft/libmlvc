// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/scale_decoder.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/schema/bundle.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

namespace libmlvc {

namespace {

// ---------------------------------------------------------------------------
// Scale decoder implementations
// ---------------------------------------------------------------------------

class ScaleDecoderBase : public IScaleDecoder {
public:
    const ScaleValues& GetScales() const override { return m_scales; }

protected:
    ScaleDecoderBase(const int modelWidth, const int modelHeight, const int latentChannels, const int downsampleLatent)
        : m_yChannels{ latentChannels / 2 }
        , m_yWidth{ (modelWidth + downsampleLatent - 1) / downsampleLatent }
        , m_yHeight{ (modelHeight + downsampleLatent - 1) / downsampleLatent }
    {
    }

    const int m_yChannels;
    const int m_yWidth;
    const int m_yHeight;
    ScaleValues m_scales;
};

class UpsampleScaleDecoder : public ScaleDecoderBase {
public:
    UpsampleScaleDecoder(const int modelWidth, const int modelHeight, const int latentChannels,
                         const int downsampleLatent, const int downsampleHyperprior, const int yScaleRepeat)
        : ScaleDecoderBase{ modelWidth, modelHeight, latentChannels, downsampleLatent }
        , m_downsampleHyperprior{ downsampleHyperprior }
        , m_yScaleRepeat{ yScaleRepeat }
    {
    }

    expected<void> ExtractScales(const Tensor<int32_t, 3>& zRaw) override
    {
        if (!zRaw.IsContiguous()) {
            MLVC_LOG_ERROR("z_raw tensor must be contiguous");
            return make_error_code(Error::invalid_argument);
        }

        if (m_downsampleHyperprior < 2 || m_downsampleHyperprior % 2 != 0) {
            MLVC_LOG_ERROR("downsample_hyperprior must be even and >= 2: %d", m_downsampleHyperprior);
            return make_error_code(Error::invalid_argument);
        }

        const auto [zChannels, zHeight, zWidth] = zRaw.Shape();
        if (((m_yChannels - 1) / m_yScaleRepeat >= zChannels) || ((m_yHeight - 1) / m_downsampleHyperprior > zHeight)
            || ((m_yWidth - 1) / m_downsampleHyperprior > zWidth)) {
            MLVC_LOG_ERROR(
                "Invalid shape, z_raw: [%d, %d, %d], latent: [%d, %d, %d], y_scale_repeat: %d, downsample_hyperprior: "
                "%d",
                zChannels, zHeight, zWidth, m_yChannels, m_yHeight, m_yWidth, m_yScaleRepeat, m_downsampleHyperprior);
            return make_error_code(Error::invalid_argument);
        }

        m_scales.Create({ m_yChannels, m_yHeight, m_yWidth });
        MLVC_ASSERT(m_scales.scales0.IsContiguous());
        MLVC_ASSERT(m_scales.scales1.IsContiguous());

        const auto zRawData = zRaw.Data();
        const auto zRawStrides = zRaw.Strides();
        const auto scales0Data = m_scales.scales0.Data();
        const auto scales0Strides = m_scales.scales0.Strides();
        const auto scales1Data = m_scales.scales1.Data();
        const auto scales1Strides = m_scales.scales1.Strides();
        const int scaleChannels = m_yChannels / m_yScaleRepeat;
        for (int chDs = 0; chDs < scaleChannels; chDs++) {
            for (int yDs = 0; yDs < zHeight; yDs++) {
                for (int xDs = 0; xDs < zWidth; xDs++) {
                    const int index0 = chDs * zRawStrides[0] + yDs * zRawStrides[1] + xDs;
                    const int index1 = index0 + scaleChannels * zRawStrides[0];
                    const int32_t value0 = std::abs(zRawData[index0]);
                    const int32_t value1 = std::abs(zRawData[index1]);
                    for (int ch = chDs * m_yScaleRepeat; ch < (chDs + 1) * m_yScaleRepeat; ch++) {
                        for (int y = yDs * m_downsampleHyperprior;
                             y < std::min((yDs + 1) * m_downsampleHyperprior, m_yHeight); y++) {
                            for (int x = xDs * m_downsampleHyperprior;
                                 x < std::min((xDs + 1) * m_downsampleHyperprior, m_yWidth); x++) {
                                const bool cond = (y + x) % 2 == 0;
                                scales0Data[ch * scales0Strides[0] + y * scales0Strides[1] + x] = cond ? value0 : value1;
                                scales1Data[ch * scales1Strides[0] + y * scales1Strides[1] + x] = cond ? value1 : value0;
                            }
                        }
                    }
                }
            }
        }
        return {};
    }

private:
    const int m_downsampleHyperprior;
    const int m_yScaleRepeat;
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

expected<std::unique_ptr<IScaleDecoder>> CreateScaleDecoder(const ScaleDecoderType scaleDecoderType,
                                                            const ScaleDecoderData& data, const int modelWidth,
                                                            const int modelHeight, const int latentChannels,
                                                            const int downsampleLatent, const int downsampleHyperprior,
                                                            const std::optional<int> yScaleRepeat, const int scaleMaxIdx)
{
    (void)data;
    (void)scaleMaxIdx;

    if (scaleDecoderType == ScaleDecoderType::UPSAMPLE) {
        if (!yScaleRepeat.has_value()) {
            MLVC_LOG_ERROR("UPSAMPLE scale decoder requires y_scale_repeat");
            return make_error_code(Error::invalid_argument);
        }

        return std::make_unique<UpsampleScaleDecoder>(modelWidth, modelHeight, latentChannels, downsampleLatent,
                                                      downsampleHyperprior, yScaleRepeat.value());
    }

    MLVC_LOG_ERROR("Unknown scale decoder type: %d", static_cast<int>(scaleDecoderType));
    return make_error_code(Error::invalid_argument);
}

}  // namespace libmlvc
