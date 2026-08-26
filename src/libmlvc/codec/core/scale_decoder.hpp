// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/tensor.hpp"

#include <libmlvc/expected.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace libmlvc {

enum struct ScaleDecoderType;

struct ScaleDecoderData {
    // Reserved for optional scale-decoder resources.
};

struct ScaleValues {
    Tensor<int32_t, 3> scales0;
    Tensor<int32_t, 3> scales1;

    ScaleValues() : scales0{ "scales_0" }, scales1{ "scales_1" } {}
    void Create(const std::array<int, 3>& shape)
    {
        scales0.Create(shape);
        scales1.Create(shape);
    }
};

class IScaleDecoder {
public:
    virtual ~IScaleDecoder() = default;
    virtual expected<void> ExtractScales(const Tensor<int32_t, 3>& zRaw) = 0;
    virtual const ScaleValues& GetScales() const = 0;
};

expected<std::unique_ptr<IScaleDecoder>> CreateScaleDecoder(const ScaleDecoderType scaleDecoderType,
                                                            const ScaleDecoderData& data, const int modelWidth,
                                                            const int modelHeight, const int latentChannels,
                                                            const int downsampleLatent, const int downsampleHyperprior,
                                                            const std::optional<int> yScaleRepeat, const int scaleMaxIdx);

}  // namespace libmlvc
