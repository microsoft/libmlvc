// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/scale_decoder.hpp"
#include "libmlvc/common/tensor.hpp"
#include "libmlvc/schema/bundle.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

using namespace libmlvc;

TEST(UnitTestScaleDecoder, UpsampleAbsAndCheckerboard)
{
    const auto decoder = CreateScaleDecoder(ScaleDecoderType::UPSAMPLE, ScaleDecoderData{}, /*modelWidth=*/4,
                                            /*modelHeight=*/4, /*latentChannels=*/4, /*downsampleLatent=*/2,
                                            /*downsampleHyperprior=*/2, /*yScaleRepeat=*/1, /*scaleMaxIdx=*/127);
    ASSERT_TRUE(static_cast<bool>(decoder));

    // modelWidth/Height=4, downsampleLatent=2 => yWidth=yHeight=2; latentChannels=4 => yChannels=2;
    // downsampleHyperprior=2, yScaleRepeat=1 => scaleChannels=2. z_raw is [4,1,1] so a single hyperprior
    // position feeds the whole 2x2 output per channel, split by the (y+x) checkerboard.
    Tensor<int32_t, 3> zRaw({ 4, 1, 1 });
    const auto d = zRaw.Data();
    d[0] = 10;
    d[1] = 20;
    d[2] = -30;
    d[3] = 40;
    ASSERT_TRUE(static_cast<bool>(decoder.value()->ExtractScales(zRaw)));
    const ScaleValues& scales = decoder.value()->GetScales();

    // value0 = |z[chDs]|, value1 = |z[chDs + scaleChannels]|; scales0 takes value0 on even (y+x).
    EXPECT_EQ(scales.scales0(0, 0, 0), 10);
    EXPECT_EQ(scales.scales0(0, 0, 1), 30);
    EXPECT_EQ(scales.scales0(0, 1, 0), 30);
    EXPECT_EQ(scales.scales0(0, 1, 1), 10);
    EXPECT_EQ(scales.scales0(1, 0, 0), 20);
    EXPECT_EQ(scales.scales0(1, 0, 1), 40);

    // scales1 swaps value0/value1 relative to scales0 at each position.
    EXPECT_EQ(scales.scales1(0, 0, 0), 30);
    EXPECT_EQ(scales.scales1(0, 0, 1), 10);
    EXPECT_EQ(scales.scales1(1, 0, 1), 20);
    EXPECT_EQ(scales.scales1(1, 1, 1), 40);
}

TEST(UnitTestScaleDecoder, FactoryUpsampleRequiresYScaleRepeat)
{
    const auto decoder = CreateScaleDecoder(ScaleDecoderType::UPSAMPLE, ScaleDecoderData{}, /*modelWidth=*/4,
                                            /*modelHeight=*/4, /*latentChannels=*/4, /*downsampleLatent=*/2,
                                            /*downsampleHyperprior=*/2, /*yScaleRepeat=*/std::nullopt,
                                            /*scaleMaxIdx=*/127);
    EXPECT_FALSE(static_cast<bool>(decoder));
}
