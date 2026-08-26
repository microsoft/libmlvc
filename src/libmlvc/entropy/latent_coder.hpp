// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/tensor.hpp"
#include "libmlvc/entropy/rans_coder.hpp"
#include "libmlvc/schema/pmf.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace libmlvc {

class GaussianEncoder {
public:
    GaussianEncoder(const std::shared_ptr<const GaussianCoderPmf>& pmf) : m_pmf(pmf) {}
    expected<void> Initialize();
    expected<void> Encode(rans::RansEncoderStream& stream, const Tensor<int32_t, 3>& symbols,
                          const Tensor<int32_t, 3>& scales);

private:
    const std::shared_ptr<const GaussianCoderPmf> m_pmf;
    rans::EntropyEncoder m_encoder;
};

class GaussianDecoder {
public:
    GaussianDecoder(const std::shared_ptr<const GaussianCoderPmf>& pmf) : m_pmf(pmf) {}
    expected<void> Initialize();
    expected<void> Decode(rans::RansDecoderStream& stream, const Tensor<int32_t, 3>& scales, Tensor<int32_t, 3>& res);

private:
    const std::shared_ptr<const GaussianCoderPmf> m_pmf;
    rans::EntropyDecoder m_decoder{};
};

class BitEstimatorBase {
public:
    BitEstimatorBase(const std::shared_ptr<const BitEstimatorPmf>& pmf);
    expected<void> Initialize();

protected:
    const std::shared_ptr<const BitEstimatorPmf> m_pmf;
    std::vector<int32_t> m_indices;

    void UpdateIndices(const std::array<int, 3>& shape, const int qp);
};

class BitEstimatorEncoder : public BitEstimatorBase {
public:
    using BitEstimatorBase::BitEstimatorBase;
    expected<void> Initialize();
    expected<void> Encode(rans::RansEncoderStream& stream, const Tensor<int32_t, 3>& symbols, const int qp);

private:
    rans::EntropyEncoder m_encoder;
};

class BitEstimatorDecoder : public BitEstimatorBase {
public:
    using BitEstimatorBase::BitEstimatorBase;
    expected<void> Initialize();
    expected<void> Decode(rans::RansDecoderStream& stream, const int qp, Tensor<int32_t, 3>& res);

private:
    rans::EntropyDecoder m_decoder;
};

}  // namespace libmlvc
