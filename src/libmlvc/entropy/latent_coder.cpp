// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/entropy/latent_coder.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"

namespace libmlvc {

static constexpr int RANS_SYMBOL_BITS = 16;
static constexpr int RANS_BYPASS_BITS = 2;

expected<void> GaussianEncoder::Initialize()
{
    if (!m_pmf->indexSpace) {
        MLVC_LOG_ERROR("Gaussian PMF is not index-space; only index-space scales are supported");
        return make_error_code(Error::invalid_argument);
    }
    auto errorCode = m_encoder.Initialize(rans::RansVariant::RansByte, m_pmf->pmfLengths, m_pmf->pmfOffsets,
                                          m_pmf->pmfTable, RANS_SYMBOL_BITS, RANS_BYPASS_BITS);
    if (errorCode) {
        MLVC_LOG_ERROR("Failed to initialize entropy encoder: %s", errorCode.message().c_str());
        return make_error_code(Error::entropy_coder_init_error);
    }
    return {};
}

expected<void> GaussianEncoder::Encode(rans::RansEncoderStream& stream, const Tensor<int32_t, 3>& symbols,
                                       const Tensor<int32_t, 3>& scales)
{
    if (!symbols.IsContiguous()) {
        MLVC_LOG_ERROR("Symbols tensor must be contiguous");
        return make_error_code(Error::entropy_coder_encode_error);
    }
    if (!scales.IsContiguous()) {
        MLVC_LOG_ERROR("Scales tensor must be contiguous");
        return make_error_code(Error::entropy_coder_encode_error);
    }

    auto errorCode = m_encoder.Encode(stream, scales.Data(), symbols.Data());
    if (errorCode) {
        MLVC_LOG_WARN("Failed to encode data: %s", errorCode.message().c_str());
        return make_error_code(Error::entropy_coder_encode_error);
    }
    return {};
}

expected<void> GaussianDecoder::Initialize()
{
    if (!m_pmf->indexSpace) {
        MLVC_LOG_ERROR("Gaussian PMF is not index-space; only index-space scales are supported");
        return make_error_code(Error::invalid_argument);
    }
    auto errorCode = m_decoder.Initialize(rans::RansVariant::RansByte, m_pmf->pmfLengths, m_pmf->pmfOffsets,
                                          m_pmf->pmfTable, RANS_SYMBOL_BITS, RANS_BYPASS_BITS);
    if (errorCode) {
        MLVC_LOG_ERROR("Failed to initialize entropy decoder: %s", errorCode.message().c_str());
        return make_error_code(Error::entropy_coder_init_error);
    }
    return {};
}

expected<void> GaussianDecoder::Decode(rans::RansDecoderStream& stream, const Tensor<int32_t, 3>& scales,
                                       Tensor<int32_t, 3>& res)
{
    if (!scales.IsContiguous()) {
        MLVC_LOG_ERROR("Scales tensor must be contiguous");
        return make_error_code(Error::entropy_coder_decode_error);
    }

    // Make sure the res tensor is contiguous
    res.Create(scales.Shape(), StridesFromShape<3>(scales.Shape()));

    auto errorCode = m_decoder.Decode(res.Data(), scales.Data(), stream);
    if (errorCode) {
        MLVC_LOG_WARN("Failed to decode data: %s", errorCode.message().c_str());
        return make_error_code(Error::entropy_coder_decode_error);
    }
    return {};
}

BitEstimatorBase::BitEstimatorBase(const std::shared_ptr<const BitEstimatorPmf>& pmf) : m_pmf(pmf) {}

expected<void> BitEstimatorBase::Initialize()
{
    return {};
}

void BitEstimatorBase::UpdateIndices(const std::array<int, 3>& shape, const int qp)
{
    const int h = shape[1];
    const int w = shape[2];
    m_indices.resize(shape[0] * h * w);
    for (int ch = 0; ch < shape[0]; ch++) {
        for (int i = 0; i < h * w; i++) {
            m_indices[ch * h * w + i] = ch + qp * m_pmf->channels;
        }
    }
}

expected<void> BitEstimatorEncoder::Initialize()
{
    if (auto ret = BitEstimatorBase::Initialize(); !ret) {
        return ret.error();
    }

    auto errorCode = m_encoder.Initialize(rans::RansVariant::RansByte, m_pmf->pmfLengths, m_pmf->pmfOffsets,
                                          m_pmf->pmfTable, RANS_SYMBOL_BITS, RANS_BYPASS_BITS);
    if (errorCode) {
        MLVC_LOG_ERROR("Failed to initialize entropy encoder: %s", errorCode.message().c_str());
        return make_error_code(Error::entropy_coder_init_error);
    }
    return {};
}

expected<void> BitEstimatorEncoder::Encode(rans::RansEncoderStream& stream, const Tensor<int32_t, 3>& symbols, const int qp)
{
    if (!symbols.IsContiguous()) {
        MLVC_LOG_ERROR("Symbols tensor must be contiguous");
        return make_error_code(Error::entropy_coder_encode_error);
    }

    UpdateIndices(symbols.Shape(), qp);

    auto errorCode = m_encoder.Encode(stream, m_indices, symbols.Data());
    if (errorCode) {
        MLVC_LOG_WARN("Failed to encode data: %s", errorCode.message().c_str());
        return make_error_code(Error::entropy_coder_encode_error);
    }
    return {};
}

expected<void> BitEstimatorDecoder::Initialize()
{
    if (auto ret = BitEstimatorBase::Initialize(); !ret) {
        return ret.error();
    }

    auto errorCode = m_decoder.Initialize(rans::RansVariant::RansByte, m_pmf->pmfLengths, m_pmf->pmfOffsets,
                                          m_pmf->pmfTable, RANS_SYMBOL_BITS, RANS_BYPASS_BITS);
    if (errorCode) {
        MLVC_LOG_ERROR("Failed to initialize entropy decoder: %s", errorCode.message().c_str());
        return make_error_code(Error::entropy_coder_init_error);
    }
    return {};
}

expected<void> BitEstimatorDecoder::Decode(rans::RansDecoderStream& stream, const int qp, Tensor<int32_t, 3>& res)
{
    if (!res.IsContiguous()) {
        MLVC_LOG_ERROR("Res tensor must be contiguous");
        return make_error_code(Error::entropy_coder_decode_error);
    }

    const auto& shape = res.Shape();
    UpdateIndices(shape, qp);
    MLVC_ASSERT(res.Size() == m_indices.size());

    auto errorCode = m_decoder.Decode(res.Data(), m_indices, stream);
    if (errorCode) {
        MLVC_LOG_WARN("Failed to decode data: %s", errorCode.message().c_str());
        return make_error_code(Error::entropy_coder_decode_error);
    }
    return {};
}

}  // namespace libmlvc
