// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/reference_data.hpp"

namespace libmlvc {

ReferenceData::ReferenceData(const InferenceSessionPtr& session) : m_session(session) {}
expected<void> ReferenceData::Initialize()
{
    if (auto ret = Allocate("ref_feature", m_refFeature); !ret) return ret.error();
    return {};
}

expected<void> ReferenceData::Allocate(std::string_view name, InferenceTensorPtr& output)
{
    auto inferenceTensor = m_session->CreateTensor(TensorIoType::INPUT, name, true);
    if (!inferenceTensor) {
        MLVC_LOG_ERROR("Failed to allocate reference tensor (%s): %s", std::string(name).c_str(),
                       inferenceTensor.error().message().c_str());
        return inferenceTensor.error();
    }

    // Initialize to zero
    auto data = inferenceTensor.value()->Data();
    if (!data) {
        MLVC_LOG_ERROR("Failed to get data for reference tensor (%s): %s", std::string(name).c_str(),
                       data.error().message().c_str());
        return data.error();
    }
    std::fill(data.value().begin(), data.value().end(), std::byte{ 0 });

    output = std::move(inferenceTensor.value());
    return {};
}

}  // namespace libmlvc
