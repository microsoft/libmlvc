// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/inference/interface.hpp"

#include <libmlvc/expected.hpp>

namespace libmlvc {

// For current model we only have one reference tensor, but this struct
// can be extended in the future if needed.

struct ReferenceData {
    ReferenceData(const InferenceSessionPtr& session);
    expected<void> Initialize();
    const InferenceTensorPtr& GetRefFeature() const { return m_refFeature; }

private:
    InferenceSessionPtr m_session{};
    InferenceTensorPtr m_refFeature;
    expected<void> Allocate(std::string_view name, InferenceTensorPtr& output);
};

}  // namespace libmlvc
