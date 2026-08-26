// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/inference/interface.hpp"

#include <memory>

namespace libmlvc {

struct CoreMlEngineParams {
    ComputeUnit computeUnit{ ComputeUnit::NPU };
};

expected<std::unique_ptr<IInferenceEngine>> MakeCoreMlEngine(const CoreMlEngineParams& params);

}  // namespace libmlvc
