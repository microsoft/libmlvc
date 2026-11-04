// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/inference/interface.hpp"

#include <memory>

namespace libmlvc {

class ModelCache;

struct CoreMlEngineParams {
    ComputeUnit computeUnit{ ComputeUnit::NPU };
    std::shared_ptr<ModelCache> modelCache{};
};

expected<std::unique_ptr<IInferenceEngine>> MakeCoreMlEngine(const CoreMlEngineParams& params);

}  // namespace libmlvc
