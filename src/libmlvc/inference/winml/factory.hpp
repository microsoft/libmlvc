// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/inference/interface.hpp"

#include <memory>

namespace libmlvc {

struct WinMlEngineParams {
    WinMlInitMode initMode{ WinMlInitMode::AppSdk };
    ComputeUnit computeUnit{ ComputeUnit::CPU };
    OnnxExecutionProvider onnxExecutionProvider{ OnnxExecutionProvider::CPU };
};

expected<std::unique_ptr<IInferenceEngine>> MakeWinMlEngine(const WinMlEngineParams& params, CancelToken cancelToken = {},
                                                            const InitializeProgressCallback& progressCallback = nullptr);

}  // namespace libmlvc
