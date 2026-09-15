// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/tensor.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace libmlvc {

enum struct TensorIoType { INPUT, OUTPUT };
enum struct TensorDataType { FP16, INT32 };

// Tensor wrapper for external tensors allocated by the inference engine.
// Explicit downloading and uploading to/from device memory is not currently implemented;
// it occurs automatically when tensor is created with hostAccessible = true.
// If required, explicit upload/download methods can be implemented
class IInferenceTensor {
public:
    virtual ~IInferenceTensor() = default;
    virtual expected<std::string_view> Name() const = 0;
    virtual expected<TensorDataType> DataType() const = 0;
    virtual expected<std::span<const int>> Shape() const = 0;
    virtual expected<std::span<const int>> Strides() const = 0;
    virtual expected<std::span<std::byte>> Data() = 0;

    // Create a typed view of the tensor data
    template <typename T, int N>
    expected<Tensor<T, N>> View();
};
using InferenceTensorPtr = std::shared_ptr<IInferenceTensor>;

class IInferenceSession {
public:
    virtual ~IInferenceSession() = default;
    // Allocates a tensor for input / output with the specified name.
    // By default, the tensor will have CPU-accessible backing / copy (hostAccessible = true).
    // If hostAccessible = false is specified, the tensor may use device-only backing (e.g., GPU memory).
    virtual expected<std::span<const std::string>> GetInputNames() = 0;
    virtual expected<std::span<const std::string>> GetOutputNames() = 0;
    virtual expected<InferenceTensorPtr> CreateTensor(const TensorIoType ioType, std::string_view name,
                                                      const bool hostAccessible = true) = 0;
    // Runs inference with the specified input and output tensors.
    // Tensors must be ordered to match GetInputNames() / GetOutputNames() (positional contract).
    // Should be thread-safe.
    virtual expected<void> Run(const std::vector<InferenceTensorPtr>& inputs,
                               const std::vector<InferenceTensorPtr>& outputs) = 0;
};
using InferenceSessionPtr = std::shared_ptr<IInferenceSession>;

struct InferenceEngineInfo {
    InferenceBackend inferenceBackend{};
    ComputeUnit computeUnit{};

    // Only used when inferenceBackend == WINDOWSML (ONNX)
    OnnxExecutionProvider onnxExecutionProvider{};
    std::string windowsAppRuntimeVersion{};
    std::string windowsAppRuntimeEpVersion{};  // EP package version parsed from library_path
    std::string onnxRuntimeVersion{};
    std::string onnxRuntimeEpVersion{};
    int64_t deviceLuid{};
    std::string driverVersion{};
    std::string driverDate{};
};

class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    virtual expected<InferenceSessionPtr>
    CreateSession(std::string_view name, std::span<const std::byte> modelData,
                  const std::map<std::string, std::span<const std::byte>>& weightsData, std::string_view cacheKey,
                  const std::optional<std::string>& functionName = std::nullopt) = 0;
    virtual const InferenceEngineInfo& GetInfo() const = 0;
};

// -------------------------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------------------------

inline size_t TensorDataTypeSize(const TensorDataType dtype)
{
    switch (dtype) {
    case TensorDataType::FP16:
        return 2;
    case TensorDataType::INT32:
        return 4;
    }
    MLVC_ASSERT(false);
    return 0;
}

template <typename T, int N>
expected<Tensor<T, N>> IInferenceTensor::View()
{
    auto name = Name();
    if (!name) return name.error();
    auto dtype = DataType();
    if (!dtype) return dtype.error();
    auto shape = Shape();
    if (!shape) return shape.error();
    auto strides = Strides();
    if (!strides) return strides.error();
    auto data = Data();
    if (!data) return data.error();

    if (TensorDataTypeSize(dtype.value()) != sizeof(T)) {
        MLVC_LOG_ERROR("Type mismatch: %zu != %zu", TensorDataTypeSize(dtype.value()), sizeof(T));
        return make_error_code(Error::invalid_argument);
    }

    for (int i = 0; i < static_cast<int>(shape.value().size() - N); i++) {
        if (shape.value()[i] != 1) {
            MLVC_LOG_ERROR("Shape mismatch: unexpected dimension %d (size %d)", i, shape.value()[i]);
            return make_error_code(Error::invalid_argument);
        }
    }

    std::array<int, N> tensorShape;
    std::array<int, N> tensorStrides;
    for (int i = 0; i < N; i++) {
        tensorShape[N - i - 1] = shape.value()[shape.value().size() - 1 - i];
        tensorStrides[N - i - 1] = strides.value()[strides.value().size() - 1 - i];
    }
    const size_t tensorSize = static_cast<size_t>(tensorShape[0]) * tensorStrides[0];
    const std::span<T> tensorData{ reinterpret_cast<T*>(data.value().data()), tensorSize };
    return Tensor<T, N>{ tensorShape, tensorStrides, tensorData, name.value() };
}

}  // namespace libmlvc
