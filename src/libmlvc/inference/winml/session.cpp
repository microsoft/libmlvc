// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/inference/winml/session.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/inference/winml/ort_helpers.hpp"

#include <cstring>
#include <string>
#include <vector>

using namespace libmlvc;

namespace libmlvc_winml {

static ONNXTensorElementDataType TensorDataTypeToOnnxType(TensorDataType dt)
{
    switch (dt) {
    case TensorDataType::FP16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TensorDataType::INT32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    default:
        MLVC_LOG_ABORT("Unsupported TensorDataType: %d", static_cast<int>(dt));
    }
}

static TensorDataType OnnxTypeToTensorDataType(ONNXTensorElementDataType onnxType)
{
    switch (onnxType) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return TensorDataType::FP16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return TensorDataType::INT32;
    default:
        MLVC_LOG_ABORT("Unsupported ONNX tensor type: %d", static_cast<int>(onnxType));
    }
}

static expected<void> ExtractTensorMetadata(const OrtApi* ortApi, OrtSession* session, bool isInput,
                                            std::vector<std::string>& names, std::vector<std::vector<int64_t>>& shapes,
                                            std::vector<TensorDataType>& types)
{
    OrtAllocator* allocator = nullptr;
    if (auto ret = CheckOrtStatus(ortApi, ortApi->GetAllocatorWithDefaultOptions(&allocator)); !ret) {
        MLVC_LOG_ERROR("Failed to get allocator: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    size_t count = 0;
    if (auto ret = CheckOrtStatus(ortApi, isInput ? ortApi->SessionGetInputCount(session, &count)
                                                  : ortApi->SessionGetOutputCount(session, &count));
        !ret) {
        MLVC_LOG_ERROR("Failed to get %s count: %s", isInput ? "input" : "output", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }

    names.reserve(count);
    shapes.reserve(count);
    types.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        // Get name
        char* name = nullptr;
        if (auto ret = CheckOrtStatus(ortApi, isInput ? ortApi->SessionGetInputName(session, i, allocator, &name)
                                                      : ortApi->SessionGetOutputName(session, i, allocator, &name));
            !ret) {
            MLVC_LOG_ERROR("Failed to get %s name: %s", isInput ? "input" : "output", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
        names.push_back(name);
        allocator->Free(allocator, name);

        // Get type info
        OrtTypeInfo* typeInfo = nullptr;
        if (auto ret = CheckOrtStatus(ortApi, isInput ? ortApi->SessionGetInputTypeInfo(session, i, &typeInfo)
                                                      : ortApi->SessionGetOutputTypeInfo(session, i, &typeInfo));
            !ret) {
            MLVC_LOG_ERROR("Failed to get %s type info: %s", isInput ? "input" : "output", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
        auto typeInfoPtr = MakeOrtTypeInfo(ortApi, typeInfo);

        // Get tensor info
        const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
        if (auto ret = CheckOrtStatus(ortApi, ortApi->CastTypeInfoToTensorInfo(typeInfoPtr.get(), &tensorInfo)); !ret) {
            MLVC_LOG_ERROR("Failed to cast type info to tensor info: %s", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }

        // Get element type
        ONNXTensorElementDataType onnxType;
        if (auto ret = CheckOrtStatus(ortApi, ortApi->GetTensorElementType(tensorInfo, &onnxType)); !ret) {
            MLVC_LOG_ERROR("Failed to get tensor element type: %s", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
        types.push_back(OnnxTypeToTensorDataType(onnxType));

        // Get shape
        size_t dimCount = 0;
        if (auto ret = CheckOrtStatus(ortApi, ortApi->GetDimensionsCount(tensorInfo, &dimCount)); !ret) {
            MLVC_LOG_ERROR("Failed to get dimensions count: %s", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }

        std::vector<int64_t> shape(dimCount);
        if (auto ret = CheckOrtStatus(ortApi, ortApi->GetDimensions(tensorInfo, shape.data(), dimCount)); !ret) {
            MLVC_LOG_ERROR("Failed to get dimensions: %s", ret.error().c_str());
            return make_error_code(Error::model_init_error);
        }
        shapes.push_back(shape);
    }

    return {};
}

// --------------------------------------------------------------------------------------------
// WinMlInferenceTensor Implementation
// --------------------------------------------------------------------------------------------

expected<std::shared_ptr<WinMlInferenceTensor>>
WinMlInferenceTensor::Create(const OrtApi* ortApi, OrtAllocator* allocator, const std::string& name,
                             TensorDataType dataType, const std::vector<int64_t>& shape)
{
    // Calculate total data size
    size_t totalElements = 1;
    for (auto dim : shape) {
        totalElements *= static_cast<size_t>(dim);
    }
    const size_t elementSize = TensorDataTypeSize(dataType);
    const size_t dataSize = totalElements * elementSize;

    // Create ORT tensor using the allocator (ORT allocates the memory)
    const auto onnxType = TensorDataTypeToOnnxType(dataType);
    OrtValue* value = nullptr;
    if (auto ret = CheckOrtStatus(
            ortApi, ortApi->CreateTensorAsOrtValue(allocator, shape.data(), shape.size(), onnxType, &value));
        !ret) {
        MLVC_LOG_ERROR("Failed to create ORT tensor: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }
    auto ortValue = MakeOrtValue(ortApi, value);

    // Resolve and cache the data pointer once (ORT guarantees it remains stable)
    void* ptr = nullptr;
    if (auto ret = CheckOrtStatus(ortApi, ortApi->GetTensorMutableData(ortValue.get(), &ptr)); !ret) {
        MLVC_LOG_ERROR("Failed to get tensor data pointer: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }
    auto data = std::span<std::byte>(static_cast<std::byte*>(ptr), dataSize);

    // Zero-initialize (ORT does not zero-initialize allocated memory)
    std::memset(data.data(), 0, data.size());

    return std::shared_ptr<WinMlInferenceTensor>(new WinMlInferenceTensor(name, dataType, shape, data, std::move(ortValue)));
}

WinMlInferenceTensor::WinMlInferenceTensor(const std::string& name, TensorDataType dataType,
                                           const std::vector<int64_t>& shape, std::span<std::byte> data,
                                           OrtValuePtr ortValue)
    : m_name(name), m_dataType(dataType), m_shape(shape), m_data(data), m_ortValue(std::move(ortValue))
{
    // Calculate strides (row-major order)
    m_strides.resize(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        m_strides[i] = stride;
        stride *= shape[i];
    }

    // Pre-compute int versions for Shape()/Strides() to avoid mutation on each call
    m_shapeInt.reserve(m_shape.size());
    for (auto dim : m_shape) {
        m_shapeInt.push_back(static_cast<int>(dim));
    }
    m_stridesInt.reserve(m_strides.size());
    for (auto s : m_strides) {
        m_stridesInt.push_back(static_cast<int>(s));
    }
}

WinMlInferenceTensor::~WinMlInferenceTensor() {}

expected<std::string_view> WinMlInferenceTensor::Name() const
{
    return std::string_view(m_name);
}

expected<TensorDataType> WinMlInferenceTensor::DataType() const
{
    return m_dataType;
}

expected<std::span<const int>> WinMlInferenceTensor::Shape() const
{
    return std::span<const int>(m_shapeInt);
}

expected<std::span<const int>> WinMlInferenceTensor::Strides() const
{
    return std::span<const int>(m_stridesInt);
}

expected<std::span<std::byte>> WinMlInferenceTensor::Data()
{
    return m_data;
}

// --------------------------------------------------------------------------------------------
// WinMlInferenceSession Implementation
// --------------------------------------------------------------------------------------------

expected<std::shared_ptr<WinMlInferenceSession>> WinMlInferenceSession::Create(std::string_view name,
                                                                               OrtSessionPtr session, const OrtApi* ortApi,
                                                                               const OrtMemoryInfo* epMemoryInfo)
{
    auto res = std::shared_ptr<WinMlInferenceSession>(new WinMlInferenceSession(name, std::move(session), ortApi));
    if (auto ret = res->Initialize(epMemoryInfo); !ret) {
        return ret.error();
    }
    return res;
}

expected<std::span<const std::string>> WinMlInferenceSession::GetInputNames()
{
    return std::span<const std::string>(m_inputNames);
}

expected<std::span<const std::string>> WinMlInferenceSession::GetOutputNames()
{
    return std::span<const std::string>(m_outputNames);
}

expected<InferenceTensorPtr> WinMlInferenceSession::CreateTensor(const TensorIoType ioType, std::string_view name,
                                                                 const bool hostAccessible)
{
    (void)hostAccessible;  // All tensors are CPU accessible

    std::string tensorName(name);
    TensorDataType dataType = TensorDataType::FP16;
    std::vector<int64_t> shape;

    // Find tensor in input or output metadata
    const auto& names = (ioType == TensorIoType::INPUT) ? m_inputNames : m_outputNames;
    const auto& shapes = (ioType == TensorIoType::INPUT) ? m_inputShapes : m_outputShapes;
    const auto& types = (ioType == TensorIoType::INPUT) ? m_inputTypes : m_outputTypes;

    size_t tensorIndex = SIZE_MAX;
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == tensorName) {
            tensorIndex = i;
            break;
        }
    }

    if (tensorIndex == SIZE_MAX) {
        MLVC_LOG_ERROR("[%s] Failed to find tensor: %s", m_name.c_str(), tensorName.c_str());
        return make_error_code(Error::model_init_error);
    }

    // Get shape and type from metadata
    shape = shapes[tensorIndex];
    dataType = types[tensorIndex];

    // Handle dynamic dimensions
    for (auto& dim : shape) {
        if (dim < 0) dim = 1;
    }

    return WinMlInferenceTensor::Create(m_ortApi, m_allocator, tensorName, dataType, shape);
}

expected<void> WinMlInferenceSession::Run(const std::vector<InferenceTensorPtr>& inputs,
                                          const std::vector<InferenceTensorPtr>& outputs)
{
    // mutex since we may be using a shared OrtSession for multiple decoders.
    // TODO: measure if shared session is better than per-decoder session
    std::lock_guard lock(m_mutex);

    // Tensors are expected in the same order as GetInputNames() / GetOutputNames()
    if (inputs.size() != m_inputNames.size()) {
        MLVC_LOG_ERROR("[%s] Input tensor count mismatch: %zu != %zu", m_name.c_str(), inputs.size(), m_inputNames.size());
        return make_error_code(Error::model_inference_error);
    }
    if (outputs.size() != m_outputNames.size()) {
        MLVC_LOG_ERROR("[%s] Output tensor count mismatch: %zu != %zu", m_name.c_str(), outputs.size(),
                       m_outputNames.size());
        return make_error_code(Error::model_inference_error);
    }

    // Extract OrtValue handles directly from tensors (zero-copy)
    for (size_t i = 0; i < inputs.size(); ++i) {
        m_inputValuesRaw[i] = static_cast<WinMlInferenceTensor*>(inputs[i].get())->GetOrtValue();
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        m_outputValuesRaw[i] = static_cast<WinMlInferenceTensor*>(outputs[i].get())->GetOrtValue();
    }

    // Run inference with pre-bound outputs (zero-copy)
    if (auto ret = CheckOrtStatus(m_ortApi,
                                  m_ortApi->Run(m_session.get(),
                                                nullptr,  // run options
                                                m_inputNamesC.data(), m_inputValuesRaw.data(), m_inputValuesRaw.size(),
                                                m_outputNamesC.data(), m_outputNamesC.size(), m_outputValuesRaw.data()));
        !ret) {
        MLVC_LOG_ERROR("[%s] ONNX Runtime inference failed: %s", m_name.c_str(), ret.error().c_str());
        return make_error_code(Error::model_inference_error);
    }

    return {};
}

WinMlInferenceSession::WinMlInferenceSession(std::string_view name, OrtSessionPtr session, const OrtApi* ortApi)
    : m_name(name), m_ortApi(ortApi), m_session(std::move(session))
{
}

expected<void> WinMlInferenceSession::Initialize(const OrtMemoryInfo* epMemoryInfo)
{
    if (auto ret = InitializeAllocator(epMemoryInfo); !ret) {
        return ret;
    }

    // Extract input metadata
    if (auto ret = ExtractTensorMetadata(m_ortApi, m_session.get(), true, m_inputNames, m_inputShapes, m_inputTypes); !ret) {
        return ret;
    }

    // Extract output metadata
    if (auto ret = ExtractTensorMetadata(m_ortApi, m_session.get(), false, m_outputNames, m_outputShapes, m_outputTypes);
        !ret) {
        return ret;
    }

    // Cache name pointers and pre-size value vectors (stable after init)
    m_inputNamesC.reserve(m_inputNames.size());
    for (const auto& name : m_inputNames) {
        m_inputNamesC.push_back(name.c_str());
    }
    m_outputNamesC.reserve(m_outputNames.size());
    for (const auto& name : m_outputNames) {
        m_outputNamesC.push_back(name.c_str());
    }
    m_inputValuesRaw.resize(m_inputNames.size());
    m_outputValuesRaw.resize(m_outputNames.size());

    return {};
}

expected<void> WinMlInferenceSession::InitializeAllocator(const OrtMemoryInfo* epMemoryInfo)
{
    // Try using an execution-provider-specific allocator to enable zero-copy
    if (epMemoryInfo) {
        OrtAllocator* allocator = nullptr;
        if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->CreateAllocator(m_session.get(), epMemoryInfo, &allocator)); ret) {
            m_epAllocator = MakeOrtAllocator(m_ortApi, allocator);
            m_allocator = m_epAllocator.get();
            return {};
        } else {
            MLVC_LOG_WARN("CreateAllocator failed: %s", ret.error().c_str());
        }
    }

    // Fallback to the default allocator
    if (auto ret = CheckOrtStatus(m_ortApi, m_ortApi->GetAllocatorWithDefaultOptions(&m_allocator)); !ret) {
        MLVC_LOG_ERROR("Failed to get default allocator: %s", ret.error().c_str());
        return make_error_code(Error::model_init_error);
    }
    MLVC_LOG_INFO("[%s] Using default CPU allocator", m_name.c_str());
    return {};
}

}  // namespace libmlvc_winml
