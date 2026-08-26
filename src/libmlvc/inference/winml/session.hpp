// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/inference/interface.hpp"
#include "libmlvc/inference/winml/ort_helpers.hpp"

#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

namespace libmlvc_winml {

using namespace libmlvc;

class WinMlInferenceTensor : public IInferenceTensor {
public:
    static expected<std::shared_ptr<WinMlInferenceTensor>> Create(const OrtApi* ortApi, OrtAllocator* allocator,
                                                                  const std::string& name, TensorDataType dataType,
                                                                  const std::vector<int64_t>& shape);
    ~WinMlInferenceTensor();
    expected<std::string_view> Name() const override;
    expected<TensorDataType> DataType() const override;
    expected<std::span<const int>> Shape() const override;
    expected<std::span<const int>> Strides() const override;
    expected<std::span<std::byte>> Data() override;

    OrtValue* GetOrtValue() const { return m_ortValue.get(); }

private:
    WinMlInferenceTensor(const std::string& name, TensorDataType dataType, const std::vector<int64_t>& shape,
                         std::span<std::byte> data, OrtValuePtr ortValue);

    std::string m_name;
    TensorDataType m_dataType;
    std::vector<int64_t> m_shape;
    std::vector<int64_t> m_strides;
    std::span<std::byte> m_data;
    OrtValuePtr m_ortValue;
    std::vector<int> m_shapeInt;
    std::vector<int> m_stridesInt;
};

class WinMlInferenceSession : public IInferenceSession {
public:
    static expected<std::shared_ptr<WinMlInferenceSession>> Create(std::string_view name, OrtSessionPtr session,
                                                                   const OrtApi* ortApi, const OrtMemoryInfo* epMemoryInfo);

    expected<std::span<const std::string>> GetInputNames() override;
    expected<std::span<const std::string>> GetOutputNames() override;
    expected<InferenceTensorPtr> CreateTensor(const TensorIoType ioType, std::string_view name,
                                              const bool hostAccessible) override;
    expected<void> Run(const std::vector<InferenceTensorPtr>& inputs, const std::vector<InferenceTensorPtr>& outputs) override;

private:
    std::string m_name;
    const OrtApi* m_ortApi;
    std::mutex m_mutex;
    OrtSessionPtr m_session;
    OrtAllocatorPtr m_epAllocator;
    OrtAllocator* m_allocator = nullptr;
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::vector<const char*> m_inputNamesC;
    std::vector<const char*> m_outputNamesC;
    std::vector<OrtValue*> m_inputValuesRaw;
    std::vector<OrtValue*> m_outputValuesRaw;
    std::vector<std::vector<int64_t>> m_inputShapes;
    std::vector<std::vector<int64_t>> m_outputShapes;
    std::vector<TensorDataType> m_inputTypes;
    std::vector<TensorDataType> m_outputTypes;

    WinMlInferenceSession(std::string_view name, OrtSessionPtr session, const OrtApi* ortApi);
    expected<void> Initialize(const OrtMemoryInfo* epMemoryInfo);
    expected<void> InitializeAllocator(const OrtMemoryInfo* epMemoryInfo);
};

}  // namespace libmlvc_winml
