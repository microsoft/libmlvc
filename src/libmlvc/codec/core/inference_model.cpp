// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/core/inference_model.hpp"
#include "libmlvc/codec/core/reference_data.hpp"
#include "libmlvc/common/macros.hpp"

#include <libmlvc/error_codes.hpp>

namespace libmlvc {

MlvcBaseModel::BaseTensorVector::BaseTensorVector(const InferenceSessionPtr& session, TensorIoType ioType)
    : m_session(session), m_ioType(ioType)
{
}

template <typename T, int N>
expected<void> MlvcBaseModel::BaseTensorVector::AllocateTensor(Tensor<T, N>& output, const bool hostAccessible)
{
    // Allocate inference tensor
    const std::string name = output.Name();
    auto inferenceTensor = m_session->CreateTensor(m_ioType, name, hostAccessible);
    if (!inferenceTensor) {
        MLVC_LOG_ERROR("Failed to allocate tensor (%s): %s", name.c_str(), inferenceTensor.error().message().c_str());
        return inferenceTensor.error();
    }

    // Create view
    auto view = inferenceTensor.value()->template View<T, N>();
    if (!view) {
        MLVC_LOG_ERROR("Failed to create view for tensor (%s): %s", name.c_str(), view.error().message().c_str());
        return view.error();
    }

    output = std::move(view.value());

    auto it = m_nameToIndex.find(name);
    if (it == m_nameToIndex.end()) {
        MLVC_LOG_ERROR("Tensor name not found in session I/O names (%s)", name.c_str());
        return make_error_code(Error::invalid_argument);
    }
    m_inferenceTensors[it->second] = std::move(inferenceTensor.value());
    return {};
}

template <typename T, int N>
expected<void> MlvcBaseModel::BaseTensorVector::SetView(const InferenceTensorPtr& inferenceTensor, Tensor<T, N>& output)
{
    const std::string name = output.Name();

    // Create view
    auto view = inferenceTensor->template View<T, N>();
    if (!view) {
        MLVC_LOG_ERROR("Failed to create view for tensor (%s): %s", name.c_str(), view.error().message().c_str());
        return view.error();
    }
    output = std::move(view.value());
    output.SetName(name);

    auto it = m_nameToIndex.find(name);
    if (it == m_nameToIndex.end()) {
        MLVC_LOG_ERROR("Tensor name not found in session I/O names (%s)", name.c_str());
        return make_error_code(Error::invalid_argument);
    }
    m_inferenceTensors[it->second] = inferenceTensor;
    return {};
}

expected<void> MlvcBaseModel::BaseTensorVector::Initialize()
{
    auto names = (m_ioType == TensorIoType::INPUT) ? m_session->GetInputNames() : m_session->GetOutputNames();
    if (!names) {
        MLVC_LOG_ERROR("Failed to get I/O names from session: %s", names.error().message().c_str());
        return names.error();
    }

    m_nameToIndex.clear();
    m_inferenceTensors.clear();
    m_inferenceTensors.resize(names.value().size(), nullptr);
    for (int i = 0; i < static_cast<int>(names.value().size()); i++) {
        m_nameToIndex[std::string(names.value()[i])] = i;
    }
    return {};
}

const std::vector<InferenceTensorPtr>& MlvcBaseModel::BaseTensorVector::GetTensors() const
{
    return m_inferenceTensors;
}

MlvcBaseModel::MlvcBaseModel(std::string_view name, const InferenceSessionPtr& session)
    : m_modelName{ name }, m_session(session)
{
}

expected<void> MlvcBaseModel::Initialize()
{
    MLVC_LOG_DEBUG("Initialize model: %s", m_modelName.c_str());
    return {};
}

expected<void> MlvcBaseModel::Run(const Inputs& inputs, Outputs& outputs)
{
    if (auto ret = m_session->Run(inputs.GetTensors(), outputs.GetTensors()); !ret) {
        MLVC_LOG_ERROR("Failed to run model (%s): %s", m_modelName.c_str(), ret.error().message().c_str());
        return make_error_code(Error::model_inference_error);
    }
    return {};
}

expected<void> MlvcEncoderModel::Inputs::Initialize()
{
    if (auto ret = BaseTensorVector::Initialize(); !ret) return ret.error();
    if (auto ret = AllocateTensor(x); !ret) return ret.error();
    if (auto ret = AllocateTensor(qIndexShifted); !ret) return ret.error();
    return {};
}

expected<void> MlvcEncoderModel::Inputs::BindReference(const ReferenceData& data)
{
    if (auto ret = SetView(data.GetRefFeature(), refFeature); !ret) return ret.error();
    return {};
}

expected<void> MlvcEncoderModel::Outputs::Initialize()
{
    if (auto ret = BaseTensorVector::Initialize(); !ret) return ret.error();
    if (auto ret = AllocateTensor(zRaw); !ret) return ret.error();
    if (auto ret = AllocateTensor(yRaw0); !ret) return ret.error();
    if (auto ret = AllocateTensor(yRaw1); !ret) return ret.error();
    return {};
}

expected<void> MlvcEncoderModel::Outputs::BindReference(const ReferenceData& data)
{
    if (auto ret = SetView(data.GetRefFeature(), feature); !ret) return ret.error();
    return {};
}

MlvcEncoderModel::MlvcEncoderModel(const InferenceSessionPtr& session) : MlvcBaseModel("MLVCEncoder", session) {}

expected<void> MlvcDecoderModel::Inputs::Initialize()
{
    if (auto ret = BaseTensorVector::Initialize(); !ret) return ret.error();
    if (auto ret = AllocateTensor(zRaw); !ret) return ret.error();
    if (auto ret = AllocateTensor(yRaw0); !ret) return ret.error();
    if (auto ret = AllocateTensor(yRaw1); !ret) return ret.error();
    if (auto ret = AllocateTensor(qIndexShifted); !ret) return ret.error();
    return {};
}

expected<void> MlvcDecoderModel::Inputs::BindReference(const ReferenceData& data)
{
    if (auto ret = SetView(data.GetRefFeature(), refFeature); !ret) return ret.error();
    return {};
}

expected<void> MlvcDecoderModel::Outputs::Initialize()
{
    if (auto ret = BaseTensorVector::Initialize(); !ret) return ret.error();
    if (auto ret = AllocateTensor(xHat); !ret) return ret.error();
    return {};
}

expected<void> MlvcDecoderModel::Outputs::BindReference(const ReferenceData& data)
{
    if (auto ret = SetView(data.GetRefFeature(), feature); !ret) return ret.error();
    return {};
}

MlvcDecoderModel::MlvcDecoderModel(const InferenceSessionPtr& session) : MlvcBaseModel("MLVCDecoder", session) {}

}  // namespace libmlvc
