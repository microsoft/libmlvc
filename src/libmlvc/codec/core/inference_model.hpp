// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/fp16.hpp"
#include "libmlvc/common/tensor.hpp"
#include "libmlvc/inference/interface.hpp"

#include <libmlvc/expected.hpp>

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace libmlvc {

struct ReferenceData;

class MlvcBaseModel {
public:
    struct BaseTensorVector {
        BaseTensorVector(const InferenceSessionPtr& session, TensorIoType ioType);

        const std::vector<InferenceTensorPtr>& GetTensors() const;

    protected:
        expected<void> Initialize();
        template <typename T, int N>
        expected<void> AllocateTensor(Tensor<T, N>& output, const bool hostAccessible = true);
        template <typename T, int N>
        expected<void> SetView(const InferenceTensorPtr& inferenceTensor, Tensor<T, N>& output);

    private:
        const InferenceSessionPtr m_session;
        const TensorIoType m_ioType;
        std::unordered_map<std::string, int> m_nameToIndex;
        std::vector<InferenceTensorPtr> m_inferenceTensors;
    };

    struct Inputs : public BaseTensorVector {
        Inputs(const InferenceSessionPtr& session) : BaseTensorVector(session, TensorIoType::INPUT) {};
        Inputs(const Inputs&) = delete;
        Inputs& operator=(const Inputs&) = delete;
    };

    struct Outputs : public BaseTensorVector {
        Outputs(const InferenceSessionPtr& session) : BaseTensorVector(session, TensorIoType::OUTPUT) {};
        Outputs(const Outputs&) = delete;
        Outputs& operator=(const Outputs&) = delete;
    };

    expected<void> Initialize();
    expected<void> Run(const Inputs& inputs, Outputs& outputs);

protected:
    const std::string m_modelName;
    InferenceSessionPtr m_session;

    MlvcBaseModel(std::string_view name, const InferenceSessionPtr& session);
};

class MlvcEncoderModel : public MlvcBaseModel {
public:
    struct Inputs : public MlvcBaseModel::Inputs {
        Tensor<mlvc_f16_t, 3> refFeature{ "ref_feature" };
        Tensor<mlvc_f16_t, 3> x{ "x" };
        Tensor<int32_t, 1> qIndexShifted{ "q_index_shifted" };

        expected<void> Initialize();
        expected<void> BindReference(const ReferenceData& data);
    };

    struct Outputs : public MlvcBaseModel::Outputs {
        Tensor<mlvc_f16_t, 3> feature{ "feature" };
        Tensor<mlvc_f16_t, 3> zRaw{ "z_raw" };
        Tensor<mlvc_f16_t, 3> yRaw0{ "y_raw_0" };
        Tensor<mlvc_f16_t, 3> yRaw1{ "y_raw_1" };

        expected<void> Initialize();
        expected<void> BindReference(const ReferenceData& data);
    };

    MlvcEncoderModel(const InferenceSessionPtr& session);
};

class MlvcDecoderModel : public MlvcBaseModel {
public:
    struct Inputs : public MlvcBaseModel::Inputs {
        Tensor<mlvc_f16_t, 3> zRaw{ "z_raw" };
        Tensor<mlvc_f16_t, 3> yRaw0{ "y_raw_0" };
        Tensor<mlvc_f16_t, 3> yRaw1{ "y_raw_1" };
        Tensor<mlvc_f16_t, 3> refFeature{ "ref_feature" };
        Tensor<int32_t, 1> qIndexShifted{ "q_index_shifted" };

        expected<void> Initialize();
        expected<void> BindReference(const ReferenceData& data);
    };

    struct Outputs : public MlvcBaseModel::Outputs {
        Tensor<mlvc_f16_t, 3> xHat{ "x_hat" };
        Tensor<mlvc_f16_t, 3> feature{ "feature" };

        expected<void> Initialize();
        expected<void> BindReference(const ReferenceData& data);
    };

    MlvcDecoderModel(const InferenceSessionPtr& session);
};

}  // namespace libmlvc
