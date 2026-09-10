// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>
#include <libmlvc/types.hpp>

#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace libmlvc {

enum struct TargetDevice { GENERIC, APPLE, INTEL, QUALCOMM };
enum struct ModelPartId { ENCODER, DECODER };
enum struct EncoderInterfaceType { FP16_SCALE_SENDING_NO_RESET_1P };
enum struct DecoderInterfaceType { FP16_SCALE_SENDING_NO_RESET_1P };
enum struct ScaleDecoderType {
    UPSAMPLE,
};

const char* TargetDeviceToString(TargetDevice device);
const char* ModelPartIdToString(const ModelPartId modelType);
const char* EncoderInterfaceTypeToString(const EncoderInterfaceType modelType);
const char* DecoderInterfaceTypeToString(const DecoderInterfaceType modelType);

struct RegistryEntry {
    std::string modelPath;
    std::optional<std::string> weightsPath;
    std::string sha256;
};

struct ModelPartManifest {
    std::string registryId;
    std::optional<std::string> functionName;
};

struct ModelManifest {
    std::string metadataPath;
    std::string gaussianPmfPath;
    std::string bitEstimatorPmfPath;
    std::map<ModelPartId, ModelPartManifest> modelParts;
    EncoderInterfaceType encoderInterfaceType;
    DecoderInterfaceType decoderInterfaceType;
};

struct ModelMetadata {
    int modelWidth{};
    int modelHeight{};
    float pixelRange{};
    int qpNum{};
    int totalQpNum{};
    std::vector<int> frameIndexMap;
    std::vector<int> qpShift;
    int featureChannels{};
    int latentChannels{};
    int hyperpriorChannels{};
    int downsampleFeature{};
    int downsampleLatent{};
    int downsampleHyperprior{};
    ScaleDecoderType scaleDecoderType{ ScaleDecoderType::UPSAMPLE };
    std::optional<int> yScaleRepeat;
    std::optional<int> iframePeriod;
    std::optional<int> resetPeriod;
    std::optional<int> ltrStartIdx;
    std::optional<int> ltrPeriod;
    std::optional<std::vector<int>> qpMapping;
};

struct BundleManifest {
    MlvcVersion mlvcVersion;
    std::string bundleName;
    std::string timestamp;
    ModelType modelType{ ModelType::UNKNOWN };
    TargetDevice targetDevice{ TargetDevice::GENERIC };
    std::map<std::string, RegistryEntry> modelRegistry;                   // registry_id -> entry
    std::map<std::string, ModelManifest> modelManifests;                  // model_id -> bundled model
    std::map<std::string, std::shared_ptr<ModelMetadata>> modelMetadata;  // model_id -> metadata

    static expected<BundleManifest> FromJsonBuffer(std::span<const std::byte> buffer);
};

}  // namespace libmlvc
