// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/schema/bundle.hpp"
#include "libmlvc/common/logging.hpp"

#include <libmlvc/error_codes.hpp>

#include <boost/json.hpp>

#include <limits>

// On macOS builds, exceptions are disabled, so Boost requires a user-defined implementation of boost::throw_exception.
// For code using non-throwing JSON parsing functions, this should never be invoked.
// TODO: Find a better approach to avoid duplicate definitions
#ifdef BOOST_NO_EXCEPTIONS
    #include <boost/assert/source_location.hpp>

    #include <exception>

namespace boost {

void throw_exception(const std::exception& e, const boost::source_location& loc)
{
    MLVC_LOG_ABORT("Boost exception at %s:%d: %s", loc.file_name(), loc.line(), e.what());
    std::terminate();
}

}  // namespace boost
#endif

namespace libmlvc {

// ----------------------------------------------------------------
// Enum parsing helpers
// ----------------------------------------------------------------

expected<TargetDevice> ParseTargetDevice(std::string_view value)
{
    if (value == "generic") {
        return TargetDevice::GENERIC;
    } else if (value == "apple") {
        return TargetDevice::APPLE;
    } else if (value == "intel") {
        return TargetDevice::INTEL;
    } else if (value == "qualcomm") {
        return TargetDevice::QUALCOMM;
    }
    MLVC_LOG_ERROR("Unknown target device: %s", std::string(value).c_str());
    return make_error_code(Error::invalid_argument);
}

const char* TargetDeviceToString(const TargetDevice device)
{
    switch (device) {
    case TargetDevice::GENERIC:
        return "generic";
    case TargetDevice::APPLE:
        return "apple";
    case TargetDevice::INTEL:
        return "intel";
    case TargetDevice::QUALCOMM:
        return "qualcomm";
    }
    return "unknown";
}

expected<ModelType> ParseModelType(std::string_view v)
{
    if (v == "coreml") {
        return ModelType::COREML;
    } else if (v == "onnx") {
        return ModelType::ONNX;
    }
    MLVC_LOG_ERROR("Unknown model type: %s", std::string(v).c_str());
    return make_error_code(Error::invalid_argument);
}

expected<ModelPartId> ParseModelPartId(std::string_view v)
{
    if (v == "MLVCEncoder") {
        return ModelPartId::ENCODER;
    } else if (v == "MLVCDecoder") {
        return ModelPartId::DECODER;
    }
    MLVC_LOG_ERROR("Unknown model part id: %s", std::string(v).c_str());
    return make_error_code(Error::invalid_argument);
}

const char* ModelPartIdToString(const ModelPartId modelPartId)
{
    switch (modelPartId) {
    case ModelPartId::ENCODER:
        return "MLVCEncoder";
    case ModelPartId::DECODER:
        return "MLVCDecoder";
    default:
        return "unknown";
    }
}

expected<EncoderInterfaceType> ParseEncoderInterfaceType(std::string_view v)
{
    if (v == "fp16_scale_sending_no_reset_1p") {
        return EncoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P;
    }
    MLVC_LOG_ERROR("Unknown encoder interface type: %s", std::string(v).c_str());
    return make_error_code(Error::invalid_argument);
}

const char* EncoderInterfaceTypeToString(const EncoderInterfaceType interfaceType)
{
    switch (interfaceType) {
    case EncoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P:
        return "fp16_scale_sending_no_reset_1p";
    default:
        return "unknown";
    }
}

expected<DecoderInterfaceType> ParseDecoderInterfaceType(std::string_view v)
{
    if (v == "fp16_scale_sending_no_reset_1p") {
        return DecoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P;
    }
    MLVC_LOG_ERROR("Unknown encoder interface type: %s", std::string(v).c_str());
    return make_error_code(Error::invalid_argument);
}

const char* DecoderInterfaceTypeToString(const DecoderInterfaceType interfaceType)
{
    switch (interfaceType) {
    case DecoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P:
        return "fp16_scale_sending_no_reset_1p";
    default:
        return "unknown";
    }
}

expected<ScaleDecoderType> ParseScaleDecoderType(std::string_view v)
{
    if (v == "upsample") {
        return ScaleDecoderType::UPSAMPLE;
    }
    MLVC_LOG_ERROR("Unsupported scale decoder type: %s", std::string(v).c_str());
    return make_error_code(Error::invalid_argument);
}

// ----------------------------------------------------------------
// JSON parsing helpers
// ----------------------------------------------------------------

template <typename Key>
expected<Key> ConvertJsonKey(std::string_view sv)
{
    if constexpr (std::is_same_v<Key, std::string>) {
        return std::string(sv);
    } else if constexpr (std::is_same_v<Key, ModelPartId>) {
        return ParseModelPartId(sv);
    } else {
        static_assert(!sizeof(Key), "ConvertJsonKey not specialized for this Key type");
    }
}

template <typename T>
std::error_code ParseRequired(const boost::json::object& parent, std::string_view key, T& out)
{
    if (auto v = parent.if_contains(key)) {
        auto res = boost::json::try_value_to<T>(*v);
        if (!res) {
            MLVC_LOG_ERROR("Failed to parse %s", std::string(key).c_str());
            return res.error();
        }
        out = std::move(res.value());
        return std::error_code{};
    } else {
        MLVC_LOG_ERROR("JSON object missing key %s", std::string(key).c_str());
        return make_error_code(Error::json_parse_error);
    }
}

template <typename T>
std::error_code ParseOptional(const boost::json::object& parent, std::string_view key, std::optional<T>& out)
{
    if (auto v = parent.if_contains(key)) {
        if (v->is_null()) {
            out = std::nullopt;
            return {};
        }
        auto res = boost::json::try_value_to<T>(*v);
        if (!res) {
            MLVC_LOG_ERROR("Failed to parse %s", std::string(key).c_str());
            return res.error();
        }
        out = std::move(res.value());
        return {};
    }
    out = std::nullopt;
    return {};
}

template <typename T>
std::error_code ParseOptional(const boost::json::object& parent, std::string_view key, T& out)
{
    // Parses an optional field into a plain value; leaves `out` unchanged when the key is absent or null
    if (auto v = parent.if_contains(key)) {
        if (v->is_null()) {
            return {};
        }
        auto res = boost::json::try_value_to<T>(*v);
        if (!res) {
            MLVC_LOG_ERROR("Failed to parse %s", std::string(key).c_str());
            return res.error();
        }
        out = std::move(res.value());
    }
    return {};
}

template <typename Key, typename Value>
std::error_code ParseMap(const boost::json::object& parent, std::string_view key, std::map<Key, Value>& out)
{
    auto v = parent.if_contains(key);
    if (!v) {
        MLVC_LOG_ERROR("Missing %s key", std::string(key).c_str());
        return make_error_code(Error::json_parse_error);
    }
    if (!v->is_object()) {
        MLVC_LOG_ERROR("%s JSON value is not an object", std::string(key).c_str());
        return make_error_code(Error::json_parse_error);
    }
    const auto& obj = v->as_object();
    for (const auto& kv : obj) {
        auto keyRes = ConvertJsonKey<Key>(kv.key());
        if (!keyRes) {
            return keyRes.error();
        }
        auto valRes = boost::json::try_value_to<Value>(kv.value());
        if (!valRes) {
            return valRes.error();
        }
        out.emplace(keyRes.value(), std::move(valRes.value()));
    }
    return {};
}

// ----------------------------------------------------------------
// Struct parsing helpers
// ----------------------------------------------------------------

boost::json::result_for<ModelType, boost::json::value>::type tag_invoke(const boost::json::try_value_to_tag<ModelType>&,
                                                                        const boost::json::value& jv)
{
    if (!jv.is_string()) {
        MLVC_LOG_ERROR("ModelType JSON value is not a string");
        return make_error_code(Error::json_parse_error);
    }
    auto s = jv.as_string();
    auto ret = ParseModelType({ s.data(), s.size() });
    if (!ret) return ret.error();
    return ret.value();
}

boost::json::result_for<TargetDevice, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<TargetDevice>&, const boost::json::value& jv)
{
    if (!jv.is_string()) {
        MLVC_LOG_ERROR("TargetDevice JSON value is not a string");
        return make_error_code(Error::json_parse_error);
    }
    auto s = jv.as_string();
    auto ret = ParseTargetDevice({ s.data(), s.size() });
    if (!ret) return ret.error();
    return ret.value();
}

boost::json::result_for<ModelPartId, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<ModelPartId>&, const boost::json::value& jv)
{
    if (!jv.is_string()) {
        MLVC_LOG_ERROR("ModelPartId JSON value is not a string");
        return make_error_code(Error::json_parse_error);
    }
    auto s = jv.as_string();
    auto ret = ParseModelPartId({ s.data(), s.size() });
    if (!ret) return ret.error();
    return ret.value();
}

boost::json::result_for<EncoderInterfaceType, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<EncoderInterfaceType>&, const boost::json::value& jv)
{
    if (!jv.is_string()) {
        MLVC_LOG_ERROR("EncoderInterfaceType JSON value is not a string");
        return make_error_code(Error::json_parse_error);
    }
    auto s = jv.as_string();
    auto ret = ParseEncoderInterfaceType({ s.data(), s.size() });
    if (!ret) return ret.error();
    return ret.value();
}

boost::json::result_for<DecoderInterfaceType, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<DecoderInterfaceType>&, const boost::json::value& jv)
{
    if (!jv.is_string()) {
        MLVC_LOG_ERROR("DecoderInterfaceType JSON value is not a string");
        return make_error_code(Error::json_parse_error);
    }
    auto s = jv.as_string();
    auto ret = ParseDecoderInterfaceType({ s.data(), s.size() });
    if (!ret) return ret.error();
    return ret.value();
}

boost::json::result_for<ScaleDecoderType, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<ScaleDecoderType>&, const boost::json::value& jv)
{
    if (!jv.is_string()) {
        MLVC_LOG_ERROR("ScaleDecoderType JSON value is not a string");
        return make_error_code(Error::json_parse_error);
    }
    auto s = jv.as_string();
    auto ret = ParseScaleDecoderType({ s.data(), s.size() });
    if (!ret) return ret.error();
    return ret.value();
}

boost::json::result_for<MlvcVersion, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<MlvcVersion>&, const boost::json::value& jv)
{
    if (!jv.is_object()) {
        MLVC_LOG_ERROR("MlvcVersion JSON value is not an object");
        return make_error_code(Error::json_parse_error);
    }
    const auto& obj = jv.as_object();

    MlvcVersion v{};
    if (auto ec = ParseRequired(obj, "major", v.major); ec) return ec;
    if (auto ec = ParseRequired(obj, "minor", v.minor); ec) return ec;
    return v;
}

boost::json::result_for<RegistryEntry, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<RegistryEntry>&, const boost::json::value& jv)
{
    if (!jv.is_object()) {
        MLVC_LOG_ERROR("RegistryEntry JSON value is not an object");
        return make_error_code(Error::json_parse_error);
    }
    const auto& obj = jv.as_object();

    RegistryEntry e{};
    if (auto ec = ParseRequired(obj, "model_path", e.modelPath); ec) return ec;
    if (auto ec = ParseOptional(obj, "weights_path", e.weightsPath); ec) return ec;
    if (auto ec = ParseRequired(obj, "sha256", e.sha256); ec) return ec;
    return e;
}

boost::json::result_for<ModelPartManifest, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<ModelPartManifest>&, const boost::json::value& jv)
{
    if (!jv.is_object()) {
        MLVC_LOG_ERROR("ModelPartManifest JSON value is not an object");
        return make_error_code(Error::json_parse_error);
    }
    const auto& obj = jv.as_object();

    ModelPartManifest m{};
    if (auto ec = ParseRequired(obj, "registry_id", m.registryId); ec) return ec;
    if (auto ec = ParseOptional(obj, "function_name", m.functionName); ec) return ec;
    return m;
}

boost::json::result_for<ModelMetadata, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<ModelMetadata>&, const boost::json::value& jv)
{
    if (!jv.is_object()) {
        MLVC_LOG_ERROR("ModelMetadata JSON value is not an object");
        return make_error_code(Error::json_parse_error);
    }
    const auto& obj = jv.as_object();

    ModelMetadata m{};
    if (auto ec = ParseRequired(obj, "model_width", m.modelWidth); ec) return ec;
    if (auto ec = ParseRequired(obj, "model_height", m.modelHeight); ec) return ec;
    if (auto ec = ParseRequired(obj, "pixel_range", m.pixelRange); ec) return ec;
    if (auto ec = ParseRequired(obj, "qp_num", m.qpNum); ec) return ec;
    if (auto ec = ParseRequired(obj, "total_qp_num", m.totalQpNum); ec) return ec;
    if (auto ec = ParseRequired(obj, "frame_index_map", m.frameIndexMap); ec) return ec;
    if (auto ec = ParseRequired(obj, "qp_shift", m.qpShift); ec) return ec;
    if (auto ec = ParseRequired(obj, "feature_channels", m.featureChannels); ec) return ec;
    if (auto ec = ParseRequired(obj, "latent_channels", m.latentChannels); ec) return ec;
    if (auto ec = ParseRequired(obj, "hyperprior_channels", m.hyperpriorChannels); ec) return ec;
    if (auto ec = ParseRequired(obj, "downsample_feature", m.downsampleFeature); ec) return ec;
    if (auto ec = ParseRequired(obj, "downsample_latent", m.downsampleLatent); ec) return ec;
    if (auto ec = ParseRequired(obj, "downsample_hyperprior", m.downsampleHyperprior); ec) return ec;
    if (auto ec = ParseOptional(obj, "scale_decoder_type", m.scaleDecoderType); ec) return ec;
    if (auto ec = ParseOptional(obj, "y_scale_repeat", m.yScaleRepeat); ec) return ec;
    if (auto ec = ParseOptional(obj, "iframe_period", m.iframePeriod); ec) return ec;
    if (auto ec = ParseOptional(obj, "reset_period", m.resetPeriod); ec) return ec;
    if (auto ec = ParseOptional(obj, "ltr_start_idx", m.ltrStartIdx); ec) return ec;
    if (auto ec = ParseOptional(obj, "ltr_period", m.ltrPeriod); ec) return ec;
    if (auto ec = ParseOptional(obj, "qp_mapping", m.qpMapping); ec) return ec;
    return m;
}

boost::json::result_for<std::shared_ptr<ModelMetadata>, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<std::shared_ptr<ModelMetadata>>&, const boost::json::value& jv)
{
    auto r = boost::json::try_value_to<ModelMetadata>(jv);
    if (!r) return r.error();
    return std::make_shared<ModelMetadata>(std::move(r.value()));
}

boost::json::result_for<ModelManifest, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<ModelManifest>&, const boost::json::value& jv)
{
    if (!jv.is_object()) {
        MLVC_LOG_ERROR("ModelManifest JSON value is not an object");
        return make_error_code(Error::json_parse_error);
    }
    const auto& obj = jv.as_object();

    ModelManifest m{};
    if (auto ec = ParseRequired(obj, "metadata_path", m.metadataPath); ec) return ec;
    if (auto ec = ParseRequired(obj, "gaussian_pmf_path", m.gaussianPmfPath); ec) return ec;
    if (auto ec = ParseRequired(obj, "bit_estimator_pmf_path", m.bitEstimatorPmfPath); ec) return ec;
    if (auto ec = ParseMap(obj, "model_parts", m.modelParts); ec) return ec;
    if (auto ec = ParseRequired(obj, "encoder_interface_type", m.encoderInterfaceType); ec) return ec;
    if (auto ec = ParseRequired(obj, "decoder_interface_type", m.decoderInterfaceType); ec) return ec;
    return m;
}

boost::json::result_for<BundleManifest, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<BundleManifest>&, const boost::json::value& jv)
{
    if (!jv.is_object()) {
        MLVC_LOG_ERROR("BundleManifest JSON value is not an object");
        return make_error_code(Error::json_parse_error);
    }
    const auto& obj = jv.as_object();

    BundleManifest b{};
    if (auto ec = ParseRequired(obj, "mlvc_version", b.mlvcVersion); ec) return ec;
    if (auto ec = ParseRequired(obj, "bundle_name", b.bundleName); ec) return ec;
    if (auto ec = ParseRequired(obj, "timestamp", b.timestamp); ec) return ec;
    if (auto ec = ParseRequired(obj, "model_type", b.modelType); ec) return ec;
    if (auto ec = ParseRequired(obj, "target_device", b.targetDevice); ec) return ec;
    if (auto ec = ParseMap(obj, "model_registry", b.modelRegistry)) return ec;
    if (auto ec = ParseMap(obj, "model_manifests", b.modelManifests)) return ec;
    if (auto ec = ParseMap(obj, "model_metadata", b.modelMetadata)) return ec;
    return b;
}

// ----------------------------------------------------------------
// BundleManifest factory from JSON buffer
// ----------------------------------------------------------------

expected<BundleManifest> BundleManifest::FromJsonBuffer(std::span<const std::byte> buffer)
{
    std::error_code ec;
    auto jv = boost::json::parse({ reinterpret_cast<const char*>(buffer.data()), buffer.size() }, ec);
    if (ec) {
        MLVC_LOG_ERROR("Failed to parse BundleManifest JSON data: %s", ec.message().c_str());
        return make_error_code(Error::json_parse_error);
    }

    // Peek MLVC major version
    int mlvcMajorVersion = 0;
    {
        auto majorVersionJv = jv.find_pointer("/mlvc_version/major", ec);
        if (ec) {
            MLVC_LOG_ERROR("Failed to read mlvc_version.major from BundleManifest JSON");
            return make_error_code(Error::json_parse_error);
        }

        auto majorRet = boost::json::try_value_to<int>(*majorVersionJv);
        if (!majorRet) {
            MLVC_LOG_ERROR("Failed to convert mlvc_version.major to int");
            return make_error_code(Error::json_parse_error);
        }
        mlvcMajorVersion = majorRet.value();
    }

    // Validate MLVC version
    if (mlvcMajorVersion < MIN_SUPPORTED_MLVC_VERSION || mlvcMajorVersion > MAX_SUPPORTED_MLVC_VERSION) {
        MLVC_LOG_ERROR("Unsupported MLVC major version: %d (supported range %d-%d)", mlvcMajorVersion,
                       MIN_SUPPORTED_MLVC_VERSION, MAX_SUPPORTED_MLVC_VERSION);
        return make_error_code(Error::invalid_argument);
    }

    auto manifest = boost::json::try_value_to<BundleManifest>(jv);
    if (!manifest) {
        MLVC_LOG_ERROR("Failed to convert JSON value to BundleManifest (MLVC major version: %d)", mlvcMajorVersion);
        return make_error_code(Error::json_parse_error);
    }
    return std::move(manifest.value());
}

}  // namespace libmlvc
