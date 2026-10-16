// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/bundle.hpp"
#include "libmlvc/common/tar.hpp"
#include "libmlvc/schema/bundle.hpp"
#include "support/test_config.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/libmlvc.hpp>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace libmlvc;

namespace {

boost::json::object MakeManifest(const boost::json::object& layoutFields = {})
{
    boost::json::object metadata{
        { "model_width", 320 },
        { "model_height", 192 },
        { "pixel_range", 1.0 },
        { "qp_num", 64 },
        { "total_qp_num", 80 },
        { "frame_index_map", boost::json::array{ 0, 1, 2, 1 } },
        { "qp_shift", boost::json::array{ 0, 4, 8 } },
        { "feature_channels", 96 },
        { "latent_channels", 128 },
        { "hyperprior_channels", 32 },
        { "downsample_feature", 2 },
        { "downsample_latent", 8 },
        { "downsample_hyperprior", 4 },
        { "y_scale_repeat", 16 },
        { "iframe_period", 256 },
        { "qp_mapping", boost::json::array{ 0, 2, 4 } },
    };
    for (const auto& field : layoutFields) {
        metadata[field.key()] = field.value();
    }
    return {
        { "mlvc_version", boost::json::object{ { "major", 0 }, { "minor", 101 } } },
        { "bundle_name", "layout-test" },
        { "timestamp", "2026-09-17T00:00:00Z" },
        { "model_type", "onnx" },
        { "target_device", "qualcomm" },
        { "model_registry",
          boost::json::object{
              { "encoder",
                boost::json::object{
                    { "model_path", "encoder.onnx" },
                    { "weights_path", "encoder.data" },
                    { "sha256", std::string(64, 'a') },
                } },
              { "decoder",
                boost::json::object{
                    { "model_path", "decoder.onnx" },
                    { "sha256", std::string(64, 'b') },
                } },
          } },
        { "model_manifests",
          boost::json::object{
              { "320x192",
                boost::json::object{
                    { "metadata_path", "metadata.json" },
                    { "gaussian_pmf_path", "gaussian.bin" },
                    { "bit_estimator_pmf_path", "bit_estimator.bin" },
                    { "encoder_interface_type", "fp16_scale_sending_no_reset_1p" },
                    { "decoder_interface_type", "fp16_scale_sending_no_reset_1p" },
                    { "model_parts",
                      boost::json::object{
                          { "MLVCEncoder",
                            boost::json::object{
                                { "registry_id", "encoder" },
                                { "function_name", "encode" },
                            } },
                          { "MLVCDecoder", boost::json::object{ { "registry_id", "decoder" } } },
                      } },
                } },
          } },
        { "model_metadata", boost::json::object{ { "320x192", metadata } } },
    };
}

expected<BundleManifest> ParseManifest(const boost::json::value& manifest)
{
    const auto json = boost::json::serialize(manifest);
    return BundleManifest::FromJsonBuffer(std::as_bytes(std::span{ json.data(), json.size() }));
}

expected<std::vector<BundleManifest>> LoadManifests(std::span<const MlvcVersion> versions,
                                                    const InferenceEngineInfo& engineInfo)
{
    const auto bundles = LoadModelBundles(GetDefaultModelBundlesDir(), versions, engineInfo);
    if (!bundles) return bundles.error();
    std::vector<BundleManifest> manifests;
    for (const auto& bundle : bundles.value()) {
        const auto files = ExtractTar(bundle);
        if (!files) return files.error();
        auto manifest = BundleManifest::FromJsonBuffer(files->at("bundle_manifest.json"));
        if (!manifest) return manifest.error();
        manifests.push_back(std::move(manifest.value()));
    }
    return manifests;
}
}  // namespace

TEST(UnitTestBundle, ParsesCompleteManifest)
{
    const auto result = ParseManifest(MakeManifest());
    ASSERT_TRUE(result) << result.error().message();
    const auto& manifest = result.value();
    EXPECT_EQ(manifest.mlvcVersion, (MlvcVersion{ 0, 101 }));
    EXPECT_EQ(manifest.bundleName, "layout-test");
    EXPECT_EQ(manifest.timestamp, "2026-09-17T00:00:00Z");
    EXPECT_EQ(manifest.modelType, ModelType::ONNX);
    EXPECT_EQ(manifest.targetDevice, TargetDevice::QUALCOMM);

    ASSERT_EQ(manifest.modelRegistry.size(), 2);
    const auto& encoder = manifest.modelRegistry.at("encoder");
    EXPECT_EQ(encoder.modelPath, "encoder.onnx");
    EXPECT_EQ(encoder.weightsPath, "encoder.data");
    EXPECT_EQ(encoder.sha256, std::string(64, 'a'));
    const auto& decoder = manifest.modelRegistry.at("decoder");
    EXPECT_EQ(decoder.modelPath, "decoder.onnx");
    EXPECT_FALSE(decoder.weightsPath.has_value());
    EXPECT_EQ(decoder.sha256, std::string(64, 'b'));

    ASSERT_EQ(manifest.modelManifests.size(), 1);
    const auto& model = manifest.modelManifests.at("320x192");
    EXPECT_EQ(model.metadataPath, "metadata.json");
    EXPECT_EQ(model.gaussianPmfPath, "gaussian.bin");
    EXPECT_EQ(model.bitEstimatorPmfPath, "bit_estimator.bin");
    EXPECT_EQ(model.encoderInterfaceType, EncoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P);
    EXPECT_EQ(model.decoderInterfaceType, DecoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P);
    ASSERT_EQ(model.modelParts.size(), 2);
    EXPECT_EQ(model.modelParts.at(ModelPartId::ENCODER).registryId, "encoder");
    EXPECT_EQ(model.modelParts.at(ModelPartId::ENCODER).functionName, "encode");
    EXPECT_EQ(model.modelParts.at(ModelPartId::DECODER).registryId, "decoder");
    EXPECT_FALSE(model.modelParts.at(ModelPartId::DECODER).functionName.has_value());

    ASSERT_EQ(manifest.modelMetadata.size(), 1);
    ASSERT_TRUE(manifest.modelMetadata.at("320x192"));
    const auto& metadata = *manifest.modelMetadata.at("320x192");
    EXPECT_EQ(metadata.modelWidth, 320);
    EXPECT_EQ(metadata.modelHeight, 192);
    EXPECT_FLOAT_EQ(metadata.pixelRange, 1.0f);
    EXPECT_EQ(metadata.qpNum, 64);
    EXPECT_EQ(metadata.totalQpNum, 80);
    EXPECT_EQ(metadata.frameIndexMap, (std::vector<int>{ 0, 1, 2, 1 }));
    EXPECT_EQ(metadata.qpShift, (std::vector<int>{ 0, 4, 8 }));
    EXPECT_EQ(metadata.featureChannels, 96);
    EXPECT_EQ(metadata.latentChannels, 128);
    EXPECT_EQ(metadata.hyperpriorChannels, 32);
    EXPECT_EQ(metadata.downsampleFeature, 2);
    EXPECT_EQ(metadata.downsampleLatent, 8);
    EXPECT_EQ(metadata.downsampleHyperprior, 4);
    EXPECT_EQ(metadata.scaleDecoderType, ScaleDecoderType::UPSAMPLE);
    EXPECT_EQ(metadata.yScaleRepeat, 16);
    EXPECT_EQ(metadata.iframePeriod, 256);
    EXPECT_FALSE(metadata.resetPeriod.has_value());
    EXPECT_FALSE(metadata.ltrStartIdx.has_value());
    EXPECT_FALSE(metadata.ltrPeriod.has_value());
    ASSERT_TRUE(metadata.qpMapping.has_value());
    EXPECT_EQ(*metadata.qpMapping, (std::vector<int>{ 0, 2, 4 }));
}

TEST(UnitTestBundle, DefaultLayoutsAreNchw)
{
    struct LayoutCase {
        boost::json::object fields;
        TensorLayout image;
        TensorLayout feature;
    };
    const LayoutCase cases[]{
        { {}, TensorLayout::NCHW, TensorLayout::NCHW },
        { { { "image_layout", nullptr }, { "feature_layout", nullptr } }, TensorLayout::NCHW, TensorLayout::NCHW },
        { { { "image_layout", "nhwc" } }, TensorLayout::NHWC, TensorLayout::NCHW },
        { { { "feature_layout", "nhwc" } }, TensorLayout::NCHW, TensorLayout::NHWC },
        { { { "image_layout", "nhwc" }, { "feature_layout", nullptr } }, TensorLayout::NHWC, TensorLayout::NCHW },
        { { { "image_layout", nullptr }, { "feature_layout", "nhwc" } }, TensorLayout::NCHW, TensorLayout::NHWC },
    };
    for (const auto& testCase : cases) {
        SCOPED_TRACE(boost::json::serialize(testCase.fields));
        const auto result = ParseManifest(MakeManifest(testCase.fields));
        ASSERT_TRUE(result) << result.error().message();
        const auto& metadata = *result.value().modelMetadata.at("320x192");
        EXPECT_EQ(metadata.imageLayout, testCase.image);
        EXPECT_EQ(metadata.featureLayout, testCase.feature);
    }
}

TEST(UnitTestBundle, ParsesIndependentLayouts)
{
    for (const auto imageLayout : { TensorLayout::NCHW, TensorLayout::NHWC }) {
        for (const auto featureLayout : { TensorLayout::NCHW, TensorLayout::NHWC }) {
            const boost::json::object fields{
                { "image_layout", imageLayout == TensorLayout::NCHW ? "nchw" : "nhwc" },
                { "feature_layout", featureLayout == TensorLayout::NCHW ? "nchw" : "nhwc" },
            };
            SCOPED_TRACE(boost::json::serialize(fields));
            const auto result = ParseManifest(MakeManifest(fields));
            ASSERT_TRUE(result) << result.error().message();
            const auto& metadata = *result.value().modelMetadata.at("320x192");
            EXPECT_EQ(metadata.imageLayout, imageLayout);
            EXPECT_EQ(metadata.featureLayout, featureLayout);
        }
    }
}

TEST(UnitTestBundle, RejectsInvalidLayouts)
{
    const boost::json::array invalidValues{
        "nwhc", "", "NCHW", 42, 0.5, true, boost::json::array{}, boost::json::object{}
    };
    for (const auto* field : { "image_layout", "feature_layout" }) {
        for (const auto& value : invalidValues) {
            const boost::json::object fields{ { field, value } };
            SCOPED_TRACE(boost::json::serialize(fields));
            const auto result = ParseManifest(MakeManifest(fields));
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error(), make_error_code(Error::json_parse_error));
        }
    }
}

TEST(UnitTestBundle, RejectsMalformedJson)
{
    for (const std::string_view json : { "", "{", "{} trailing", "{\"mlvc_version\":" }) {
        SCOPED_TRACE(json);
        const auto result = BundleManifest::FromJsonBuffer(std::as_bytes(std::span{ json.data(), json.size() }));
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), make_error_code(Error::json_parse_error));
    }
}

TEST(UnitTestBundle, RejectsMissingRequiredFields)
{
    struct RequiredFields {
        const char* parent;
        std::initializer_list<const char*> names;
    };
    const RequiredFields groups[]{
        { "",
          { "mlvc_version", "bundle_name", "timestamp", "model_type", "target_device", "model_registry",
            "model_manifests", "model_metadata" } },
        { "/mlvc_version", { "major", "minor" } },
        { "/model_registry/encoder", { "model_path", "sha256" } },
        { "/model_manifests/320x192",
          { "metadata_path", "gaussian_pmf_path", "bit_estimator_pmf_path", "model_parts", "encoder_interface_type",
            "decoder_interface_type" } },
        { "/model_manifests/320x192/model_parts/MLVCEncoder", { "registry_id" } },
        { "/model_metadata/320x192",
          { "model_width", "model_height", "pixel_range", "qp_num", "total_qp_num", "frame_index_map", "qp_shift",
            "feature_channels", "latent_channels", "hyperprior_channels", "downsample_feature", "downsample_latent",
            "downsample_hyperprior" } },
    };
    for (const auto& group : groups) {
        for (const auto* name : group.names) {
            SCOPED_TRACE(::testing::Message() << group.parent << '/' << name);
            boost::json::value manifest = MakeManifest();
            ASSERT_EQ(manifest.at_pointer(group.parent).as_object().erase(name), 1);
            const auto result = ParseManifest(manifest);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error(), make_error_code(Error::json_parse_error));
        }
    }
}

TEST(UnitTestBundle, RejectsInvalidManifestFields)
{
    struct InvalidField {
        const char* path;
        boost::json::value value;
        Error error{ Error::json_parse_error };
    };
    const InvalidField cases[]{
        { "", nullptr },
        { "", boost::json::array{} },
        { "/mlvc_version", "v0.101" },
        { "/mlvc_version/major", "0" },
        { "/mlvc_version/minor", "101" },
        { "/mlvc_version/major", MIN_SUPPORTED_MLVC_VERSION - 1, Error::invalid_argument },
        { "/mlvc_version/major", MAX_SUPPORTED_MLVC_VERSION + 1, Error::invalid_argument },
        { "/bundle_name", 42 },
        { "/model_type", "unknown" },
        { "/target_device", "unknown" },
        { "/model_registry", boost::json::array{} },
        { "/model_registry/encoder", false },
        { "/model_registry/encoder/model_path", 42 },
        { "/model_registry/encoder/weights_path", 42 },
        { "/model_manifests", boost::json::array{} },
        { "/model_manifests/320x192", nullptr },
        { "/model_manifests/320x192/model_parts", boost::json::array{} },
        { "/model_manifests/320x192/model_parts",
          boost::json::object{
              { "UnknownPart", boost::json::object{ { "registry_id", "encoder" } } },
          } },
        { "/model_manifests/320x192/model_parts/MLVCEncoder", false },
        { "/model_manifests/320x192/model_parts/MLVCEncoder/function_name", 42 },
        { "/model_manifests/320x192/encoder_interface_type", "unknown" },
        { "/model_manifests/320x192/decoder_interface_type", "unknown" },
        { "/model_metadata", boost::json::array{} },
        { "/model_metadata/320x192", nullptr },
        { "/model_metadata/320x192/model_width", "320" },
        { "/model_metadata/320x192/frame_index_map", boost::json::array{ "invalid" } },
        { "/model_metadata/320x192/y_scale_repeat", "two" },
        { "/model_metadata/320x192/qp_mapping", boost::json::array{ "invalid" } },
    };
    for (const auto& testCase : cases) {
        SCOPED_TRACE(::testing::Message() << testCase.path << " = " << boost::json::serialize(testCase.value));
        boost::json::value manifest = MakeManifest();
        manifest.at_pointer(testCase.path) = testCase.value;
        const auto result = ParseManifest(manifest);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), make_error_code(testCase.error));
    }
}

TEST(UnitTestLoadModelBundles, SelectsBundleForInferenceEngine)
{
    struct TestCase {
        InferenceEngineInfo engineInfo;
        TargetDevice targetDevice;
    };
    const TestCase cases[] = {
        { { .inferenceBackend = InferenceBackend::WINDOWSML, .onnxExecutionProvider = OnnxExecutionProvider::OPENVINO },
          TargetDevice::INTEL },
        { { .inferenceBackend = InferenceBackend::WINDOWSML, .onnxExecutionProvider = OnnxExecutionProvider::QNN },
          TargetDevice::QUALCOMM },
        { { .inferenceBackend = InferenceBackend::COREML }, TargetDevice::APPLE },
    };
    const std::array versions{ GetTestConfig().mlvcVersion };
    for (const auto& testCase : cases) {
        SCOPED_TRACE(TargetDeviceToString(testCase.targetDevice));
        const auto manifests = LoadManifests(versions, testCase.engineInfo);
        ASSERT_TRUE(manifests) << manifests.error().message();
        ASSERT_EQ(manifests->size(), 1u);
        EXPECT_EQ(manifests->front().mlvcVersion, versions.front());
        EXPECT_EQ(manifests->front().targetDevice, testCase.targetDevice);
    }
}

TEST(UnitTestLoadModelBundles, ReportsMissingVersion)
{
    const InferenceEngineInfo engineInfo{ .inferenceBackend = InferenceBackend::COREML };
    const auto manifests = LoadManifests(std::array{ MlvcVersion{ 99, 99 } }, engineInfo);
    ASSERT_FALSE(manifests);
    EXPECT_EQ(manifests.error(), make_error_code(Error::general_failure));
}
