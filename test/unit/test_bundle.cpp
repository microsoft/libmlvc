// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/bundle.hpp"
#include "libmlvc/common/tar.hpp"
#include "libmlvc/common/thread_pool.hpp"
#include "libmlvc/schema/bundle.hpp"
#include "support/test_config.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/libmlvc.hpp>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
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
    const auto bundles = ReadBundleBlobs(GetDefaultModelBundlesDir(), versions, engineInfo);
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

class ScopedTempDir {
public:
    ScopedTempDir()
        : path(std::filesystem::temp_directory_path()
               / ("libmlvc_bundles_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directory(path);
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    std::filesystem::path path;
};
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

    // Overflows the stack without Boost.JSON's depth limit
    const auto tooDeep = std::string(100000, '[') + std::string(100000, ']');
    const auto result = BundleManifest::FromJsonBuffer(std::as_bytes(std::span{ tooDeep.data(), tooDeep.size() }));
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), make_error_code(Error::json_parse_error));
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

TEST(UnitTestReadBundleBlobs, SelectsBundleForInferenceEngine)
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

TEST(UnitTestReadBundleBlobs, ReportsMissingVersion)
{
    const InferenceEngineInfo engineInfo{ .inferenceBackend = InferenceBackend::COREML };
    const auto manifests = LoadManifests(std::array{ MlvcVersion{ 99, 99 } }, engineInfo);
    ASSERT_FALSE(manifests);
    EXPECT_EQ(manifests.error(), make_error_code(Error::general_failure));
}

TEST(UnitTestReadBundleBlobs, HandlesFileNamesOutsideCodePage)
{
    // On Windows, path::string() throws for names that the ANSI code page can't represent
    const InferenceEngineInfo engineInfo{ .inferenceBackend = InferenceBackend::COREML };
    const std::array versions{ GetTestConfig().mlvcVersion };
    const auto original = ReadBundleBlobs(GetDefaultModelBundlesDir(), versions, engineInfo);
    ASSERT_TRUE(original) << original.error().message();
    ASSERT_EQ(original->size(), 1u);
    const auto& bundle = original->front();

    const ScopedTempDir dir;
    ASSERT_TRUE(std::ofstream(dir.path / std::filesystem::path(u8"\U0001F600.tar"), std::ios::binary)
                    .write(reinterpret_cast<const char*>(bundle.data()), static_cast<std::streamsize>(bundle.size()))
                    .good());
    for (const auto* name : { u8"\U0001F600\U0001F600.tar", u8"\U0001F600.txt" }) {
        ASSERT_TRUE((std::ofstream(dir.path / std::filesystem::path(name)) << "not a bundle").good());
    }
    const auto bundles = ReadBundleBlobs(dir.path, versions, engineInfo);
    ASSERT_TRUE(bundles) << bundles.error().message();
    ASSERT_EQ(bundles->size(), 1u);
    EXPECT_TRUE(bundles->front() == bundle);
}

namespace {

class FakeInferenceSession : public IInferenceSession {
public:
    expected<std::span<const std::string>> GetInputNames() override { return std::span<const std::string>{}; }
    expected<std::span<const std::string>> GetOutputNames() override { return std::span<const std::string>{}; }
    expected<InferenceTensorPtr> CreateTensor(const TensorIoType, std::string_view, const bool) override
    {
        return make_error_code(Error::general_failure);
    }
    expected<void> Run(const std::vector<InferenceTensorPtr>&, const std::vector<InferenceTensorPtr>&) override
    {
        return {};
    }
};

// Holds the creation of sessions whose name starts with the blocked prefix until Release()
class BlockingInferenceEngine : public IInferenceEngine {
public:
    struct Creation {
        std::string name;
        std::thread::id threadId;
    };

    explicit BlockingInferenceEngine(InferenceEngineInfo info) : m_info(std::move(info)) {}

    expected<InferenceSessionPtr> CreateSession(std::string_view name, std::span<const std::byte>,
                                                const std::map<std::string, std::span<const std::byte>>&,
                                                std::string_view, const std::optional<std::string>&) override
    {
        std::unique_lock lock(m_mutex);
        if (!m_failedPrefix.empty() && name.starts_with(m_failedPrefix)) {
            m_failedCreations++;
            return make_error_code(Error::general_failure);
        }
        if (!m_blockedPrefix.empty() && name.starts_with(m_blockedPrefix)) {
            m_blockedCreations++;
            m_cv.notify_all();
            m_cv.wait(lock, [this] { return m_released; });
        }
        m_creations.push_back({ .name = std::string(name), .threadId = std::this_thread::get_id() });
        m_cv.notify_all();
        return std::make_shared<FakeInferenceSession>();
    }

    expected<void> ValidateModelCache(std::string_view) const override { return {}; }

    expected<bool> PrepareModelCache(std::string_view, std::span<const std::byte>,
                                     const std::map<std::string, std::span<const std::byte>>&, std::string_view) override
    {
        return false;
    }

    const InferenceEngineInfo& GetInfo() const override { return m_info; }

    void Block(std::string prefix)
    {
        std::lock_guard lock(m_mutex);
        m_blockedPrefix = std::move(prefix);
    }

    bool WaitUntilBlocked()
    {
        std::unique_lock lock(m_mutex);
        return m_cv.wait_for(lock, std::chrono::seconds(10), [this] { return m_blockedCreations > 0; });
    }

    void Release()
    {
        {
            std::lock_guard lock(m_mutex);
            m_released = true;
        }
        m_cv.notify_all();
    }

    int GetBlockedCreations()
    {
        std::lock_guard lock(m_mutex);
        return m_blockedCreations;
    }

    void Fail(std::string prefix)
    {
        std::lock_guard lock(m_mutex);
        m_failedPrefix = std::move(prefix);
    }

    int GetFailedCreations()
    {
        std::lock_guard lock(m_mutex);
        return m_failedCreations;
    }

    // Waits up to 10 s until at least `minCount` sessions are created, and returns the creations so far
    std::vector<Creation> GetCreations(const size_t minCount = 0)
    {
        std::unique_lock lock(m_mutex);
        m_cv.wait_for(lock, std::chrono::seconds(10), [&] { return m_creations.size() >= minCount; });
        return m_creations;
    }

private:
    InferenceEngineInfo m_info;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::string m_blockedPrefix;
    int m_blockedCreations = 0;
    bool m_released = false;
    std::string m_failedPrefix;
    int m_failedCreations = 0;
    std::vector<Creation> m_creations;
};

expected<std::unique_ptr<ModelBundle>> CreateTestBundle(IInferenceEngine& engine, const SessionPreloadMode mode,
                                                        ThreadPool* sessionCreationPool = nullptr)
{
    const auto blobs =
        ReadBundleBlobs(GetDefaultModelBundlesDir(), std::array{ GetTestConfig().mlvcVersion }, engine.GetInfo());
    if (!blobs) return blobs.error();
    if (blobs->size() != 1) return make_error_code(Error::invalid_argument);
    return ModelBundle::Create(engine, sessionCreationPool, blobs->front(), false, mode, {}, nullptr);
}

std::string DecoderSessionName(const std::string& modelId)
{
    return modelId + "/" + ModelPartIdToString(ModelPartId::DECODER);
}

size_t NumSessions(const BundleManifest& manifest)
{
    size_t numSessions = 0;
    for (const auto& model : manifest.modelManifests | std::views::values) {
        numSessions += model.modelParts.size();
    }
    return numSessions;
}

class UnitTestModelBundleInBackground : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto pool = ThreadPool::Create(1);
        ASSERT_TRUE(pool) << pool.error().message();
        m_pool = std::move(pool.value());
        auto bundle = CreateTestBundle(m_engine, SessionPreloadMode::ON_FIRST_USE, m_pool.get());
        ASSERT_TRUE(bundle) << bundle.error().message();
        m_bundle = std::move(bundle.value());
    }

    BlockingInferenceEngine m_engine{ InferenceEngineInfo{ .inferenceBackend = InferenceBackend::WINDOWSML,
                                                           .onnxExecutionProvider = OnnxExecutionProvider::CPU } };
    std::unique_ptr<ModelBundle> m_bundle;
    std::unique_ptr<ThreadPool> m_pool;  // Declared after the bundle, so it stops before the bundle is destroyed

    int Area(const std::string& modelId) const
    {
        const auto& metadata = m_bundle->GetManifest().modelMetadata.at(modelId);
        return metadata->modelWidth * metadata->modelHeight;
    }

    std::string SmallestModelId() const
    {
        return std::ranges::min(m_bundle->GetManifest().modelManifests | std::views::keys, {},
                                [this](const std::string& id) { return Area(id); });
    }
    std::string LargestModelId() const
    {
        return std::ranges::max(m_bundle->GetManifest().modelManifests | std::views::keys, {},
                                [this](const std::string& id) { return Area(id); });
    }
};

}  // namespace

TEST(UnitTestModelBundle, SessionPreloadModeStrings)
{
    for (const auto& [mode, name] : { std::pair{ SessionPreloadMode::AT_INIT, "at-init" },
                                      std::pair{ SessionPreloadMode::ON_FIRST_USE, "on-first-use" },
                                      std::pair{ SessionPreloadMode::NONE, "none" } }) {
        EXPECT_STREQ(SessionPreloadModeToString(mode), name);
        const auto parsed = StringToSessionPreloadMode(name);
        ASSERT_TRUE(parsed) << parsed.error().message();
        EXPECT_EQ(parsed.value(), mode);
    }

    const auto unknown = StringToSessionPreloadMode("unknown");
    ASSERT_FALSE(unknown);
    EXPECT_EQ(unknown.error(), make_error_code(Error::invalid_argument));
}

TEST(UnitTestModelBundle, CreatesSessionsWithoutBlockingOtherRequests)
{
    BlockingInferenceEngine engine(
        { .inferenceBackend = InferenceBackend::WINDOWSML, .onnxExecutionProvider = OnnxExecutionProvider::CPU });
    const auto bundle = CreateTestBundle(engine, SessionPreloadMode::NONE);
    ASSERT_TRUE(bundle) << bundle.error().message();

    const auto& models = bundle.value()->GetManifest().modelManifests;
    ASSERT_GE(models.size(), 2u);
    const auto& readyModelId = models.begin()->first;
    const auto& blockedModelId = std::next(models.begin())->first;
    const auto ready = bundle.value()->GetInferenceSession(readyModelId, ModelPartId::DECODER);
    ASSERT_TRUE(ready) << ready.error().message();

    engine.Block(blockedModelId + "/");
    auto blockedRequests = std::vector<std::future<expected<InferenceSessionPtr>>>{};
    for (int i = 0; i < 2; i++) {
        blockedRequests.push_back(std::async(std::launch::async, [&] {
            return bundle.value()->GetInferenceSession(blockedModelId, ModelPartId::DECODER);
        }));
    }
    EXPECT_TRUE(engine.WaitUntilBlocked());

    auto readyRequest = std::async(
        std::launch::async, [&] { return bundle.value()->GetInferenceSession(readyModelId, ModelPartId::DECODER); });
    const auto readyReturned = readyRequest.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    engine.Release();
    EXPECT_TRUE(readyReturned) << "Request for a created session waited for another session's creation";
    const auto readyAgain = readyRequest.get();
    ASSERT_TRUE(readyAgain) << readyAgain.error().message();
    EXPECT_EQ(readyAgain.value(), ready.value());

    auto blockedSessions = std::vector<InferenceSessionPtr>{};
    for (auto& request : blockedRequests) {
        const auto session = request.get();
        ASSERT_TRUE(session) << session.error().message();
        blockedSessions.push_back(session.value());
    }
    EXPECT_EQ(blockedSessions.front(), blockedSessions.back());
    EXPECT_EQ(engine.GetBlockedCreations(), 1);
}

TEST(UnitTestModelBundle, CreatesSessionsAtInitOrOnDemand)
{
    const InferenceEngineInfo info{ .inferenceBackend = InferenceBackend::WINDOWSML,
                                    .onnxExecutionProvider = OnnxExecutionProvider::CPU };
    BlockingInferenceEngine atInitEngine(info);
    const auto atInit = CreateTestBundle(atInitEngine, SessionPreloadMode::AT_INIT);
    ASSERT_TRUE(atInit) << atInit.error().message();
    EXPECT_EQ(atInitEngine.GetCreations().size(), NumSessions(atInit.value()->GetManifest()));

    BlockingInferenceEngine onDemandEngine(info);
    const auto onDemand = CreateTestBundle(onDemandEngine, SessionPreloadMode::NONE);
    ASSERT_TRUE(onDemand) << onDemand.error().message();
    EXPECT_TRUE(onDemandEngine.GetCreations().empty());
    const auto& modelId = onDemand.value()->GetManifest().modelManifests.begin()->first;
    const auto session = onDemand.value()->GetInferenceSession(modelId, ModelPartId::DECODER);
    ASSERT_TRUE(session) << session.error().message();
    const auto creations = onDemandEngine.GetCreations();
    ASSERT_EQ(creations.size(), 1u);
    EXPECT_TRUE(creations.front().name.starts_with(DecoderSessionName(modelId)));
    EXPECT_EQ(creations.front().threadId, std::this_thread::get_id());
}

TEST(UnitTestModelBundle, RequiresSessionCreationPoolInBackground)
{
    BlockingInferenceEngine engine(
        { .inferenceBackend = InferenceBackend::WINDOWSML, .onnxExecutionProvider = OnnxExecutionProvider::CPU });
    const auto blobs =
        ReadBundleBlobs(GetDefaultModelBundlesDir(), std::array{ GetTestConfig().mlvcVersion }, engine.GetInfo());
    ASSERT_TRUE(blobs) << blobs.error().message();
    ASSERT_EQ(blobs->size(), 1u);
    const auto bundle =
        ModelBundle::Create(engine, nullptr, blobs->front(), false, SessionPreloadMode::ON_FIRST_USE, {}, nullptr);
    ASSERT_FALSE(bundle);
    EXPECT_EQ(bundle.error(), make_error_code(Error::invalid_argument));
}

TEST(UnitTestModelBundle, QueuesRequestedSessionFirst)
{
    // A pool without threads creates each session as it's queued, so the creations show the queue's order
    const auto pool = ThreadPool::Create(0);
    ASSERT_TRUE(pool) << pool.error().message();
    BlockingInferenceEngine engine(
        { .inferenceBackend = InferenceBackend::WINDOWSML, .onnxExecutionProvider = OnnxExecutionProvider::CPU });
    const auto bundle = CreateTestBundle(engine, SessionPreloadMode::ON_FIRST_USE, pool.value().get());
    ASSERT_TRUE(bundle) << bundle.error().message();

    // An unknown session fails without queueing anything
    const auto unknown = bundle.value()->GetInferenceSession("unknown", ModelPartId::DECODER);
    ASSERT_FALSE(unknown);
    EXPECT_EQ(unknown.error(), make_error_code(Error::invalid_argument));
    EXPECT_TRUE(engine.GetCreations().empty());

    // As a decoder's first frame does, a request before any preload queues every session, its own first
    const auto& modelId = bundle.value()->GetManifest().modelManifests.begin()->first;
    const auto session = bundle.value()->GetInferenceSession(modelId, ModelPartId::DECODER);
    ASSERT_TRUE(session) << session.error().message();
    const auto creations = engine.GetCreations();
    ASSERT_EQ(creations.size(), NumSessions(bundle.value()->GetManifest()));
    EXPECT_TRUE(creations.front().name.starts_with(DecoderSessionName(modelId))) << creations.front().name;
}

TEST_F(UnitTestModelBundleInBackground, CreatesSessionsLargestFirst)
{
    // Every session of the bundle is created on one background thread, largest resolution first, and a model's
    // encoder before its decoder
    m_bundle->PreloadSessionsInBackground();
    const auto& manifest = m_bundle->GetManifest();
    const auto numSessions = NumSessions(manifest);
    const auto creations = m_engine.GetCreations(numSessions);
    ASSERT_EQ(creations.size(), numSessions);
    std::vector<std::string> createdNames;
    for (const auto& creation : creations) {
        EXPECT_NE(creation.threadId, std::this_thread::get_id());
        EXPECT_EQ(creation.threadId, creations.front().threadId);
        createdNames.push_back(creation.name);
    }
    const auto area = [this](const std::string& sessionName) {
        return Area(sessionName.substr(0, sessionName.find('/')));
    };
    EXPECT_TRUE(std::ranges::is_sorted(createdNames, std::ranges::greater{}, area))
        << ::testing::PrintToString(createdNames);
    for (const auto& modelId : manifest.modelManifests | std::views::keys) {
        const auto created = [&](const ModelPartId partId) {
            return std::ranges::find_if(createdNames, [&](const std::string& name) {
                return name.starts_with(modelId + "/" + ModelPartIdToString(partId));
            });
        };
        EXPECT_LT(created(ModelPartId::ENCODER), created(ModelPartId::DECODER)) << modelId;
    }
}

TEST_F(UnitTestModelBundleInBackground, RunsRequestedSessionNext)
{
    const auto smallest = SmallestModelId();
    const auto largest = LargestModelId();
    ASSERT_NE(largest, smallest);

    // Hold the first creation, of the largest encoder, so the other sessions stay queued
    m_engine.Block(largest + "/");
    m_bundle->PreloadSessionsInBackground();
    ASSERT_TRUE(m_engine.WaitUntilBlocked());

    // The request has time to move its session to the front of the queue before the held creation ends
    auto request =
        std::async(std::launch::async, [&] { return m_bundle->GetInferenceSession(smallest, ModelPartId::DECODER); });
    EXPECT_EQ(request.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    m_engine.Release();
    const auto session = request.get();
    ASSERT_TRUE(session) << session.error().message();

    const auto creations = m_engine.GetCreations(2);
    ASSERT_GE(creations.size(), 2u);
    EXPECT_TRUE(creations[0].name.starts_with(largest + "/")) << creations[0].name;
    EXPECT_TRUE(creations[1].name.starts_with(DecoderSessionName(smallest))) << creations[1].name;
}

TEST_F(UnitTestModelBundleInBackground, QueuesEachSessionOnce)
{
    const auto& manifest = m_bundle->GetManifest();
    const auto& modelId = manifest.modelManifests.begin()->first;

    auto preload = std::async(std::launch::async, [&] { m_bundle->PreloadSessionsInBackground(); });
    auto another = std::async(std::launch::async, [&] { m_bundle->PreloadSessionsInBackground(); });
    auto request =
        std::async(std::launch::async, [&] { return m_bundle->GetInferenceSession(modelId, ModelPartId::DECODER); });
    preload.get();
    another.get();
    const auto session = request.get();
    ASSERT_TRUE(session) << session.error().message();
    m_bundle->PreloadSessionsInBackground();

    // The pool's one thread runs a LOW-priority task after every task queued before it
    const auto drained = m_pool->Submit("drained", []() -> expected<void> { return {}; }, ThreadPool::Priority::LOW);
    ASSERT_EQ(drained->Wait(), ThreadPool::Task::Status::COMPLETED);
    std::vector<std::string> createdNames;
    for (const auto& creation : m_engine.GetCreations()) {
        createdNames.push_back(creation.name);
    }
    EXPECT_EQ(createdNames.size(), NumSessions(manifest)) << ::testing::PrintToString(createdNames);
    std::ranges::sort(createdNames);
    EXPECT_EQ(std::ranges::adjacent_find(createdNames), createdNames.end()) << ::testing::PrintToString(createdNames);
}

TEST_F(UnitTestModelBundleInBackground, DoesNotRetryFailedSession)
{
    const auto& modelId = m_bundle->GetManifest().modelManifests.begin()->first;

    m_engine.Fail(DecoderSessionName(modelId));
    for (int i = 0; i < 2; i++) {
        const auto session = m_bundle->GetInferenceSession(modelId, ModelPartId::DECODER);
        ASSERT_FALSE(session);
        EXPECT_EQ(session.error(), make_error_code(Error::general_failure));
    }
    EXPECT_EQ(m_engine.GetFailedCreations(), 1);
}

TEST_F(UnitTestModelBundleInBackground, CancelsQueuedSessionsOnShutdown)
{
    const auto smallest = SmallestModelId();
    const auto largest = LargestModelId();

    // Hold the first creation, of the largest encoder, so the other sessions stay queued
    m_engine.Block(largest + "/");
    m_bundle->PreloadSessionsInBackground();
    ASSERT_TRUE(m_engine.WaitUntilBlocked());
    auto request =
        std::async(std::launch::async, [&] { return m_bundle->GetInferenceSession(smallest, ModelPartId::DECODER); });

    // Stopping the pool cancels the queued sessions, which ends the request, then waits for the held creation
    auto stopping = std::async(std::launch::async, [&] { m_pool.reset(); });
    const auto session = request.get();
    m_engine.Release();
    stopping.get();
    ASSERT_FALSE(session);
    EXPECT_EQ(session.error(), make_error_code(Error::operation_cancelled));
    const auto creations = m_engine.GetCreations();
    ASSERT_EQ(creations.size(), 1u);
    EXPECT_TRUE(creations.front().name.starts_with(largest + "/")) << creations.front().name;
}
