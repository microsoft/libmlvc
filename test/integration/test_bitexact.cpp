// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/codec/codec.hpp"
#include "libmlvc/codec/core/core.hpp"
#include "libmlvc/codec/manager.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/platform_names.hpp"
#include "libmlvc_support/compress.hpp"
#include "libmlvc_support/encoder_overrides.hpp"
#include "libmlvc_support/hash.hpp"
#include "libmlvc_support/mlvc_io.hpp"
#include "libmlvc_support/test_data.hpp"
#include "libmlvc_support/video_io.hpp"
#include "support/recovery_controller.hpp"
#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <span>
#include <vector>

using namespace libmlvc;

namespace {

expected<std::vector<std::span<const std::byte>>> SplitToNalus(std::span<const std::byte> buffer)
{
    static constexpr size_t startCodeLength = 4;
    const auto isNaluStartCode = [buffer](const size_t pos) {
        return pos + startCodeLength <= buffer.size() && buffer[pos] == std::byte{ 0x00 }
               && buffer[pos + 1] == std::byte{ 0x00 } && buffer[pos + 2] == std::byte{ 0x00 }
               && buffer[pos + 3] == std::byte{ 0x01 };
    };

    std::vector<std::span<const std::byte>> nalus;
    size_t offset = 0;
    while (offset < buffer.size()) {
        if (!isNaluStartCode(offset)) {
            std::cerr << "Error: Failed to find NALU start code at offset " << offset << '\n';
            return make_error_code(Error::io_error);
        }

        size_t nextOffset = offset + startCodeLength;
        while (nextOffset < buffer.size() && !isNaluStartCode(nextOffset)) {
            nextOffset++;
        }
        nalus.emplace_back(buffer.data() + offset, nextOffset - offset);
        offset = nextOffset;
    }
    return nalus;
}

// -----------------------------------------------------------------------------
// JSON utilities
// -----------------------------------------------------------------------------

template <typename T>
inline T ParseJson(const boost::json::value& v)
{
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>) {
        if (!v.is_int64()) {
            MLVC_LOG_ABORT("Failed to parse json int64");
        }
        return static_cast<T>(v.as_int64());
    } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        if (!v.is_double() && !v.is_int64()) {
            MLVC_LOG_ABORT("Failed to parse json number");
        }
        return static_cast<T>(v.is_double() ? v.as_double() : static_cast<double>(v.as_int64()));
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!v.is_string()) {
            MLVC_LOG_ABORT("Failed to parse json string");
        }
        return std::string(v.as_string().c_str());
    } else if constexpr (std::is_same_v<T, boost::json::object>) {
        if (!v.is_object()) {
            MLVC_LOG_ABORT("Failed to parse json object");
        }
        return v.as_object();
    } else if constexpr (std::is_same_v<T, boost::json::array>) {
        if (!v.is_array()) {
            MLVC_LOG_ABORT("Failed to parse json array");
        }
        return v.as_array();
    } else {
        static_assert(sizeof(T) == 0, "Unsupported ParseJson<T> specialization");
    }
}

template <typename T>
inline T ParseJson(const boost::json::object& obj, const std::string& key)
{
    auto it = obj.find(key);
    if (it == obj.end()) {
        MLVC_LOG_ABORT("Key not found: %s", key.c_str());
    }
    return ParseJson<T>(it->value());
}

// -----------------------------------------------------------------------------
// Hashing and tensor utilities
// -----------------------------------------------------------------------------

template <typename T, int N>
void ToContiguous(const Tensor<T, N>& input, Tensor<T, N>& output)
{
    const std::array<int, N>& shape = input.Shape();
    output.Create(shape, StridesFromShape<N>(shape));
    if constexpr (N == 2) {
        for (int y = 0; y < shape[0]; y++) {
            for (int x = 0; x < shape[1]; x++) {
                output(y, x) = input(y, x);
            }
        }
    } else if constexpr (N == 3) {
        for (int ch = 0; ch < shape[0]; ch++) {
            for (int y = 0; y < shape[1]; y++) {
                for (int x = 0; x < shape[2]; x++) {
                    output(ch, y, x) = input(ch, y, x);
                }
            }
        }
    } else if constexpr (N == 4) {
        for (int ch = 0; ch < shape[0]; ch++) {
            for (int y = 0; y < shape[1]; y++) {
                for (int x = 0; x < shape[2]; x++) {
                    for (int z = 0; z < shape[3]; z++) {
                        output(ch, y, x, z) = input(ch, y, x, z);
                    }
                }
            }
        }
    } else {
        static_assert(N <= 4, "Not implemented");
    }
}

template <typename T, int N>
std::span<const T> GetContiguousData(const Tensor<T, N>& tensor, Tensor<T, N>& contiguous)
{
    if (!tensor.IsContiguous()) {
        ToContiguous(tensor, contiguous);
        return contiguous.Data();
    }
    return tensor.Data();
}

inline std::string HashSha256(std::span<const std::byte> bytes)
{
    Sha256 hash;
    if (auto res = hash.Initialize(); !res) {
        MLVC_LOG_ABORT("Failed to initialize SHA256 hash: %s", res.error().message().c_str());
    }

    if (auto ret = hash.Update(bytes.data(), bytes.size()); !ret) {
        MLVC_LOG_ABORT("Failed to update SHA256 hash: %s", ret.error().message().c_str());
    }

    auto res = hash.Finalize();
    if (!res) {
        MLVC_LOG_ABORT("Failed to finalize SHA256 hash: %s", res.error().message().c_str());
    }
    return res.value();
}

template <typename T, int N>
std::string HashSha256(const Tensor<T, N>& tensor)
{
    Tensor<T, N> contiguous;
    const auto data = GetContiguousData(tensor, contiguous);
    return HashSha256({ reinterpret_cast<const std::byte*>(data.data()), data.size_bytes() });
}

// NVIDIA GPU models can differ; this opt-out allows regenerating platform-specific test data.
inline bool SkipBitExactness()
{
    static const bool skipBitExactness = []() {
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)  // disable getenv unsafe warning for windows in test code
#endif
        const char* env = std::getenv("MLVC_SKIP_BIT_EXACTNESS");
        return env != nullptr && std::string(env) == "1";
#ifdef _MSC_VER
    #pragma warning(pop)
#endif
    }();
    return skipBitExactness;
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

class TensorCheck {
public:
    TensorCheck(const boost::json::object& obj)
        : m_itemsize{ static_cast<size_t>(ParseJson<int>(obj, "itemsize")) }
        , m_size{ static_cast<size_t>(ParseJson<int>(obj, "size")) }
        , m_sum{ ParseJson<double>(obj, "sum") }
        , m_sha256{ ParseJson<std::string>(obj, "sha256") }
    {
    }

    template <typename T, int N>
    void Validate(const Tensor<T, N>& tensor) const
    {
        double sum = 0.0;
        if constexpr (std::is_same_v<T, uint16_t>) {
            // Float16
            for (auto v : tensor.Data()) {
                sum += static_cast<double>(Fp16To32(v));
            }
        } else {
            for (auto v : tensor.Data()) {
                sum += static_cast<double>(v);
            }
        }

        ASSERT_EQ(sizeof(T), m_itemsize);
        ASSERT_EQ(tensor.Size(), m_size);
        if (!SkipBitExactness()) {
            ASSERT_DOUBLE_EQ(sum, m_sum);
            ASSERT_EQ(HashSha256(tensor), m_sha256);
        } else {
            // Approximate check using atol + rtol * |reference| tolerance model
            // rtol: relative tolerance, atol: absolute tolerance per element
            // For sums, atol scales with element count to account for accumulated per-element errors
            // Using looser tolerances to accommodate GPU numerical differences
            constexpr double rtol = 0.10;  // 10% relative tolerance
            constexpr double atol = 0.05;  // absolute tolerance per element
            const double tolerance = static_cast<double>(m_size) * atol + rtol * std::abs(m_sum);
            const double diff = std::abs(sum - m_sum);
            const char* typeName = (m_itemsize == 2) ? "float16" : (m_itemsize == 4) ? "float32" : "unknown";
            ASSERT_LE(diff, tolerance) << "Sum mismatch: diff=" << diff << ", allowed=" << tolerance
                                       << ", actual=" << sum << ", expected=" << m_sum << ", size=" << m_size
                                       << ", type=" << typeName << ", tensor=" << tensor.Name();
        }
    }

private:
    size_t m_itemsize;
    size_t m_size;
    double m_sum;
    std::string m_sha256;
};

class BitStreamCheck {
public:
    BitStreamCheck(const boost::json::object& obj)
        : m_itemsize{ static_cast<size_t>(ParseJson<int>(obj, "itemsize")) }
        , m_size{ static_cast<size_t>(ParseJson<int>(obj, "size")) }
        , m_sum{ ParseJson<double>(obj, "sum") }
        , m_sha256{ ParseJson<std::string>(obj, "sha256") }
    {
    }

    void Validate(std::span<const std::byte> bytes) const
    {
        double sum = 0.0;
        for (const auto b : bytes) {
            sum += static_cast<double>(b);
        }

        ASSERT_EQ(1, m_itemsize);
        if (!SkipBitExactness()) {
            ASSERT_EQ(bytes.size(), m_size);
            ASSERT_DOUBLE_EQ(sum, m_sum);
            ASSERT_EQ(HashSha256(bytes), m_sha256);
        } else {
            // Approximate check: verify bitstream size is within tolerance
            // Size can vary significantly due to entropy coding differences on different GPUs
            // Use the larger of: 25% relative tolerance OR minimum 20 bytes for small bitstreams
            constexpr double sizeRtol = 0.25;
            constexpr double minAbsTolerance = 20.0;
            const double sizeTolerance = std::max(minAbsTolerance, sizeRtol * static_cast<double>(m_size));
            const double sizeDiff = std::abs(static_cast<double>(bytes.size()) - static_cast<double>(m_size));
            ASSERT_LE(sizeDiff, sizeTolerance)
                << "Bitstream size mismatch: diff=" << sizeDiff << ", allowed=" << sizeTolerance
                << ", actual=" << bytes.size() << ", expected=" << m_size;
        }
    }

private:
    size_t m_itemsize;
    size_t m_size;
    double m_sum;
    std::string m_sha256;
};

class TensorCheckGroup {
public:
    TensorCheckGroup(const boost::json::object& obj)
    {
        for (const auto& kv : obj) {
            const auto& name = kv.key();
            const auto& value = kv.value();
            m_checks.insert({ std::string(name), TensorCheck{ ParseJson<boost::json::object>(value) } });
        }
    }

    void Validate(const MlvcEncoderModel::Inputs& inputs) const
    {
        ASSERT_NO_FATAL_FAILURE(Test(inputs.x));
        ASSERT_NO_FATAL_FAILURE(Test(inputs.refFeature));
        ASSERT_NO_FATAL_FAILURE(Test(inputs.qIndexShifted));
    }

    void Validate(const MlvcEncoderModel::Outputs& outputs) const
    {
        ASSERT_NO_FATAL_FAILURE(Test(outputs.feature));
        ASSERT_NO_FATAL_FAILURE(Test(outputs.zRaw));
        ASSERT_NO_FATAL_FAILURE(Test(outputs.yRaw0));
        ASSERT_NO_FATAL_FAILURE(Test(outputs.yRaw1));
    }

    void Validate(const ScaleValues& scales) const
    {
        ASSERT_NO_FATAL_FAILURE(Test(scales.scales0));
        ASSERT_NO_FATAL_FAILURE(Test(scales.scales1));
    }

    void Validate(const MlvcDecoderModel::Inputs& inputs) const
    {
        ASSERT_NO_FATAL_FAILURE(Test(inputs.zRaw));
        ASSERT_NO_FATAL_FAILURE(Test(inputs.yRaw0));
        ASSERT_NO_FATAL_FAILURE(Test(inputs.yRaw1));
        ASSERT_NO_FATAL_FAILURE(Test(inputs.refFeature));
        ASSERT_NO_FATAL_FAILURE(Test(inputs.qIndexShifted));
    }

    void Validate(const MlvcDecoderModel::Outputs& outputs) const
    {
        ASSERT_NO_FATAL_FAILURE(Test(outputs.xHat));
        ASSERT_NO_FATAL_FAILURE(Test(outputs.feature));
    }

    void Validate(std::string_view name, std::span<const std::byte> bytes) const
    {
        // NOLINTNEXTLINE: const_cast needed because Tensor takes std::span<T>, not std::span<const T>
        std::span<uint8_t> tmp{ reinterpret_cast<uint8_t*>(const_cast<std::byte*>(bytes.data())), bytes.size() };
        Tensor<uint8_t, 1> tensor{ { static_cast<int>(bytes.size()) }, { 1 }, tmp, name };
        ASSERT_NO_FATAL_FAILURE(Test(tensor));
    }

    template <typename T, int N>
    void Test(const Tensor<T, N>& tensor) const
    {
        auto it = m_checks.find(tensor.Name());
        if (it == m_checks.end()) {
            FAIL() << "No checks for tensor: " << tensor.Name();
        }
        ASSERT_NO_FATAL_FAILURE(it->second.Validate(tensor));
    }

protected:
    std::map<std::string, TensorCheck> m_checks;
};

class EncoderChecks {
public:
    EncoderChecks(const boost::json::object& obj)
        : m_encoderInputChecks{ ParseJson<boost::json::object>(obj, "input") }
        , m_encoderOutputChecks{ ParseJson<boost::json::object>(obj, "output") }
        , m_coderChecks{ ParseJson<boost::json::object>(obj, "coder") }
        , m_bitStreamCheck{ ParseJson<boost::json::object>(obj, "bit_stream") }
    {
    }

    void Validate(const MlvcEncoderCore& encoder, std::span<const std::byte> bitStream) const
    {
        ASSERT_NO_FATAL_FAILURE(m_encoderInputChecks.Validate(encoder.GetModelInputs()));
        ASSERT_NO_FATAL_FAILURE(m_encoderOutputChecks.Validate(encoder.GetModelOutputs()));
        ASSERT_NO_FATAL_FAILURE(m_coderChecks.Validate(encoder.GetScales()));
        ASSERT_NO_FATAL_FAILURE(m_bitStreamCheck.Validate(bitStream));
    }

private:
    TensorCheckGroup m_encoderInputChecks;
    TensorCheckGroup m_encoderOutputChecks;
    TensorCheckGroup m_coderChecks;
    BitStreamCheck m_bitStreamCheck;
};

class DecoderChecks {
public:
    DecoderChecks(const boost::json::object& obj)
        : m_decoderInputChecks{ ParseJson<boost::json::object>(obj, "input") }
        , m_decoderOutputChecks{ ParseJson<boost::json::object>(obj, "output") }
        , m_outputFrameChecks{ ParseJson<boost::json::object>(obj, "output_frame") }
    {
    }

    void Validate(const MlvcDecoderCore& decoder, const DecodedFrame& decodedFrame) const
    {
        ASSERT_NO_FATAL_FAILURE(m_decoderInputChecks.Validate(decoder.GetModelInputs()));
        ASSERT_NO_FATAL_FAILURE(m_decoderOutputChecks.Validate(decoder.GetModelOutputs()));

        ASSERT_EQ(decodedFrame.frame.YPlane().data() + decodedFrame.frame.YPlane().size(),
                  decodedFrame.frame.UvPlane().data())
            << "NV12 plane data not contiguous";
        std::span<const std::byte> frameDataSpan{ reinterpret_cast<const std::byte*>(decodedFrame.frame.YPlane().data()),
                                                  decodedFrame.frame.YPlane().size() + decodedFrame.frame.UvPlane().size() };
        ASSERT_NO_FATAL_FAILURE(m_outputFrameChecks.Validate(frameDataSpan));
    }

private:
    TensorCheckGroup m_decoderInputChecks;
    TensorCheckGroup m_decoderOutputChecks;
    BitStreamCheck m_outputFrameChecks;
};

// -----------------------------------------------------------------------------
// Test driver
// -----------------------------------------------------------------------------

inline std::string GetDefaultReferenceName()
{
#if defined(MLVC_PLATFORM_APPLE) && defined(MLVC_ARCH_ARM64)
    // Apple Silicon
    if (GetPlatformInfo().cpuSeries == CPU_SERIES_M1 || GetPlatformInfo().cpuSeries == CPU_SERIES_M2
        || GetPlatformInfo().cpuSeries == CPU_SERIES_M3 || GetPlatformInfo().cpuSeries == CPU_SERIES_M4) {
        return "apple_m1_npu";
    }
    return "apple_m5_npu";
#elif defined(MLVC_PLATFORM_WINCLASSIC) && defined(MLVC_ARCH_ARM64)
    // Qualcomm NPU
    return "qualcomm_npu";
#elif defined(MLVC_PLATFORM_WINCLASSIC) && defined(MLVC_ARCH_X86_64)
    const auto& platform = GetPlatformInfo();
    if (platform.cpuVendor == VENDOR_INTEL && platform.cpuSeries == CPU_SERIES_LUNAR_LAKE) {
        return "intel_npu4";
    } else if (platform.cpuVendor == VENDOR_INTEL && platform.cpuSeries == CPU_SERIES_PANTHER_LAKE) {
        return "intel_npu5";
    } else {
        // NVIDIA GPU fallback
        return "nvidia_gpu";
    }
#endif
    return "";
}

inline size_t PingPongSourceIndex(size_t outputIdx, size_t numClipFrames)
{
    if (numClipFrames <= 1) {
        return 0;
    }
    const size_t period = 2 * (numClipFrames - 1);
    const size_t j = outputIdx % period;
    return j < numClipFrames ? j : period - j;
}

class BitExactTestRunner {
public:
    BitExactTestRunner(std::string_view testName, std::string_view filename, const int qp,
                       std::shared_ptr<MlvcManagerImpl> manager, const MlvcVersion& mlvcVersion, int width, int height,
                       const EncoderConfigOverrides& overrides, const std::filesystem::path& dataDir,
                       std::string_view referenceName)
        : m_testName{ testName }
        , m_inputVideoName{ filename }
        , m_manager{ std::move(manager) }
        , m_dataDir{ dataDir }
        , m_referenceName{ referenceName }
        , m_qp{ qp }
        , m_mlvcVersion{ mlvcVersion }
        , m_width{ width }
        , m_height{ height }
        , m_overrides{ overrides }
    {
        m_referenceDataDir = m_dataDir / "bitexact" / m_mlvcVersion.ToString() / referenceName / m_testName;
    }

    void SetLostFrameIds(const std::set<int>& frameIds) { m_lostFrameIds = frameIds; }

    void SetNumTemporalLayersSchedule(const std::map<int, int>& schedule) { m_numTemporalLayersSchedule = schedule; }

    template <typename T>
    expected<std::vector<T>> LoadTensorChecks(std::string_view name) const
    {
        auto checksPath = m_referenceDataDir / (std::string(name) + ".json.gz");
        MLVC_LOG_INFO("Read checks (%s)...", checksPath.string().c_str());

        const auto data = ReadGzipFile(checksPath);
        if (!data) {
            MLVC_LOG_INFO("Failed to read file: %s", checksPath.string().c_str());
            return data.error();
        }

        std::error_code ec;
        auto parsed = boost::json::parse({ reinterpret_cast<const char*>(data->data()), data->size() }, ec);
        if (ec) {
            MLVC_LOG_ERROR("Failed to parse JSON data: %s", ec.message().c_str());
            return make_error_code(Error::json_parse_error);
        }
        if (!parsed.is_array()) {
            MLVC_LOG_ERROR("Expected top-level JSON array for checks");
            return make_error_code(Error::json_parse_error);
        }
        const auto& framesVar = parsed.as_array();
        std::vector<T> res;
        res.reserve(framesVar.size());
        for (const auto& frameVal : framesVar) {
            if (!frameVal.is_object()) {
                MLVC_LOG_ERROR("Frame element not an object");
                return make_error_code(Error::json_parse_error);
            }
            res.push_back({ frameVal.as_object() });
        }
        return res;
    }

    void TestEncoder() const
    {
        // Read input video
        auto videoPath = m_dataDir / "clips" / m_inputVideoName;

        MLVC_LOG_INFO("Read frames (%s)...", videoPath.string().c_str());
        auto frames = LoadNv12Frames(videoPath, { .rawFrameWidth = m_width, .rawFrameHeight = m_height });
        if (!frames) {
            MLVC_LOG_INFO("Failed to read video: %s", videoPath.string().c_str());
            FAIL() << "Failed to read video: " << videoPath;
        }

        // Load checks
        const auto frameChecks = LoadTensorChecks<EncoderChecks>("encoder_checks");
        if (!frameChecks) {
            MLVC_LOG_INFO("Failed to load checks: %s", frameChecks.error().message().c_str());
            FAIL() << "Failed to load checks: " << frameChecks.error().message();
        }

        RecoveryController recoveryController(m_lostFrameIds);

        auto config = m_manager->GetDefaultEncoderConfig(m_mlvcVersion);
        if (!config) {
            MLVC_LOG_ABORT("Failed to get default encoder config: %s", config.error().message().c_str());
        }
        config.value().SetSize(m_width, m_height);
        m_overrides.Apply(config.value());

        auto encoder = m_manager->CreateEncoder(config.value());
        if (!encoder) {
            MLVC_LOG_ABORT("Failed to create encoder: %s", encoder.error().message().c_str());
        }

        // Parser for extracting LTR slots from bitstream
        MlvcParser parser;

        for (int frameId = 0; frameId < static_cast<int>(frameChecks.value().size()); frameId++) {
            MLVC_LOG_DEBUG("Processing frame %d ...", frameId);

            if (auto it = m_numTemporalLayersSchedule.find(frameId); it != m_numTemporalLayersSchedule.end()) {
                config.value().numTemporalLayers = it->second;
                ASSERT_TRUE(encoder.value()->Configure(config.value()));
            }

            const auto recoveryInfo = recoveryController.ComputeRecovery(frameId);

            const size_t srcIdx = PingPongSourceIndex(static_cast<size_t>(frameId), frames.value().size());
            const Nv12FrameView frame{ m_width, m_height, frames.value()[srcIdx].data };
            auto encodedFrame = encoder.value()->Encode(frame, {
                                                                   .qp = m_qp,
                                                                   .forceIdr = recoveryInfo.idr,
                                                                   .useLtrSlotIdx = recoveryInfo.useLtrSlotIdx,
                                                               });
            ASSERT_TRUE(encodedFrame) << "Failed to encode frame: " << frameId << ": " << encodedFrame.error().message();

            const auto& encoderCore = encoder.value()->GetCore();

            // Parse bitstream to get LTR slots
            auto frameData = parser.Parse(encodedFrame.value().bitStream);
            ASSERT_TRUE(frameData) << "Failed to parse bitstream: " << frameData.error().message();
            recoveryController.Update(frameId, frameData.value().ltrSlots);

            const auto& checks = frameChecks.value()[frameId];
            switch (encoderCore.GetEncoderInterfaceType()) {
            case EncoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P:
                ASSERT_NO_FATAL_FAILURE(
                    checks.Validate(static_cast<const MlvcEncoderCore&>(encoderCore), encodedFrame.value().bitStream))
                    << "Encoder checks failed at frame " << frameId;
                break;
            default:
                MLVC_LOG_ABORT("Unknown encoder core type");
            }
        }
    }

    void TestDecoder() const
    {
        // Read input video
        auto videoPath = m_referenceDataDir / "output.mlvc";

        MLVC_LOG_INFO("Read frames (%s)...", videoPath.string().c_str());
        auto frames = ReadMlvcAccessUnits(videoPath);
        ASSERT_TRUE(frames) << "Failed to read MLVC access units: " << videoPath;
        MLVC_LOG_INFO("Read frames: %zu", frames.value().size());

        // Load checks
        const auto frameChecks = LoadTensorChecks<DecoderChecks>("decoder_checks");
        if (!frameChecks) {
            MLVC_LOG_INFO("Failed to load checks: %s", frameChecks.error().message().c_str());
            FAIL() << "Failed to load checks: " << frameChecks.error().message();
        }

        auto decoder = m_manager->CreateDecoder();
        if (!decoder) {
            MLVC_LOG_ABORT("Failed to create decoder: %s", decoder.error().message().c_str());
        }

        auto Validate = [&decoder, &frameChecks](const expected<DecodedFrame>& decodedFrame, const size_t frameId) {
            const auto& decoderCore = decoder.value()->GetCore();
            const auto& checks = frameChecks.value()[frameId];
            ASSERT_TRUE(decodedFrame) << "Failed to decode frame: " << frameId << ": " << decodedFrame.error().message();

            switch (decoderCore.GetDecoderInterfaceType()) {
            case DecoderInterfaceType::FP16_SCALE_SENDING_NO_RESET_1P:
                ASSERT_NO_FATAL_FAILURE(
                    checks.Validate(static_cast<const MlvcDecoderCore&>(decoderCore), decodedFrame.value()));
                break;
            default:
                MLVC_LOG_ABORT("Unknown decoder core type");
            }
        };

        for (size_t frameId = 0; frameId < frames.value().size(); frameId++) {
            if (frameId >= frameChecks.value().size()) {
                break;
            }
            const auto& frameData = frames.value()[frameId].data;
            MLVC_LOG_DEBUG("Processing frame %zu ...", frameId);
            const auto nalus = SplitToNalus(frameData);
            if (!nalus) {
                MLVC_LOG_ERROR("Failed to split NALUs: %s", nalus.error().message().c_str());
                FAIL() << "Failed to split NALUs: " << nalus.error().message();
            }
            for (const auto& nalu : nalus.value()) {
                auto decodedFrame = decoder.value()->Decode(nalu);
                if (!decodedFrame && IsPartialAccessUnitError(decodedFrame.error())) {
                    continue;
                }
                ASSERT_NO_FATAL_FAILURE(Validate(decodedFrame, frameId)) << "Decoder checks failed at frame " << frameId;
            }
        }
    }

private:
    // Identity & paths
    const std::string m_testName;
    const std::string m_inputVideoName;
    const std::shared_ptr<MlvcManagerImpl> m_manager;
    const std::filesystem::path m_dataDir;
    const std::string m_referenceName;
    std::filesystem::path m_referenceDataDir;

    // Encoding config
    const int m_qp{};
    const MlvcVersion m_mlvcVersion;
    const int m_width{};
    const int m_height{};
    const EncoderConfigOverrides m_overrides;

    // Recovery simulation
    std::set<int> m_lostFrameIds;

    // Per-frame mid-stream numTemporalLayers schedule (frameId -> value)
    std::map<int, int> m_numTemporalLayersSchedule;
};

}  // anonymous namespace

class BitExactTests : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (m_referenceName.empty()) {
            GTEST_SKIP() << "No reference data for this platform";
        }

        const auto referenceBaseDir = m_testDataDir / "bitexact" / m_mlvcVersion.ToString() / m_referenceName;
        std::error_code ec;
        if (!std::filesystem::exists(referenceBaseDir, ec)) {
            GTEST_SKIP() << "Reference data not found: " << referenceBaseDir << (ec ? " (" + ec.message() + ")" : "");
        }

        ManagerParams managerParams;
        managerParams.computeUnit = GetTestConfig().computeUnit;
        managerParams.enableSessionCaching = false;
        managerParams.winmlInitMode = GetTestConfig().winmlInitMode;
        m_manager = std::make_shared<MlvcManagerImpl>(managerParams);
        auto ret = m_manager->Initialize({}, std::array{ m_mlvcVersion });
        ASSERT_TRUE(ret) << "Failed to initialize MlvcManagerImpl: " << ret.error().message();
    }

    BitExactTestRunner MakeTest(std::string_view testName, std::string_view filename, int width, int height, int qp,
                                const EncoderConfigOverrides& overrides = {})
    {
        return { testName, filename, qp,        m_manager,     m_mlvcVersion,
                 width,    height,   overrides, m_testDataDir, m_referenceName };
    }

    std::filesystem::path m_testDataDir = GetTestDataDir();
    MlvcVersion m_mlvcVersion = GetTestConfig().mlvcVersion;
    std::string m_referenceName = GetDefaultReferenceName();
    std::shared_ptr<MlvcManagerImpl> m_manager;
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

TEST_F(BitExactTests, Encoder_960x540_30fps_qp0)
{
    auto test = MakeTest("960x540_30fps_qp0", "VCD_s1_0380a3_960x540_30fps.nv12.gz", 960, 540, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_960x540_30fps_qp0)
{
    auto test = MakeTest("960x540_30fps_qp0", "VCD_s1_0380a3_960x540_30fps.nv12.gz", 960, 540, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_960x540_30fps_qp0_long)
{
    auto test = MakeTest("960x540_30fps_qp0_long", "VCD_s1_0380a3_960x540_30fps.nv12.gz", 960, 540, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_960x540_30fps_qp0_long)
{
    auto test = MakeTest("960x540_30fps_qp0_long", "VCD_s1_0380a3_960x540_30fps.nv12.gz", 960, 540, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0)
{
    auto test = MakeTest("640x360_30fps_qp0", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0)
{
    auto test = MakeTest("640x360_30fps_qp0", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp30)
{
    auto test = MakeTest("640x360_30fps_qp30", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 30);
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp30)
{
    auto test = MakeTest("640x360_30fps_qp30", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 30);
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp51)
{
    auto test = MakeTest("640x360_30fps_qp51", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 51);
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp51)
{
    auto test = MakeTest("640x360_30fps_qp51", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 51);
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_426x240_30fps_qp0)
{
    auto test = MakeTest("426x240_30fps_qp0", "VCD_s1_0380a3_426x240_30fps.nv12.gz", 426, 240, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_426x240_30fps_qp0)
{
    auto test = MakeTest("426x240_30fps_qp0", "VCD_s1_0380a3_426x240_30fps.nv12.gz", 426, 240, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_tl22)
{
    auto test = MakeTest("640x360_30fps_qp0_tl22", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2 });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_tl22)
{
    auto test = MakeTest("640x360_30fps_qp0_tl22", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2 });
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_tl21)
{
    auto test = MakeTest("640x360_30fps_qp0_tl21", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2 });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_tl21)
{
    auto test = MakeTest("640x360_30fps_qp0_tl21", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2 });
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_tl_switch_multi)
{
    auto test = MakeTest("640x360_30fps_qp0_tl_switch_multi", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0);
    test.SetNumTemporalLayersSchedule({ { 0, 1 }, { 11, 2 }, { 20, 1 }, { 27, 2 }, { 37, 1 }, { 44, 2 }, { 51, 1 } });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_tl_switch_multi)
{
    auto test = MakeTest("640x360_30fps_qp0_tl_switch_multi", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0);
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_idr_recovery)
{
    auto test = MakeTest("640x360_30fps_qp0_idr_recovery", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .ltrPeriod = 0 });
    test.SetLostFrameIds({ 40 });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_idr_recovery)
{
    auto test = MakeTest("640x360_30fps_qp0_idr_recovery", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .ltrPeriod = 0 });
    test.SetLostFrameIds({ 40 });
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_tl22_idr_recovery)
{
    auto test = MakeTest("640x360_30fps_qp0_tl22_idr_recovery", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2, .ltrPeriod = 0 });
    test.SetLostFrameIds({ 40 });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_tl22_idr_recovery)
{
    auto test = MakeTest("640x360_30fps_qp0_tl22_idr_recovery", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2, .ltrPeriod = 0 });
    test.SetLostFrameIds({ 40 });
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_ltr_recovery)
{
    auto test = MakeTest("640x360_30fps_qp0_ltr_recovery", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .ltrStartIdx = 8, .ltrPeriod = 32, .ltrRecoveryPeriod = 0 });
    test.SetLostFrameIds({ 40 });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_ltr_recovery)
{
    auto test = MakeTest("640x360_30fps_qp0_ltr_recovery", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .ltrStartIdx = 8, .ltrPeriod = 32, .ltrRecoveryPeriod = 0 });
    test.SetLostFrameIds({ 40 });
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_tl22_ltr_recovery)
{
    auto test = MakeTest("640x360_30fps_qp0_tl22_ltr_recovery", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2, .ltrPeriod = 32, .ltrRecoveryPeriod = 0 });
    test.SetLostFrameIds({ 40 });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_tl22_ltr_recovery)
{
    auto test = MakeTest("640x360_30fps_qp0_tl22_ltr_recovery", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2, .ltrPeriod = 32, .ltrRecoveryPeriod = 0 });
    test.SetLostFrameIds({ 40 });
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_proactive_ltr)
{
    auto test = MakeTest("640x360_30fps_qp0_proactive_ltr", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .ltrPeriod = 16, .ltrRecoveryPeriod = 2 });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_proactive_ltr)
{
    auto test = MakeTest("640x360_30fps_qp0_proactive_ltr", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .ltrPeriod = 16, .ltrRecoveryPeriod = 2 });
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}

TEST_F(BitExactTests, Encoder_640x360_30fps_qp0_tl22_proactive_ltr)
{
    auto test = MakeTest("640x360_30fps_qp0_tl22_proactive_ltr", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2, .ltrPeriod = 16, .ltrRecoveryPeriod = 2 });
    ASSERT_NO_FATAL_FAILURE(test.TestEncoder());
}

TEST_F(BitExactTests, Decoder_640x360_30fps_qp0_tl22_proactive_ltr)
{
    auto test = MakeTest("640x360_30fps_qp0_tl22_proactive_ltr", "VCD_s1_0380a3_640x360_30fps.nv12.gz", 640, 360, 0,
                         { .numTemporalLayers = 2, .ltrPeriod = 16, .ltrRecoveryPeriod = 2 });
    ASSERT_NO_FATAL_FAILURE(test.TestDecoder());
}
