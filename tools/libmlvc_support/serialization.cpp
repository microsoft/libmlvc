// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/serialization.hpp"
#include "libmlvc_support/compress.hpp"
#include "libmlvc_support/interop.hpp"

#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>
#include <libmlvc/platform_info.hpp>

#include <boost/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <span>
#include <string>

namespace libmlvc {

static constexpr const char* SNAPSHOT_FILENAME = "snapshot.json";
static constexpr const char* FRAME_METRICS_FILENAME = "frame_metrics.json.gz";

namespace {

expected<void> WriteJsonValue(const std::filesystem::path& path, const boost::json::value& jv) noexcept
{
    const auto jsonStr = boost::json::serialize(jv);

    // .extension() returns the last extension, so "file.json.gz" → ".gz"
    if (path.extension() == ".gz") {
        auto data = std::span<const std::byte>(reinterpret_cast<const std::byte*>(jsonStr.data()), jsonStr.size());
        auto compressed = Compress(data);
        if (!compressed) {
            std::cerr << "Error: Failed to compress JSON for: " << path.string() << '\n';
            return compressed.error();
        }

        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to open file for writing: " << path.string() << '\n';
            return make_error_code(Error::io_error);
        }
        file.write(reinterpret_cast<const char*>(compressed.value().data()),
                   static_cast<std::streamsize>(compressed.value().size()));
        if (!file) {
            std::cerr << "Error: Failed to write file: " << path.string() << '\n';
            return make_error_code(Error::io_error);
        }
        return {};
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file for writing: " << path.string() << '\n';
        return make_error_code(Error::io_error);
    }
    file << jsonStr;
    if (!file) {
        std::cerr << "Error: Failed to write file: " << path.string() << '\n';
        return make_error_code(Error::io_error);
    }
    return {};
}

expected<boost::json::value> ReadJsonValue(const std::filesystem::path& path)
{
    std::string content;

    // .extension() returns the last extension, so "file.json.gz" → ".gz"
    if (path.extension() == ".gz") {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to open gzip JSON file: " << path.string() << '\n';
            return make_error_code(Error::io_error);
        }
        std::vector<char> compressed((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (file.bad()) {
            std::cerr << "Error: Failed to read gzip JSON file: " << path.string() << '\n';
            return make_error_code(Error::io_error);
        }
        auto data = std::span<const std::byte>(reinterpret_cast<const std::byte*>(compressed.data()), compressed.size());
        auto decompressed = Uncompress(data);
        if (!decompressed) {
            std::cerr << "Error: Failed to decompress JSON file: " << path.string() << '\n';
            return decompressed.error();
        }
        content.assign(reinterpret_cast<const char*>(decompressed.value().data()), decompressed.value().size());
    } else {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to open JSON file: " << path.string() << '\n';
            return make_error_code(Error::io_error);
        }
        content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (file.bad()) {
            std::cerr << "Error: Failed to read JSON file: " << path.string() << '\n';
            return make_error_code(Error::io_error);
        }
    }

    std::error_code ec;
    auto jv = boost::json::parse(content, ec);
    if (ec) {
        std::cerr << "Error: Failed to parse JSON: " << ec.message() << '\n';
        return make_error_code(Error::json_parse_error);
    }
    return jv;
}

// Non-throwing JSON field accessors

double GetDouble(const boost::json::object& obj, const char* key)
{
    const auto* p = obj.if_contains(key);
    if (!p) return 0.0;
    if (const auto* d = p->if_double()) return *d;
    if (const auto* i = p->if_int64()) return static_cast<double>(*i);
    return 0.0;
}

std::string GetString(const boost::json::object& obj, const char* key)
{
    const auto* p = obj.if_contains(key);
    return (p && p->is_string()) ? std::string(p->as_string()) : std::string{};
}

int GetInt(const boost::json::object& obj, const char* key)
{
    const auto* p = obj.if_contains(key);
    return (p && p->is_int64()) ? static_cast<int>(p->as_int64()) : 0;
}

int64_t GetInt64(const boost::json::object& obj, const char* key)
{
    const auto* p = obj.if_contains(key);
    if (p && p->is_int64()) return p->as_int64();
    if (p && p->is_uint64()) return static_cast<int64_t>(p->as_uint64());
    return 0;
}

bool GetBool(const boost::json::object& obj, const char* key)
{
    const auto* p = obj.if_contains(key);
    return p && p->is_bool() && p->as_bool();
}

template <typename T>
boost::json::result_for<T, boost::json::value>::type TryGet(const boost::json::object& obj, const char* key)
{
    const auto* p = obj.if_contains(key);
    if (!p) return make_error_code(Error::json_parse_error);
    return boost::json::try_value_to<T>(*p);
}

template <typename T>
boost::json::result_for<T, boost::json::value>::type TryGetOptional(const boost::json::object& obj, const char* key)
{
    const auto* p = obj.if_contains(key);
    if (!p) return T{};
    return boost::json::try_value_to<T>(*p);
}

// Enum FromString helpers

InferenceBackend InferenceBackendFromString(const std::string& s)
{
    if (s == "coreml") return InferenceBackend::COREML;
    if (s == "windowsml") return InferenceBackend::WINDOWSML;
    std::cerr << "Warning: Unknown InferenceBackend '" << s << "', defaulting to WINDOWSML\n";
    return InferenceBackend::WINDOWSML;
}

ModelType ModelTypeFromString(const std::string& s)
{
    if (s == "coreml") return ModelType::COREML;
    if (s == "onnx") return ModelType::ONNX;
    std::cerr << "Warning: Unknown ModelType '" << s << "', defaulting to UNKNOWN\n";
    return ModelType::UNKNOWN;
}

ComputeUnit ComputeUnitFromString(const std::string& s)
{
    if (s == "cpu") return ComputeUnit::CPU;
    if (s == "gpu") return ComputeUnit::GPU;
    if (s == "npu") return ComputeUnit::NPU;
    if (s != "auto") std::cerr << "Warning: Unknown ComputeUnit '" << s << "', defaulting to AUTO\n";
    return ComputeUnit::AUTO;
}

OnnxExecutionProvider OnnxExecutionProviderFromString(const std::string& s)
{
    if (s == "directml") return OnnxExecutionProvider::DIRECTML;
    if (s == "qnn") return OnnxExecutionProvider::QNN;
    if (s == "openvino") return OnnxExecutionProvider::OPENVINO;
    if (s != "cpu") {
        std::cerr << "Warning: Unknown OnnxExecutionProvider '" << s << "', defaulting to CPU\n";
    }
    return OnnxExecutionProvider::CPU;
}

LtrMode LtrModeFromString(const std::string& s)
{
    if (s == "external") return LtrMode::EXTERNAL;
    if (s != "internal") std::cerr << "Warning: Unknown LtrMode '" << s << "', defaulting to INTERNAL\n";
    return LtrMode::INTERNAL;
}

AcceleratorType AcceleratorTypeFromString(const std::string& s)
{
    if (s == "npu") return AcceleratorType::NPU;
    if (s != "gpu") std::cerr << "Warning: Unknown AcceleratorType '" << s << "', defaulting to GPU\n";
    return AcceleratorType::GPU;
}

OsVersion ParseOsVersion(const std::string& s)
{
    OsVersion v{};
#ifdef _MSC_VER
    sscanf_s(s.c_str(), "%d.%d.%d", &v.major, &v.minor, &v.build);
#else
    std::sscanf(s.c_str(), "%d.%d.%d", &v.major, &v.minor, &v.build);
#endif
    return v;
}

MlvcVersion ParseMlvcVersion(const std::string& s)
{
    MlvcVersion v{};
#ifdef _MSC_VER
    sscanf_s(s.c_str(), "v%d.%d", &v.major, &v.minor);
#else
    std::sscanf(s.c_str(), "v%d.%d", &v.major, &v.minor);
#endif
    return v;
}

}  // namespace

// Serialization

// Platform & config

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const PlatformInfo& p)
{
    jv = {
        { "os_name", p.osName },
        { "os_version", p.osVersion.ToString() },
        { "hw_manufacturer", p.hwManufacturer },
        { "hw_model", p.hwModel },
        { "cpu_arch", p.cpuArch },
        { "cpu_vendor", p.cpuVendor },
        { "cpu_series", p.cpuSeries },
        { "cpu_name", p.cpuName },
    };

    if (!p.accelerators.empty()) {
        boost::json::array accs;
        for (const auto& a : p.accelerators) {
            accs.push_back({
                { "type", AcceleratorTypeToString(a.type) },
                { "name", a.name },
                { "vendor_name", a.vendorName },
                { "vendor_id", a.vendorId },
                { "device_id", a.deviceId },
                { "driver_version", a.driverVersion },
                { "driver_date", a.driverDate },
                { "luid", a.luid },
            });
        }
        jv.as_object()["accelerators"] = std::move(accs);
    }
}

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const ManagerInfo& c)
{
    jv = {
        { "enable_model_cache", c.enableModelCache },
        { "enable_session_caching", c.enableSessionCaching },
        { "inference_backend", InferenceBackendToString(c.inferenceBackend) },
        { "model_type", ModelTypeToString(c.modelType) },
        { "compute_unit", ComputeUnitToString(c.computeUnit) },
        { "onnx_execution_provider", OnnxExecutionProviderToString(c.onnxExecutionProvider) },
        { "windows_app_runtime_version", c.windowsAppRuntimeVersion },
        { "windows_app_runtime_ep_version", c.windowsAppRuntimeEpVersion },
        { "onnx_runtime_version", c.onnxRuntimeVersion },
        { "onnx_runtime_ep_version", c.onnxRuntimeEpVersion },
        { "device_luid", c.deviceLuid },
        { "driver_version", c.driverVersion },
        { "driver_date", c.driverDate },
    };
}

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const EncoderConfig& c)
{
    jv = {
        { "mlvc_version", c.mlvcVersion.ToString() },
        { "width", c.width },
        { "height", c.height },
        { "iframe_period", c.iframePeriod },
        { "num_temporal_layers", c.numTemporalLayers },
        { "ltr_mode", LtrModeToString(c.ltrMode) },
        { "ltr_start_idx", c.ltrStartIdx },
        { "ltr_period", c.ltrPeriod },
        { "ltr_num_slots", c.ltrNumSlots },
        { "ltr_recovery_period", c.ltrRecoveryPeriod },
    };
}

// Metrics

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const FrameMetrics& m)
{
    jv = {
        { "psnr", m.psnr },    { "psnr_y", m.psnrY }, { "psnr_u", m.psnrU },
        { "psnr_v", m.psnrV }, { "bpp", m.bpp },      { "kbps", m.kbps },
    };
}

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const AggregatedMetrics& m)
{
    jv = {
        { "psnr", m.psnr },        { "psnr_y", m.psnrY },     { "psnr_u", m.psnrU },
        { "psnr_v", m.psnrV },     { "bpp", m.bpp },          { "kbps", m.kbps },
        { "psnr_min", m.psnrMin }, { "psnr_max", m.psnrMax }, { "count", m.count },
    };
}

// Stats

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const OpTimerStats& t)
{
    jv = {
        { "latest_ms", t.LatestMs() }, { "total_ms", t.TotalMs() }, { "sum_squared_ms", t.SumSquaredMs() },
        { "min_ms", t.MinMs() },       { "max_ms", t.MaxMs() },     { "count", t.Count() },
    };
}

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const EncoderStats& s)
{
    jv = {
        { "num_frames_attempted", s.numFramesAttempted },
        { "num_frames_encoded", s.numFramesEncoded },
        { "num_idrs_forced", s.numIdrsForced },
        { "num_idrs_total", s.numIdrsTotal },
        { "num_ltr_recoveries_forced", s.numLtrRecoveriesForced },
        { "num_ltr_recoveries_total", s.numLtrRecoveriesTotal },
        { "reconfigure", boost::json::value_from(s.reconfigure) },
        { "preprocess", boost::json::value_from(s.preprocess) },
        { "inference", boost::json::value_from(s.inference) },
        { "scale_decoder", boost::json::value_from(s.scaleDecoder) },
        { "entropy_coding", boost::json::value_from(s.entropyCoding) },
        { "total", boost::json::value_from(s.total) },
    };
}

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const DecoderStats& s)
{
    jv = {
        { "num_decode_calls", s.numDecodeCalls },
        { "num_frames_attempted", s.numFramesAttempted },
        { "num_frames_decoded", s.numFramesDecoded },
        { "num_idrs", s.numIdrs },
        { "num_ltr_recoveries", s.numLtrRecoveries },
        { "reconfigure", boost::json::value_from(s.reconfigure) },
        { "entropy_coding", boost::json::value_from(s.entropyCoding) },
        { "scale_decoder", boost::json::value_from(s.scaleDecoder) },
        { "inference", boost::json::value_from(s.inference) },
        { "postprocess", boost::json::value_from(s.postprocess) },
        { "total", boost::json::value_from(s.total) },
    };
}

// Snapshot

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const BuildInfo& b)
{
    jv = {
        { "libmlvc_version", b.libmlvcVersion },
        { "git_hash", b.gitHash },
        { "git_short_hash", b.gitShortHash },
        { "git_branch", b.gitBranch },
    };
}

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const SnapshotClipResult& c)
{
    jv = {
        { "scenario", c.scenario },
        { "clip_name", c.clipName },
        { "qp", c.qp },
        { "bitstream_path", c.bitstreamPath },
        { "bitstream_hash", c.bitstreamHash },
        { "reconstruction_hash", c.reconstructionHash },
        { "encoder_stats", boost::json::value_from(c.encoderStats) },
        { "decoder_stats", boost::json::value_from(c.decoderStats) },
        { "metrics", boost::json::value_from(c.metrics) },
        // frame_metrics intentionally omitted — saved separately as frame_metrics.json.gz
    };
}

void tag_invoke(boost::json::value_from_tag, boost::json::value& jv, const SnapshotResult& r)
{
    jv = {
        { "snapshot_name", r.snapshotName },
        { "timestamp", r.timestamp },
        { "reference_name", r.referenceName },
        { "dataset_name", r.datasetName },
        { "build_info", boost::json::value_from(r.buildInfo) },
        { "platform", boost::json::value_from(r.platformInfo) },
        { "manager_info", boost::json::value_from(r.managerInfo) },
        { "encoder_config", boost::json::value_from(r.encoderConfig) },
        { "clip_results", boost::json::value_from(r.clipResults) },
    };
}

// Deserialization

// Platform & config

boost::json::result_for<PlatformInfo, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<PlatformInfo>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();
    PlatformInfo info{
        .osName = GetString(obj, "os_name"),
        .osVersion = ParseOsVersion(GetString(obj, "os_version")),
        .hwManufacturer = GetString(obj, "hw_manufacturer"),
        .hwModel = GetString(obj, "hw_model"),
        .cpuArch = GetString(obj, "cpu_arch"),
        .cpuVendor = GetString(obj, "cpu_vendor"),
        .cpuSeries = GetString(obj, "cpu_series"),
        .cpuName = GetString(obj, "cpu_name"),
    };

    if (const auto* arr = obj.if_contains("accelerators"); arr && arr->is_array()) {
        for (const auto& elem : arr->as_array()) {
            if (!elem.is_object()) continue;
            const auto& ao = elem.as_object();
            info.accelerators.push_back({
                .type = AcceleratorTypeFromString(GetString(ao, "type")),
                .name = GetString(ao, "name"),
                .vendorName = GetString(ao, "vendor_name"),
                .vendorId = GetString(ao, "vendor_id"),
                .deviceId = GetString(ao, "device_id"),
                .driverVersion = GetString(ao, "driver_version"),
                .driverDate = GetString(ao, "driver_date"),
                .luid = GetInt64(ao, "luid"),
            });
        }
    }

    return info;
}

boost::json::result_for<ManagerInfo, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<ManagerInfo>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();
    return ManagerInfo{
        .enableModelCache = GetBool(obj, "enable_model_cache"),
        .enableSessionCaching = GetBool(obj, "enable_session_caching"),
        .inferenceBackend = InferenceBackendFromString(GetString(obj, "inference_backend")),
        .modelType = ModelTypeFromString(GetString(obj, "model_type")),
        .computeUnit = ComputeUnitFromString(GetString(obj, "compute_unit")),
        .onnxExecutionProvider = OnnxExecutionProviderFromString(GetString(obj, "onnx_execution_provider")),
        .windowsAppRuntimeVersion = GetString(obj, "windows_app_runtime_version"),
        .windowsAppRuntimeEpVersion = GetString(obj, "windows_app_runtime_ep_version"),
        .onnxRuntimeVersion = GetString(obj, "onnx_runtime_version"),
        .onnxRuntimeEpVersion = GetString(obj, "onnx_runtime_ep_version"),
        .deviceLuid = GetInt64(obj, "device_luid"),
        .driverVersion = GetString(obj, "driver_version"),
        .driverDate = GetString(obj, "driver_date"),
    };
}

boost::json::result_for<EncoderConfig, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<EncoderConfig>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();
    return EncoderConfig{
        .mlvcVersion = ParseMlvcVersion(GetString(obj, "mlvc_version")),
        .width = GetInt(obj, "width"),
        .height = GetInt(obj, "height"),
        .iframePeriod = GetInt(obj, "iframe_period"),
        .numTemporalLayers = GetInt(obj, "num_temporal_layers"),
        .ltrMode = LtrModeFromString(GetString(obj, "ltr_mode")),
        .ltrStartIdx = GetInt(obj, "ltr_start_idx"),
        .ltrPeriod = GetInt(obj, "ltr_period"),
        .ltrNumSlots = GetInt(obj, "ltr_num_slots"),
        .ltrRecoveryPeriod = GetInt(obj, "ltr_recovery_period"),
    };
}

// Metrics

boost::json::result_for<FrameMetrics, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<FrameMetrics>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();
    return FrameMetrics{
        .psnr = GetDouble(obj, "psnr"),
        .psnrY = GetDouble(obj, "psnr_y"),
        .psnrU = GetDouble(obj, "psnr_u"),
        .psnrV = GetDouble(obj, "psnr_v"),
        .bpp = GetDouble(obj, "bpp"),
        .kbps = GetDouble(obj, "kbps"),
    };
}

boost::json::result_for<AggregatedMetrics, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<AggregatedMetrics>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();
    return AggregatedMetrics{
        .psnr = GetDouble(obj, "psnr"),
        .psnrY = GetDouble(obj, "psnr_y"),
        .psnrU = GetDouble(obj, "psnr_u"),
        .psnrV = GetDouble(obj, "psnr_v"),
        .bpp = GetDouble(obj, "bpp"),
        .kbps = GetDouble(obj, "kbps"),
        .psnrMin = GetDouble(obj, "psnr_min"),
        .psnrMax = GetDouble(obj, "psnr_max"),
        .count = GetInt(obj, "count"),
    };
}

// Stats

boost::json::result_for<OpTimerStats, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<OpTimerStats>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();
    return OpTimerStats::FromState(GetDouble(obj, "latest_ms"), GetDouble(obj, "total_ms"),
                                   GetDouble(obj, "sum_squared_ms"), GetDouble(obj, "min_ms"), GetDouble(obj, "max_ms"),
                                   GetInt(obj, "count"));
}

boost::json::result_for<EncoderStats, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<EncoderStats>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();

    auto reconfigure = TryGet<OpTimerStats>(obj, "reconfigure");
    if (!reconfigure) return reconfigure.error();
    auto preprocess = TryGet<OpTimerStats>(obj, "preprocess");
    if (!preprocess) return preprocess.error();
    auto inference = TryGet<OpTimerStats>(obj, "inference");
    if (!inference) return inference.error();
    auto entropyCoding = TryGet<OpTimerStats>(obj, "entropy_coding");
    if (!entropyCoding) return entropyCoding.error();
    auto scaleDecoder = TryGetOptional<OpTimerStats>(obj, "scale_decoder");
    if (!scaleDecoder) return scaleDecoder.error();
    auto total = TryGet<OpTimerStats>(obj, "total");
    if (!total) return total.error();

    return EncoderStats{
        .numFramesAttempted = GetInt(obj, "num_frames_attempted"),
        .numFramesEncoded = GetInt(obj, "num_frames_encoded"),
        .numIdrsForced = GetInt(obj, "num_idrs_forced"),
        .numIdrsTotal = GetInt(obj, "num_idrs_total"),
        .numLtrRecoveriesForced = GetInt(obj, "num_ltr_recoveries_forced"),
        .numLtrRecoveriesTotal = GetInt(obj, "num_ltr_recoveries_total"),
        .reconfigure = std::move(reconfigure.value()),
        .preprocess = std::move(preprocess.value()),
        .inference = std::move(inference.value()),
        .scaleDecoder = std::move(scaleDecoder.value()),
        .entropyCoding = std::move(entropyCoding.value()),
        .total = std::move(total.value()),
    };
}

boost::json::result_for<DecoderStats, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<DecoderStats>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();

    auto reconfigure = TryGet<OpTimerStats>(obj, "reconfigure");
    if (!reconfigure) return reconfigure.error();
    auto entropyCoding = TryGet<OpTimerStats>(obj, "entropy_coding");
    if (!entropyCoding) return entropyCoding.error();
    auto scaleDecoder = TryGetOptional<OpTimerStats>(obj, "scale_decoder");
    if (!scaleDecoder) return scaleDecoder.error();
    auto inference = TryGet<OpTimerStats>(obj, "inference");
    if (!inference) return inference.error();
    auto postprocess = TryGet<OpTimerStats>(obj, "postprocess");
    if (!postprocess) return postprocess.error();
    auto total = TryGet<OpTimerStats>(obj, "total");
    if (!total) return total.error();

    return DecoderStats{
        .numDecodeCalls = GetInt(obj, "num_decode_calls"),
        .numFramesAttempted = GetInt(obj, "num_frames_attempted"),
        .numFramesDecoded = GetInt(obj, "num_frames_decoded"),
        .numIdrs = GetInt(obj, "num_idrs"),
        .numLtrRecoveries = GetInt(obj, "num_ltr_recoveries"),
        .reconfigure = std::move(reconfigure.value()),
        .entropyCoding = std::move(entropyCoding.value()),
        .scaleDecoder = std::move(scaleDecoder.value()),
        .inference = std::move(inference.value()),
        .postprocess = std::move(postprocess.value()),
        .total = std::move(total.value()),
    };
}

// Snapshot

boost::json::result_for<BuildInfo, boost::json::value>::type tag_invoke(const boost::json::try_value_to_tag<BuildInfo>&,
                                                                        const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();
    return BuildInfo{
        .libmlvcVersion = GetString(obj, "libmlvc_version"),
        .gitHash = GetString(obj, "git_hash"),
        .gitShortHash = GetString(obj, "git_short_hash"),
        .gitBranch = GetString(obj, "git_branch"),
    };
}

boost::json::result_for<SnapshotClipResult, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<SnapshotClipResult>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();

    auto encoderStats = TryGet<EncoderStats>(obj, "encoder_stats");
    if (!encoderStats) return encoderStats.error();
    auto decoderStats = TryGet<DecoderStats>(obj, "decoder_stats");
    if (!decoderStats) return decoderStats.error();
    auto metrics = TryGet<AggregatedMetrics>(obj, "metrics");
    if (!metrics) return metrics.error();

    return SnapshotClipResult{
        .scenario = GetString(obj, "scenario"),
        .clipName = GetString(obj, "clip_name"),
        .qp = GetInt(obj, "qp"),
        .bitstreamPath = GetString(obj, "bitstream_path"),
        .bitstreamHash = GetString(obj, "bitstream_hash"),
        .reconstructionHash = GetString(obj, "reconstruction_hash"),
        .encoderStats = std::move(encoderStats.value()),
        .decoderStats = std::move(decoderStats.value()),
        .metrics = std::move(metrics.value()),
        // frameMetrics not loaded here — stored separately in frame_metrics.json.gz
    };
}

boost::json::result_for<SnapshotResult, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<SnapshotResult>&, const boost::json::value& jv)
{
    if (!jv.is_object()) return make_error_code(Error::json_parse_error);
    const auto& obj = jv.as_object();

    auto buildInfo = TryGet<BuildInfo>(obj, "build_info");
    if (!buildInfo) return buildInfo.error();

    auto platform = TryGet<PlatformInfo>(obj, "platform");
    if (!platform) return platform.error();

    auto managerInfo = TryGet<ManagerInfo>(obj, "manager_info");
    if (!managerInfo) return managerInfo.error();

    auto encoderConfig = TryGet<EncoderConfig>(obj, "encoder_config");
    if (!encoderConfig) return encoderConfig.error();

    auto clipResults = TryGet<std::vector<SnapshotClipResult>>(obj, "clip_results");
    if (!clipResults) return clipResults.error();

    return SnapshotResult{
        .snapshotName = GetString(obj, "snapshot_name"),
        .timestamp = GetString(obj, "timestamp"),
        .referenceName = GetString(obj, "reference_name"),
        .datasetName = GetString(obj, "dataset_name"),
        .buildInfo = std::move(buildInfo.value()),
        .platformInfo = std::move(platform.value()),
        .managerInfo = std::move(managerInfo.value()),
        .encoderConfig = std::move(encoderConfig.value()),
        .clipResults = std::move(clipResults.value()),
    };
}

// Public API

expected<void> SaveSnapshot(const std::filesystem::path& dir, const SnapshotResult& result)
{
    // Build parallel JSON array of per-frame metrics (index matches clip_results order)
    boost::json::array frameMetricsArray;
    for (const auto& clip : result.clipResults) {
        frameMetricsArray.push_back(boost::json::value_from(clip.frameMetrics));
    }

    // Write frame_metrics.json.gz first, snapshot.json last.
    // snapshot.json acts as the commit marker checked by ListSnapshots,
    // so a partial write (e.g. disk full) won't produce a discoverable incomplete snapshot.
    if (auto rc = WriteJsonValue(dir / FRAME_METRICS_FILENAME, boost::json::value(std::move(frameMetricsArray))); !rc) {
        return rc.error();
    }
    return WriteJsonValue(dir / SNAPSHOT_FILENAME, boost::json::value_from(result));
}

expected<SnapshotResult> LoadSnapshot(const std::filesystem::path& dir, bool loadFrameMetrics)
{
    auto jv = ReadJsonValue(dir / SNAPSHOT_FILENAME);
    if (!jv) return jv.error();

    auto parsed = boost::json::try_value_to<SnapshotResult>(jv.value());
    if (!parsed) {
        std::cerr << "Error: Failed to parse snapshot JSON: " << parsed.error().message() << '\n';
        return make_error_code(Error::json_parse_error);
    }
    auto result = std::move(parsed.value());

    // Optionally load per-frame metrics from the separate compressed file
    if (loadFrameMetrics) {
        const auto fmPath = dir / FRAME_METRICS_FILENAME;
        std::error_code ec;
        if (!std::filesystem::exists(fmPath, ec)) {
            std::cerr << "Warning: Frame metrics file not found: " << fmPath.string() << " ("
                      << (ec ? ec.message() : std::string{ "not found" }) << ")\n";
        } else {
            auto fmJv = ReadJsonValue(fmPath);
            if (!fmJv) {
                std::cerr << "Warning: Failed to load frame metrics: " << fmPath.string() << '\n';
            } else if (fmJv.value().is_array()) {
                const auto& arr = fmJv.value().as_array();
                for (size_t i = 0; i < arr.size() && i < result.clipResults.size(); ++i) {
                    auto fm = boost::json::try_value_to<std::vector<FrameMetrics>>(arr[i]);
                    if (fm) {
                        result.clipResults[i].frameMetrics = std::move(fm.value());
                    }
                }
            }
        }
    }

    return result;
}

}  // namespace libmlvc
