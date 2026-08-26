// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace libmlvc {

// --------------------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------------------

constexpr int MIN_SUPPORTED_MLVC_VERSION = 0;
constexpr int MAX_SUPPORTED_MLVC_VERSION = 0;
constexpr int MAX_TEMPORAL_LAYERS = 2;
constexpr int MAX_LTR_SLOTS = 8;
constexpr int MIN_QP = 0;
constexpr int MAX_QP = 51;
constexpr int DEFAULT_QP = 26;

// --------------------------------------------------------------------------------------
// Common types
// --------------------------------------------------------------------------------------

enum struct LogLevel { Debug, Info, Warn, Error, Fatal };
inline const char* LogLevelToString(LogLevel level)
{
    switch (level) {
    case LogLevel::Debug:
        return "DEBUG";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Warn:
        return "WARN";
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Fatal:
        return "FATAL";
    }
    return "UNKNOWN";
}

using LogHandler = std::function<void(LogLevel level, const std::source_location& loc, std::string_view msg)>;

struct MlvcVersion {
    int major{};
    int minor{};

    auto operator<=>(const MlvcVersion&) const = default;
    std::string ToString() const { return "v" + std::to_string(major) + "." + std::to_string(minor); }
};

struct Nv12FrameView {
    Nv12FrameView() = default;
    Nv12FrameView(int width, int height, std::span<const std::byte> buffer)
        : Nv12FrameView(width, height, width, buffer.subspan(0, static_cast<size_t>(width) * height),
                        buffer.subspan(static_cast<size_t>(width) * height))
    {
    }

    Nv12FrameView(int width, int height, int stride, const std::byte* const yPlane, const std::byte* const uvPlane)
        : Nv12FrameView{ width,
                         height,
                         stride,
                         { yPlane, static_cast<size_t>(stride) * height },
                         { uvPlane, static_cast<size_t>(stride) * height / 2 } }

    {
    }

    Nv12FrameView(int width, int height, int stride, std::span<const std::byte> yPlane, std::span<const std::byte> uvPlane)
        : m_width{ width }, m_height{ height }, m_stride{ stride }, m_yPlane{ yPlane }, m_uvPlane{ uvPlane }
    {
    }

    int Width() const { return m_width; }
    int Height() const { return m_height; }
    int Stride() const { return m_stride; }
    std::span<const std::byte> YPlane() const { return m_yPlane; }
    std::span<const std::byte> UvPlane() const { return m_uvPlane; }

private:
    int m_width{};
    int m_height{};
    int m_stride{};
    std::span<const std::byte> m_yPlane;
    std::span<const std::byte> m_uvPlane;
};

class OpTimerStats {
public:
    void Append(double durationMs)
    {
        m_latestMs = durationMs;
        m_totalMs += durationMs;
        m_sumSquaredMs += durationMs * durationMs;
        m_minMs = (m_count == 0) ? durationMs : std::min(m_minMs, durationMs);
        m_maxMs = (m_count == 0) ? durationMs : std::max(m_maxMs, durationMs);
        m_count++;
    }

    double LatestMs() const { return m_latestMs; }
    double TotalMs() const { return m_totalMs; }
    double SumSquaredMs() const { return m_sumSquaredMs; }
    double MinMs() const { return m_minMs; }
    double MaxMs() const { return m_maxMs; }
    int Count() const { return m_count; }
    double AverageMs() const { return m_count > 0 ? m_totalMs / static_cast<double>(m_count) : 0.0; }
    double StdDevMs() const
    {
        if (m_count <= 1) return 0.0;
        const double mean = AverageMs();
        return std::sqrt(std::max(0.0, m_sumSquaredMs / static_cast<double>(m_count) - mean * mean));
    }

    static OpTimerStats FromState(double latestMs, double totalMs, double sumSquaredMs, double minMs, double maxMs, int count)
    {
        OpTimerStats s;
        s.m_latestMs = latestMs;
        s.m_totalMs = totalMs;
        s.m_sumSquaredMs = sumSquaredMs;
        s.m_minMs = minMs;
        s.m_maxMs = maxMs;
        s.m_count = count;
        return s;
    }

private:
    double m_latestMs{};
    double m_totalMs{};
    double m_sumSquaredMs{};
    double m_minMs{};
    double m_maxMs{};
    int m_count{};
};

// --------------------------------------------------------------------------------------
// Encoder types
// --------------------------------------------------------------------------------------

enum struct FrameType {
    I_FRAME,
    P_FRAME,
    LTR_RECOVERY,
};

enum struct LtrMode {
    INTERNAL,
    EXTERNAL,
};

inline const char* LtrModeToString(const LtrMode mode)
{
    switch (mode) {
    case LtrMode::INTERNAL:
        return "internal";
    case LtrMode::EXTERNAL:
        return "external";
    }
    return "unknown";
}

struct EncoderConfig {
    MlvcVersion mlvcVersion{};
    int width{};
    int height{};
    int iframePeriod{};  // 0 = no periodic I-frames
    int numTemporalLayers{ 1 };
    LtrMode ltrMode{ LtrMode::INTERNAL };

    // The following LTR parameters are only used when ltrMode == INTERNAL
    int ltrStartIdx{};        // skip first n frames in periodic LTR marking
    int ltrPeriod{};          // 0 = no periodic LTR frames
    int ltrNumSlots{};        // number of LTR slots available
    int ltrRecoveryPeriod{};  // 0 = disabled, in LTR period units

    bool operator==(const EncoderConfig&) const = default;

    EncoderConfig& SetSize(int w, int h)
    {
        width = w;
        height = h;
        return *this;
    }
};

struct QpParams {
    QpParams(const int qp = DEFAULT_QP) { qps.fill(qp); }
    QpParams(const int qpBase, const int qpRest) : QpParams(qpRest) { qps[0] = qpBase; }
    std::array<int, MAX_TEMPORAL_LAYERS> qps{};
};

struct EncodeParams {
    QpParams qp{};
    bool forceIdr{};
    std::optional<int> useLtrSlotIdx;
};

struct FrameInfo {
    FrameType frameType{};
    int frameIdx{};
    int refFrameIdx{};
    int temporalId{};
    int predictionChainLength{};
};

struct EncodedFrame {
    FrameInfo info{};
    std::size_t payloadBytes{};
    std::span<const std::byte> bitStream;
};

struct EncoderStats {
    // Frames
    int numFramesAttempted{};
    int numFramesEncoded{};

    // Per-layer
    std::array<int, MAX_TEMPORAL_LAYERS> framesPerLayer{};
    std::array<std::size_t, MAX_TEMPORAL_LAYERS> bytesPerLayer{};

    // IDRs
    int numIdrsForced{};
    int numIdrsTotal{};  // forced + internal

    // LTR
    int numLtrRecoveriesForced{};
    int numLtrRecoveriesTotal{};  // forced + internal

    // Timers
    OpTimerStats reconfigure;
    OpTimerStats preprocess;
    OpTimerStats inference;
    OpTimerStats scaleDecoder;
    OpTimerStats entropyCoding;
    OpTimerStats total;
    OpTimerStats frameInterval;
};

// --------------------------------------------------------------------------------------
// Decoder types
// --------------------------------------------------------------------------------------

struct DecodedFrame {
    FrameInfo info{};
    int qp{};
    Nv12FrameView frame{};
};

struct DecoderStats {
    // Frames
    int numDecodeCalls{};
    int numFramesAttempted{};
    int numFramesDecoded{};

    // Per-layer
    std::array<int, MAX_TEMPORAL_LAYERS> framesPerLayer{};
    std::array<std::size_t, MAX_TEMPORAL_LAYERS> bytesPerLayer{};

    // IDRs
    int numIdrs{};

    // LTR
    int numLtrRecoveries{};

    // Timers
    OpTimerStats reconfigure;
    OpTimerStats entropyCoding;
    OpTimerStats scaleDecoder;
    OpTimerStats inference;
    OpTimerStats postprocess;
    OpTimerStats total;
    OpTimerStats frameInterval;
};

// --------------------------------------------------------------------------------------
// Manager types
// --------------------------------------------------------------------------------------

class CancelToken {
public:
    CancelToken() : m_flag(std::make_shared<std::atomic<bool>>(false)) {}
    void Cancel() noexcept { m_flag->store(true, std::memory_order_release); }
    [[nodiscard]] bool IsCancelled() const noexcept { return m_flag->load(std::memory_order_acquire); }

private:
    std::shared_ptr<std::atomic<bool>> m_flag;
};

struct StatusMessage {
    std::string message;
};

struct WindowsAppRuntimeVersionAvailable {
    std::string version;
};

struct WindowsAppRuntimeEPInfoAvailable {
    std::string version;
    std::optional<uint32_t> downloadTimeMs;
};

using InitializeProgressEvent =
    std::variant<StatusMessage, WindowsAppRuntimeVersionAvailable, WindowsAppRuntimeEPInfoAvailable>;

// Progress callback for Initialize - emits variant events with version/status information
using InitializeProgressCallback = std::function<void(const InitializeProgressEvent& event)>;

// Append-only: don't reuse or change values
enum struct ComputeUnit {
    AUTO = 0,
    CPU = 1,
    GPU = 2,
    NPU = 3,
};
inline const char* ComputeUnitToString(const ComputeUnit computeUnit)
{
    switch (computeUnit) {
    case ComputeUnit::AUTO:
        return "auto";
    case ComputeUnit::CPU:
        return "cpu";
    case ComputeUnit::GPU:
        return "gpu";
    case ComputeUnit::NPU:
        return "npu";
    }
    return "unknown";
}

enum struct InferenceBackend { WINDOWSML, COREML };
inline const char* InferenceBackendToString(const InferenceBackend engine)
{
    switch (engine) {
    case InferenceBackend::WINDOWSML:
        return "windowsml";
    case InferenceBackend::COREML:
        return "coreml";
    }
    return "unknown";
}

enum struct ModelType { COREML, ONNX, UNKNOWN };
inline const char* ModelTypeToString(const ModelType type)
{
    switch (type) {
    case ModelType::COREML:
        return "coreml";
    case ModelType::ONNX:
        return "onnx";
    case ModelType::UNKNOWN:
        return "unknown";
    }
    return "unknown";
}

enum struct OnnxExecutionProvider { CPU, DIRECTML, QNN, OPENVINO, TENSOR_RT };
inline const char* OnnxExecutionProviderToString(const OnnxExecutionProvider ep)
{
    switch (ep) {
    case OnnxExecutionProvider::CPU:
        return "cpu";
    case OnnxExecutionProvider::DIRECTML:
        return "directml";
    case OnnxExecutionProvider::QNN:
        return "qnn";
    case OnnxExecutionProvider::OPENVINO:
        return "openvino";
    case OnnxExecutionProvider::TENSOR_RT:
        return "tensorrt";
    }
    return "unknown";
}

enum struct WinMlInitMode { AppSdk, SelfContained };

inline const char* WinMlInitModeToString(WinMlInitMode mode)
{
    switch (mode) {
    case WinMlInitMode::AppSdk:
        return "appsdk";
    case WinMlInitMode::SelfContained:
        return "self-contained";
    }
    return "unknown";
}

inline expected<WinMlInitMode> StringToWinMlInitMode(std::string_view value)
{
    if (value == "appsdk") {
        return WinMlInitMode::AppSdk;
    }
    if (value == "self-contained" || value == "selfcontained") {
        return WinMlInitMode::SelfContained;
    }
    return make_error_code(Error::invalid_argument);
}

struct ManagerParams {
    ComputeUnit computeUnit{ ComputeUnit::AUTO };
    std::filesystem::path cacheDir{};  // Empty means use platform app data dir
    bool enableModelCache{ true };
    bool enableSessionCaching{ true };
    WinMlInitMode winmlInitMode{ WinMlInitMode::AppSdk };  // Windows-only; ignored on other platforms
};

struct Capabilities {
    int maxWidth{};
    int maxHeight{};
    float maxFps{};
    int maxNumOfTemporalLayers{};
    int maxNumOfLtrFrames{};
};

struct ManagerInfo {
    bool enableModelCache{};
    bool enableSessionCaching{};

    InferenceBackend inferenceBackend{ InferenceBackend::WINDOWSML };
    ModelType modelType{ ModelType::UNKNOWN };
    ComputeUnit computeUnit{ ComputeUnit::CPU };

    // Only used when inferenceBackend == WINDOWSML (ONNX)
    OnnxExecutionProvider onnxExecutionProvider{};
    std::string windowsAppRuntimeVersion{};
    std::string windowsAppRuntimeEpVersion{};
    std::string onnxRuntimeVersion{};
    std::string onnxRuntimeEpVersion{};

    // Device info (populated by LUID matching with PlatformInfo accelerators)
    int64_t deviceLuid{};  // 0 if unavailable
    std::string driverVersion{};
    std::string driverDate{};
};

// --------------------------------------------------------------------------------------
// Parser / bitstream types
// --------------------------------------------------------------------------------------

struct CropOffsets {
    int left{ 0 };
    int right{ 0 };
    int top{ 0 };
    int bottom{ 0 };

    bool IsEmpty() const { return left == 0 && right == 0 && top == 0 && bottom == 0; }
};

enum struct NaluType {
    TRAIL_N = 0,    // non-reference trailing picture
    TRAIL_R = 1,    // trailing picture
    TSA_N = 2,      // non-reference temporal sub-layer picture
    TSA_R = 3,      // temporal sub-layer picture
    IDR_N_LP = 20,  // idr picture
    SPS = 33,       // sequence parameter set
    PPS = 34,       // picture parameter set
};

struct NaluHeader {
    int forbiddenZero{ 0 };                   // f(1)
    NaluType naluType{ NaluType::IDR_N_LP };  // u(6)
    int layerId{ 0 };                         // u(6)
    int temporalIdPlus1{ 1 };                 // u(3)

    NaluHeader(NaluType type = NaluType::IDR_N_LP) : naluType(type) {}
    bool operator==(const NaluHeader&) const = default;
};

struct SpsNalu {
    NaluHeader naluHeader{ NaluType::SPS };  // NALU header
    int mlvcVersionMajor{ 0 };               // u(8)
    int mlvcVersionMinor{ 0 };               // u(8)
    int spsId{ -1 };                         // ue(v)
    int modelWidthDiv2{ 0 };                 // u(12)
    int modelHeightDiv2{ 0 };                // u(12)
    bool transposeFlag{ false };             // u(1)
    bool cropFlag{ false };                  // u(1)
    int cropLeftDiv2{ 0 };                   // ue(v), if cropFlag
    int cropRightDiv2{ 0 };                  // ue(v), if cropFlag
    int cropTopDiv2{ 0 };                    // ue(v), if cropFlag
    int cropBottomDiv2{ 0 };                 // ue(v), if cropFlag
    int maxTemporalLayersMinus1{ 0 };        // u(3)
    int frameIdxBitsMinus8{ 0 };             // ue(v)

    bool operator==(const SpsNalu&) const = default;
};

struct PpsNalu {
    NaluHeader naluHeader{ NaluType::PPS };  // NALU header
    int ppsId{ -1 };                         // ue(v)
    int spsId{ -2 };                         // ue(v)
    int initQpMinus26{ 0 };                  // se(v)
};

struct LtrSlotInfo {
    bool empty{ true };
    int frameIdx{ 0 };

    LtrSlotInfo() = default;
    explicit LtrSlotInfo(int idx) : empty{ false }, frameIdx{ idx } {}

    bool HasValue() const { return !empty; }
    bool operator==(const LtrSlotInfo&) const = default;
};

struct FrameHeader {
    int ppsId{ 0 };                                   // ue(v)
    int qpDelta{ 0 };                                 // se(v)
    std::array<LtrSlotInfo, MAX_LTR_SLOTS> ltrSlots;  // size: u(4), values: u(1): slot+empty, u(v): frame_idx (if slot used)
    int frameIdx{ 0 };                                // u(10), if !idr
    int refFrameIdx{ 0 };                             // u(10), if !idr
    bool featureResetFlag{ false };                   // u(1), if !idr
};

struct FrameNalu {
    NaluHeader naluHeader{ NaluType::IDR_N_LP };  // NALU header
    FrameHeader frameHeader;                      // frame header
    std::span<const std::byte> payload;           // payload bytes
};

struct CustomNalu {
    NaluHeader naluHeader;               // NALU header
    std::span<const std::byte> payload;  // payload bytes
};

struct FrameData {
    MlvcVersion mlvcVersion{};
    int modelWidth{};
    int modelHeight{};
    CropOffsets cropOffsets{};
    bool transposeFlag{};
    int frameIdxBits{};
    int temporalId{};
    int maxTemporalLayers{};
    FrameType frameType{};
    int qp{};
    bool featureResetFlag{};
    int curFrameIdx{};
    int refFrameIdx{};
    std::array<LtrSlotInfo, MAX_LTR_SLOTS> ltrSlots;
    std::span<const std::byte> payload;

    int DisplayWidth() const
    {
        if (!transposeFlag) {
            return modelWidth - cropOffsets.left - cropOffsets.right;
        } else {
            return modelHeight - cropOffsets.top - cropOffsets.bottom;
        }
    }

    int DisplayHeight() const
    {
        if (!transposeFlag) {
            return modelHeight - cropOffsets.top - cropOffsets.bottom;
        } else {
            return modelWidth - cropOffsets.left - cropOffsets.right;
        }
    }
};

}  // namespace libmlvc
