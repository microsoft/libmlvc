// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/// @file
/// Public configuration, result, statistics, and bitstream types.

#pragma once
#include <libmlvc/error_codes.hpp>
#include <libmlvc/expected.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
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

/// Severity of a libmlvc log message.
enum struct LogLevel { Debug, Info, Warn, Error, Fatal };

/// Returns the string name of a log level.
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

/// Log callback; `loc` and `msg` are valid only for the duration of the call.
using LogHandler = std::function<void(LogLevel level, const std::source_location& loc, std::string_view msg)>;

/// Major/minor identifier embedded in a model bundle and encoded bitstream.
struct MlvcVersion {
    int major{};
    int minor{};

    auto operator<=>(const MlvcVersion&) const = default;
    std::string ToString() const { return "v" + std::to_string(major) + "." + std::to_string(minor); }
};

/// View of caller-owned NV12 pixels.
///
/// The backing storage must outlive the view. Stride is measured in bytes.
struct Nv12FrameView {
    Nv12FrameView() = default;

    /// Creates a tightly packed view. `buffer` must contain at least `width * height * 3 / 2` bytes.
    Nv12FrameView(int width, int height, std::span<const std::byte> buffer)
        : Nv12FrameView(width, height, width, buffer.subspan(0, static_cast<size_t>(width) * height),
                        buffer.subspan(static_cast<size_t>(width) * height))
    {
    }

    /// Creates a view from plane pointers, reading `stride * height` Y bytes and `stride * height / 2` UV bytes.
    Nv12FrameView(int width, int height, int stride, const std::byte* const yPlane, const std::byte* const uvPlane)
        : Nv12FrameView{ width,
                         height,
                         stride,
                         { yPlane, static_cast<size_t>(stride) * height },
                         { uvPlane, static_cast<size_t>(stride) * height / 2 } }

    {
    }

    /// Creates a view over explicitly sized Y and UV planes.
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

/// Aggregate timing samples measured in milliseconds.
///
/// `StdDevMs()` reports population standard deviation and returns zero for fewer than two samples.
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

    /// Reconstructs timing statistics from aggregate values in accessor order.
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

/// Prediction role assigned to an encoded or decoded frame.
enum struct FrameType {
    I_FRAME,       ///< Independently decodable IDR frame.
    P_FRAME,       ///< Frame predicted from a reference frame.
    LTR_RECOVERY,  ///< Frame recovering from a long-term reference.
};

/// Long-term reference management mode.
enum struct LtrMode {
    INTERNAL,  ///< The encoder manages LTR marking and recovery.
    EXTERNAL,  ///< The caller manages LTR marking and recovery.
};

/// Returns the string name of an LTR mode.
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

/// Encoder configuration for one MLVC model version and display resolution.
struct EncoderConfig {
    MlvcVersion mlvcVersion{};             ///< Must identify a model version loaded by the manager.
    int width{};                           ///< Display width in pixels; must be positive and even.
    int height{};                          ///< Display height in pixels; must be positive and even.
    int iframePeriod{};                    ///< I-frame period in frames; zero disables periodic I-frames.
    int numTemporalLayers{ 1 };            ///< Range: 1 to `MAX_TEMPORAL_LAYERS`.
    LtrMode ltrMode{ LtrMode::INTERNAL };  ///< Long-term reference management mode.

    int ltrStartIdx{};        ///< Initial frames skipped by internal LTR marking.
    int ltrPeriod{};          ///< Internal LTR period in frames; zero disables marking.
    int ltrNumSlots{};        ///< Slots available to internal or external LTR management.
    int ltrRecoveryPeriod{};  ///< Internal recovery interval as a multiple of `ltrPeriod`; zero disables it.

    bool operator==(const EncoderConfig&) const = default;

    EncoderConfig& SetSize(int w, int h)
    {
        width = w;
        height = h;
        return *this;
    }
};

/// Per-temporal-layer quantization parameters.
/// Default construction sets every layer to `DEFAULT_QP`.
struct QpParams {
    /// Sets the same quantization parameter for every temporal layer.
    QpParams(const int qp = DEFAULT_QP) { qps.fill(qp); }
    /// Sets layer zero to `qpBase` and the remaining layers to `qpRest`.
    QpParams(const int qpBase, const int qpRest) : QpParams(qpRest) { qps[0] = qpBase; }
    std::array<int, MAX_TEMPORAL_LAYERS> qps{};  ///< Indexed by temporal ID; range: 0 to 51.
};

/// Per-frame encoder controls.
struct EncodeParams {
    QpParams qp{};    ///< Quantization parameters by temporal layer.
    bool forceIdr{};  ///< Whether to encode this frame as an IDR.
    /// LTR recovery slot. Out-of-range indices make `Encode()` fail; unpopulated slots are ignored.
    std::optional<int> useLtrSlotIdx{};
};

/// Prediction and dependency information for one frame.
struct FrameInfo {
    FrameType frameType{ FrameType::I_FRAME };  ///< Prediction role for this frame.
    int frameIdx{};                             ///< Current frame index.
    int refFrameIdx{};                          ///< Primary reference index; meaningful only for non-IDR frames.
    int temporalId{};                           ///< Temporal-layer index; range: 0 to the configured count minus 1.
    int predictionChainLength{};                ///< Number of prediction steps back to an I-frame.
};

/// Result of encoding one frame.
struct EncodedFrame {
    FrameInfo info{};
    std::size_t payloadBytes{};            ///< Entropy payload size, excluding NALU framing.
    std::span<const std::byte> bitStream;  ///< Invalidated by the next `Encode()` or encoder destruction.
};

/// Cumulative encoder counters and timing statistics.
struct EncoderStats {
    // Frames
    int numFramesAttempted{};  ///< Encode attempts that passed input validation.
    int numFramesEncoded{};    ///< Frames encoded successfully.

    // Per-layer
    std::array<int, MAX_TEMPORAL_LAYERS> framesPerLayer{};         ///< Successful frames by temporal ID.
    std::array<std::size_t, MAX_TEMPORAL_LAYERS> bytesPerLayer{};  ///< Encoded access-unit bytes by temporal ID.

    // IDRs
    int numIdrsForced{};  ///< Successful IDRs requested by `EncodeParams::forceIdr`.
    int numIdrsTotal{};   ///< Total successful IDRs.

    // LTR
    int numLtrRecoveriesForced{};  ///< Successful recoveries requested with `useLtrSlotIdx`.
    int numLtrRecoveriesTotal{};   ///< Total successful LTR recovery frames.

    // Timers
    OpTimerStats reconfigure;    ///< Encoder model initialization and replacement time.
    OpTimerStats preprocess;     ///< NV12 input preprocessing time.
    OpTimerStats inference;      ///< Model inference time.
    OpTimerStats scaleDecoder;   ///< Latent scale decoding time.
    OpTimerStats entropyCoding;  ///< Entropy encoding time.
    OpTimerStats total;          ///< Successful `Encode()` call time.
    OpTimerStats frameInterval;  ///< Time between successfully encoded frames.
};

// --------------------------------------------------------------------------------------
// Decoder types
// --------------------------------------------------------------------------------------

/// Result of decoding one frame.
struct DecodedFrame {
    FrameInfo info{};
    int qp{};               ///< Range: 0 to 51.
    Nv12FrameView frame{};  ///< Invalidated by the next `Decode()` or decoder destruction.
};

/// Cumulative decoder counters and timing statistics.
struct DecoderStats {
    // Frames
    int numDecodeCalls{};      ///< Total `Decode()` calls.
    int numFramesAttempted{};  ///< Frames attempted after successful bitstream parsing.
    int numFramesDecoded{};    ///< Frames decoded successfully.

    // Per-layer
    std::array<int, MAX_TEMPORAL_LAYERS> framesPerLayer{};         ///< Successful frames by temporal ID.
    std::array<std::size_t, MAX_TEMPORAL_LAYERS> bytesPerLayer{};  ///< Bytes from successful calls by temporal ID.

    // IDRs
    int numIdrs{};  ///< Successfully decoded IDRs.

    // LTR
    int numLtrRecoveries{};  ///< Successfully decoded LTR recovery frames.

    // Timers
    OpTimerStats reconfigure;    ///< Decoder model initialization and replacement time.
    OpTimerStats entropyCoding;  ///< Entropy decoding time.
    OpTimerStats scaleDecoder;   ///< Latent scale decoding time.
    OpTimerStats inference;      ///< Model inference time.
    OpTimerStats postprocess;    ///< Output frame postprocessing time.
    OpTimerStats total;          ///< Successful `Decode()` call time.
    OpTimerStats frameInterval;  ///< Time between successfully decoded frames.
};

// --------------------------------------------------------------------------------------
// Manager types
// --------------------------------------------------------------------------------------

/// Requested inference compute unit. Values are append-only.
enum struct ComputeUnit {
    AUTO = 0,  ///< Resolves to NPU on supported platforms and GPU otherwise.
    CPU = 1,   ///< Uses the CPU.
    GPU = 2,   ///< Uses the GPU.
    NPU = 3,   ///< Uses the NPU.
};

/// Returns the string name of a compute unit.
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

/// Inference API used by an initialized manager.
enum struct InferenceBackend { WINDOWSML, COREML };

/// Returns the string name of an inference backend.
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

/// Serialization format of the loaded inference models.
enum struct ModelType { COREML, ONNX, UNKNOWN };

/// Returns the string name of a model type.
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

/// ONNX Runtime execution provider selected by the Windows ML backend.
enum struct OnnxExecutionProvider { CPU, DIRECTML, QNN, OPENVINO, TENSOR_RT };

/// Returns the string name of an ONNX execution provider.
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

/// Windows ML runtime deployment mode.
enum struct WinMlInitMode {
    AppSdk,         ///< Uses Windows App SDK deployment.
    SelfContained,  ///< Loads runtime DLLs beside the libmlvc module.
};

/// Returns the string name of a Windows ML initialization mode.
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

/// Parses `appsdk`, `self-contained`, or `selfcontained` as a Windows ML initialization mode.
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

/// Copyable token used to cancel manager creation; copies share cancellation state.
/// Cancellation takes effect at the next check between initialization steps.
class CancelToken {
public:
    CancelToken() : m_flag(std::make_shared<std::atomic<bool>>(false)) {}
    void Cancel() noexcept { m_flag->store(true, std::memory_order_release); }
    [[nodiscard]] bool IsCancelled() const noexcept { return m_flag->load(std::memory_order_acquire); }

private:
    std::shared_ptr<std::atomic<bool>> m_flag;
};

/// Human-readable initialization progress update.
struct StatusMessage {
    std::string message;
};

/// Windows App Runtime version discovered during Windows ML initialization.
struct WindowsAppRuntimeVersionAvailable {
    std::string version;
};

/// Windows execution-provider package information discovered during initialization.
/// Successful catalog-provider initialization emits this event when a callback is supplied, including
/// when the provider is already installed.
struct WindowsAppRuntimeEPInfoAvailable {
    OnnxExecutionProvider provider{ OnnxExecutionProvider::CPU };
    std::string epPackageVersion;
    std::optional<uint32_t> downloadTimeMs;  ///< Download duration in milliseconds, when measured.
};

/// Platform-dependent initialization progress event.
using InitializeProgressEvent =
    std::variant<StatusMessage, WindowsAppRuntimeVersionAvailable, WindowsAppRuntimeEPInfoAvailable>;

/// Callback receiving initialization progress events.
///
/// During a Windows execution-provider download, calls may arrive on a WinML worker thread; otherwise
/// they run on the thread creating the manager. All calls finish before manager creation returns.
/// The callback must be safe to call from either thread and must not throw.
using InitializeProgressCallback = std::function<void(const InitializeProgressEvent& event)>;

/// Parameters controlling manager and inference initialization.
struct ManagerParams {
    ComputeUnit computeUnit{ ComputeUnit::NPU };  ///< Inference compute unit to request.
    std::filesystem::path cacheDir{};             ///< Root cache directory; empty selects a platform default.
    bool enableModelCache{ true };                ///< Enables persistent caching of compiled inference models.
    bool enableSessionCaching{ true };  ///< Creates sessions during initialization and retains them for reuse.
    WinMlInitMode winmlInitMode{ WinMlInitMode::AppSdk };  ///< Windows ML deployment mode; ignored outside Windows.
};

/// Model-bundle limits reported for an MLVC model version.
struct Capabilities {
    int maxWidth{};                ///< Largest model width in pixels; display dimensions may be transposed to fit.
    int maxHeight{};               ///< Largest model height in pixels; display dimensions may be transposed to fit.
    float maxFps{};                ///< Maximum frames per second.
    int maxNumOfTemporalLayers{};  ///< Maximum temporal-layer count.
    int maxNumOfLtrFrames{};       ///< Maximum long-term reference slot count.
};

/// Effective manager configuration and initialized backend information.
/// Windows App Runtime and ONNX fields are meaningful only for `InferenceBackend::WINDOWSML`.
/// Version and driver strings are empty when unavailable.
struct ManagerInfo {
    bool enableModelCache{};      ///< Whether model caching is enabled.
    bool enableSessionCaching{};  ///< Whether inference session caching is enabled.

    InferenceBackend inferenceBackend{ InferenceBackend::WINDOWSML };  ///< Initialized inference backend.
    ModelType modelType{ ModelType::UNKNOWN };                         ///< Loaded model format.
    ComputeUnit computeUnit{ ComputeUnit::CPU };                       ///< Resolved inference compute unit.

    OnnxExecutionProvider onnxExecutionProvider{ OnnxExecutionProvider::CPU };  ///< Selected ONNX execution provider.
    std::string windowsAppRuntimeVersion{};                                     ///< Windows App Runtime version.
    std::string windowsAppRuntimeEpVersion{};  ///< Windows App Runtime execution-provider version.
    std::string onnxRuntimeVersion{};          ///< ONNX Runtime version.
    std::string onnxRuntimeEpVersion{};        ///< ONNX execution-provider version.

    int64_t deviceLuid{};         ///< Windows adapter locally unique identifier, or zero when unavailable.
    std::string driverVersion{};  ///< Selected accelerator driver version.
    std::string driverDate{};     ///< Selected accelerator driver date.
};

// --------------------------------------------------------------------------------------
// Parser / bitstream types
// --------------------------------------------------------------------------------------

/// Number of pixels cropped from each edge of the model output.
struct CropOffsets {
    int left{ 0 };
    int right{ 0 };
    int top{ 0 };
    int bottom{ 0 };

    bool IsEmpty() const { return left == 0 && right == 0 && top == 0 && bottom == 0; }
};

/// MLVC network abstraction layer unit type.
enum struct NaluType {
    TRAIL_N = 0,    ///< Non-reference trailing picture.
    TRAIL_R = 1,    ///< Reference trailing picture.
    TSA_N = 2,      ///< Non-reference temporal sub-layer picture.
    TSA_R = 3,      ///< Reference temporal sub-layer picture.
    IDR_N_LP = 20,  ///< Instantaneous decoding refresh picture.
    SPS = 33,       ///< Sequence parameter set.
    PPS = 34,       ///< Picture parameter set.
};

/// Common MLVC NALU header fields.
struct NaluHeader {
    int forbiddenZero{ 0 };                   ///< One-bit field that must be zero.
    NaluType naluType{ NaluType::IDR_N_LP };  ///< Six-bit NALU type.
    int layerId{ 0 };                         ///< Six-bit layer identifier.
    int temporalIdPlus1{ 1 };                 ///< Three-bit temporal identifier plus one.

    NaluHeader(NaluType type = NaluType::IDR_N_LP) : naluType(type) {}
    bool operator==(const NaluHeader&) const = default;
};

/// Parsed MLVC sequence parameter set.
struct SpsNalu {
    NaluHeader naluHeader{ NaluType::SPS };  ///< NALU header.
    int mlvcVersionMajor{ 0 };               ///< Eight-bit MLVC model major version.
    int mlvcVersionMinor{ 0 };               ///< Eight-bit MLVC model minor version.
    int spsId{ -1 };                         ///< Unsigned Exp-Golomb sequence parameter set ID; negative means unset.
    int modelWidthDiv2{ 0 };                 ///< Twelve-bit model width divided by two.
    int modelHeightDiv2{ 0 };                ///< Twelve-bit model height divided by two.
    bool transposeFlag{ false };             ///< Whether model and display orientations are transposed.
    bool cropFlag{ false };                  ///< Whether crop offsets are present.
    int cropLeftDiv2{ 0 };                   ///< Left crop divided by two when `cropFlag` is set.
    int cropRightDiv2{ 0 };                  ///< Right crop divided by two when `cropFlag` is set.
    int cropTopDiv2{ 0 };                    ///< Top crop divided by two when `cropFlag` is set.
    int cropBottomDiv2{ 0 };                 ///< Bottom crop divided by two when `cropFlag` is set.
    int maxTemporalLayersMinus1{ 0 };        ///< Three-bit maximum temporal-layer count minus one.
    int frameIdxBitsMinus8{ 0 };             ///< Frame-index field width minus eight.

    bool operator==(const SpsNalu&) const = default;
};

/// Parsed MLVC picture parameter set.
struct PpsNalu {
    NaluHeader naluHeader{ NaluType::PPS };  ///< NALU header.
    int ppsId{ -1 };                         ///< Unsigned Exp-Golomb picture parameter set ID; negative means unset.
    int spsId{ -2 };                         ///< Referenced sequence parameter set ID; negative means unset.
    int initQpMinus26{ 0 };                  ///< Signed Exp-Golomb initial QP minus 26.
};

/// State of one long-term reference slot.
struct LtrSlotInfo {
    bool empty{ true };
    int frameIdx{ 0 };  ///< Meaningful only when `empty` is false.

    LtrSlotInfo() = default;
    explicit LtrSlotInfo(int idx) : empty{ false }, frameIdx{ idx } {}

    bool HasValue() const { return !empty; }
    bool operator==(const LtrSlotInfo&) const = default;
};

/// Parsed MLVC frame-header fields.
struct FrameHeader {
    int ppsId{ 0 };                                   ///< Referenced picture parameter set ID.
    int qpDelta{ 0 };                                 ///< QP difference from the initial PPS value.
    std::array<LtrSlotInfo, MAX_LTR_SLOTS> ltrSlots;  ///< Signaled LTR slot state.
    int frameIdx{ 0 };                                ///< Current frame index for non-IDR pictures.
    int refFrameIdx{ 0 };                             ///< Reference frame index for non-IDR pictures.
    bool featureResetFlag{ false };                   ///< Whether decoder feature state is reset.
};

/// Parsed frame NALU.
struct FrameNalu {
    NaluHeader naluHeader{ NaluType::IDR_N_LP };  ///< NALU header.
    FrameHeader frameHeader;                      ///< Frame header.
    std::span<const std::byte> payload;           ///< Entropy payload stored outside this object.
};

/// Parsed custom NALU.
struct CustomNalu {
    NaluHeader naluHeader;               ///< NALU header.
    std::span<const std::byte> payload;  ///< Payload stored outside this object.
};

/// Parsed description and entropy payload for one MLVC frame.
struct FrameData {
    MlvcVersion mlvcVersion{};                        ///< Model version required to decode the frame.
    int modelWidth{};                                 ///< Model width before cropping and display transposition.
    int modelHeight{};                                ///< Model height before cropping and display transposition.
    CropOffsets cropOffsets{};                        ///< Pixels cropped from each edge of the model output.
    bool transposeFlag{};                             ///< Whether model and display orientations are transposed.
    int frameIdxBits{};                               ///< Number of bits used to encode frame indices.
    int temporalId{};                                 ///< Temporal-layer index for this frame.
    int maxTemporalLayers{};                          ///< Temporal-layer count declared by the sequence.
    FrameType frameType{ FrameType::I_FRAME };        ///< Prediction role for this frame.
    int qp{};                                         ///< Quantization parameter in the range 0 to 51.
    bool featureResetFlag{};                          ///< Whether decoder feature state is reset for this frame.
    int curFrameIdx{};                                ///< Current frame index.
    int refFrameIdx{};                                ///< Primary reference frame index.
    std::array<LtrSlotInfo, MAX_LTR_SLOTS> ltrSlots;  ///< LTR slot state signaled by the frame.
    std::span<const std::byte> payload;  ///< Entropy payload; from the parser, valid until the next `Parse()`.

    /// Returns the display width in pixels.
    int DisplayWidth() const
    {
        if (!transposeFlag) {
            return modelWidth - cropOffsets.left - cropOffsets.right;
        } else {
            return modelHeight - cropOffsets.top - cropOffsets.bottom;
        }
    }

    /// Returns the display height in pixels.
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
