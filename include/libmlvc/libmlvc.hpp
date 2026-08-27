// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/// @file
/// Encoder, decoder, manager, parser, and logging API.

#pragma once
#include <libmlvc/build_info.hpp>
#include <libmlvc/expected.hpp>
#include <libmlvc/export.hpp>
#include <libmlvc/platform_info.hpp>
#include <libmlvc/types.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4251)  // PIMPL smart-pointer member of a dll-interface class (private, safe)
#endif

/// APIs for loading MLVC models and encoding, decoding, and parsing video.
namespace libmlvc {

// Forward declarations
class MlvcEncoderImpl;
class MlvcDecoderImpl;
class MlvcManagerImpl;
class MlvcParserImpl;

/// Sets the process-wide minimum log level.
LIBMLVC_EXPORT void SetLogLevel(const LogLevel level);

/// Replaces the process-wide log handler.
///
/// An empty handler disables logging. Calls are serialized. The handler must not throw, call
/// `SetLogHandler`, or invoke libmlvc operations that may log; doing so can deadlock. Its arguments
/// are valid only for the call.
LIBMLVC_EXPORT void SetLogHandler(LogHandler&& handler);

/// Returns the model-bundle directory used when no directory is supplied.
///
/// Uses the `LIBMLVC_MODEL_BUNDLES_DIR` environment variable when set, otherwise `./data/model_bundles`.
LIBMLVC_EXPORT std::filesystem::path GetDefaultModelBundlesDir();

/// Returns the MLVC model version selected when no version is specified.
LIBMLVC_EXPORT MlvcVersion GetDefaultModelVersion();

/// Move-only MLVC encoder.
///
/// Keeps the shared manager state alive and may outlive all `MlvcManager` handles. Instances are
/// not thread-safe.
class LIBMLVC_EXPORT MlvcEncoder {
public:
    ~MlvcEncoder();
    MlvcEncoder(const MlvcEncoder&) = delete;
    MlvcEncoder& operator=(const MlvcEncoder&) = delete;
    MlvcEncoder(MlvcEncoder&&) noexcept;
    MlvcEncoder& operator=(MlvcEncoder&&) noexcept;

    /// Updates the encoder configuration.
    /// @param config Configuration whose structural fields are checked immediately. Model version and
    /// resolution support are checked by the next `Encode()`.
    /// @return Success or `Error::invalid_argument`. A changed configuration clears a pending LTR request.
    expected<void> Configure(const EncoderConfig& config);

    /// Requests that the next temporal-layer-0 frame be stored in an external LTR slot.
    ///
    /// Only external LTR mode honors the request; internal mode ignores it. In external mode, an IDR
    /// is stored in slot zero and discards any pending request.
    /// @param ltrSlotIdx Destination slot, from zero to `ltrNumSlots - 1`. Validation occurs on the
    /// next `Encode()`; an invalid request remains pending until replaced or cleared by configuration.
    expected<void> MarkNextFrameAsLtr(const int ltrSlotIdx);

    /// Predicts the next frame without changing encoder state.
    /// @param params Per-frame controls to use for the prediction.
    /// @return Predicted frame information or an error. Before the first encode or after changing version
    /// or size, only `frameType` is populated and the other fields are zero.
    expected<FrameInfo> GetNextFrameInfo(const EncodeParams& params = {}) const;

    /// Encodes one NV12 frame.
    /// @param frame Frame whose dimensions match the configuration, stride is at least its width, and
    /// planes contain `stride * height` Y bytes and `stride * height / 2` UV bytes. Storage must remain
    /// valid until this call returns.
    /// @param params Per-frame quantization, IDR, and LTR recovery controls.
    /// @return An encoded frame, or an error. Its bitstream remains valid until the next `Encode()` or
    /// encoder destruction.
    expected<EncodedFrame> Encode(const Nv12FrameView& frame, const EncodeParams& params = {});

    /// Returns the currently stored encoder configuration.
    EncoderConfig GetConfig() const;

    /// Returns cumulative statistics; the reference remains valid until encoder destruction.
    const EncoderStats& GetStats() const;

private:
    friend class MlvcManager;
    MlvcEncoder(std::unique_ptr<MlvcEncoderImpl>&& impl);
    std::unique_ptr<MlvcEncoderImpl> m_impl;
};

/// Move-only MLVC decoder.
///
/// Keeps the shared manager state alive and may outlive all `MlvcManager` handles. Instances are
/// not thread-safe.
class LIBMLVC_EXPORT MlvcDecoder {
public:
    ~MlvcDecoder();
    MlvcDecoder(const MlvcDecoder&) = delete;
    MlvcDecoder& operator=(const MlvcDecoder&) = delete;
    MlvcDecoder(MlvcDecoder&&) noexcept;
    MlvcDecoder& operator=(MlvcDecoder&&) noexcept;

    /// Decodes MLVC NALUs until a complete frame is available.
    /// @param bytes One or more complete NALUs; a frame NALU, if present, must be last. Parameter-set
    /// state is retained across calls, but incomplete NALU bytes are not buffered. Storage must remain
    /// valid until this call returns.
    /// @return A decoded frame, `Error::bit_stream_partial_access_unit_error` if the input ends before
    /// a complete frame is available, or another error. The frame remains valid until the next
    /// `Decode()` or decoder destruction. Use `IsRecoverableDecoderError()` to identify errors
    /// recoverable by IDR.
    expected<DecodedFrame> Decode(std::span<const std::byte> bytes);

    /// Returns cumulative statistics; the reference remains valid until decoder destruction.
    const DecoderStats& GetStats() const;

private:
    friend class MlvcManager;
    MlvcDecoder(std::unique_ptr<MlvcDecoderImpl>&& impl);
    std::unique_ptr<MlvcDecoderImpl> m_impl;
};

/// Copyable, thread-safe handle to loaded models and the inference backend.
///
/// Copies share state. Manager calls and separate encoders or decoders may run concurrently; each
/// encoder or decoder instance must be used by only one thread at a time.
class LIBMLVC_EXPORT MlvcManager {
public:
    /// Creates a manager from model bundles in a directory.
    /// @param params Compute, cache, and Windows ML initialization settings.
    /// @param bundlesDir Bundle directory, or empty to use `GetDefaultModelBundlesDir()`.
    /// @param versions Model versions to load, or empty to use `GetDefaultModelVersion()`.
    /// @param cancelToken Token checked for cancellation between initialization steps.
    /// @param progressCallback Optional initialization progress callback.
    /// @return An initialized manager, or an initialization error.
    static expected<MlvcManager> CreateFromDirectory(const ManagerParams& params,
                                                     const std::filesystem::path& bundlesDir = {},
                                                     std::span<const MlvcVersion> versions = {},
                                                     CancelToken cancelToken = {},
                                                     InitializeProgressCallback progressCallback = nullptr);

    /// Creates a manager from in-memory model bundles.
    /// @param params Compute, cache, and Windows ML initialization settings.
    /// @param bundleBlobs Model bundles to copy. Their storage must remain valid and unchanged until this call returns.
    /// @param cancelToken Token checked for cancellation between initialization steps.
    /// @param progressCallback Optional initialization progress callback.
    /// @return An initialized manager, or an initialization error.
    static expected<MlvcManager> CreateFromBlobs(const ManagerParams& params,
                                                 std::span<const std::span<const std::byte>> bundleBlobs,
                                                 CancelToken cancelToken = {},
                                                 InitializeProgressCallback progressCallback = nullptr);
    MlvcManager(const MlvcManager&) = default;
    MlvcManager& operator=(const MlvcManager&) = default;
    MlvcManager(MlvcManager&&) noexcept = default;
    MlvcManager& operator=(MlvcManager&&) noexcept = default;
    ~MlvcManager();

    /// Returns loaded MLVC model versions in ascending major/minor order.
    std::vector<MlvcVersion> GetAvailableVersions() const;

    /// Returns capabilities for an exact loaded MLVC model version.
    expected<Capabilities> GetCapabilities(const MlvcVersion mlvcVersion) const;

    /// Returns the default encoder configuration for an exact loaded MLVC model version.
    expected<EncoderConfig> GetDefaultEncoderConfig(const MlvcVersion mlvcVersion) const;

    /// Creates an encoder; its model is initialized by the first `Encode()`.
    expected<MlvcEncoder> CreateEncoder(const EncoderConfig& config) const;

    /// Creates a decoder; its model is selected and initialized from the first decodable frame.
    expected<MlvcDecoder> CreateDecoder() const;

    /// Returns initialized backend details.
    /// The reference remains valid while the shared state is held by a manager, encoder, or decoder.
    const ManagerInfo& GetInfo() const;

private:
    MlvcManager(std::shared_ptr<MlvcManagerImpl> impl);
    std::shared_ptr<MlvcManagerImpl> m_impl;
};

/// Move-only, single-threaded parser for MLVC access units.
class LIBMLVC_EXPORT MlvcParser {
public:
    MlvcParser();
    ~MlvcParser();
    MlvcParser(MlvcParser&&) noexcept;
    MlvcParser& operator=(MlvcParser&&) noexcept;

    /// Parses MLVC NALUs until a complete frame is available without decoding its video payload.
    /// @param bytes One or more complete NALUs; a frame NALU, if present, must be last. Parameter-set
    /// state is retained across calls, but incomplete NALU bytes are not buffered. Storage must remain
    /// valid until this call returns.
    /// @return Parsed frame data, `Error::bit_stream_partial_access_unit_error` if the input ends before
    /// a complete frame is available, or another error. The payload remains valid until the next
    /// `Parse()` or parser destruction.
    expected<FrameData> Parse(std::span<const std::byte> bytes);

    /// Returns parser-owned sequence parameters that may change after `Parse()`.
    const SpsNalu& GetSps() const;

    /// Returns parser-owned picture parameters that may change after `Parse()`.
    const PpsNalu& GetPps() const;

private:
    std::unique_ptr<MlvcParserImpl> m_impl;
};

}  // namespace libmlvc

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif
