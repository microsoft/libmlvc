// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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

namespace libmlvc {

// Forward declarations
class MlvcEncoderImpl;
class MlvcDecoderImpl;
class MlvcManagerImpl;
class MlvcParserImpl;

LIBMLVC_EXPORT void SetLogLevel(const LogLevel level);
LIBMLVC_EXPORT void SetLogHandler(LogHandler&& handler);
LIBMLVC_EXPORT std::filesystem::path GetDefaultModelBundlesDir();
LIBMLVC_EXPORT MlvcVersion GetDefaultModelVersion();

class LIBMLVC_EXPORT MlvcEncoder {
public:
    ~MlvcEncoder();
    MlvcEncoder(const MlvcEncoder&) = delete;
    MlvcEncoder& operator=(const MlvcEncoder&) = delete;
    MlvcEncoder(MlvcEncoder&&) noexcept;
    MlvcEncoder& operator=(MlvcEncoder&&) noexcept;
    expected<void> Configure(const EncoderConfig& config);
    expected<void> MarkNextFrameAsLtr(const int ltrSlotIdx);
    expected<FrameInfo> GetNextFrameInfo(const EncodeParams& params = {}) const;
    expected<EncodedFrame> Encode(const Nv12FrameView& frame, const EncodeParams& params = {});
    EncoderConfig GetConfig() const;
    const EncoderStats& GetStats() const;

private:
    friend class MlvcManager;
    MlvcEncoder(std::unique_ptr<MlvcEncoderImpl>&& impl);
    std::unique_ptr<MlvcEncoderImpl> m_impl;
};

class LIBMLVC_EXPORT MlvcDecoder {
public:
    ~MlvcDecoder();
    MlvcDecoder(const MlvcDecoder&) = delete;
    MlvcDecoder& operator=(const MlvcDecoder&) = delete;
    MlvcDecoder(MlvcDecoder&&) noexcept;
    MlvcDecoder& operator=(MlvcDecoder&&) noexcept;
    expected<DecodedFrame> Decode(std::span<const std::byte> bytes);
    const DecoderStats& GetStats() const;

private:
    friend class MlvcManager;
    MlvcDecoder(std::unique_ptr<MlvcDecoderImpl>&& impl);
    std::unique_ptr<MlvcDecoderImpl> m_impl;
};

// MlvcManager is a copyable, thread-safe handle. Copies share the same manager state.
// Created encoders and decoders are not thread-safe; each instance must be used from a single thread.
class LIBMLVC_EXPORT MlvcManager {
public:
    static expected<MlvcManager> CreateFromDirectory(const ManagerParams& params,
                                                     const std::filesystem::path& bundlesDir = {},
                                                     std::span<const MlvcVersion> versions = {},
                                                     CancelToken cancelToken = {},
                                                     InitializeProgressCallback progressCallback = nullptr);
    static expected<MlvcManager> CreateFromBlobs(const ManagerParams& params,
                                                 std::span<const std::span<const std::byte>> bundleBlobs,
                                                 CancelToken cancelToken = {},
                                                 InitializeProgressCallback progressCallback = nullptr);
    MlvcManager(const MlvcManager&) = default;
    MlvcManager& operator=(const MlvcManager&) = default;
    MlvcManager(MlvcManager&&) noexcept = default;
    MlvcManager& operator=(MlvcManager&&) noexcept = default;
    ~MlvcManager();
    std::vector<MlvcVersion> GetAvailableVersions() const;
    expected<Capabilities> GetCapabilities(const MlvcVersion mlvcVersion) const;
    expected<EncoderConfig> GetDefaultEncoderConfig(const MlvcVersion mlvcVersion) const;
    expected<MlvcEncoder> CreateEncoder(const EncoderConfig& config) const;
    expected<MlvcDecoder> CreateDecoder() const;
    const ManagerInfo& GetInfo() const;

private:
    MlvcManager(std::shared_ptr<MlvcManagerImpl> impl);
    std::shared_ptr<MlvcManagerImpl> m_impl;
};

class LIBMLVC_EXPORT MlvcParser {
public:
    MlvcParser();
    ~MlvcParser();
    MlvcParser(MlvcParser&&) noexcept;
    MlvcParser& operator=(MlvcParser&&) noexcept;
    expected<FrameData> Parse(std::span<const std::byte> bytes);
    const SpsNalu& GetSps() const;
    const PpsNalu& GetPps() const;

private:
    std::unique_ptr<MlvcParserImpl> m_impl;
};

}  // namespace libmlvc

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif
