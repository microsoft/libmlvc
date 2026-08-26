// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/bitstream/coder.hpp"
#include "libmlvc/codec/codec.hpp"
#include "libmlvc/codec/manager.hpp"
#include "libmlvc/common/logging.hpp"

#include <libmlvc/libmlvc.hpp>

namespace libmlvc {

void SetLogLevel(const LogLevel level)
{
    Logger::Instance().SetLogLevel(level);
}

void SetLogHandler(LogHandler&& handler)
{
    Logger::Instance().SetHandler(std::move(handler));
}

std::filesystem::path GetDefaultModelBundlesDir()
{
    return MlvcManagerImpl::GetDefaultModelBundlesDir();
}

MlvcVersion GetDefaultModelVersion()
{
    return MlvcManagerImpl::GetDefaultModelVersion();
}

MlvcEncoder::MlvcEncoder(std::unique_ptr<MlvcEncoderImpl>&& impl) : m_impl{ std::move(impl) } {}

MlvcEncoder::~MlvcEncoder() = default;

MlvcEncoder::MlvcEncoder(MlvcEncoder&&) noexcept = default;

MlvcEncoder& MlvcEncoder::operator=(MlvcEncoder&&) noexcept = default;

expected<void> MlvcEncoder::Configure(const EncoderConfig& config)
{
    return m_impl->Configure(config);
}

expected<void> MlvcEncoder::MarkNextFrameAsLtr(const int ltrSlotIdx)
{
    return m_impl->MarkNextFrameAsLtr(ltrSlotIdx);
}

expected<FrameInfo> MlvcEncoder::GetNextFrameInfo(const EncodeParams& params) const
{
    return m_impl->GetNextFrameInfo(params);
}

expected<EncodedFrame> MlvcEncoder::Encode(const Nv12FrameView& frame, const EncodeParams& params)
{
    return m_impl->Encode(frame, params);
}

EncoderConfig MlvcEncoder::GetConfig() const
{
    return m_impl->GetConfig();
}

const EncoderStats& MlvcEncoder::GetStats() const
{
    return m_impl->GetStats();
}

MlvcDecoder::MlvcDecoder(std::unique_ptr<MlvcDecoderImpl>&& impl) : m_impl{ std::move(impl) } {}

MlvcDecoder::~MlvcDecoder() = default;

MlvcDecoder::MlvcDecoder(MlvcDecoder&&) noexcept = default;

MlvcDecoder& MlvcDecoder::operator=(MlvcDecoder&&) noexcept = default;

expected<DecodedFrame> MlvcDecoder::Decode(std::span<const std::byte> bytes)
{
    return m_impl->Decode(bytes);
}

const DecoderStats& MlvcDecoder::GetStats() const
{
    return m_impl->GetStats();
}

expected<MlvcManager> MlvcManager::CreateFromDirectory(const ManagerParams& params, const std::filesystem::path& bundlesDir,
                                                       std::span<const MlvcVersion> versions, CancelToken cancelToken,
                                                       InitializeProgressCallback progressCallback)
{
    auto impl = std::make_shared<MlvcManagerImpl>(params);
    if (auto ret = impl->Initialize(bundlesDir, versions, std::move(cancelToken), std::move(progressCallback)); !ret) {
        return ret.error();
    }
    return MlvcManager(std::move(impl));
}

expected<MlvcManager> MlvcManager::CreateFromBlobs(const ManagerParams& params,
                                                   std::span<const std::span<const std::byte>> bundleBlobs,
                                                   CancelToken cancelToken, InitializeProgressCallback progressCallback)
{
    auto impl = std::make_shared<MlvcManagerImpl>(params);
    if (auto ret = impl->Initialize(bundleBlobs, std::move(cancelToken), std::move(progressCallback)); !ret) {
        return ret.error();
    }
    return MlvcManager(std::move(impl));
}

MlvcManager::MlvcManager(std::shared_ptr<MlvcManagerImpl> impl) : m_impl{ std::move(impl) } {}

MlvcManager::~MlvcManager() = default;

std::vector<MlvcVersion> MlvcManager::GetAvailableVersions() const
{
    return m_impl->GetAvailableVersions();
}

expected<Capabilities> MlvcManager::GetCapabilities(const MlvcVersion mlvcVersion) const
{
    return m_impl->GetCapabilities(mlvcVersion);
}

expected<EncoderConfig> MlvcManager::GetDefaultEncoderConfig(const MlvcVersion mlvcVersion) const
{
    return m_impl->GetDefaultEncoderConfig(mlvcVersion);
}

expected<MlvcEncoder> MlvcManager::CreateEncoder(const EncoderConfig& config) const
{
    auto impl = m_impl->CreateEncoder(config);
    if (!impl) {
        return impl.error();
    }
    return MlvcEncoder(std::move(impl.value()));
}

expected<MlvcDecoder> MlvcManager::CreateDecoder() const
{
    auto impl = m_impl->CreateDecoder();
    if (!impl) {
        return impl.error();
    }
    return MlvcDecoder(std::move(impl.value()));
}

const ManagerInfo& MlvcManager::GetInfo() const
{
    return m_impl->GetInfo();
}

class MlvcParserImpl : public BitstreamDecoder {};

MlvcParser::MlvcParser() : m_impl{ std::make_unique<MlvcParserImpl>() } {}

MlvcParser::~MlvcParser() = default;

MlvcParser::MlvcParser(MlvcParser&&) noexcept = default;

MlvcParser& MlvcParser::operator=(MlvcParser&&) noexcept = default;

expected<FrameData> MlvcParser::Parse(std::span<const std::byte> bytes)
{
    return m_impl->Decode(bytes);
}

const SpsNalu& MlvcParser::GetSps() const
{
    return m_impl->GetSps();
}

const PpsNalu& MlvcParser::GetPps() const
{
    return m_impl->GetPps();
}

}  // namespace libmlvc
