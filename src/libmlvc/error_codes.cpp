// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/macros.hpp"

#include <libmlvc/error_codes.hpp>

namespace libmlvc {

class LibMlvcErrorCategory final : public std::error_category {
    const char* name() const noexcept override { return "libmlvc"; }

    std::string message(int ev) const noexcept override
    {
        switch (static_cast<Error>(ev)) {
        case Error::io_error:
            return "I/O error";
        case Error::general_failure:
            return "general failure";
        case Error::operation_cancelled:
            return "operation cancelled";
        case Error::invalid_argument:
            return "invalid argument";
        case Error::json_parse_error:
            return "JSON parse error";
        case Error::not_initialized:
            return "not initialized";
        case Error::command_line_parse_error:
            return "command line parse error";
        case Error::ep_download_error:
            return "execution provider download error";
        case Error::ep_register_error:
            return "execution provider registration error";
        case Error::model_init_error:
            return "model initialization error";
        case Error::model_inference_error:
            return "model inference error";
        case Error::reference_error:
            return "Reference error";
        case Error::bit_stream_unexpected_error:
            return "bit stream unexpected error";
        case Error::bit_stream_missing_sps_error:
            return "bit stream missing SPS error";
        case Error::bit_stream_missing_pps_error:
            return "bit stream missing PPS error";
        case Error::bit_stream_partial_access_unit_error:
            return "bit stream partial access unit error";
        case Error::entropy_coder_init_error:
            return "entropy coder initialization error";
        case Error::entropy_coder_encode_error:
            return "entropy coder encode error";
        case Error::entropy_coder_decode_error:
            return "entropy coder decode error";
        case Error::incompatible_driver_version_error:
            return "incompatible driver version error";
        case Error::unsupported_platform_error:
            return "unsupported platform error";
        case Error::windows_app_runtime_unavailable_error:
            return "Windows App Runtime unavailable or below minimum version";
        case Error::incompatible_ep_version_error:
            return "incompatible execution provider version error";
        default:
            MLVC_ASSERT(false);
            return "unknown error code";
        }
    }
};
}  // namespace libmlvc

const std::error_category& libmlvc::error_category() noexcept
{
    static const LibMlvcErrorCategory category;
    return category;
}
