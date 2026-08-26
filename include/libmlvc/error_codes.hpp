// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/export.hpp>

#include <system_error>

namespace libmlvc {

// Append-only: don't reuse or change values
enum class Error : int {
    // Generic errors
    io_error = 1,
    general_failure = 2,
    operation_cancelled = 3,
    invalid_argument = 4,
    json_parse_error = 5,
    not_initialized = 6,
    command_line_parse_error = 7,
    // Execution provider errors (WinML catalog/download/register)
    ep_download_error = 8,
    ep_register_error = 9,
    // Model/inference errors
    model_init_error = 10,
    model_inference_error = 11,
    // Decoder runtime errors
    reference_error = 12,
    // Bitstream errors
    bit_stream_unexpected_error = 13,
    bit_stream_missing_sps_error = 14,
    bit_stream_missing_pps_error = 15,
    bit_stream_partial_access_unit_error = 16,
    // Entropy coder errors
    entropy_coder_init_error = 17,
    entropy_coder_encode_error = 18,
    entropy_coder_decode_error = 19,
    // Other
    incompatible_driver_version_error = 20,
    unsupported_platform_error = 21,
    windows_app_runtime_unavailable_error = 22,
    incompatible_ep_version_error = 23,
};

LIBMLVC_EXPORT const std::error_category& error_category() noexcept;

inline std::error_code make_error_code(Error e) noexcept
{
    return std::error_code(static_cast<int>(e), error_category());
}

inline bool IsPartialAccessUnitError(const std::error_code& ec) noexcept
{
    return ec == make_error_code(Error::bit_stream_partial_access_unit_error);
}

inline bool IsRecoverableDecoderError(const std::error_code& ec) noexcept
{
    // Decoder errors that can be recovered by receiving a fresh IDR frame.
    // Excludes fatal errors (model init, inference, encoder-side) that require re-initialization.
    return ec == make_error_code(Error::reference_error) || ec == make_error_code(Error::bit_stream_missing_sps_error)
           || ec == make_error_code(Error::bit_stream_missing_pps_error)
           || ec == make_error_code(Error::bit_stream_partial_access_unit_error)
           || ec == make_error_code(Error::bit_stream_unexpected_error)
           || ec == make_error_code(Error::entropy_coder_decode_error);
}
}  // namespace libmlvc

template <>
struct std::is_error_code_enum<libmlvc::Error> : true_type {};
