// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/schema/pmf.hpp"
#include "libmlvc/common/logging.hpp"

#include <libmlvc/error_codes.hpp>

#include <boost/json.hpp>

#include <string>
#include <string_view>

namespace libmlvc {
namespace {

template <typename T>
std::error_code ParseRequired(const boost::json::object& parent, std::string_view key, T& out)
{
    const auto value = parent.if_contains(key);
    if (!value) {
        MLVC_LOG_ERROR("Missing required field %s", std::string(key).c_str());
        return make_error_code(Error::json_parse_error);
    }

    auto result = boost::json::try_value_to<T>(*value);
    if (!result) {
        MLVC_LOG_ERROR("Failed to parse %s", std::string(key).c_str());
        return result.error();
    }
    out = std::move(result.value());
    return {};
}

}  // namespace

boost::json::result_for<GaussianCoderPmf, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<GaussianCoderPmf>&, const boost::json::value& value)
{
    if (!value.is_object()) {
        MLVC_LOG_ERROR("GaussianCoderPmf JSON value is not an object");
        return make_error_code(Error::json_parse_error);
    }
    const auto& object = value.as_object();

    GaussianCoderPmf result{};
    if (auto error = ParseRequired(object, "pmf_lengths", result.pmfLengths); error) return error;
    if (auto error = ParseRequired(object, "pmf_offsets", result.pmfOffsets); error) return error;
    if (auto error = ParseRequired(object, "pmf_table", result.pmfTable); error) return error;
    if (auto error = ParseRequired(object, "scale_min", result.scaleMin); error) return error;
    if (auto error = ParseRequired(object, "scale_max", result.scaleMax); error) return error;
    if (auto error = ParseRequired(object, "scale_levels", result.scaleLevels); error) return error;
    if (auto error = ParseRequired(object, "index_space", result.indexSpace); error) return error;
    return result;
}

boost::json::result_for<BitEstimatorPmf, boost::json::value>::type
tag_invoke(const boost::json::try_value_to_tag<BitEstimatorPmf>&, const boost::json::value& value)
{
    if (!value.is_object()) {
        MLVC_LOG_ERROR("BitEstimatorPmf JSON value is not an object");
        return make_error_code(Error::json_parse_error);
    }
    const auto& object = value.as_object();

    BitEstimatorPmf result{};
    if (auto error = ParseRequired(object, "pmf_lengths", result.pmfLengths); error) return error;
    if (auto error = ParseRequired(object, "pmf_offsets", result.pmfOffsets); error) return error;
    if (auto error = ParseRequired(object, "pmf_table", result.pmfTable); error) return error;
    if (auto error = ParseRequired(object, "qp_num", result.qpNum); error) return error;
    if (auto error = ParseRequired(object, "channels", result.channels); error) return error;
    return result;
}

expected<GaussianCoderPmf> GaussianCoderPmf::FromJsonBuffer(std::span<const std::byte> buffer)
{
    std::error_code error;
    auto value = boost::json::parse({ reinterpret_cast<const char*>(buffer.data()), buffer.size() }, error);
    if (error) {
        MLVC_LOG_ERROR("Failed to parse GaussianCoderPmf JSON data: %s", error.message().c_str());
        return make_error_code(Error::json_parse_error);
    }

    auto result = boost::json::try_value_to<GaussianCoderPmf>(value);
    if (!result) {
        MLVC_LOG_ERROR("Failed to convert JSON value to GaussianCoderPmf");
        return make_error_code(Error::json_parse_error);
    }
    return result.value();
}

expected<BitEstimatorPmf> BitEstimatorPmf::FromJsonBuffer(std::span<const std::byte> buffer)
{
    std::error_code error;
    auto value = boost::json::parse({ reinterpret_cast<const char*>(buffer.data()), buffer.size() }, error);
    if (error) {
        MLVC_LOG_ERROR("Failed to parse BitEstimatorPmf JSON data: %s", error.message().c_str());
        return make_error_code(Error::json_parse_error);
    }

    auto result = boost::json::try_value_to<BitEstimatorPmf>(value);
    if (!result) {
        MLVC_LOG_ERROR("Failed to convert JSON value to BitEstimatorPmf");
        return make_error_code(Error::json_parse_error);
    }
    return result.value();
}

}  // namespace libmlvc
