// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/entropy/rans_coder.hpp"
#include "libmlvc/schema/pmf.hpp"

#include <libmlvc/error_codes.hpp>

#include <boost/json.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace libmlvc;

namespace {

// Valid as both PMF types: two PMFs over three values and the bypass tail mass
constexpr std::string_view kValidPmf = R"({
    "pmf_lengths": [4, 4], "pmf_offsets": [1, 2],
    "pmf_table": [8192, 40960, 8192, 8192, 16384, 16384, 16384, 16384],
    "scale_min": 0.11, "scale_max": 16.0, "scale_levels": 2, "index_space": true, "qp_num": 1, "channels": 2
})";

// kValidPmf with fields overridden
boost::json::object MakePmf(std::string_view fields)
{
    auto pmf = boost::json::parse(kValidPmf).as_object();
    const auto overrides = boost::json::parse(fields);
    for (const auto& field : overrides.as_object()) {
        pmf[field.key()] = field.value();
    }
    return pmf;
}

template <typename Pmf>
expected<Pmf> ParsePmf(std::string_view fields = "{}")
{
    const auto json = boost::json::serialize(MakePmf(fields));
    return Pmf::FromJsonBuffer(std::as_bytes(std::span{ json.data(), json.size() }));
}

template <typename Coder>
bool RansAccepts(std::string_view fields = "{}")
{
    const auto pmf = MakePmf(fields);
    const auto get = [&pmf](std::string_view key) { return boost::json::value_to<std::vector<int32_t>>(pmf.at(key)); };
    Coder coder;
    return !coder.Initialize(rans::RansVariant::RansByte, get("pmf_lengths"), get("pmf_offsets"), get("pmf_table"),
                             BasePmf::kPrecisionBits);
}

template <typename Pmf>
constexpr std::string_view PmfName()
{
    if constexpr (std::is_same_v<Pmf, GaussianCoderPmf>) {
        return "GaussianCoderPmf";
    } else {
        static_assert(std::is_same_v<Pmf, BitEstimatorPmf>);
        return "BitEstimatorPmf";
    }
}

template <typename Pmf>
void ExpectInvalid(std::string_view fields, Error error = Error::invalid_argument)
{
    SCOPED_TRACE(testing::Message() << PmfName<Pmf>() << ' ' << fields);
    const auto result = ParsePmf<Pmf>(fields);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), make_error_code(error));
}

template <typename Pmf>
void ExpectParseError(std::string_view json)
{
    SCOPED_TRACE(PmfName<Pmf>());
    const auto result = Pmf::FromJsonBuffer(std::as_bytes(std::span{ json.data(), json.size() }));
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), make_error_code(Error::json_parse_error));
}

}  // namespace

TEST(UnitTestPmf, ParsesValidPmfs)
{
    const auto gaussian = ParsePmf<GaussianCoderPmf>();
    ASSERT_TRUE(gaussian) << gaussian.error().message();
    EXPECT_EQ(gaussian.value().scaleLevels, 2);
    const auto bitEstimator = ParsePmf<BitEstimatorPmf>();
    ASSERT_TRUE(bitEstimator) << bitEstimator.error().message();
    EXPECT_EQ(bitEstimator.value().channels, 2);
    EXPECT_TRUE(RansAccepts<rans::EntropyEncoder>());
    EXPECT_TRUE(RansAccepts<rans::EntropyDecoder>());
}

TEST(UnitTestPmf, RejectsInvalidPmfs)
{
    // The tables that the rANS coder rejects
    for (const std::string_view fields : {
             // With a matching scale_levels, so that the Gaussian count check doesn't also reject it
             R"({"pmf_lengths": [], "pmf_offsets": [], "pmf_table": [], "scale_levels": 0})",
             R"({"pmf_offsets": [1]})",
             R"({"pmf_lengths": [1, 7]})",
             R"({"pmf_offsets": [-1, 2]})",
             R"({"pmf_offsets": [1, 3]})",
             R"({"pmf_table": [8192, 40960, 8192, 8192, 16384, 16384, 16384]})",
             R"({"pmf_table": [8192, 40960, 8192, 8192, 16384, 16384, 16384, 16384, 16384]})",
             R"({"pmf_table": [8192, 40960, 8192, 8192, 0, 16384, 16384, 32768]})",
             R"({"pmf_table": [8192, 40960, 8192, 8192, -16384, 16384, 32768, 32768]})",
             R"({"pmf_table": [8192, 40960, 8192, 8193, 16384, 16384, 16384, 16384]})",
         }) {
        EXPECT_FALSE(RansAccepts<rans::EntropyEncoder>(fields)) << fields;
        EXPECT_FALSE(RansAccepts<rans::EntropyDecoder>(fields)) << fields;
        ExpectInvalid<GaussianCoderPmf>(fields);
        ExpectInvalid<BitEstimatorPmf>(fields);
    }

    // One PMF per scale level, and per QP and channel
    for (const std::string_view fields : { R"({"scale_levels": 0})", R"({"scale_levels": 3})" }) {
        ExpectInvalid<GaussianCoderPmf>(fields);
    }
    for (const std::string_view fields : { R"({"qp_num": 2})", R"({"channels": 1})", R"({"qp_num": -1, "channels": -2})" }) {
        ExpectInvalid<BitEstimatorPmf>(fields);
    }
    // qp_num * channels overflows int to 2
    ExpectInvalid<BitEstimatorPmf>(R"({"qp_num": 65538, "channels": 1073709057})");

    for (const std::string_view fields : { R"({"pmf_offsets": [1, 1.5]})", R"({"pmf_offsets": [1, 2147483648]})" }) {
        ExpectInvalid<GaussianCoderPmf>(fields, Error::json_parse_error);
        ExpectInvalid<BitEstimatorPmf>(fields, Error::json_parse_error);
    }

    // Overflows the stack without Boost.JSON's depth limit
    const auto tooDeep = std::string(100000, '[') + std::string(100000, ']');
    for (const std::string_view json : { "[]", "{}", tooDeep.c_str() }) {
        ExpectParseError<GaussianCoderPmf>(json);
        ExpectParseError<BitEstimatorPmf>(json);
    }
}
