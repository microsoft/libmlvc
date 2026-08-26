// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <charconv>
#include <iostream>
#include <regex>

namespace libmlvc {

namespace {

bool ParseMlvcVersion(const std::string_view value, MlvcVersion& version)
{
    static const std::regex versionRegex("^v(\\d+)\\.(\\d+)$");
    std::string str{ value };
    std::smatch match;
    if (!std::regex_match(str, match, versionRegex) || match.size() != 3) {
        std::cerr << "Invalid format for --mlvc-version. Expected v<major>.<minor>, got " << value << std::endl;
        return false;
    }
    int major = 0;
    int minor = 0;
    const auto majorStr = match[1].str();
    const auto minorStr = match[2].str();
    std::from_chars(majorStr.data(), majorStr.data() + majorStr.size(), major);
    std::from_chars(minorStr.data(), minorStr.data() + minorStr.size(), minor);
    version = MlvcVersion{ major, minor };
    return true;
}

bool ParseLogLevel(const std::string_view value, LogLevel& level)
{
    if (value == "debug") {
        level = LogLevel::Debug;
    } else if (value == "info") {
        level = LogLevel::Info;
    } else if (value == "warn") {
        level = LogLevel::Warn;
    } else if (value == "error") {
        level = LogLevel::Error;
    } else if (value == "fatal") {
        level = LogLevel::Fatal;
    } else {
        std::cerr << "Unknown value for --log-level: " << value << std::endl;
        return false;
    }
    return true;
}

}  // anonymous namespace

static TestConfig s_testConfig{ GetDefaultModelVersion() };

bool ParseTestConfig(std::span<char* const> args)
{
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg{ args[i] };
        if (arg == "--mlvc-version") {
            if (i + 1 >= args.size()) {
                std::cerr << "Missing value for --mlvc-version" << std::endl;
                return false;
            }
            ++i;
            if (!ParseMlvcVersion(args[i], s_testConfig.mlvcVersion)) {
                return false;
            }
        } else if (arg == "--log-level") {
            if (i + 1 >= args.size()) {
                std::cerr << "Missing value for --log-level" << std::endl;
                return false;
            }
            ++i;
            if (!ParseLogLevel(args[i], s_testConfig.logLevel)) {
                return false;
            }
        } else if (arg == "--winml-init") {
            if (i + 1 >= args.size()) {
                std::cerr << "Missing value for --winml-init" << std::endl;
                return false;
            }
            ++i;
            auto parsed = StringToWinMlInitMode(args[i]);
            if (!parsed) {
                std::cerr << "Unknown value for --winml-init: " << args[i] << " (expected 'appsdk' or 'self-contained')"
                          << std::endl;
                return false;
            }
            s_testConfig.winmlInitMode = *parsed;
        } else {
            std::cerr << "Unknown test argument: " << arg << std::endl;
            return false;
        }
    }

    return true;
}

void PrintTestConfigHelp()
{
    std::cerr << "Test config options:" << std::endl;
    std::cerr << "  --mlvc-version <v#.#>  MLVC model version (default: " << s_testConfig.mlvcVersion.ToString() << ")"
              << std::endl;
    std::cerr << "  --log-level <level>    Log level: debug, info, warn, error, fatal (default: warn)" << std::endl;
    std::cerr << "  --winml-init <mode>    WinML init mode: appsdk, self-contained (default: "
              << WinMlInitModeToString(s_testConfig.winmlInitMode) << ")" << std::endl;
}

const TestConfig& GetTestConfig()
{
    return s_testConfig;
}

}  // namespace libmlvc
