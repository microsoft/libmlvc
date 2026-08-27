// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <regex>

namespace libmlvc {

namespace {

bool ParseMlvcVersion(std::string_view value, MlvcVersion& version)
{
    static const std::regex versionRegex("^v(\\d+)\\.(\\d+)$");
    std::string str{ value };
    std::smatch match;
    if (!std::regex_match(str, match, versionRegex) || match.size() != 3) {
        std::cerr << "Invalid MLVC version: " << value << " (expected v<major>.<minor>)" << std::endl;
        return false;
    }
    int major = 0;
    int minor = 0;
    const auto majorStr = match[1].str();
    const auto minorStr = match[2].str();
    if (std::from_chars(majorStr.data(), majorStr.data() + majorStr.size(), major).ec != std::errc{}
        || std::from_chars(minorStr.data(), minorStr.data() + minorStr.size(), minor).ec != std::errc{}) {
        std::cerr << "Invalid MLVC version number: " << value << std::endl;
        return false;
    }
    version = MlvcVersion{ major, minor };
    return true;
}

bool ParseComputeUnit(std::string_view value, ComputeUnit& computeUnit)
{
    if (value == "auto") {
        computeUnit = ComputeUnit::AUTO;
    } else if (value == "cpu") {
        computeUnit = ComputeUnit::CPU;
    } else if (value == "gpu") {
        computeUnit = ComputeUnit::GPU;
    } else if (value == "npu") {
        computeUnit = ComputeUnit::NPU;
    } else {
        std::cerr << "Unknown compute unit: " << value << " (expected 'auto', 'cpu', 'gpu', or 'npu')" << std::endl;
        return false;
    }
    return true;
}

bool ParseWinMlInitMode(std::string_view value, WinMlInitMode& mode)
{
    auto parsed = StringToWinMlInitMode(value);
    if (!parsed) {
        std::cerr << "Unknown WinML init mode: " << value << " (expected 'appsdk' or 'self-contained')" << std::endl;
        return false;
    }
    mode = *parsed;
    return true;
}

bool ParseLogLevel(std::string_view value, LogLevel& level)
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
        std::cerr << "Unknown log level: " << value << " (expected 'debug', 'info', 'warn', 'error', or 'fatal')"
                  << std::endl;
        return false;
    }
    return true;
}

const char* GetEnvironmentVariable(const char* name)
{
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif
    const char* value = std::getenv(name);
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif
    return value;
}

}  // anonymous namespace

static TestConfig s_testConfig{ GetDefaultModelVersion() };

bool ParseTestConfig(std::span<const std::string_view> args)
{
    const char* mlvcVersion = GetEnvironmentVariable("LIBMLVC_TEST_MLVC_VERSION");
    if (mlvcVersion && !ParseMlvcVersion(mlvcVersion, s_testConfig.mlvcVersion)) {
        return false;
    }
    const char* computeUnit = GetEnvironmentVariable("LIBMLVC_TEST_COMPUTE_UNIT");
    if (computeUnit && !ParseComputeUnit(computeUnit, s_testConfig.computeUnit)) {
        return false;
    }
    const char* winMlInit = GetEnvironmentVariable("LIBMLVC_TEST_WINML_INIT");
    if (winMlInit && !ParseWinMlInitMode(winMlInit, s_testConfig.winmlInitMode)) {
        return false;
    }
    const char* logLevel = GetEnvironmentVariable("LIBMLVC_TEST_LOG_LEVEL");
    if (logLevel && !ParseLogLevel(logLevel, s_testConfig.logLevel)) {
        return false;
    }

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--mlvc-version") {
            if (i + 1 >= args.size()) {
                std::cerr << "Missing value for --mlvc-version" << std::endl;
                return false;
            }
            ++i;
            if (!ParseMlvcVersion(args[i], s_testConfig.mlvcVersion)) {
                return false;
            }
        } else if (arg == "--compute-unit") {
            if (i + 1 >= args.size()) {
                std::cerr << "Missing value for --compute-unit" << std::endl;
                return false;
            }
            ++i;
            if (!ParseComputeUnit(args[i], s_testConfig.computeUnit)) {
                return false;
            }
        } else if (arg == "--winml-init") {
            if (i + 1 >= args.size()) {
                std::cerr << "Missing value for --winml-init" << std::endl;
                return false;
            }
            ++i;
            if (!ParseWinMlInitMode(args[i], s_testConfig.winmlInitMode)) {
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
    std::cerr << "  --compute-unit <unit>  Compute unit: auto, cpu, gpu, npu (default: "
              << ComputeUnitToString(s_testConfig.computeUnit) << ")" << std::endl;
    std::cerr << "  --winml-init <mode>    WinML init mode: appsdk, self-contained (default: "
              << WinMlInitModeToString(s_testConfig.winmlInitMode) << ")" << std::endl;
    std::cerr << "  --log-level <level>    Log level: debug, info, warn, error, fatal (default: "
              << LogLevelToString(s_testConfig.logLevel) << ")" << std::endl;
    std::cerr << "Environment defaults: LIBMLVC_TEST_MLVC_VERSION, LIBMLVC_TEST_COMPUTE_UNIT, "
                 "LIBMLVC_TEST_WINML_INIT, LIBMLVC_TEST_LOG_LEVEL (CLI overrides environment)"
              << std::endl;
}

const TestConfig& GetTestConfig()
{
    return s_testConfig;
}

}  // namespace libmlvc
