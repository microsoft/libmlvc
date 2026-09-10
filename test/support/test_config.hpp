// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/types.hpp>

#include <span>
#include <string_view>

namespace libmlvc {

struct TestConfig {
    MlvcVersion mlvcVersion;
    ComputeUnit computeUnit = ComputeUnit::NPU;
    WinMlInitMode winmlInitMode = WinMlInitMode::AppSdk;
    LogLevel logLevel = LogLevel::Warn;
};

bool ParseTestConfig(std::span<const std::string_view> args);
void PrintTestConfigHelp();
const TestConfig& GetTestConfig();

}  // namespace libmlvc
