// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/types.hpp>

#include <span>

namespace libmlvc {

struct TestConfig {
    MlvcVersion mlvcVersion;
    LogLevel logLevel = LogLevel::Warn;
    WinMlInitMode winmlInitMode = WinMlInitMode::AppSdk;
};

bool ParseTestConfig(std::span<char* const> args);
void PrintTestConfigHelp();
const TestConfig& GetTestConfig();

}  // namespace libmlvc
