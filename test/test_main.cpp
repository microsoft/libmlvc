// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

using namespace libmlvc;

namespace {

std::vector<char*> g_benchmarkArgs;

void StdoutLogHandler(LogLevel level, const std::source_location& loc, std::string_view msg) noexcept
{
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf;
#if defined(_WIN32)
    localtime_s(&tmBuf, &timeNow);
#else
    localtime_r(&timeNow, &tmBuf);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tmBuf, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count();

    const char* basename = loc.file_name();
    for (const char* p = loc.file_name(); *p; ++p)
        if (*p == '/' || *p == '\\') basename = p + 1;

    std::fprintf(stdout, "%s [%5s][%s:%d]: %.*s\n", ss.str().c_str(), LogLevelToString(level), basename, loc.line(),
                 static_cast<int>(msg.size()), msg.data());
}

TEST(BENCHMARKS, Run)
{
    SetLogLevel(LogLevel::Warn);

    int sz = static_cast<int>(g_benchmarkArgs.size()) - 1;
    ::benchmark::Initialize(&sz, g_benchmarkArgs.data());
    ::benchmark::ConsoleReporter cr(::benchmark::ConsoleReporter::OO_ColorTabular);
    ::benchmark::RunSpecifiedBenchmarks(&cr);

    // memory leak in google benchmark workaround: https://github.com/google/benchmark/issues/782
    delete (const_cast<::benchmark::CPUInfo*>(&::benchmark::CPUInfo::Get()));
    delete (const_cast<::benchmark::SystemInfo*>(&::benchmark::SystemInfo::Get()));

    SetLogLevel(GetTestConfig().logLevel);
}

}  // anonymous namespace

int testmain(int argc, char** argv)
{
    // Split gtest, gbenchmark, and custom command line arguments
    std::vector<char*> gtestArgs = { argv[0] };
    g_benchmarkArgs.push_back(argv[0]);
    std::vector<std::string_view> configArgs;
    bool helpRequested = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{ argv[i] };
        if (arg == "--help") {
            helpRequested = true;
            gtestArgs.push_back(argv[i]);
        } else if (arg.starts_with("--benchmark")) {
            g_benchmarkArgs.push_back(argv[i]);
        } else if (arg.starts_with("--gtest") || arg.starts_with("--seed")) {
            gtestArgs.push_back(argv[i]);
        } else {
            configArgs.push_back(arg);
        }
    }
    gtestArgs.push_back(nullptr);
    g_benchmarkArgs.push_back(nullptr);

    if (helpRequested) {
        PrintTestConfigHelp();
    }

    if (!ParseTestConfig(configArgs)) {
        PrintTestConfigHelp();
        return 1;
    }
    const auto& testConfig = GetTestConfig();
    SetLogLevel(testConfig.logLevel);
    SetLogHandler(StdoutLogHandler);

    int sz = static_cast<int>(gtestArgs.size()) - 1;
    ::testing::InitGoogleTest(&sz, gtestArgs.data());

    if (!IsPlatformSupported()) {  // checks minimum OS version and CPU features (FP16, AVX2)
        std::cerr << "Platform is not supported for MLVC, some tests may fail\n";
    }

    int ret = RUN_ALL_TESTS();
    g_benchmarkArgs.clear();
    return ret;
}

// Provide the real entry point unless an external test harness supplies its own main().
#if defined(LIBMLVC_PROVIDE_TESTMAIN_ENTRY)
int main(int argc, char** argv)
{
    return testmain(argc, argv);
}
#endif
