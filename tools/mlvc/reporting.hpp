// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <string>
#include <vector>

namespace libmlvc {

struct BuildInfo;
struct PlatformInfo;
struct ManagerInfo;
struct MlvcVersion;
struct Capabilities;
struct EncoderConfig;
struct SnapshotResult;
struct InteropResult;
struct ValidationTestSummary;
class BenchmarkResults;
class Dataset;

// Utilities
const char* ColorCode(const char* code);

// Building blocks
void PrintBuildInfo(const BuildInfo& buildInfo);
void PrintPlatformInfo(const PlatformInfo& platform);
void PrintEnvironment(const char* label, const BuildInfo& buildInfo, const PlatformInfo& platform, const ManagerInfo& info);
void PrintManagerInfo(const ManagerInfo& info);
void PrintCapabilities(const std::vector<MlvcVersion>& versions, const std::vector<Capabilities>& capabilities);
void PrintEncoderConfig(const EncoderConfig& config, const char* indent = "");
void PrintDataset(const Dataset& dataset);

// Composite printers
void PrintRuntimeInfo();
void PrintSnapshotResult(const SnapshotResult& result);
void PrintInteropResult(const InteropResult& interopResult);
void PrintInteropSummaryTable(const std::vector<InteropResult>& results);
void PrintValidationResultsTable(const std::map<std::string, ValidationTestSummary>& summaries);
void PrintBenchmarkSummary(const BenchmarkResults& results);

// CLI
void PrintHelp();
void PrintPassFail(bool passed, const char* suffix = nullptr);

}  // namespace libmlvc
