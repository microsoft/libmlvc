// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/export.hpp>
#include <libmlvc/types.hpp>

namespace libmlvc {

// --------------------------------------------------------------------------------------
// Platform information types
// --------------------------------------------------------------------------------------

struct OsVersion {
    int major{};
    int minor{};
    int build{};  // Windows: build number (e.g. 26100); unused on macOS/iOS

    auto operator<=>(const OsVersion&) const = default;
    std::string ToString() const
    {
        auto s = std::to_string(major) + "." + std::to_string(minor);
        if (build != 0) s += "." + std::to_string(build);
        return s;
    }
};

enum struct AcceleratorType { GPU, NPU };
inline const char* AcceleratorTypeToString(AcceleratorType type)
{
    switch (type) {
    case AcceleratorType::GPU:
        return "gpu";
    case AcceleratorType::NPU:
        return "npu";
    }
    return "unknown";
}

struct AcceleratorInfo {
    // Device
    AcceleratorType type{ AcceleratorType::GPU };
    std::string name;
    std::string vendorName;  // VENDOR_* constants ("Intel", "Qualcomm", ...)
    std::string vendorId;    // PCI ("8086") or ACPI ("QCOM")
    std::string deviceId;

    // Driver
    std::string driverVersion;
    std::string driverDate;

    // Windows-only
    int64_t luid{};  // adapter LUID (0 if unavailable)
};

struct PlatformInfo {
    // OS / hardware
    std::string osName;          // "macOS", "iOS", "Windows"
    OsVersion osVersion;         // e.g. 15.2 on macOS, 10.0.26100 on Windows
    std::string hwManufacturer;  // e.g. "Apple", "LENOVO", "Microsoft Corporation"
    std::string hwModel;         // e.g. "Mac16,1", "iPhone17,3", "83ED"

    // CPU
    std::string cpuArch;    // "arm64", "x86_64"
    std::string cpuVendor;  // "Intel", "AMD", "Apple", "Qualcomm", etc.
    std::string cpuSeries;  // series name: "M4", "Arrow Lake", "Snapdragon X1"
    std::string cpuName;    // full name: "Apple M4 Pro", "12th Gen Intel Core i7-12700K"

    // CPU features
    bool hasFP16{};     // x86/ARM64, F16C (x86) / FEAT_FP16 (ARM64)
    bool hasAVX2{};     // x86, AVX2
    bool hasAVXVNNI{};  // x86, AVX-VNNI (VEX)
    bool hasDotProd{};  // ARM64, FEAT_DotProd / SDOT

    // Accelerators (GPUs and NPUs)
    std::vector<AcceleratorInfo> accelerators;
};

// --------------------------------------------------------------------------------------
// Platform information APIs
// --------------------------------------------------------------------------------------

LIBMLVC_EXPORT const PlatformInfo& GetPlatformInfo();

// Returns true if the current platform meets minimum requirements for MLVC (OS version, CPU features).
LIBMLVC_EXPORT bool IsPlatformSupported();

// Returns true if the current platform has a known supported NPU (Apple ANE, Intel NPU, or Qualcomm NPU).
LIBMLVC_EXPORT bool HasSupportedNpu();

}  // namespace libmlvc
