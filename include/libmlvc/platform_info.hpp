// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/// @file
/// Operating-system, processor, and accelerator discovery API.

#pragma once
#include <libmlvc/export.hpp>
#include <libmlvc/types.hpp>

namespace libmlvc {

// --------------------------------------------------------------------------------------
// Platform information types
// --------------------------------------------------------------------------------------

/// Operating-system version components.
struct OsVersion {
    int major{};
    int minor{};
    int build{};  ///< Windows build number; zero on macOS and iOS.

    auto operator<=>(const OsVersion&) const = default;
    std::string ToString() const
    {
        auto s = std::to_string(major) + "." + std::to_string(minor);
        if (build != 0) s += "." + std::to_string(build);
        return s;
    }
};

/// Hardware accelerator category.
enum struct AcceleratorType { GPU, NPU };

/// Returns the string name of an accelerator type.
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

/// Discovered accelerator and driver information.
///
/// Fields may be empty when the operating system does not report the corresponding property.
struct AcceleratorInfo {
    // Device
    AcceleratorType type{ AcceleratorType::GPU };
    std::string name;
    std::string vendorName;
    std::string vendorId;  ///< PCI or ACPI vendor identifier.
    std::string deviceId;

    // Driver
    std::string driverVersion;
    std::string driverDate;

    int64_t luid{};  ///< Windows adapter LUID, or zero when unavailable.
};

/// Operating-system, processor, and accelerator information.
struct PlatformInfo {
    // OS and hardware
    std::string osName;  ///< `macOS`, `iOS`, or `Windows`.
    OsVersion osVersion;
    std::string hwManufacturer;
    std::string hwModel;

    // CPU
    std::string cpuArch;  ///< `arm64` or `x86_64`.
    std::string cpuVendor;
    std::string cpuSeries;  ///< CPU family used by support checks, or `unknown`.
    std::string cpuName;    ///< Device identifier on iOS, where the CPU brand is unavailable.

    // CPU features
    bool hasFP16{};     ///< Whether F16C is usable on x86 or FP16 arithmetic is available on ARM64.
    bool hasAVX2{};     ///< Whether AVX2 is usable on x86.
    bool hasAVXVNNI{};  ///< Whether VEX-encoded AVX-VNNI is usable on x86.
    bool hasDotProd{};  ///< Whether ARM64 dot-product instructions are available (`FEAT_DotProd`/`SDOT`).

    std::vector<AcceleratorInfo> accelerators;  ///< Windows GPUs and NPUs; empty on Apple platforms.
};

// --------------------------------------------------------------------------------------
// Platform information APIs
// --------------------------------------------------------------------------------------

/// Returns detected OS, CPU, and accelerator information.
/// The returned reference remains valid until the process exits.
LIBMLVC_EXPORT const PlatformInfo& GetPlatformInfo();

/// Returns whether the OS version and required CPU features meet libmlvc minimums.
///
/// This does not test inference provider readiness or detect an NPU. The result is calculated once.
LIBMLVC_EXPORT bool IsPlatformSupported();

/// Returns whether the detected CPU is on libmlvc's list of platforms with supported NPUs.
/// This does not detect an NPU or verify that an inference provider can use it.
LIBMLVC_EXPORT bool HasSupportedNpu();

}  // namespace libmlvc
