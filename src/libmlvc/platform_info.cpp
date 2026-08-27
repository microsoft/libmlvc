// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/platform_names.hpp"

#include <libmlvc/build_info.hpp>
#include <libmlvc/platform_info.hpp>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(MLVC_ARCH_X86_64)
    #include <cpuid.h>
#endif

#if defined(MLVC_PLATFORM_APPLE)
    #include <sys/sysctl.h>
#endif

// clang-format off
#if defined(MLVC_PLATFORM_WINCLASSIC)
    #include <windows.h>
    #ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
        #define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
    #endif
    #include <initguid.h>  // must precede devpkey.h / devguid.h
    #include <SetupAPI.h>
    #include <devguid.h>
    #include <devpkey.h>
#endif
// clang-format on

namespace libmlvc {

using namespace std::string_view_literals;

namespace {

// ---------------------------------------------------------------------------
// OS version and hardware model detection
// ---------------------------------------------------------------------------

#if defined(MLVC_PLATFORM_APPLE)
std::string SysctlString(const char* name)
{
    char buf[256] = "";
    size_t size = sizeof(buf);
    if (::sysctlbyname(name, buf, &size, nullptr, 0) == 0) {
        return buf;
    }
    return {};
}

int32_t SysctlInt32(const char* name, int32_t defaultValue = 0)
{
    int32_t val = defaultValue;
    size_t size = sizeof(val);
    ::sysctlbyname(name, &val, &size, nullptr, 0);
    return val;
}
#endif

OsVersion GetOsVersion()
{
#if defined(MLVC_PLATFORM_APPLE)
    OsVersion v{};
    const auto str = SysctlString("kern.osproductversion");
    if (!str.empty()) {
        std::sscanf(str.c_str(), "%d.%d", &v.major, &v.minor);
    }
    return v;
#elif defined(MLVC_PLATFORM_WINCLASSIC)
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    OSVERSIONINFOW osvi{};
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    if (const auto ntdll = ::GetModuleHandleW(L"ntdll.dll")) {
        const auto fn = reinterpret_cast<RtlGetVersionFn>(::GetProcAddress(ntdll, "RtlGetVersion"));
        if (fn && fn(&osvi) == 0) {
            return { static_cast<int>(osvi.dwMajorVersion), static_cast<int>(osvi.dwMinorVersion),
                     static_cast<int>(osvi.dwBuildNumber) };
        }
    }
    return {};
#else
    #error "Unsupported platform"
#endif
}

std::string GetHwManufacturer()
{
#if defined(MLVC_PLATFORM_APPLE)
    return "Apple";
#elif defined(MLVC_PLATFORM_WINCLASSIC)
    HKEY hKey{};
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\BIOS", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buf[256] = "";
        DWORD size = sizeof(buf);
        DWORD type = REG_SZ;
        if (RegQueryValueExA(hKey, "SystemManufacturer", nullptr, &type, reinterpret_cast<LPBYTE>(buf), &size)
            == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return std::string(buf);
        }
        RegCloseKey(hKey);
    }
    return {};
#else
    #error "Unsupported platform"
#endif
}

std::string GetHwModel()
{
#if defined(MLVC_PLATFORM_MACOSX)
    return SysctlString("hw.model");
#elif defined(MLVC_PLATFORM_IOS)
    // hw.model returns board ID (e.g. "D93AP"); hw.machine gives "iPhone17,3" style identifiers
    return SysctlString("hw.machine");
#elif defined(MLVC_PLATFORM_WINCLASSIC)
    HKEY hKey{};
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\BIOS", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buf[256] = "";
        DWORD size = sizeof(buf);
        DWORD type = REG_SZ;
        if (RegQueryValueExA(hKey, "SystemProductName", nullptr, &type, reinterpret_cast<LPBYTE>(buf), &size)
            == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return std::string(buf);
        }
        RegCloseKey(hKey);
    }
    return {};
#else
    #error "Unsupported platform"
#endif
}

// ---------------------------------------------------------------------------
// Architecture-specific CPU detection
// ---------------------------------------------------------------------------

#if defined(MLVC_PLATFORM_APPLE)

std::string GetAppleCpuSeries(std::string_view cpuName)
{
    // "Apple M4 Pro" → "M4", "Apple M3" → "M3", "Apple A18 Pro" → "A18"
    for (const auto prefix : { "Apple M"sv, "Apple A"sv }) {
        const auto pos = cpuName.find(prefix);
        if (pos == std::string_view::npos) continue;

        const auto letterPos = pos + prefix.size() - 1;  // 'M' or 'A'
        auto numEnd = letterPos + 1;
        while (numEnd < cpuName.size() && std::isdigit(static_cast<unsigned char>(cpuName[numEnd]))) {
            numEnd++;
        }
        if (numEnd > letterPos + 1) {
            return std::string(cpuName.substr(letterPos, numEnd - letterPos));
        }
    }
    return "unknown";
}

#elif defined(MLVC_ARCH_X86_64)

using CPUIDRegisters = std::array<int, 4>;

MLVC_FORCE_INLINE void Cpuid(CPUIDRegisters& regs, int func)
{
    #if defined(_MSC_VER)
    __cpuid(regs.data(), func);
    #else
    __cpuid(func, regs[0], regs[1], regs[2], regs[3]);
    #endif
}

MLVC_FORCE_INLINE void Cpuid(CPUIDRegisters& regs, int func, int subfunc)
{
    #if defined(_MSC_VER)
    __cpuidex(regs.data(), func, subfunc);
    #else
    __cpuid_count(func, subfunc, regs[0], regs[1], regs[2], regs[3]);
    #endif
}

struct X86Feature {
    int reg;
    int bit;
};

constexpr bool HasFeature(const CPUIDRegisters& regs, X86Feature f)
{
    return (regs[f.reg] & (1U << f.bit)) != 0;
}

// cpuid function 1, ECX
constexpr X86Feature kXSAVE{ .reg = 2, .bit = 26 };
constexpr X86Feature kOSXSAVE{ .reg = 2, .bit = 27 };
constexpr X86Feature kF16C{ .reg = 2, .bit = 29 };
// cpuid function 7, sub function 0, EBX
constexpr X86Feature kAVX2{ .reg = 1, .bit = 5 };
// cpuid function 7, sub function 1, EAX
constexpr X86Feature kAVXVNNI{ .reg = 0, .bit = 4 };

MLVC_FORCE_INLINE uint32_t Xgetbv(uint32_t xcr)
{
    #if defined(_MSC_VER)
    return static_cast<uint32_t>(_xgetbv(xcr));
    #else
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
    return eax;
    #endif
}

bool HasAvxOsSupport(const CPUIDRegisters& regs)
{
    if (!HasFeature(regs, kXSAVE) || !HasFeature(regs, kOSXSAVE)) return false;
    constexpr uint32_t kXmmYmmMask = 0x6;  // bits 1 (XMM) and 2 (YMM)
    return (Xgetbv(0) & kXmmYmmMask) == kXmmYmmMask;
}

std::string GetX86CpuName()
{
    CPUIDRegisters regs;
    Cpuid(regs, static_cast<int>(0x80000000));
    if (static_cast<unsigned>(regs[0]) < 0x80000004) return {};

    std::array<char, 49> brand{};
    for (unsigned i = 0; i < 3; i++) {
        Cpuid(regs, static_cast<int>(0x80000002 + i));
        std::memcpy(brand.data() + i * 16, regs.data(), 16);
    }

    std::string_view sv{ brand.data() };
    const auto start = sv.find_first_not_of(' ');
    const auto end = sv.find_last_not_of(' ');
    if (start == std::string_view::npos) return {};
    return std::string(sv.substr(start, end - start + 1));
}

std::string GetIntelCpuSeries(unsigned cpuFamily, unsigned cpuModel)
{
    if (cpuFamily != 6) return "unknown";

    switch (cpuModel) {
    case 0x97:  // Alder Lake S
    case 0x9A:  // Alder Lake P/H/U
    case 0xBE:  // Alder Lake N
        return CPU_SERIES_ALDER_LAKE;
    case 0xB7:  // Raptor Lake S/H
    case 0xBA:  // Raptor Lake P/U
    case 0xBF:  // Raptor Lake S refresh
        return CPU_SERIES_RAPTOR_LAKE;
    case 0xAA:  // Meteor Lake M/P
        return CPU_SERIES_METEOR_LAKE;
    case 0xC5:  // Arrow Lake S
    case 0xC6:  // Arrow Lake H
        return CPU_SERIES_ARROW_LAKE;
    case 0xBC:  // Lunar Lake
    case 0xBD:  // Lunar Lake
        return CPU_SERIES_LUNAR_LAKE;
    case 0xCC:  // Panther Lake
        return CPU_SERIES_PANTHER_LAKE;
    default:
        return "unknown";
    }
}

#elif defined(MLVC_ARCH_ARM64)

std::string GetQualcommCpuSeries(std::string_view cpuName)
{
    // "Snapdragon(R) X Elite - X1E80100" → "Snapdragon X1"
    // "Snapdragon(R) X Plus - X1P64100" → "Snapdragon X1"
    if (cpuName.find("X1E"sv) != std::string_view::npos || cpuName.find("X1P"sv) != std::string_view::npos) {
        return CPU_SERIES_SNAPDRAGON_X1;
    }
    return "unknown";
}

std::string ReadCpuRegString(const char* valueName)
{
    HKEY hKey{};
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey)
        == ERROR_SUCCESS) {
        char buf[256] = "";
        DWORD size = sizeof(buf);
        DWORD type = REG_SZ;
        if (RegQueryValueExA(hKey, valueName, nullptr, &type, reinterpret_cast<LPBYTE>(buf), &size) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return buf;
        }
        RegCloseKey(hKey);
    }
    return {};
}
#endif

// ---------------------------------------------------------------------------
// Accelerator (GPU / NPU) enumeration
// ---------------------------------------------------------------------------

#if defined(MLVC_PLATFORM_WINCLASSIC)

struct ParsedHwIds {
    std::string vendorId;
    std::string deviceId;
};

ParsedHwIds ParseHardwareIds(const wchar_t* hardwareId)
{
    ParsedHwIds ids{};
    const std::wstring_view sv{ hardwareId };

    auto extractValue = [&](std::wstring_view prefix) -> std::string {
        const auto pos = sv.find(prefix);
        if (pos == std::wstring_view::npos) return {};
        const auto start = pos + prefix.size();
        auto end = start;
        while (end < sv.size() && sv[end] != L'&' && sv[end] != L'\\')
            end++;
        std::string result;
        for (auto i = start; i < end; i++)
            result += static_cast<char>(sv[i]);
        return result;
    };

    ids.vendorId = extractValue(L"VEN_");
    ids.deviceId = extractValue(L"DEV_");
    return ids;
}

std::string WideToUtf8(const wchar_t* wide)
{
    if (!wide || !wide[0]) return {};
    const int len = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string result(static_cast<size_t>(len - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, result.data(), len, nullptr, nullptr);
    return result;
}

std::string FormatFileTime(const FILETIME& ft)
{
    SYSTEMTIME st{};
    if (FileTimeToSystemTime(&ft, &st)) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", st.wYear, st.wMonth, st.wDay);
        return buf;
    }
    return {};
}

// DEVPKEY_GPU_LUID — adapter LUID property key.
static const DEVPROPKEY kDevPropKeyLuid = {
    { 0x60B193CB, 0x5276, 0x4D0F, { 0x96, 0xFC, 0xF1, 0x73, 0xAB, 0xAD, 0x3E, 0xC6 } }, 2
};

std::string VendorIdToName(const std::string& vendorId)
{
    if (vendorId == "8086") return VENDOR_INTEL;
    if (vendorId == "1002" || vendorId == "1022") return VENDOR_AMD;
    if (vendorId == "10DE") return VENDOR_NVIDIA;
    if (vendorId == "QCOM") return VENDOR_QUALCOMM;
    if (vendorId == "106B") return VENDOR_APPLE;
    return {};
}

std::vector<AcceleratorInfo> EnumerateDevices(const GUID& classGuid, AcceleratorType type)
{
    std::vector<AcceleratorInfo> devices;

    const HDEVINFO devInfoSet = SetupDiGetClassDevsW(&classGuid, nullptr, nullptr, DIGCF_PRESENT);
    if (devInfoSet == INVALID_HANDLE_VALUE) return devices;

    struct Guard {
        HDEVINFO h;
        ~Guard() { SetupDiDestroyDeviceInfoList(h); }
    } guard{ devInfoSet };

    SP_DEVINFO_DATA devInfoData{};
    devInfoData.cbSize = sizeof(devInfoData);

    for (DWORD i = 0; SetupDiEnumDeviceInfo(devInfoSet, i, &devInfoData); i++) {
        AcceleratorInfo acc{};
        acc.type = type;

        {
            wchar_t buf[256]{};
            if (SetupDiGetDeviceRegistryPropertyW(devInfoSet, &devInfoData, SPDRP_DEVICEDESC, nullptr,
                                                  reinterpret_cast<PBYTE>(buf), sizeof(buf), nullptr)) {
                acc.name = WideToUtf8(buf);
            }
        }

        {
            wchar_t buf[512]{};
            if (SetupDiGetDeviceRegistryPropertyW(devInfoSet, &devInfoData, SPDRP_HARDWAREID, nullptr,
                                                  reinterpret_cast<PBYTE>(buf), sizeof(buf), nullptr)) {
                const auto ids = ParseHardwareIds(buf);
                acc.vendorName = VendorIdToName(ids.vendorId);
                acc.vendorId = ids.vendorId;
                acc.deviceId = ids.deviceId;
            }
        }

        {
            wchar_t buf[128]{};
            DEVPROPTYPE propType = 0;
            if (SetupDiGetDevicePropertyW(devInfoSet, &devInfoData, &DEVPKEY_Device_DriverVersion, &propType,
                                          reinterpret_cast<PBYTE>(buf), sizeof(buf), nullptr, 0)) {
                acc.driverVersion = WideToUtf8(buf);
            }
        }

        {
            FILETIME ft{};
            DEVPROPTYPE propType = 0;
            if (SetupDiGetDevicePropertyW(devInfoSet, &devInfoData, &DEVPKEY_Device_DriverDate, &propType,
                                          reinterpret_cast<PBYTE>(&ft), sizeof(ft), nullptr, 0)) {
                acc.driverDate = FormatFileTime(ft);
            }
        }

        {
            uint64_t luidVal = 0;
            DEVPROPTYPE propType = 0;
            if (SetupDiGetDevicePropertyW(devInfoSet, &devInfoData, &kDevPropKeyLuid, &propType,
                                          reinterpret_cast<PBYTE>(&luidVal), sizeof(luidVal), nullptr, 0)) {
                acc.luid = static_cast<int64_t>(luidVal);
            }
        }

        if (!acc.name.empty() && !acc.vendorId.empty()) {
            devices.push_back(std::move(acc));
        }
    }

    return devices;
}

std::vector<AcceleratorInfo> GetAccelerators()
{
    auto result = EnumerateDevices(GUID_DEVCLASS_DISPLAY, AcceleratorType::GPU);
    auto npus = EnumerateDevices(GUID_DEVCLASS_COMPUTEACCELERATOR, AcceleratorType::NPU);
    result.insert(result.end(), std::make_move_iterator(npus.begin()), std::make_move_iterator(npus.end()));
    return result;
}

#endif  // MLVC_PLATFORM_WINCLASSIC

// ---------------------------------------------------------------------------
// Platform support policy
// ---------------------------------------------------------------------------

bool IsSupportedOs(const PlatformInfo& platform)
{
#if defined(MLVC_PLATFORM_MACOSX)
    constexpr OsVersion kMinMacOS{ .major = 15 };  // macOS 15.0 (Sequoia)
    if (platform.osVersion < kMinMacOS) {
        MLVC_LOG_WARN("macOS %s is below the minimum required version (%s)", platform.osVersion.ToString().c_str(),
                      kMinMacOS.ToString().c_str());
        return false;
    }
    return true;
#elif defined(MLVC_PLATFORM_IOS)
    constexpr OsVersion kMinIOS{ .major = 18 };  // iOS 18.0
    if (platform.osVersion < kMinIOS) {
        MLVC_LOG_WARN("iOS %s is below the minimum required version (%s)", platform.osVersion.ToString().c_str(),
                      kMinIOS.ToString().c_str());
        return false;
    }
    return true;
#elif defined(MLVC_PLATFORM_WINCLASSIC)
    constexpr int kMinWindowsBuild = 26100;  // Windows 11 24H2
    if (platform.osVersion.build < kMinWindowsBuild) {
        MLVC_LOG_WARN("Windows build %d is below the minimum required build (%d)", platform.osVersion.build,
                      kMinWindowsBuild);
        return false;
    }
    return true;
#else
    #error "Unsupported platform"
#endif
}

struct SupportedNpu {
    const char* cpuVendor;
    const char* cpuSeries;
};

constexpr SupportedNpu SUPPORTED_NPUS[] = {
    // Apple (all Apple Silicon has ANE)
    { VENDOR_APPLE, CPU_SERIES_M1 },
    { VENDOR_APPLE, CPU_SERIES_M2 },
    { VENDOR_APPLE, CPU_SERIES_M3 },
    { VENDOR_APPLE, CPU_SERIES_M4 },
    { VENDOR_APPLE, CPU_SERIES_M5 },
    // Intel
    { VENDOR_INTEL, CPU_SERIES_LUNAR_LAKE },
    { VENDOR_INTEL, CPU_SERIES_PANTHER_LAKE },
    // Qualcomm
    { VENDOR_QUALCOMM, CPU_SERIES_SNAPDRAGON_X1 },
};

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const PlatformInfo& GetPlatformInfo()
{
    static const auto info = []() {
        const auto t0 = std::chrono::steady_clock::now();
        PlatformInfo info{};

#if defined(MLVC_PLATFORM_MACOSX)
        info.osName = OS_NAME_MACOS;
#elif defined(MLVC_PLATFORM_IOS)
        info.osName = OS_NAME_IOS;
#elif defined(MLVC_PLATFORM_WINCLASSIC)
        info.osName = OS_NAME_WINDOWS;
#else
    #error "Unsupported platform for OS name"
#endif
        info.osVersion = GetOsVersion();
        info.hwManufacturer = GetHwManufacturer();
        info.hwModel = GetHwModel();

#if defined(MLVC_ARCH_X86_64)
        info.cpuArch = "x86_64";

        CPUIDRegisters regs;
        Cpuid(regs, 0);
        const auto maxFunc = regs[0];

        // Vendor string is EBX-EDX-ECX; swap ECX/EDX ([2]/[3]) to get the correct byte order
        std::swap(regs[2], regs[3]);
        const std::string_view rawVendor{ reinterpret_cast<const char*>(&regs[1]), 12 };

        if (rawVendor == "GenuineIntel")
            info.cpuVendor = VENDOR_INTEL;
        else if (rawVendor == "AuthenticAMD")
            info.cpuVendor = VENDOR_AMD;
        else
            info.cpuVendor = rawVendor;

        const bool isIntel = (info.cpuVendor == VENDOR_INTEL);
        const bool isAmd = (info.cpuVendor == VENDOR_AMD);

        unsigned cpuFamily = 0;
        unsigned cpuModel = 0;

        if (maxFunc >= 1) {
            Cpuid(regs, 1);

            const bool avxOs = HasAvxOsSupport(regs);
            info.hasFP16 = avxOs && HasFeature(regs, kF16C);

            cpuFamily = (regs[0] & 0xf00U) >> 8;
            cpuModel = (regs[0] & 0xf0U) >> 4;
            if (isAmd || isIntel) {
                if ((isIntel && cpuFamily == 0x6U) || cpuFamily == 0xfU) cpuModel |= (regs[0] & 0xf0000U) >> 12;
                if (cpuFamily == 0xfU) cpuFamily += (regs[0] & 0xff00000U) >> 20;
            }

            if (maxFunc >= 7) {
                Cpuid(regs, 7, 0);
                info.hasAVX2 = avxOs && HasFeature(regs, kAVX2);

                Cpuid(regs, 7, 1);  // AVX-VNNI (VEX) lives in leaf 7 sub-leaf 1; same YMM OS state as AVX2
                info.hasAVXVNNI = avxOs && HasFeature(regs, kAVXVNNI);
            }
        }

        info.cpuName = GetX86CpuName();
        info.cpuSeries = isIntel ? GetIntelCpuSeries(cpuFamily, cpuModel) : "unknown";

#elif defined(MLVC_ARCH_ARM64)
        info.cpuArch = "arm64";

    #if defined(MLVC_PLATFORM_MACOSX)
        info.cpuName = SysctlString("machdep.cpu.brand_string");
        info.cpuVendor = VENDOR_APPLE;
        info.cpuSeries = GetAppleCpuSeries(info.cpuName);
    #elif defined(MLVC_PLATFORM_IOS)
        // machdep.cpu.brand_string is not available in the iOS sandbox.
        // Use hw.machine (already in hwModel, e.g. "iPhone17,3") as a fallback identifier.
        info.cpuName = info.hwModel;
        info.cpuVendor = VENDOR_APPLE;
        info.cpuSeries = "unknown";
    #elif defined(MLVC_PLATFORM_WINCLASSIC)
        info.cpuName = ReadCpuRegString("ProcessorNameString");
        const auto vendorId = ReadCpuRegString("VendorIdentifier");
        if (vendorId.find("Qualcomm"sv) != std::string_view::npos
            || info.cpuName.find("Snapdragon"sv) != std::string_view::npos) {
            info.cpuVendor = VENDOR_QUALCOMM;
            info.cpuSeries = GetQualcommCpuSeries(info.cpuName);
        } else {
            info.cpuVendor = "unknown";
            info.cpuSeries = "unknown";
        }
    #else
        #error "Unsupported platform for ARM64 CPU detection"
    #endif

    #if defined(MLVC_PLATFORM_APPLE)
        info.hasFP16 = SysctlInt32("hw.optional.arm.FEAT_FP16") != 0;
    #elif defined(MLVC_PLATFORM_WINCLASSIC)
        // All Windows ARM64 chips are ARMv8.2+ which mandates FP16 support
        info.hasFP16 = true;
    #else
        #error "Unsupported platform for ARM64 FP16 detection"
    #endif

    #if defined(MLVC_PLATFORM_APPLE)
        info.hasDotProd = SysctlInt32("hw.optional.arm.FEAT_DotProd") != 0;
    #elif defined(MLVC_PLATFORM_WINCLASSIC)
        info.hasDotProd = ::IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
    #else
        #error "Unsupported platform for ARM64 DotProd detection"
    #endif

#else
    #error "Unsupported architecture for libmlvc platforminfo"
#endif

#if defined(MLVC_PLATFORM_WINCLASSIC)
        info.accelerators = GetAccelerators();
#endif

        const auto dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto& buildInfo = GetBuildInfo();
        MLVC_LOG_INFO("Build: libmlvc %s (%s, %s)", buildInfo.libmlvcVersion.c_str(), buildInfo.gitShortHash.c_str(),
                      buildInfo.gitBranch.c_str());
        MLVC_LOG_INFO("Platform: OS=%s %s, HW=%s %s", info.osName.c_str(), info.osVersion.ToString().c_str(),
                      info.hwManufacturer.c_str(), info.hwModel.c_str());
#if defined(MLVC_ARCH_X86_64)
        MLVC_LOG_INFO("Platform: CPU=%s (%s %s, %s), FP16=%s, AVX2=%s, AVX-VNNI=%s", info.cpuName.c_str(),
                      info.cpuVendor.c_str(), info.cpuSeries.c_str(), info.cpuArch.c_str(), info.hasFP16 ? "yes" : "no",
                      info.hasAVX2 ? "yes" : "no", info.hasAVXVNNI ? "yes" : "no");
#elif defined(MLVC_ARCH_ARM64)
        MLVC_LOG_INFO("Platform: CPU=%s (%s %s, %s), FP16=%s, DotProd=%s", info.cpuName.c_str(), info.cpuVendor.c_str(),
                      info.cpuSeries.c_str(), info.cpuArch.c_str(), info.hasFP16 ? "yes" : "no",
                      info.hasDotProd ? "yes" : "no");
#endif
        for (const auto& acc : info.accelerators) {
            auto label = std::string(AcceleratorTypeToString(acc.type));
            for (auto& c : label)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            MLVC_LOG_INFO("Platform: %s=%s (driver=%s, date=%s, vendor=%s, vendor_id=%s, device_id=%s, luid=%lld)",
                          label.c_str(), acc.name.c_str(), acc.driverVersion.c_str(), acc.driverDate.c_str(),
                          acc.vendorName.c_str(), acc.vendorId.c_str(), acc.deviceId.c_str(),
                          static_cast<long long>(acc.luid));
        }

        MLVC_LOG_INFO("Platform: enumeration time: %.1f ms", dt);

        return info;
    }();
    return info;
}

bool IsPlatformSupported()
{
    static bool supported = []() {
        const auto& platform = GetPlatformInfo();
        if (!IsSupportedOs(platform)) {
            return false;
        }
        if (!platform.hasFP16) {
            MLVC_LOG_WARN("CPU does not support FP16 (required for MLVC)");
            return false;
        }
#if defined(MLVC_ARCH_X86_64)
        if (!platform.hasAVX2) {
            MLVC_LOG_WARN("CPU does not support AVX2 (required for MLVC on x86_64)");
            return false;
        }
#elif defined(MLVC_ARCH_ARM64)
        if (!platform.hasDotProd) {
            MLVC_LOG_WARN("CPU does not support dotprod (required for MLVC on arm64)");
            return false;
        }
#endif
        MLVC_LOG_DEBUG("Platform is supported for MLVC");
        return true;
    }();
    return supported;
}

bool HasSupportedNpu()
{
    const auto& platform = GetPlatformInfo();
    for (const auto& npu : SUPPORTED_NPUS) {
        if (platform.cpuVendor == npu.cpuVendor && platform.cpuSeries == npu.cpuSeries) {
            return true;
        }
    }
    return false;
}

}  // namespace libmlvc
