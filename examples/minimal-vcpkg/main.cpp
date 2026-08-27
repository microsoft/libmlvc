// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <libmlvc/libmlvc.hpp>

#include <iostream>
#include <string_view>

int main(int argc, char* argv[])
{
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <model-bundles-dir> [auto|cpu|gpu|npu]\n";
        return 2;
    }

    libmlvc::ManagerParams managerParams;
    if (argc == 3) {
        const std::string_view computeUnit{ argv[2] };
        if (computeUnit == "auto") {
            managerParams.computeUnit = libmlvc::ComputeUnit::AUTO;
        } else if (computeUnit == "cpu") {
            managerParams.computeUnit = libmlvc::ComputeUnit::CPU;
        } else if (computeUnit == "gpu") {
            managerParams.computeUnit = libmlvc::ComputeUnit::GPU;
        } else if (computeUnit == "npu") {
            managerParams.computeUnit = libmlvc::ComputeUnit::NPU;
        } else {
            std::cerr << "Unknown compute unit: " << computeUnit << '\n';
            return 2;
        }
    }

    const auto& buildInfo = libmlvc::GetBuildInfo();
    std::cout << "libmlvc " << buildInfo.libmlvcVersion << " (" << buildInfo.gitShortHash << ", "
              << buildInfo.gitBranch << ")\n";

    auto manager = libmlvc::MlvcManager::CreateFromDirectory(managerParams, argv[1]);
    if (!manager) {
        std::cerr << "Failed to initialize libmlvc: " << manager.error().message() << '\n';
        return 1;
    }

    std::cout << "Initialized libmlvc model versions:";
    for (const auto& version : manager.value().GetAvailableVersions()) {
        std::cout << ' ' << version.ToString();
    }
    std::cout << '\n';
    return 0;
}
