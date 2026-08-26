// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <libmlvc/libmlvc.hpp>

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model-bundles-dir>\n";
        return 2;
    }

    const auto& buildInfo = libmlvc::GetBuildInfo();
    std::cout << "libmlvc " << buildInfo.libmlvcVersion << " (" << buildInfo.gitShortHash << ", "
              << buildInfo.gitBranch << ")\n";

    auto manager = libmlvc::MlvcManager::CreateFromDirectory(libmlvc::ManagerParams{}, argv[1]);
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
