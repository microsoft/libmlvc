// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace libmlvc {

struct ClipMetadata {
    enum class Type { YUV420, NV12 };

    std::string scenario{};
    std::string name{};
    std::filesystem::path path{};
    Type type{};
    int width{};
    int height{};
    int frames{};
};

class Dataset {
public:
    static expected<Dataset> Load(const std::filesystem::path& configPath,
                                  const std::vector<std::string>& scenarioNames = {}, int numClipsLimit = 0);
    const std::filesystem::path& ConfigPath() const { return m_configPath; }
    std::string GetName() const { return m_configPath.parent_path().filename().string(); }
    const std::set<std::string>& GetScenarios() const { return m_scenarios; }
    const std::vector<ClipMetadata>& Clips() const { return m_clips; }
    double GetFps() const { return m_fps; }
    size_t NumClips() const { return m_clips.size(); }

private:
    Dataset(const std::filesystem::path& configPath, std::vector<ClipMetadata> clips);

    std::filesystem::path m_configPath;
    std::vector<ClipMetadata> m_clips;
    std::set<std::string> m_scenarios;
    double m_fps{ 30.0 };
};

}  // namespace libmlvc
