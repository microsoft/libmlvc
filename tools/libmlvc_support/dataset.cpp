// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/dataset.hpp"

#include <libmlvc/error_codes.hpp>

#include <boost/json.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>

namespace libmlvc {

namespace {

expected<ClipMetadata::Type> ParseType(const std::string& value)
{
    if (value == "yuv420") {
        return ClipMetadata::Type::YUV420;
    } else if (value == "nv12") {
        return ClipMetadata::Type::NV12;
    }

    std::cerr << "Error: Unsupported src_type '" << value << "'\n";
    return make_error_code(Error::invalid_argument);
}

}  // namespace

expected<Dataset> Dataset::Load(const std::filesystem::path& configPath, const std::vector<std::string>& scenarioNames,
                                int numClipsLimit)
{
    // Read the JSON file
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open config file: " << configPath.string() << '\n';
        return make_error_code(Error::io_error);
    }

    std::string jsonContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) {
        std::cerr << "Error: Failed to read config file: " << configPath.string() << '\n';
        return make_error_code(Error::io_error);
    }

    // Parse JSON
    std::error_code ec;
    boost::json::value root = boost::json::parse(jsonContent, ec);
    if (ec) {
        std::cerr << "Error: Failed to parse JSON config: " << ec.message() << '\n';
        return make_error_code(Error::json_parse_error);
    }

    const auto* rootObj = root.if_object();
    if (!rootObj) {
        std::cerr << "Error: JSON config root is not an object\n";
        return make_error_code(Error::json_parse_error);
    }

    const auto* testClassesVal = rootObj->if_contains("test_classes");
    const auto* testClasses = testClassesVal ? testClassesVal->if_object() : nullptr;
    if (!testClasses) {
        std::cerr << "Error: Missing or invalid 'test_classes' object in config\n";
        return make_error_code(Error::json_parse_error);
    }

    const auto basePath = configPath.parent_path();

    std::vector<ClipMetadata> clips;
    for (const auto& [scenarioName, scenarioValue] : *testClasses) {
        // Filter by scenario names if provided (empty means include all)
        if (!scenarioNames.empty()) {
            const auto it = std::find(scenarioNames.begin(), scenarioNames.end(), std::string(scenarioName));
            if (it == scenarioNames.end()) {
                continue;
            }
        }

        const auto* scenarioObj = scenarioValue.if_object();
        if (!scenarioObj) {
            continue;
        }

        std::string scenarioBasePath;
        if (const auto* basePathPtr = scenarioObj->if_contains("base_path")) {
            if (const auto* basePathStr = basePathPtr->if_string()) {
                scenarioBasePath = *basePathStr;
            }
        }

        std::string srcTypeStr;
        if (const auto* srcTypePtr = scenarioObj->if_contains("src_type")) {
            if (const auto* srcTypeVal = srcTypePtr->if_string()) {
                srcTypeStr = *srcTypeVal;
            }
        }
        auto typeRes = ParseType(srcTypeStr);
        if (!typeRes) {
            return typeRes.error();
        }
        const auto type = *typeRes;

        const auto* sequencesPtr = scenarioObj->if_contains("sequences");
        if (!sequencesPtr) {
            continue;
        }

        const auto* sequences = sequencesPtr->if_object();
        if (!sequences) {
            continue;
        }

        // Iterate over sequences
        int scenarioClipCount = 0;
        for (const auto& [sequenceName, sequenceValue] : *sequences) {
            if (numClipsLimit > 0 && scenarioClipCount >= numClipsLimit) {
                break;
            }

            const auto* seqObj = sequenceValue.if_object();
            if (!seqObj) {
                continue;
            }

            const auto* widthPtr = seqObj->if_contains("width");
            const auto* heightPtr = seqObj->if_contains("height");
            const auto* framesPtr = seqObj->if_contains("frames");

            if (!widthPtr || !heightPtr || !framesPtr) {
                std::cerr << "Error: Missing width/height/frames in sequence '" << std::string(sequenceName) << "'\n";
                return make_error_code(Error::json_parse_error);
            }

            const auto* widthVal = widthPtr->if_int64();
            const auto* heightVal = heightPtr->if_int64();
            const auto* framesVal = framesPtr->if_int64();

            if (!widthVal || !heightVal || !framesVal) {
                std::cerr << "Error: Invalid type for width/height/frames in sequence '" << std::string(sequenceName)
                          << "' (expected integers)\n";
                return make_error_code(Error::json_parse_error);
            }

            clips.push_back(ClipMetadata{
                .scenario = scenarioName,
                .name = sequenceName,
                .path = basePath / scenarioBasePath / std::string(sequenceName),
                .type = type,
                .width = static_cast<int>(*widthVal),
                .height = static_cast<int>(*heightVal),
                .frames = static_cast<int>(*framesVal),
            });
            ++scenarioClipCount;
        }
    }

    if (clips.empty()) {
        std::cerr << "Error: No clips found in dataset: " << configPath.string() << '\n';
        return make_error_code(Error::io_error);
    }

    return Dataset(configPath, std::move(clips));
}

Dataset::Dataset(const std::filesystem::path& configPath, std::vector<ClipMetadata> clips)
    : m_configPath(std::filesystem::absolute(configPath)), m_clips(std::move(clips))
{
    for (const auto& clip : m_clips) {
        m_scenarios.insert(clip.scenario);
    }
}

}  // namespace libmlvc
