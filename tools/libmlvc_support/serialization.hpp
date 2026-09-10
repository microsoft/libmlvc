// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <filesystem>

namespace libmlvc {

struct SnapshotResult;
expected<void> SaveSnapshot(const std::filesystem::path& dir, const SnapshotResult& result);
expected<SnapshotResult> LoadSnapshot(const std::filesystem::path& dir, bool loadFrameMetrics = false);

}  // namespace libmlvc
