// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/interop.hpp"
#include "libmlvc_support/test_data.hpp"
#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>

using namespace libmlvc;

class InteropTests : public ::testing::Test {
protected:
    void SetUp() override
    {
        // To speed up tests, disable session caching
        ManagerParams managerParams;
        managerParams.enableSessionCaching = false;
        managerParams.winmlInitMode = GetTestConfig().winmlInitMode;

        auto manager =
            MlvcManager::CreateFromDirectory(managerParams, std::filesystem::path{}, std::array{ m_mlvcVersion });
        ASSERT_TRUE(manager) << "Failed to create MlvcManager";
        m_manager = std::move(manager.value());
    }

    void RunInteropTest(const std::string& datasetName)
    {
        const std::vector<int> qpList = { 0 };

        auto configDir = m_testDataDir / "datasets" / datasetName;
        auto dataset = Dataset::Load(configDir / "test_config.json");
        ASSERT_TRUE(dataset) << "Failed to read dataset";

        auto runner = InteropTestRunner::Create(*m_manager, m_mlvcVersion, dataset.value(), qpList);
        ASSERT_TRUE(runner) << "Failed to create interop test runner";

        const auto referenceSnapshots = runner.value().ListSnapshots();
        ASSERT_FALSE(referenceSnapshots.empty()) << "No reference snapshots found";

        for (const auto& snapshotDir : referenceSnapshots) {
            SCOPED_TRACE(snapshotDir.filename().string());

            auto result = runner.value().RunInteropTest(snapshotDir);
            ASSERT_TRUE(result) << "Interop test failed";

            const auto& m = result.value().metrics;
            const auto& t = result.value().thresholds;
            ASSERT_FALSE(std::isnan(m.meanPsnr)) << "Mean PSNR is NaN";
            EXPECT_GE(m.meanPsnr, t.meanPsnrFloor) << "Mean PSNR below floor";
            EXPECT_GE(m.minPsnr, t.minPsnrFloor) << "Min PSNR below floor";
            EXPECT_LE(m.psnrDrop, t.maxPsnrDrop) << "PSNR drop exceeds limit";
            if (t.requireBitExact) {
                EXPECT_TRUE(m.bitExact) << "Bit-exactness required but not achieved";
            }
            EXPECT_TRUE(result.value().evaluationPassed) << "Interop evaluation failed";
        }
    }

    std::filesystem::path m_testDataDir = GetTestDataDir();
    MlvcVersion m_mlvcVersion = GetTestConfig().mlvcVersion;
    std::optional<MlvcManager> m_manager;
};

TEST_F(InteropTests, DefaultConfig_960x540)
{
    RunInteropTest("VCD-960x540-div");
}

TEST_F(InteropTests, DefaultConfig_640x360)
{
    RunInteropTest("VCD-640x360-div");
}

TEST_F(InteropTests, DefaultConfig_426x240)
{
    RunInteropTest("VCD-426x240-div");
}

TEST_F(InteropTests, DefaultConfig_320x180)
{
    RunInteropTest("VCD-320x180-div");
}
