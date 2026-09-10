// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/test_data.hpp"
#include "libmlvc_support/validate.hpp"
#include "support/test_config.hpp"

#include <libmlvc/libmlvc.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>

using namespace libmlvc;

class ValidationTests : public ::testing::Test {
protected:
    void SetUp() override
    {
        // To speed up tests, disable session caching
        ManagerParams managerParams;
        managerParams.computeUnit = GetTestConfig().computeUnit;
        managerParams.enableSessionCaching = false;
        managerParams.winmlInitMode = GetTestConfig().winmlInitMode;

        auto manager =
            MlvcManager::CreateFromDirectory(managerParams, std::filesystem::path{}, std::array{ m_mlvcVersion });
        ASSERT_TRUE(manager) << "Failed to create MlvcManager";
        m_manager = std::move(manager.value());
    }

    void RunValidation(const std::string& datasetName, double bdRateThreshold)
    {
        const std::vector<int> qpList = { 16, 24, 32, 40 };

        auto configDir = m_testDataDir / "datasets" / datasetName;
        auto dataset = Dataset::Load(configDir / "test_config.json");
        ASSERT_TRUE(dataset) << "Failed to read dataset";

        auto anchorMetrics = ReadAnchorMetrics(configDir / "intel_hw_hevc_lp.json", dataset.value().GetFps());
        ASSERT_TRUE(anchorMetrics) << "Failed to read anchor metrics";

        ValidationTestRunner runner(*m_manager, m_mlvcVersion, dataset.value(), qpList, {});
        ASSERT_TRUE(runner.Initialize()) << "Failed to initialize validation runner";
        auto testMetrics = runner.Run();
        ASSERT_TRUE(testMetrics) << "Validation test runner failed";

        auto summary = ComputeValidationSummary(testMetrics.value(), anchorMetrics.value());
        ASSERT_TRUE(summary) << "Failed to compute validation summary";

        ASSERT_FALSE(std::isnan(summary.value().bdRate)) << "BD-rate is NaN";
        EXPECT_LT(summary.value().bdRate, bdRateThreshold)
            << "BD-rate " << summary.value().bdRate << "% exceeds threshold " << bdRateThreshold << "%";
    }

    std::filesystem::path m_testDataDir = GetTestDataDir();
    MlvcVersion m_mlvcVersion = GetTestConfig().mlvcVersion;
    std::optional<MlvcManager> m_manager;
};

TEST_F(ValidationTests, DefaultConfig_320x180)
{
    RunValidation("VCD-320x180-s1-3s", -22.6);
}

TEST_F(ValidationTests, DefaultConfig_426x240)
{
    RunValidation("VCD-426x240-s1-3s", -13.3);
}

TEST_F(ValidationTests, DefaultConfig_640x360)
{
    RunValidation("VCD-640x360-s1-3s", -0.4);
}

TEST_F(ValidationTests, DefaultConfig_960x540)
{
    RunValidation("VCD-960x540-s1-3s", 17.7);
}
