// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace libmlvc {

class RansTestPmf {
public:
    static RansTestPmf MakeGaussianPmf();
    static RansTestPmf MakeBitEstimatorPmf();

    RansTestPmf(std::span<const int32_t> pmfLengths, std::span<const int32_t> pmfOffsets, std::span<const int32_t> pmfTable,
                std::span<const uint16_t> indexWeights, int32_t symbolBits, int32_t bypassBits);
    void GenerateSymbols(std::vector<int32_t>& indices, std::vector<int32_t>& values, int numSymbols) const;
    double CalcShannonEntropy() const;

    std::span<const int32_t> PmfLengths() const { return m_pmfLengths; }
    std::span<const int32_t> PmfOffsets() const { return m_pmfOffsets; }
    std::span<const int32_t> PmfTable() const { return m_pmfTable; }
    int32_t SymbolBits() const { return m_symbolBits; }
    int32_t BypassBits() const { return m_bypassBits; }

private:
    int32_t SampleValue(int32_t pmfIdx) const;

    std::vector<int32_t> m_pmfLengths;
    std::vector<int32_t> m_pmfOffsets;
    std::vector<int32_t> m_pmfTable;
    int32_t m_symbolBits{};
    int32_t m_bypassBits{};
    std::vector<size_t> m_tableOffsets;
    std::vector<int32_t> m_indexLut;
    std::vector<double> m_indexWeights;
    mutable std::mt19937 m_rng{ 42 };
};

}  // namespace libmlvc
