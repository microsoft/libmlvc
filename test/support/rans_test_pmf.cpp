// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "support/rans_test_pmf.hpp"

#include <cmath>
#include <numeric>

namespace libmlvc {

namespace {

// clang-format off

constexpr int32_t GAUSS_PMF_LENGTHS[] = {
    6,6,6,6,8,8,10,10,12,12,14,14,14,16,16,18,18,18,18,18,18,18
};
constexpr int32_t GAUSS_PMF_OFFSETS[] = {
    2,2,2,2,3,3,4,4,5,5,6,6,6,7,7,8,8,8,8,8,8,8
};
constexpr int32_t GAUSS_PMF_TABLE[] = {
    /* [0] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [1] len=6 off=2 */ 1,1095,63343,1095,1,1,
    /* [2] len=6 off=2 */ 1,5409,54714,5410,1,1,
    /* [3] len=6 off=2 */ 64,9861,45684,9861,65,1,
    /* [4] len=8 off=3 */ 1,457,13066,38486,13066,458,1,1,
    /* [5] len=8 off=3 */ 21,1336,14917,32986,14917,1336,22,1,
    /* [6] len=10 off=4 */ 2,119,2546,15726,28750,15726,2546,119,1,1,
    /* [7] len=10 off=4 */ 11,355,3831,15857,25426,15857,3830,355,13,1,
    /* [8] len=12 off=5 */ 2,51,747,4998,15588,22764,15588,4998,747,51,1,1,
    /* [9] len=12 off=5 */ 9,143,1261,5957,15103,20589,15103,5958,1261,143,8,1,
    /* [10] len=14 off=6 */ 1,29,301,1839,6692,14513,18784,14513,6692,1839,301,29,2,1,
    /* [11] len=14 off=6 */ 7,73,527,2426,7219,13883,17265,13883,7219,2426,527,74,6,1,
    /* [12] len=14 off=6 */ 19,151,807,2980,7573,13252,15968,13252,7574,2980,807,150,19,4,
    /* [13] len=16 off=7 */ 5,45,263,1123,3478,7790,12638,14851,12638,7790,3478,1123,263,45,5,1,
    /* [14] len=16 off=7 */ 14,87,410,1457,3906,7900,12054,13877,12054,7900,3906,1457,409,87,14,4,
    /* [15] len=18 off=8 */ 5,29,150,586,1789,4265,7929,11503,13022,11504,7929,4264,1790,586,149,30,5,1,
    /* [16] len=18 off=8 */ 11,55,234,783,2109,4556,7898,10988,12265,10987,7898,4556,2109,783,234,56,10,4,
    /* [17] len=18 off=8 */ 21,94,337,994,2405,4787,7824,10506,11591,10506,7824,4786,2406,993,337,94,22,9,
    /* [18] len=18 off=8 */ 39,145,457,1209,2675,4964,7717,10058,10987,10058,7717,4964,2675,1209,457,145,39,21,
    /* [19] len=18 off=8 */ 63,210,592,1422,2915,5094,7590,9641,10441,9641,7590,5094,2915,1422,591,210,64,41,
    /* [20] len=18 off=8 */ 97,286,735,1629,3124,5186,7447,9253,9947,9253,7447,5186,3125,1628,735,287,96,75,
    /* [21] len=18 off=8 */ 139,375,883,1824,3305,5245,7294,8892,9498,8891,7295,5244,3305,1825,883,374,139,125,
};

constexpr uint16_t GAUSS_PMF_INDEX_WEIGHTS[] = {
    46928,17197,911,229,91,53,34,25,17,13,8,7,5,4,2,3,2,3,2,1,1,0
};

constexpr int32_t BIT_EST_PMF_LENGTHS[] = {
    14,9,17,12,11,6,11,11,12,18,15,13,6,6,6,6,6,8,6,6,6,6,6,6,
    6,6,8,6,6,6,6,6,6,18,6,6,6,6,6,6,8,6,6,6,6,6,8,6
};
constexpr int32_t BIT_EST_PMF_OFFSETS[] = {
    8,5,8,3,6,2,3,6,7,8,5,7,2,2,2,2,2,3,2,2,2,2,2,2,
    2,2,3,2,2,2,2,2,2,8,2,2,2,2,2,2,3,2,2,2,2,2,3,2
};
constexpr int32_t BIT_EST_PMF_TABLE[] = {
    /* [0] len=14 off=8 */ 5,17,52,158,482,1459,4815,43629,14617,224,50,16,5,7,
    /* [1] len=9 off=5 */ 9,53,296,1799,43953,19332,77,10,7,
    /* [2] len=17 off=8 */ 165,329,652,1276,2436,4523,8602,22471,23503,1072,190,77,38,18,9,5,170,
    /* [3] len=12 off=3 */ 6,25,147,20211,41640,2677,617,156,40,11,2,4,
    /* [4] len=11 off=6 */ 5,22,97,417,2065,41107,21669,127,19,4,4,
    /* [5] len=6 off=2 */ 1,204,65128,201,1,1,
    /* [6] len=11 off=3 */ 4,18,119,19380,43222,2212,449,100,23,5,4,
    /* [7] len=11 off=6 */ 6,22,93,384,1822,34576,28382,208,30,7,6,
    /* [8] len=12 off=7 */ 4,13,52,196,751,3151,43875,17314,143,25,6,6,
    /* [9] len=18 off=8 */ 357,625,1081,1840,3059,5047,9374,28710,14092,556,155,76,42,24,13,7,5,473,
    /* [10] len=15 off=5 */ 4,9,25,70,284,12330,41096,7331,2664,1050,408,157,60,23,25,
    /* [11] len=13 off=7 */ 8,26,90,303,1021,3760,43460,16606,200,41,12,3,6,
    /* [12] len=6 off=2 */ 3,664,64239,627,2,1,
    /* [13] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [14] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [15] len=6 off=2 */ 9,1241,63064,1212,9,1,
    /* [16] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [17] len=8 off=3 */ 5,95,3223,59009,3107,91,5,1,
    /* [18] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [19] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [20] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [21] len=6 off=2 */ 1,378,64762,391,3,1,
    /* [22] len=6 off=2 */ 12,1157,63118,1234,13,2,
    /* [23] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [24] len=6 off=2 */ 1,143,65253,137,1,1,
    /* [25] len=6 off=2 */ 4,739,64025,763,4,1,
    /* [26] len=8 off=3 */ 4,84,2933,59441,2983,86,4,1,
    /* [27] len=6 off=2 */ 1,32,65467,34,1,1,
    /* [28] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [29] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [30] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [31] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [32] len=6 off=2 */ 1,264,64951,316,3,1,
    /* [33] len=18 off=8 */ 8,19,52,134,350,913,2534,10520,37373,9791,2416,875,335,129,49,19,7,12,
    /* [34] len=6 off=2 */ 1,405,64719,409,1,1,
    /* [35] len=6 off=2 */ 1,40,65454,39,1,1,
    /* [36] len=6 off=2 */ 5,992,63528,1005,5,1,
    /* [37] len=6 off=2 */ 1,136,65268,129,1,1,
    /* [38] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [39] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [40] len=8 off=3 */ 1,48,2513,60260,2660,51,2,1,
    /* [41] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [42] len=6 off=2 */ 1,46,65444,43,1,1,
    /* [43] len=6 off=2 */ 1,227,65077,229,1,1,
    /* [44] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [45] len=6 off=2 */ 1,1,65531,1,1,1,
    /* [46] len=8 off=3 */ 5,94,3172,59029,3138,92,5,1,
    /* [47] len=6 off=2 */ 1,533,64465,534,2,1,
};

constexpr uint16_t BIT_EST_PMF_INDEX_WEIGHTS[] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

// clang-format on

}  // namespace

// ---------------------------------------------------------------------------
// Factory methods
// ---------------------------------------------------------------------------

RansTestPmf RansTestPmf::MakeGaussianPmf()
{
    return RansTestPmf(GAUSS_PMF_LENGTHS, GAUSS_PMF_OFFSETS, GAUSS_PMF_TABLE, GAUSS_PMF_INDEX_WEIGHTS, 16, 2);
}

RansTestPmf RansTestPmf::MakeBitEstimatorPmf()
{
    return RansTestPmf(BIT_EST_PMF_LENGTHS, BIT_EST_PMF_OFFSETS, BIT_EST_PMF_TABLE, BIT_EST_PMF_INDEX_WEIGHTS, 16, 2);
}

// ---------------------------------------------------------------------------
// RansTestPmf implementation
// ---------------------------------------------------------------------------

RansTestPmf::RansTestPmf(std::span<const int32_t> pmfLengths, std::span<const int32_t> pmfOffsets,
                         std::span<const int32_t> pmfTable, std::span<const uint16_t> indexWeights, int32_t symbolBits,
                         int32_t bypassBits)
    : m_pmfLengths(pmfLengths.begin(), pmfLengths.end())
    , m_pmfOffsets(pmfOffsets.begin(), pmfOffsets.end())
    , m_pmfTable(pmfTable.begin(), pmfTable.end())
    , m_symbolBits(symbolBits)
    , m_bypassBits(bypassBits)
{
    m_tableOffsets.resize(m_pmfLengths.size());
    std::exclusive_scan(m_pmfLengths.begin(), m_pmfLengths.end(), m_tableOffsets.begin(), size_t{ 0 });
    for (int i = 0; i < static_cast<int>(indexWeights.size()); i++) {
        m_indexLut.insert(m_indexLut.end(), indexWeights[i], static_cast<int32_t>(i));
    }
    double wSum = static_cast<double>(m_indexLut.size());
    m_indexWeights.resize(indexWeights.size());
    for (int i = 0; i < static_cast<int>(indexWeights.size()); i++) {
        m_indexWeights[i] = indexWeights[i] / wSum;
    }
}

void RansTestPmf::GenerateSymbols(std::vector<int32_t>& indices, std::vector<int32_t>& values, int numSymbols) const
{
    indices.resize(numSymbols);
    values.resize(numSymbols);
    std::uniform_int_distribution<int> indexDist(0, static_cast<int>(m_indexLut.size()) - 1);
    for (int i = 0; i < numSymbols; i++) {
        auto pmfIdx = m_indexLut[indexDist(m_rng)];
        indices[i] = pmfIdx;
        values[i] = SampleValue(pmfIdx);
    }
}

double RansTestPmf::CalcShannonEntropy() const
{
    double h = 0.0;
    for (int i = 0; i < static_cast<int>(m_pmfLengths.size()); i++) {
        double w = m_indexWeights[i];
        if (w == 0.0) continue;
        for (int32_t j = 0; j < m_pmfLengths[i]; j++) {
            if (auto c = m_pmfTable[m_tableOffsets[i] + j]; c > 0) {
                double p = c / static_cast<double>(1 << m_symbolBits);
                h -= w * p * std::log2(p);
            }
        }
    }
    return h;
}

int32_t RansTestPmf::SampleValue(int32_t pmfIdx) const
{
    auto len = static_cast<int32_t>(m_pmfLengths[pmfIdx]);
    auto off = m_pmfOffsets[pmfIdx];
    auto base = m_tableOffsets[pmfIdx];
    std::uniform_int_distribution<int32_t> dist(0, (1 << m_symbolBits) - 1);
    auto r = dist(m_rng);
    int32_t cum = 0;
    for (int32_t j = 0; j < len - 1; j++) {
        cum += m_pmfTable[base + j];
        if (r < cum) return j - off;
    }
    return (len - 1) - off;
}

}  // namespace libmlvc
