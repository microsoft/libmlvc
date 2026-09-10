// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "libmlvc/common/macros.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace libmlvc {

template <int N>
constexpr std::array<int, N> StridesFromShape(const std::array<int, N>& shape)
{
    std::array<int, N> strides{};
    if (!shape.empty()) {
        strides[shape.size() - 1] = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return strides;
}

template <typename T, int N>
class Tensor {
public:
    Tensor(std::string_view name = {}) : m_name(name)
    {
        // Empty tensor
    }

    Tensor(const std::array<int, N>& shape, const std::array<int, N>& strides, std::vector<T>&& data,
           std::string_view name = {}) noexcept
        : m_shape(shape), m_strides(strides), m_data(std::move(data)), m_view(m_data->data(), m_data->size()), m_name(name)
    {
        // Take ownership of the data (move existing data)
    }

    Tensor(const std::array<int, N>& shape, const std::array<int, N>& strides, std::span<T> view,
           std::string_view name = {}) noexcept
        : m_shape(shape), m_strides(strides), m_view(view), m_name(name)
    {
        // View to the existing data (does not take ownership)
    }

    Tensor(const std::array<int, N>& shape, std::string_view name = {}) noexcept : Tensor(name)
    {
        // Zero initialized tensor (owns the newly allocated data)
        Create(shape);
    }

    Tensor(const Tensor&) = delete;  // Use CopyTo instead
    Tensor(Tensor&& other) = default;

    Tensor& operator=(const Tensor& other) = delete;
    Tensor& operator=(Tensor&& other) = default;

    void Create(const std::array<int, N>& shape, const std::array<int, N>& strides)
    {
        // Only create new data if the shape or strides have changed
        if (shape == m_shape && strides == m_strides) {
            return;
        }
        m_shape = shape;
        m_strides = strides;
        if (!m_data) {
            m_data = std::vector<T>{};
        }
        m_data->resize(shape.empty() ? 0 : m_strides[0] * m_shape[0]);
        m_view = { m_data->data(), m_data->size() };
    }

    void Create(const std::array<int, N>& shape)
    {
        // Create new data if shape has changed
        if (shape == m_shape) {
            return;
        }
        Create(shape, StridesFromShape<N>(shape));
    }

    void SetZero() { std::fill(Data().begin(), Data().end(), T{}); }

    void SetValue(const T& value)
    {
        if (IsContiguous()) {
            std::fill(Data().begin(), Data().end(), value);
            return;
        }
        // Non-contiguous: only set logical elements, skip stride padding
        auto data = Data();
        MLVC_ASSERT(m_strides[N - 1] == 1);
        if constexpr (N == 1) {
            std::fill_n(data.begin(), m_shape[0], value);
        } else if constexpr (N == 2) {
            for (int i = 0; i < m_shape[0]; i++) {
                std::fill_n(data.begin() + i * m_strides[0], m_shape[1], value);
            }
        } else if constexpr (N == 3) {
            for (int i = 0; i < m_shape[0]; i++) {
                for (int j = 0; j < m_shape[1]; j++) {
                    std::fill_n(data.begin() + i * m_strides[0] + j * m_strides[1], m_shape[2], value);
                }
            }
        } else {
            static_assert(N <= 3, "SetValue not implemented for N > 3");
        }
    }

    template <int Dim = N, typename = std::enable_if_t<Dim == 1>>
    T& operator()(const int i)
    {
        MLVC_ASSERT(i >= 0 && i < m_shape[0]);
        return m_view[i];
    }

    template <int Dim = N, typename = std::enable_if_t<Dim == 1>>
    const T& operator()(const int i) const
    {
        MLVC_ASSERT(i >= 0 && i < m_shape[0]);
        return m_view[i];
    }

    template <int Dim = N, typename = std::enable_if_t<Dim == 2>>
    T& operator()(const int y, const int x)
    {
        MLVC_ASSERT(y >= 0 && y < m_shape[0]);
        MLVC_ASSERT(x >= 0 && x < m_shape[1]);
        const size_t index = y * m_strides[0] + x * m_strides[1];
        return m_view[index];
    }

    template <int Dim = N, typename = std::enable_if_t<Dim == 2>>
    const T& operator()(const int y, const int x) const
    {
        MLVC_ASSERT(y >= 0 && y < m_shape[0]);
        MLVC_ASSERT(x >= 0 && x < m_shape[1]);
        const size_t index = y * m_strides[0] + x * m_strides[1];
        return m_view[index];
    }

    template <int Dim = N, typename = std::enable_if_t<Dim == 3>>
    T& operator()(const int ch, const int y, const int x)
    {
        MLVC_ASSERT(ch >= 0 && ch < m_shape[0]);
        MLVC_ASSERT(y >= 0 && y < m_shape[1]);
        MLVC_ASSERT(x >= 0 && x < m_shape[2]);
        const size_t index = ch * m_strides[0] + y * m_strides[1] + x * m_strides[2];
        return m_view[index];
    }

    template <int Dim = N, typename = std::enable_if_t<Dim == 3>>
    const T& operator()(const int ch, const int y, const int x) const
    {
        MLVC_ASSERT(ch >= 0 && ch < m_shape[0]);
        MLVC_ASSERT(y >= 0 && y < m_shape[1]);
        MLVC_ASSERT(x >= 0 && x < m_shape[2]);
        const size_t index = ch * m_strides[0] + y * m_strides[1] + x * m_strides[2];
        return m_view[index];
    }

    std::span<T> Data() const { return m_view; }

    const std::array<int, N>& Shape() const { return m_shape; }

    const std::array<int, N>& Strides() const { return m_strides; }

    const std::string& Name() const { return m_name; }
    void SetName(std::string_view name) { m_name = name; }

    size_t Size() const
    {
        if (m_shape.empty()) {
            return 0;
        }
        size_t res = 1;
        for (auto s : m_shape) {
            res *= s;
        }
        return res;
    }

    bool IsEmpty() const { return Size() == 0; }

    size_t SizeBytes() const { return m_view.size_bytes(); }

    bool IsContiguous() const { return m_strides == StridesFromShape<N>(m_shape); }

private:
    std::array<int, N> m_shape{};
    std::array<int, N> m_strides{};
    std::optional<std::vector<T>> m_data;
    std::span<T> m_view;
    std::string m_name;
};

}  // namespace libmlvc
