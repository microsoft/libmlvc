// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/// @file
/// Error wrapper used to construct failed `expected` results.

#pragma once
#include <type_traits>
#include <utility>

namespace libmlvc {

/// Wrapper used to construct an `expected` result containing an error.
template <typename E>
class unexpected_type {
public:
    explicit unexpected_type(const E& val) noexcept(std::is_nothrow_copy_constructible<E>::value) : val(val) {}

    explicit unexpected_type(E&& val) noexcept(std::is_nothrow_move_constructible<E>::value) : val(std::move(val)) {}

    const E& value() const noexcept { return val; }

    E& value() noexcept { return val; }

private:
    E val;
};

template <typename E>
static inline unexpected_type<typename std::decay<E>::type>
make_unexpected(E&& val) noexcept(std::is_nothrow_move_constructible<E>::value)
{
    return unexpected_type<typename std::decay<E>::type>(std::move(val));
}

template <typename E>
static inline bool operator==(const unexpected_type<E>& lhs, const unexpected_type<E>& rhs)
{
    return lhs.value() == rhs.value();
}

template <typename E>
static inline bool operator!=(const unexpected_type<E>& lhs, const unexpected_type<E>& rhs)
{
    return !(lhs == rhs);
}

}  // namespace libmlvc
