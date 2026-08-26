// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <libmlvc/unexpected.hpp>

// clang-format off
#include <cassert>
#include <system_error>
// clang-format on

namespace libmlvc {

namespace internal {
template <typename T, typename E>
class expected_storage {

protected:
    expected_storage() noexcept {}

    ~expected_storage() {}

    const T& value() const noexcept { return val; }

    T& value() noexcept { return val; }

    const E& error() const noexcept { return err; }

    E& error() noexcept { return err; }

private:
    union {
        T val;
        E err;
    };
};

template <typename E>
class expected_storage<void, E> {

protected:
    expected_storage() noexcept {}

    ~expected_storage() {}

    const E& error() const noexcept { return err; }

    E& error() noexcept { return err; }

private:
    union {
        E err;
    };
};
}  // namespace internal

template <typename T, typename E = std::error_code>
class expected : private internal::expected_storage<T, E> {

public:
    typedef T value_type;
    typedef E error_type;

    expected() noexcept(std::is_nothrow_default_constructible<T>::value) : has_value(true) { new (&value()) T; }

    expected(const expected& rhs) noexcept(std::is_nothrow_copy_constructible<T>::value
                                           && std::is_nothrow_copy_constructible<E>::value)
        : has_value(rhs.has_value)
    {
        if (has_value) {
            new (&value()) T(rhs.value());
        } else {
            new (&error()) E(rhs.error());
        }
    }

    expected(const T& v) noexcept(std::is_nothrow_copy_constructible<T>::value) : has_value(true)
    {
        new (&value()) T(v);
    }

    expected(const unexpected_type<E>& e) noexcept(std::is_nothrow_copy_constructible<E>::value) : has_value(false)
    {
        new (&error()) E(e.value());
    }

    template <typename X>
    expected(const unexpected_type<X>& e) noexcept(std::is_nothrow_copy_constructible<E>::value
                                                   && std::is_base_of<E, X>::value)
        : has_value(false)
    {
        new (&error()) E(e.value());
    }

    expected(const expected<expected, E>& rhs) noexcept(std::is_nothrow_copy_constructible<T>::value
                                                        && std::is_nothrow_copy_constructible<E>::value)
        : has_value(rhs && *rhs)
    {
        if (has_value) {
            new (&value()) T(rhs->value());
        } else {
            new (&error()) E(rhs ? rhs->error() : rhs.error());
        }
    }

    expected(expected&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value
                                      && std::is_nothrow_move_constructible<E>::value)
        : has_value(rhs.has_value)
    {
        if (has_value) {
            new (&value()) T(std::move(rhs.value()));
        } else {
            new (&error()) E(std::move(rhs.error()));
        }
    }

    expected(T&& v) noexcept(std::is_nothrow_move_constructible<T>::value) : has_value(true)
    {
        new (&value()) T(std::move(v));
    }

    expected(unexpected_type<E>&& e) noexcept(std::is_nothrow_move_constructible<E>::value) : has_value(false)
    {
        new (&error()) E(std::move(e.value()));
    }

    template <typename X>
    expected(unexpected_type<X>&& e) noexcept(std::is_nothrow_move_constructible<E>::value && std::is_base_of<E, X>::value)
        : has_value(false)
    {
        new (&error()) E(std::move(e.value()));
    }

    expected(expected<expected, E>&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value
                                                   && std::is_nothrow_move_constructible<E>::value)
        : has_value(rhs && *rhs)
    {
        if (has_value) {
            new (&value()) T(std::move(rhs->value()));
        } else {
            new (&error()) E(std::move(rhs ? rhs->error() : rhs.error()));
        }
    }

    static_assert(!std::is_same<T, std::error_code>::value, "expected type cannot be an error, it's misuse");
    template <typename EX = E, typename X = typename std::enable_if<std::is_same<EX, std::error_code>::value>::type>
    expected(const std::error_code& ec) : expected<T, std::error_code>(libmlvc::make_unexpected(ec))
    {
    }

private:
    template <typename X>
    using remove_const_ref_t = typename std::remove_cv<typename std::remove_reference<X>::type>::type;

public:
    template <typename U,
              typename = typename std::enable_if<std::is_assignable<T, U>::value && !std::is_same<T, remove_const_ref_t<U>>::value &&  // don't shadow copy and move ctor
                                                 !std::is_base_of<T, U>::value  // prevent slicing
                                                 >::type>
    expected(U&& p) : has_value(true)
    {
        new (&value()) T(std::forward<U>(p));
    }

    template <typename U, typename = typename std::enable_if<std::is_assignable<T, U>::value && std::is_constructible<T, U>::value
                                                             && !std::is_same<expected<U, E>, expected>::value &&  // don't shadow copy and move ctor
                                                             !std::is_base_of<T, U>::value  // prevent slicing
                                                             >::type>
    expected(expected<U, E>&& p) : has_value(p)
    {
        if (p)
            new (&value()) T(std::forward<U>(p.value()));
        else
            new (&error()) E(p.error());
    }

    ~expected()
    {
        if (has_value) {
            value().~T();
        } else {
            error().~E();
        }
    }

    expected&
    operator=(const expected& rhs) noexcept(std::is_nothrow_copy_constructible_v<T> && std::is_nothrow_copy_assignable_v<T>
                                            && std::is_nothrow_destructible_v<T> && std::is_nothrow_copy_constructible_v<E>
                                            && std::is_nothrow_copy_assignable_v<E> && std::is_nothrow_destructible_v<E>)
    {
        if (*this) {
            if (rhs) {
                value() = rhs.value();
            } else {
                value().~T();
                has_value = false;
                new (&error()) E(rhs.error());
            }
        } else {
            if (rhs) {
                error().~E();
                has_value = true;
                new (&value()) T(rhs.value());
            } else {
                error() = rhs.error();
            }
        }

        return *this;
    }

    expected& operator=(const unexpected_type<E>& e) noexcept(std::is_nothrow_copy_constructible_v<E>
                                                              && std::is_nothrow_copy_assignable_v<E>
                                                              && std::is_nothrow_destructible_v<E>)
    {
        if (has_value) {
            value().~T();
            has_value = false;
            new (&error()) E(e.value());
        } else {
            error() = e.value();
        }

        return *this;
    }

    expected&
    operator=(expected&& rhs) noexcept(std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>
                                       && std::is_nothrow_destructible_v<T> && std::is_nothrow_move_constructible_v<E>
                                       && std::is_nothrow_move_assignable_v<E> && std::is_nothrow_destructible_v<E>)
    {
        if (*this) {
            if (rhs) {
                value() = std::move(rhs.value());
            } else {
                value().~T();
                has_value = false;
                new (&error()) E(std::move(rhs.error()));
            }
        } else {
            if (rhs) {
                error().~E();
                has_value = true;
                new (&value()) T(std::move(rhs.value()));
            } else {
                error() = std::move(rhs.error());
            }
        }

        return *this;
    }

    expected& operator=(unexpected_type<E>&& e) noexcept(std::is_nothrow_move_constructible_v<E>
                                                         && std::is_nothrow_move_assignable_v<E>
                                                         && std::is_nothrow_destructible_v<E>)
    {
        if (has_value) {
            value().~T();
            has_value = false;
            new (&error()) E(std::move(e.value()));
        } else {
            error() = std::move(e.value());
        }

        return *this;
    }

    template <typename U, typename = typename std::enable_if<std::is_assignable<T, U>::value && !std::is_same<expected<U, E>, expected>::value
                                                             && std::is_move_assignable<T>::value>::type>
    expected& operator=(expected<U, E>&& p)
    {
        if (*this && p)
            value() = std::move(p.value());
        else if (*this && !p) {
            value().~T();
            has_value = false;
            new (&error()) E(p.error());
        } else if (p) {
            error().~E();
            has_value = true;
            new (&value()) T(std::move(p.value()));
        } else
            error() = std::move(p.error());
        return *this;
    }

    template <typename U, typename = typename std::enable_if<std::is_assignable<T, U>::value && !std::is_same<expected<U, E>, expected>::value
                                                             && std::is_copy_assignable<T>::value>::type>
    expected& operator=(expected<U, E> const& p)
    {
        if (*this && p)
            value() = p.value();
        else if (*this && !p) {
            value().~T();
            has_value = false;
            new (&error()) E(p.error());
        } else if (p) {
            error().~E();
            has_value = true;
            new (&value()) T(p.value());
        } else
            error() = p.error();
        return *this;
    }

    void swap(expected& rhs) noexcept(std::is_nothrow_copy_constructible_v<T> && std::is_nothrow_copy_assignable_v<T>
                                      && std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>
                                      && std::is_nothrow_destructible_v<T> && std::is_nothrow_copy_constructible_v<E>
                                      && std::is_nothrow_copy_assignable_v<E> && std::is_nothrow_move_constructible_v<E>
                                      && std::is_nothrow_move_assignable_v<E> && std::is_nothrow_destructible_v<E>)
    {
        using std::swap;

        if (*this) {
            if (rhs) {
                swap(value(), rhs.value());
            } else {
                E e(rhs.error());
                rhs.error().~E();
                rhs.has_value = true;

                new (&rhs.value()) T(std::move(value()));
                value().~T();
                has_value = false;

                new (&error()) E(std::move(e));
            }
        } else {
            if (rhs) {
                rhs.swap(*this);
            } else {
                swap(error(), rhs.error());
            }
        }
    }

    const T& value() const noexcept
    {
        if (!*this) std::abort();
        return internal::expected_storage<T, E>::value();
    }

    T& value() noexcept
    {
        if (!*this) std::abort();
        return internal::expected_storage<T, E>::value();
    }

    const T* operator->() const noexcept
    {
        assert(*this);
        return &value();
    }

    T* operator->() noexcept
    {
        assert(*this);
        return &value();
    }

    const T& operator*() const noexcept
    {
        assert(*this);
        return value();
    }

    T& operator*() noexcept
    {
        assert(*this);
        return value();
    }

    explicit operator bool() const noexcept { return has_value; }

    const E& error() const noexcept
    {
        if (*this) std::abort();
        return internal::expected_storage<T, E>::error();
    }

    E& error() noexcept
    {
        if (*this) std::abort();
        return internal::expected_storage<T, E>::error();
    }

    unexpected_type<E> get_unexpected() const noexcept(std::is_nothrow_move_constructible<E>::value)
    {
        assert(!*this);
        return make_unexpected(error());
    }

    template <typename U>
    T value_or(U&& v) const
        noexcept(std::is_nothrow_copy_constructible<T>::value && std::is_nothrow_move_constructible<U>::value)
    {
        if (*this)
            return value();
        else
            return static_cast<T>(std::forward<U>(v));
    }

private:
    bool has_value;
};

template <typename T, typename E>
class expected<T&, E> {
    static_assert(!std::is_reference<T&>::value, "expected<T&> is not supported");
};

template <typename E>
class expected<void, E> : private internal::expected_storage<void, E> {

public:
    typedef void value_type;
    typedef E error_type;

    expected() noexcept : has_value(true) {}

    expected(const expected& rhs) noexcept(std::is_nothrow_copy_constructible<E>::value) : has_value(rhs.has_value)
    {
        if (!has_value) {
            new (&error()) E(rhs.error());
        }
    }

    expected(const unexpected_type<E>& e) noexcept(std::is_nothrow_copy_constructible<E>::value) : has_value(false)
    {
        new (&error()) E(e.value());
    }

    template <typename X>
    expected(const unexpected_type<X>& e) noexcept(std::is_nothrow_copy_constructible<E>::value
                                                   && std::is_base_of<E, X>::value)
        : has_value(false)
    {
        new (&error()) E(e.value());
    }

    expected(const expected<expected, E>& rhs) noexcept(std::is_nothrow_copy_constructible<E>::value)
        : has_value(rhs && *rhs)
    {
        if (!has_value) {
            new (&error()) E(rhs ? rhs->error() : rhs.error());
        }
    }

    expected(expected&& rhs) noexcept(std::is_nothrow_move_constructible<E>::value) : has_value(rhs.has_value)
    {
        if (!has_value) {
            new (&error()) E(std::move(rhs.error()));
        }
    }

    expected(unexpected_type<E>&& e) noexcept(std::is_nothrow_move_constructible<E>::value) : has_value(false)
    {
        new (&error()) E(std::move(e.value()));
    }

    template <typename X>
    expected(unexpected_type<X>&& e) noexcept(std::is_nothrow_move_constructible<E>::value && std::is_base_of<E, X>::value)
        : has_value(false)
    {
        new (&error()) E(std::move(e.value()));
    }

    expected(expected<expected, E>&& rhs) noexcept(std::is_nothrow_move_constructible<E>::value)
        : has_value(rhs && *rhs)
    {
        if (!has_value) {
            new (&error()) E(std::move(rhs ? rhs->error() : rhs.error()));
        }
    }

    template <typename EX = E, typename X = typename std::enable_if<std::is_same<EX, std::error_code>::value>::type>
    expected(const std::error_code& ec) : expected<void, std::error_code>(libmlvc::make_unexpected(ec))
    {
    }

    ~expected()
    {
        if (!has_value) {
            error().~E();
        }
    }

    expected& operator=(const expected& rhs) noexcept(std::is_nothrow_copy_constructible_v<E>
                                                      && std::is_nothrow_copy_assignable_v<E>
                                                      && std::is_nothrow_destructible_v<E>)
    {
        if (*this) {
            if (!rhs) {
                has_value = false;
                new (&error()) E(rhs.error());
            }
        } else {
            if (rhs) {
                error().~E();
                has_value = true;
            } else {
                error() = rhs.error();
            }
        }

        return *this;
    }

    expected& operator=(const unexpected_type<E>& e) noexcept(std::is_nothrow_copy_constructible_v<E>
                                                              && std::is_nothrow_copy_assignable_v<E>
                                                              && std::is_nothrow_destructible_v<E>)
    {
        if (has_value) {
            has_value = false;
            new (&error()) E(e.value());
        } else {
            error() = e.value();
        }

        return *this;
    }

    expected& operator=(expected&& rhs) noexcept(std::is_nothrow_move_constructible_v<E>
                                                 && std::is_nothrow_move_assignable_v<E>
                                                 && std::is_nothrow_destructible_v<E>)
    {
        if (*this) {
            if (!rhs) {
                has_value = false;
                new (&error()) E(std::move(rhs.error()));
            }
        } else {
            if (rhs) {
                error().~E();
                has_value = true;
            } else {
                error() = std::move(rhs.error());
            }
        }

        return *this;
    }

    expected& operator=(unexpected_type<E>&& e) noexcept(std::is_nothrow_move_constructible_v<E>
                                                         && std::is_nothrow_move_assignable_v<E>
                                                         && std::is_nothrow_destructible_v<E>)
    {
        if (has_value) {
            has_value = false;
            new (&error()) E(std::move(e));
        } else {
            error() = std::move(e);
        }

        return *this;
    }

    void swap(expected& rhs) noexcept(std::is_nothrow_copy_constructible_v<E> && std::is_nothrow_copy_assignable_v<E>
                                      && std::is_nothrow_move_constructible_v<E> && std::is_nothrow_move_assignable_v<E>
                                      && std::is_nothrow_destructible_v<E>)
    {
        using std::swap;

        if (*this) {
            if (!rhs) {
                has_value = false;
                new (&error()) E(std::move(rhs.error()));
                rhs.error().~E();
                rhs.has_value = true;
            }
        } else {
            if (rhs) {
                rhs.swap(*this);
            } else {
                swap(error(), rhs.error());
            }
        }
    }

    explicit operator bool() const noexcept { return has_value; }

    void value() const noexcept { assert(*this); }

    const E& error() const noexcept
    {
        if (*this) std::abort();
        return internal::expected_storage<void, E>::error();
    }

    E& error() noexcept
    {
        if (*this) std::abort();
        return internal::expected_storage<void, E>::error();
    }

    unexpected_type<E> get_unexpected() const noexcept(std::is_nothrow_move_constructible<E>::value)
    {
        assert(!*this);
        return make_unexpected(error());
    }

private:
    bool has_value;
};

template <typename T, typename E>
static inline void swap(expected<T, E>& lhs, expected<T, E>& rhs) noexcept(std::is_nothrow_copy_constructible_v<E>
                                                                           && std::is_nothrow_copy_assignable_v<E>
                                                                           && std::is_nothrow_move_constructible_v<E>
                                                                           && std::is_nothrow_move_assignable_v<E>
                                                                           && std::is_nothrow_destructible_v<E>)
{
    lhs.swap(rhs);
}

template <typename E>
static inline expected<void, E> make_expected() noexcept
{
    return expected<void, E>();
}

template <typename T, typename E>
static inline expected<T, typename std::decay<E>::type>
make_expected_from_error(E&& e) noexcept(std::is_nothrow_move_constructible<E>::value)
{
    return make_unexpected(std::forward<E>(e));
}

template <typename T, typename E>
static inline bool operator==(const expected<T, E>& lhs, const expected<T, E>& rhs)
{
    return (lhs && rhs && *lhs == *rhs) || (!lhs && !rhs && lhs.get_unexpected() == rhs.get_unexpected());
}

template <typename E>
static inline bool operator==(const expected<void, E>& lhs, const expected<void, E>& rhs)
{
    return (lhs && rhs) || (!lhs && !rhs && lhs.get_unexpected() == rhs.get_unexpected());
}

template <typename T>
static inline bool operator==(const expected<T, std::error_code>& exp, const std::error_condition& ec)
{
    return !exp && exp.error() == ec;
}

template <typename T>
static inline bool operator==(const std::error_condition& ec, const expected<T, std::error_code>& exp)
{
    return !exp && exp.error() == ec;
}

template <typename T, typename E>
static inline bool operator==(const expected<T, E>& lhs, const unexpected_type<E>& rhs)
{
    return !lhs && lhs.get_unexpected() == rhs;
}

template <typename T, typename E>
static inline bool operator==(const unexpected_type<E>& lhs, const expected<T, E>& rhs)
{
    return rhs == lhs;
}

template <typename T, typename E>
static inline bool operator!=(const expected<T, E>& lhs, const expected<T, E>& rhs)
{
    return !(lhs == rhs);
}

template <typename E>
static inline bool operator!=(const expected<void, E>& lhs, const expected<void, E>& rhs)
{
    return !(lhs == rhs);
}

template <typename T, typename E>
static inline bool operator!=(const expected<T, E>& lhs, const unexpected_type<E>& rhs)
{
    return !(lhs == rhs);
}

template <typename T, typename E>
static inline bool operator!=(const unexpected_type<E>& lhs, const expected<T, E>& rhs)
{
    return !(lhs == rhs);
}

}  // namespace libmlvc
