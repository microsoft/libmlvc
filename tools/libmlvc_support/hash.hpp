// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <memory>
#include <string>

namespace libmlvc {

class Sha256 {
public:
    Sha256();
    ~Sha256();
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;
    Sha256(Sha256&&) noexcept = default;
    Sha256& operator=(Sha256&&) noexcept = default;

    static expected<Sha256> Create();
    expected<void> Initialize();
    expected<void> Update(const void* data, size_t len);
    expected<std::string> Finalize();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace libmlvc
