// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/hash.hpp"

#include <libmlvc/error_codes.hpp>

// clang-format off
#if defined(_WIN32)
    #include <windows.h>
    #include <bcrypt.h>
#elif defined(__APPLE__)
    #include <CommonCrypto/CommonCryptor.h>
    #include <CommonCrypto/CommonDigest.h>
#endif
// clang-format on

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace libmlvc {

#if defined(_WIN32)

class Sha256::Impl {
public:
    Impl() {}

    ~Impl()
    {
        if (m_hashHandle) {
            ::BCryptDestroyHash(m_hashHandle);
        }
        if (m_algHandle) {
            ::BCryptCloseAlgorithmProvider(m_algHandle, 0);
        }
    }

    expected<void> Initialize()
    {
        if (auto s = ::BCryptOpenAlgorithmProvider(&m_algHandle, BCRYPT_SHA256_ALGORITHM, nullptr, 0); s < 0) {
            return make_error_code(Error::general_failure);
        }

        DWORD objLen = 0;
        ULONG bytesCopied = 0;
        if (auto s = ::BCryptGetProperty(m_algHandle, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&objLen),
                                         sizeof(DWORD), &bytesCopied, 0);
            s < 0 || bytesCopied != sizeof(DWORD)) {
            return make_error_code(Error::general_failure);
        }

        DWORD hashLen = 0;
        if (auto s = ::BCryptGetProperty(m_algHandle, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hashLen),
                                         sizeof(DWORD), &bytesCopied, 0);
            s < 0 || bytesCopied != sizeof(DWORD)) {
            return make_error_code(Error::general_failure);
        }

        m_hashObject.resize(objLen);
        m_hashBuffer.resize(hashLen);
        if (auto s = ::BCryptCreateHash(m_algHandle, &m_hashHandle, m_hashObject.data(),
                                        static_cast<ULONG>(m_hashObject.size()), nullptr, 0, 0);
            s < 0) {
            return make_error_code(Error::general_failure);
        }
        m_initialized = true;
        return {};
    }

    expected<void> Update(const void* data, size_t len)
    {
        if (!m_initialized) {
            return make_error_code(Error::not_initialized);
        }
        if (!data || !len) {
            return make_error_code(Error::invalid_argument);
        }
        if (m_finalized) {
            return make_error_code(Error::general_failure);
        }

        if (auto s = ::BCryptHashData(m_hashHandle, (PUCHAR)data, (ULONG)len, 0); s < 0) {
            return make_error_code(Error::general_failure);
        }
        return {};
    }

    expected<std::string> Finalize()
    {
        if (!m_initialized) {
            return make_error_code(Error::not_initialized);
        }

        if (!m_finalized) {
            if (auto s = ::BCryptFinishHash(m_hashHandle, m_hashBuffer.data(), (ULONG)m_hashBuffer.size(), 0); s < 0) {
                return make_error_code(Error::general_failure);
            }
            m_finalized = true;
        }
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (auto b : m_hashBuffer)
            oss << std::setw(2) << static_cast<int>(b);
        return oss.str();
    }

private:
    bool m_initialized{ false };
    bool m_finalized{ false };
    BCRYPT_ALG_HANDLE m_algHandle{ nullptr };
    BCRYPT_HASH_HANDLE m_hashHandle{ nullptr };
    std::vector<unsigned char> m_hashObject;
    std::vector<unsigned char> m_hashBuffer;
};

#elif defined(__APPLE__)

class Sha256::Impl {
public:
    Impl() {}

    ~Impl() {}

    expected<void> Initialize()
    {
        if (auto ret = CC_SHA256_Init(&m_shaState); ret != 1) {
            return make_error_code(Error::general_failure);
        }
        m_initialized = true;
        return {};
    }

    expected<void> Update(const void* data, size_t len)
    {
        if (!m_initialized) {
            return make_error_code(Error::not_initialized);
        }
        if (!data || !len) {
            return make_error_code(Error::invalid_argument);
        }
        if (m_finalized) {
            return make_error_code(Error::general_failure);
        }

        if (auto ret = CC_SHA256_Update(&m_shaState, data, static_cast<CC_LONG>(len)); ret != 1) {
            return make_error_code(Error::general_failure);
        }
        return {};
    }

    expected<std::string> Finalize()
    {
        if (!m_initialized) {
            return make_error_code(Error::not_initialized);
        }

        if (!m_finalized) {
            m_hashBuffer.resize(CC_SHA256_DIGEST_LENGTH);
            if (auto ret = CC_SHA256_Final(m_hashBuffer.data(), &m_shaState); ret != 1) {
                return make_error_code(Error::general_failure);
            }
            m_finalized = true;
        }
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (auto b : m_hashBuffer) {
            oss << std::setw(2) << static_cast<int>(b);
        }
        return oss.str();
    }

private:
    bool m_initialized{ false };
    bool m_finalized{ false };
    std::vector<unsigned char> m_hashBuffer;
    CC_SHA256_CTX m_shaState;
};
#else

    #error "Unsupported platform for Sha256 implementation."

#endif

Sha256::Sha256() : m_impl(std::make_unique<Impl>()) {}

Sha256::~Sha256() = default;

expected<Sha256> Sha256::Create()
{
    Sha256 hasher;
    if (auto ret = hasher.Initialize(); !ret) return ret.error();
    return hasher;
}

expected<void> Sha256::Initialize()
{
    return m_impl->Initialize();
}

expected<void> Sha256::Update(const void* data, size_t len)
{
    return m_impl->Update(data, len);
}

expected<std::string> Sha256::Finalize()
{
    return m_impl->Finalize();
}

}  // namespace libmlvc
