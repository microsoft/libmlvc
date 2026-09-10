// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/tar.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"

#include <libmlvc/error_codes.hpp>

#include <string_view>

namespace libmlvc {

namespace {

enum class TarFileType : char { File = '0', HardLink = '1', SymLink = '2', Directory = '5' };

struct TarFileEntry {
    TarFileType type;
    std::string_view name;
    std::span<const std::byte> data;
};

class TarHeader {
public:
    TarHeader(std::span<const std::byte> data) : m_data(reinterpret_cast<const char*>(data.data()), data.size()) {}

    expected<std::string_view> GetFileName() const
    {
        auto name = m_data.substr(0, 100);
        auto end = name.find('\0');
        if (end == std::string_view::npos) {
            MLVC_LOG_ERROR("Invalid tar header: file name not null-terminated");
            return make_error_code(Error::general_failure);
        }
        return name.substr(0, end);
    }

    expected<TarFileType> GetFileType() const noexcept
    {
        const auto x = m_data[156];
        if (x != '\0' && x != '0' && x != '1' && x != '2' && x != '5') {
            MLVC_LOG_ERROR("Invalid tar header: unknown file type '%c'", x);
            return make_error_code(Error::general_failure);
        }
        return x == 0 ? TarFileType::File : static_cast<TarFileType>(x);
    }

    expected<size_t> GetFileSize() const noexcept
    {
        auto s = m_data.substr(124, 12);
        if (!(s.ends_with('\0') || s.ends_with(' '))) {
            MLVC_LOG_ERROR("Invalid tar header: file size field not null-terminated");
            return make_error_code(Error::general_failure);
        }
        s.remove_suffix(1);
        return FromOct(s);
    }

private:
    std::string_view m_data;

    static expected<size_t> FromOct(std::string_view s)
    {
        size_t res{};
        for (auto ch : s) {
            if (ch < '0' || '7' < ch) {
                MLVC_LOG_ERROR("Invalid octal digit in tar header: '%c'", ch);
                return make_error_code(Error::general_failure);
            }
            res = res * 8 + ch - '0';
        }
        return res;
    }
};

}  // namespace

expected<std::unordered_map<std::string, std::span<const std::byte>>> ExtractTar(std::span<const std::byte> data)
{
    static constexpr size_t blockSize = 512;
    auto allZeros = [](std::span<const std::byte> block) -> bool {
        for (auto value : block) {
            if (value != std::byte{}) return false;
        }
        return true;
    };

    std::unordered_map<std::string, std::span<const std::byte>> res;
    std::span<const std::byte> remainingData = data;
    while (true) {
        if (remainingData.size() < 2 * blockSize) {
            MLVC_LOG_ERROR("Invalid tar header: unexpected end of archive");
            return make_error_code(Error::general_failure);
        }

        if (allZeros(remainingData.subspan(0, 2 * blockSize))) {
            break;  // expected end of archive
        }

        const TarHeader headerView(remainingData.subspan(0, blockSize));
        auto name = headerView.GetFileName();
        if (!name) {
            return name.error();
        }

        const auto fileType = headerView.GetFileType();
        if (!fileType) {
            return fileType.error();
        }

        if (fileType.value() == TarFileType::File) {
            auto fsize = headerView.GetFileSize();
            if (!fsize) {
                return fsize.error();
            }

            const size_t paddedSize = ((fsize.value() + blockSize - 1) / blockSize) * blockSize;
            const size_t nextPos = blockSize + paddedSize;
            if (nextPos > remainingData.size()) {
                MLVC_LOG_ERROR("Invalid tar header: file size exceeds archive size");
                return make_error_code(Error::general_failure);
            }
            const auto fileData = remainingData.subspan(blockSize, fsize.value());
            res.emplace(name.value(), fileData);
            remainingData = remainingData.subspan(nextPos);
        } else if (fileType.value() == TarFileType::Directory || fileType.value() == TarFileType::HardLink
                   || fileType.value() == TarFileType::SymLink) {
            // Skip non-regular files
            remainingData = remainingData.subspan(blockSize);
        } else {
            MLVC_LOG_ERROR("Invalid tar header: unsupported file type");
            MLVC_ASSERT(false);
            return make_error_code(Error::general_failure);
        }
    }
    return res;
}

}  // namespace libmlvc
