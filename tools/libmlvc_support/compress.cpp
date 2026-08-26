// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc_support/compress.hpp"

#include <libmlvc/error_codes.hpp>

#include <zlib.h>

#include <cstring>
#include <fstream>

namespace libmlvc {

namespace {

using ZlibInitFn = int (*)(z_streamp, int);
using ZlibProcessFn = int (*)(z_streamp, int);
using ZlibEndFn = int (*)(z_streamp);

expected<std::vector<std::byte>> ReadBinaryFile(const std::filesystem::path& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return make_error_code(Error::io_error);
    }

    file.seekg(0, std::ios::end);
    const auto fileSize = file.tellg();
    if (fileSize < 0) {
        return make_error_code(Error::io_error);
    }
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> data(static_cast<size_t>(fileSize));
    if (!file.read(reinterpret_cast<char*>(data.data()), fileSize)) {
        return make_error_code(Error::io_error);
    }
    return data;
}

expected<std::vector<std::byte>> RunZlibStream(std::span<const std::byte> data, ZlibInitFn initFn, int initParam,
                                               ZlibProcessFn processFn, int flushMode, ZlibEndFn endFn) noexcept
{
    if (data.empty()) return {};

    z_stream strm{};
    strm.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(data.data()));
    strm.avail_in = static_cast<uInt>(data.size());

    if (initFn(&strm, initParam) != Z_OK) {
        return make_error_code(Error::general_failure);
    }

    std::vector<std::byte> out;
    constexpr size_t kChunkSize = 16 * 1024;
    int ret = Z_OK;
    while (ret == Z_OK) {
        std::byte buffer[kChunkSize];
        strm.next_out = reinterpret_cast<Bytef*>(buffer);
        strm.avail_out = static_cast<uInt>(kChunkSize);

        ret = processFn(&strm, flushMode);
        if (ret != Z_OK && ret != Z_STREAM_END) {
            endFn(&strm);
            return make_error_code(Error::general_failure);
        }

        const size_t produced = kChunkSize - strm.avail_out;
        if (produced) {
            const size_t oldSize = out.size();
            out.resize(oldSize + produced);
            std::memcpy(out.data() + oldSize, buffer, produced);
        }
    }

    if (endFn(&strm) != Z_OK) {
        return make_error_code(Error::general_failure);
    }
    return out;
}

}  // namespace

expected<std::vector<std::byte>> Compress(std::span<const std::byte> data) noexcept
{
    // MAX_WBITS + 16 produces gzip format (symmetric with Uncompress which uses MAX_WBITS + 32 to auto-detect)
    auto initFn = [](z_streamp strm, int /*unused*/) {
        return deflateInit2(strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, MAX_WBITS + 16, 8, Z_DEFAULT_STRATEGY);
    };
    return RunZlibStream(data, initFn, 0, deflate, Z_FINISH, deflateEnd);
}

expected<std::vector<std::byte>> Uncompress(std::span<const std::byte> data) noexcept
{
    // MAX_WBITS + 32 enables automatic gzip/zlib format detection
    auto initFn = [](z_streamp strm, int windowBits) { return inflateInit2(strm, windowBits); };
    return RunZlibStream(data, initFn, MAX_WBITS + 32, inflate, Z_NO_FLUSH, inflateEnd);
}

expected<std::vector<std::byte>> ReadGzipFile(const std::filesystem::path& filename)
{
    auto compressedBytes = ReadBinaryFile(filename);
    if (!compressedBytes) {
        return compressedBytes.error();
    }
    return Uncompress(compressedBytes.value());
}

}  // namespace libmlvc
