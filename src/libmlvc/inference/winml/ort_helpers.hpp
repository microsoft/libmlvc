// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <libmlvc/expected.hpp>

#include <winml/onnxruntime_c_api.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace libmlvc_winml {

// Type aliases using std::unique_ptr with lambda deleters
using OrtStatusPtr = std::unique_ptr<OrtStatus, std::function<void(OrtStatus*)>>;
using OrtEnvPtr = std::unique_ptr<OrtEnv, std::function<void(OrtEnv*)>>;
using OrtSessionOptionsPtr = std::unique_ptr<OrtSessionOptions, std::function<void(OrtSessionOptions*)>>;
using OrtSessionPtr = std::unique_ptr<OrtSession, std::function<void(OrtSession*)>>;
using OrtMemoryInfoPtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
using OrtValuePtr = std::unique_ptr<OrtValue, std::function<void(OrtValue*)>>;
using OrtTypeInfoPtr = std::unique_ptr<OrtTypeInfo, std::function<void(OrtTypeInfo*)>>;
using OrtRunOptionsPtr = std::unique_ptr<OrtRunOptions, std::function<void(OrtRunOptions*)>>;
using OrtAllocatorPtr = std::unique_ptr<OrtAllocator, std::function<void(OrtAllocator*)>>;

// Helper functions to create wrapped pointers
inline OrtStatusPtr MakeOrtStatus(const OrtApi* api, OrtStatus* status)
{
    return OrtStatusPtr(status, [api](OrtStatus* p) {
        if (p && api) api->ReleaseStatus(p);
    });
}

inline OrtEnvPtr MakeOrtEnv(const OrtApi* api, OrtEnv* env)
{
    return OrtEnvPtr(env, [api](OrtEnv* p) {
        if (p && api) api->ReleaseEnv(p);
    });
}

inline OrtSessionOptionsPtr MakeOrtSessionOptions(const OrtApi* api, OrtSessionOptions* options)
{
    return OrtSessionOptionsPtr(options, [api](OrtSessionOptions* p) {
        if (p && api) api->ReleaseSessionOptions(p);
    });
}

inline OrtSessionPtr MakeOrtSession(const OrtApi* api, OrtSession* session)
{
    return OrtSessionPtr(session, [api](OrtSession* p) {
        if (p && api) api->ReleaseSession(p);
    });
}

inline OrtMemoryInfoPtr MakeOrtMemoryInfo(const OrtApi* api, OrtMemoryInfo* memInfo)
{
    return OrtMemoryInfoPtr(memInfo, [api](OrtMemoryInfo* p) {
        if (p && api) api->ReleaseMemoryInfo(p);
    });
}

inline OrtValuePtr MakeOrtValue(const OrtApi* api, OrtValue* value)
{
    return OrtValuePtr(value, [api](OrtValue* p) {
        if (p && api) api->ReleaseValue(p);
    });
}

inline OrtTypeInfoPtr MakeOrtTypeInfo(const OrtApi* api, OrtTypeInfo* typeInfo)
{
    return OrtTypeInfoPtr(typeInfo, [api](OrtTypeInfo* p) {
        if (p && api) api->ReleaseTypeInfo(p);
    });
}

inline OrtRunOptionsPtr MakeOrtRunOptions(const OrtApi* api, OrtRunOptions* runOptions)
{
    return OrtRunOptionsPtr(runOptions, [api](OrtRunOptions* p) {
        if (p && api) api->ReleaseRunOptions(p);
    });
}

inline OrtAllocatorPtr MakeOrtAllocator(const OrtApi* api, OrtAllocator* allocator)
{
    return OrtAllocatorPtr(allocator, [api](OrtAllocator* p) {
        if (p && api) api->ReleaseAllocator(p);
    });
}

inline const char* GetOrtErrorMessage(OrtStatus* status, const OrtApi* api)
{
    return status ? api->GetErrorMessage(status) : "";
}

template <typename T>
std::vector<T*> ToRawPointers(const std::vector<std::unique_ptr<T, std::function<void(T*)>>>& smartPtrs)
{
    std::vector<T*> rawPtrs;
    rawPtrs.reserve(smartPtrs.size());
    for (const auto& ptr : smartPtrs) {
        rawPtrs.push_back(ptr.get());
    }
    return rawPtrs;
}

template <typename Container>
std::vector<const char*> ToRawCStrings(const Container& strings)
{
    std::vector<const char*> ptrs;
    ptrs.reserve(strings.size());
    for (const auto& str : strings) {
        ptrs.push_back(str.c_str());
    }
    return ptrs;
}

template <typename Container>
std::vector<const wchar_t*> ToRawWideStrings(const Container& strings)
{
    std::vector<const wchar_t*> ptrs;
    ptrs.reserve(strings.size());
    for (const auto& str : strings) {
        ptrs.push_back(str.c_str());
    }
    return ptrs;
}

// Checked ORT API call - returns error message on failure
static libmlvc::expected<void, std::string> CheckOrtStatus(const OrtApi* api, OrtStatus* status)
{
    auto statusPtr = MakeOrtStatus(api, status);
    if (statusPtr) {
        return libmlvc::make_unexpected(std::string(GetOrtErrorMessage(statusPtr.get(), api)));
    }
    return {};
}

}  // namespace libmlvc_winml
