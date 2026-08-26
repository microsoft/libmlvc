// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef LIBMLVC_EXPORT_HPP
#define LIBMLVC_EXPORT_HPP

#if defined(LIBMLVC_STATIC_DEFINE) || !defined(libmlvc_EXPORTS)
    #define LIBMLVC_EXPORT
    #define LIBMLVC_NO_EXPORT
#else
    #if defined(_WIN32)
        #define LIBMLVC_EXPORT __declspec(dllexport)
        #define LIBMLVC_NO_EXPORT
    #else
        #define LIBMLVC_EXPORT __attribute__((visibility("default")))
        #define LIBMLVC_NO_EXPORT __attribute__((visibility("hidden")))
    #endif
#endif

#ifndef LIBMLVC_DEPRECATED
    #if defined(_WIN32)
        #define LIBMLVC_DEPRECATED __declspec(deprecated)
    #else
        #define LIBMLVC_DEPRECATED __attribute__((__deprecated__))
    #endif
#endif

#ifndef LIBMLVC_DEPRECATED_EXPORT
    #define LIBMLVC_DEPRECATED_EXPORT LIBMLVC_EXPORT LIBMLVC_DEPRECATED
#endif

#ifndef LIBMLVC_DEPRECATED_NO_EXPORT
    #define LIBMLVC_DEPRECATED_NO_EXPORT LIBMLVC_NO_EXPORT LIBMLVC_DEPRECATED
#endif

#endif  // LIBMLVC_EXPORT_HPP
