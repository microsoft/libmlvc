// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <cstdio>
#include <cstdlib>

#if defined(__clang__) || defined(__GNUC__)
    #define MLVC_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define MLVC_FORCE_INLINE __forceinline
#else
    #define MLVC_FORCE_INLINE inline
    #error Unsupported compiler for MLVC
#endif

#if defined(_WIN32)
    #define MLVC_PLATFORM_WINCLASSIC
#elif defined(__APPLE__)
    #include <AvailabilityMacros.h>
    #include <TargetConditionals.h>

    #if TARGET_OS_IPHONE
        #define MLVC_PLATFORM_IOS
    #elif TARGET_OS_OSX
        #define MLVC_PLATFORM_MACOSX
    #else
        #error Unsupported Apple platform for MLVC
    #endif

    #define MLVC_PLATFORM_APPLE
#else
    #error Unsupported platform for MLVC
#endif

#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
    #define MLVC_ARCH_X86_64
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define MLVC_ARCH_ARM64
#else
    #error Unsupported architecture for MLVC
#endif

#if defined(MLVC_PLATFORM_APPLE) && !defined(MLVC_ARCH_ARM64)
    #error "MLVC supports ARM64 only on macOS/iOS"
#endif

#ifndef NDEBUG
    #define MLVC_ASSERT(e)                                                                        \
        if (!(e)) {                                                                               \
            std::fprintf(stderr, "Assertion failed in %s line %i: %s\n", __FILE__, __LINE__, #e); \
            std::fflush(stderr);                                                                  \
            std::abort();                                                                         \
        }
#else
    #define MLVC_ASSERT(e) ((void)0)
#endif
