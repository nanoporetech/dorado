#pragma once

// TSan's init breaks the call to __cpu_indicator_init (which determines which implementation to take)
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__clang__) && !defined(__SANITIZE_THREAD__)
#define ENABLE_AVX2_IMPL 1
#include <immintrin.h>
#else
#define ENABLE_AVX2_IMPL 0
#endif

#if defined(__APPLE__) && defined(__arm64__)
#define ENABLE_NEON_IMPL 1
#include "arm_neon.h"
#else
#define ENABLE_NEON_IMPL 0
#endif
