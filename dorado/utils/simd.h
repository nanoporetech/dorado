#pragma once

// TSan's init breaks the call to __cpu_indicator_init (which determines which implementation to take)
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__APPLE__) && !defined(__SANITIZE_THREAD__)
#define ENABLE_AVX2_IMPL 1
#else
#define ENABLE_AVX2_IMPL 0
#endif

#if ENABLE_AVX2_IMPL
#include <immintrin.h>
#endif