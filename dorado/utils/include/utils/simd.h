#pragma once

// TSan's init breaks the call to __cpu_indicator_init (which determines which implementation to take)
// __clang__ doesn't support f16c in target multiversioning.
#if defined(_MSC_VER) || (defined(__GNUC__) && defined(__x86_64__) && !defined(__clang__) && \
                          !defined(__SANITIZE_THREAD__))
#define ENABLE_AVX2_IMPL 1
#include <immintrin.h>
#else
#define ENABLE_AVX2_IMPL 0
#endif

#if defined(__arm64__) || defined(__aarch64__)
#define ENABLE_NEON_IMPL 1
#include <arm_neon.h>
#else
#define ENABLE_NEON_IMPL 0
#endif

#include <cstddef>

namespace dorado::utils::simd {

#if ENABLE_NEON_IMPL

// Neon registers have 4 floats.
static constexpr size_t kFloatsPerRegister = 4;

using FloatRegister = float32x4_t;
using HalfRegister = float16x4_t;
using Int16Register = int16x4_t;

#define simd_load_f32(ptr) vld1q_f32(ptr)
#define simd_load1_f32(ptr) vdupq_n_f32(*(ptr))
#define simd_convert_f32_f16(reg) vcvt_f16_f32(reg)
#define simd_store_f32(ptr, reg) vst1q_f32(ptr, reg)
#define simd_store1_f32(ptr, reg) *(ptr) = vgetq_lane_f32(reg, 0)

#define simd_load_f16(ptr) vld1_f16(reinterpret_cast<float16_t const *>(ptr))
#define simd_load1_f16(ptr) vdup_n_f16(*(ptr))
#define simd_convert_f16_f32(reg) vcvt_f32_f16(reg)
#define simd_add_f16(regA, regB) vadd_f16(regA, regB)
#define simd_store_f16(ptr, reg) vst1_f16(reinterpret_cast<float16_t *>(ptr), reg)
#define simd_store1_f16(ptr, reg) *(ptr) = vget_lane_f16(reg, 0)

#define simd_load_i16(ptr) vreinterpret_s16_f16(simd_load_f16(ptr))
#define simd_convert_i16_f32(reg) vcvtq_f32_s32(vmovl_s16(reg))

#elif ENABLE_AVX2_IMPL

// AVX registers have 8 floats.
static constexpr size_t kFloatsPerRegister = 8;

// Matches torch behaviour.
static constexpr int kRoundNearestEven = 0;

using FloatRegister = __m256;
using HalfRegister = __m128i;
using Int16Register = __m128i;

#define simd_load_f32(ptr) _mm256_loadu_ps(ptr)
#define simd_load1_f32(ptr) _mm256_broadcast_ss(ptr)
#define simd_convert_f32_f16(reg) _mm256_cvtps_ph(reg, dorado::utils::simd::kRoundNearestEven)

#define simd_load_f16(ptr) _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr))
#define simd_store_f16(ptr, reg) _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), reg)
#define simd_store1_f16(ptr, reg) \
    *(ptr) = c10::Half(_mm_extract_epi16(reg, 0), c10::Half::from_bits())

#define simd_load_i16(ptr) simd_load_f16(ptr)
#define simd_convert_i16_f32(reg) _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(reg))

#endif

// Helpers to emulate IFUNC on MSVC.
//
// Usage:
//   namespace {
//     IFUNC_DEFINE_DEFAULT(int, add_impl)(int x, int y) { return x + y; } // default impl
//     IFUNC_DEFINE_OPTIMISED(int, add_impl, AVX2)(int x, int y) { return x + y; } // AVX2 impl
//     IFUNC_DECLARE(add_impl, AVX2); // declare an IFUNC dispatcher (no-op on non-MSVC)
//   }
//   int add(int x, int y) { return IFUNC_GET(add_impl)(x, y); } // dispatch to the best IFUNC
//
#ifdef __GNUC__
#define IFUNC_DEFINE_DEFAULT(ret, name) [[maybe_unused]] IFUNC_EXT_DEFAULT ret name
#define IFUNC_DEFINE_OPTIMISED(ret, name, ext) [[maybe_unused]] IFUNC_EXT_##ext ret name
#define IFUNC_DECLARE(name, ext) static_assert(true)
#define IFUNC_GET(name) name
#else
#define IFUNC_DEFINE_DEFAULT(ret, name) ret name##_ifunc_default
#define IFUNC_DEFINE_OPTIMISED(ret, name, ext) ret name##_ifunc_##ext
#define IFUNC_DECLARE(name, ext) \
    const auto name##_ifunc =    \
            dorado::utils::simd::has_##ext ? name##_ifunc_##ext : name##_ifunc_default
#define IFUNC_GET(name) name##_ifunc
#endif

// Supported IFUNC extensions.
#if ENABLE_AVX2_IMPL
#define IFUNC_EXT_DEFAULT [[gnu::target("default")]]

#define IFUNC_EXT_AVX2 [[gnu::target("avx2")]]
inline bool has_AVX2() { return false; }  // TODO

#define IFUNC_EXT_AVX2_F16C [[gnu::target("avx2,f16c")]]
inline bool has_AVX2_F16C() { return false; }  // TODO

#else
// No multiversioning.
#define IFUNC_EXT_DEFAULT
#endif

}  // namespace dorado::utils::simd
