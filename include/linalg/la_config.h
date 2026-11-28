/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_config.h
 * Purpose: Compiler, architecture, SIMD feature detection, portability macros,
 *          alignment qualifiers, and numerical constants.
 */

#ifndef LA_CONFIG_H
#define LA_CONFIG_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ============================================================================
 * Library Version
 * ============================================================================ */

#define LA_VERSION_MAJOR 0
#define LA_VERSION_MINOR 1
#define LA_VERSION_PATCH 0

/* ============================================================================
 * Compiler Detection
 * ============================================================================ */

#if defined(__clang__)
    #define LA_COMPILER_CLANG 1
#elif defined(__GNUC__)
    #define LA_COMPILER_GCC 1
#elif defined(_MSC_VER)
    #define LA_COMPILER_MSVC 1
#else
    #define LA_COMPILER_UNKNOWN 1
#endif

/* ============================================================================
 * SIMD Backend Detection
 * Define LA_FORCE_SCALAR=1 before including to force the pure C99 scalar path.
 * ============================================================================ */

#if defined(LA_FORCE_SCALAR) && (LA_FORCE_SCALAR != 0)
    #define LA_SIMD_SCALAR 1
    #define LA_SIMD_NAME "Scalar (forced)"
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    #define LA_SIMD_NEON 1
    #define LA_SIMD_NAME "ARM NEON"
    #if defined(__ARM_FEATURE_FMA) || defined(__aarch64__) || defined(_M_ARM64)
        #define LA_SIMD_HAS_FMA 1
    #endif
#elif defined(__AVX2__)
    #define LA_SIMD_AVX2 1
    #define LA_SIMD_SSE41 1
    #define LA_SIMD_SSE2 1
    #if defined(__FMA__)
        #define LA_SIMD_HAS_FMA 1
        #define LA_SIMD_NAME "x86 AVX2+FMA"
    #else
        #define LA_SIMD_NAME "x86 AVX2"
    #endif
#elif defined(__SSE4_1__)
    #define LA_SIMD_SSE41 1
    #define LA_SIMD_SSE2 1
    #define LA_SIMD_NAME "x86 SSE4.1"
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #define LA_SIMD_SSE2 1
    #define LA_SIMD_NAME "x86 SSE2"
#elif defined(__wasm_simd128__)
    #define LA_SIMD_WASM 1
    #define LA_SIMD_NAME "WASM SIMD128"
#else
    #define LA_SIMD_SCALAR 1
    #define LA_SIMD_NAME "Scalar"
#endif

#ifndef LA_SIMD_HAS_FMA
    #define LA_SIMD_HAS_FMA 0
#endif

/* ============================================================================
 * Linkage, Inlining, Alignment & Optimization Qualifiers
 * ============================================================================ */

#if defined(LA_NO_FORCE_INLINE)
    #define LA_INLINE static inline
#elif defined(LA_COMPILER_CLANG) || defined(LA_COMPILER_GCC)
    #define LA_INLINE static inline __attribute__((always_inline))
#elif defined(LA_COMPILER_MSVC)
    #define LA_INLINE static __forceinline
#else
    #define LA_INLINE static inline
#endif

/*
 * LA_API controls linkage for heavier batch kernels and iterative solvers:
 *   - Default: header-only (static inline)
 *   - Define LA_SEPARATE_IMPL to emit declarations only, and define
 *     LINALG_IMPLEMENTATION in exactly one translation unit to emit definitions.
 */
#if defined(LA_SEPARATE_IMPL)
    #ifdef __cplusplus
        #define LA_API extern "C"
    #else
        #define LA_API extern
    #endif
    #if defined(LINALG_IMPLEMENTATION)
        #define LA_IMPL_ENABLED 1
    #else
        #define LA_IMPL_ENABLED 0
    #endif
#else
    #define LA_API static inline
    #define LA_IMPL_ENABLED 1
#endif

/* Restrict pointer qualifier for C99 and C++17 */
#if defined(__cplusplus)
    #if defined(LA_COMPILER_CLANG) || defined(LA_COMPILER_GCC)
        #define LA_RESTRICT __restrict__
    #elif defined(LA_COMPILER_MSVC)
        #define LA_RESTRICT __restrict
    #else
        #define LA_RESTRICT
    #endif
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
    #define LA_RESTRICT restrict
#elif defined(LA_COMPILER_MSVC)
    #define LA_RESTRICT __restrict
#else
    #define LA_RESTRICT
#endif

/* Alignment specifier */
#if defined(__cplusplus) && (__cplusplus >= 201103L)
    #define LA_ALIGN(n) alignas(n)
    #define LA_ALIGNOF(T) alignof(T)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    #define LA_ALIGN(n) _Alignas(n)
    #define LA_ALIGNOF(T) _Alignof(T)
#elif defined(LA_COMPILER_CLANG) || defined(LA_COMPILER_GCC)
    #define LA_ALIGN(n) __attribute__((aligned(n)))
    #define LA_ALIGNOF(T) __alignof__(T)
#elif defined(LA_COMPILER_MSVC)
    #define LA_ALIGN(n) __declspec(align(n))
    #define LA_ALIGNOF(T) __alignof(T)
#else
    #define LA_ALIGN(n)
    #define LA_ALIGNOF(T) (sizeof(void *))
#endif

/* Branch prediction hints */
#if defined(LA_COMPILER_CLANG) || defined(LA_COMPILER_GCC)
    #define LA_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define LA_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define LA_UNUSED_ATTR __attribute__((unused))
#else
    #define LA_LIKELY(x)   (x)
    #define LA_UNLIKELY(x) (x)
    #define LA_UNUSED_ATTR
#endif

#define LA_UNUSED(x) ((void)(x))

/* Static assertion portable across C99, C11, and C++11+ */
#define LA_CONCAT_IMPL(a, b) a##b
#define LA_CONCAT(a, b) LA_CONCAT_IMPL(a, b)

#if defined(__cplusplus) && (__cplusplus >= 201103L)
    #define LA_STATIC_ASSERT(cond, name) static_assert((cond), #name)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    #define LA_STATIC_ASSERT(cond, name) _Static_assert((cond), #name)
#else
    #define LA_STATIC_ASSERT(cond, name) \
        typedef char LA_CONCAT(la_static_assert_, name)[(cond) ? 1 : -1] LA_UNUSED_ATTR
#endif

/* C++ extern "C" guard helpers */
#ifdef __cplusplus
    #define LA_EXTERN_C_BEGIN extern "C" {
    #define LA_EXTERN_C_END   }
#else
    #define LA_EXTERN_C_BEGIN
    #define LA_EXTERN_C_END
#endif

/* ============================================================================
 * Numerical Constants
 * ============================================================================ */

#define LA_PI          3.14159265358979323846f
#define LA_TAU         6.28318530717958647692f
#define LA_HALF_PI     1.57079632679489661923f
#define LA_QUARTER_PI  0.78539816339744830962f
#define LA_INV_PI      0.31830988618379067154f
#define LA_INV_TAU     0.15915494309189533577f
#define LA_DEG2RAD     0.01745329251994329577f
#define LA_RAD2DEG     57.2957795130823208768f
#define LA_SQRT2       1.41421356237309504880f
#define LA_INV_SQRT2   0.70710678118654752440f
#define LA_SQRT3       1.73205080756887729353f

#ifndef LA_EPSILON
    #define LA_EPSILON    1.0e-6f
#endif

#ifndef LA_EPSILON_SQ
    #define LA_EPSILON_SQ 1.0e-12f
#endif

#endif /* LA_CONFIG_H */
