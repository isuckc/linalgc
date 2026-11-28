/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_simd.h
 * Purpose: Portable 4-wide (and AVX2 8-wide) SIMD vector abstraction across
 *          ARM64 NEON, x86-64 SSE2/SSE4.1/AVX2+FMA, WASM SIMD128, and C99 Scalar.
 *          All struct load/store helpers avoid strict-aliasing UB.
 */

#ifndef LA_SIMD_H
#define LA_SIMD_H

#include "la_types.h"
#include "la_scalar.h"

#if defined(LA_SIMD_NEON)
    #include <arm_neon.h>
#elif defined(LA_SIMD_AVX2)
    #include <immintrin.h>
#elif defined(LA_SIMD_SSE41)
    #include <smmintrin.h>
#elif defined(LA_SIMD_SSE2)
    #include <emmintrin.h>
#elif defined(LA_SIMD_WASM)
    #include <wasm_simd128.h>
#endif

LA_EXTERN_C_BEGIN

/* ============================================================================
 * Backend 1: ARM NEON (AArch64 & ARMv7)
 * ============================================================================ */
#if defined(LA_SIMD_NEON)

typedef float32x4_t la_f32x4;
typedef uint32x4_t  la_u32x4;

LA_INLINE la_f32x4 la_f32x4_zero(void) {
    return vdupq_n_f32(0.0f);
}

LA_INLINE la_f32x4 la_f32x4_splat(float s) {
    return vdupq_n_f32(s);
}

LA_INLINE la_f32x4 la_f32x4_set(float x, float y, float z, float w) {
    float tmp[4];
    tmp[0] = x;
    tmp[1] = y;
    tmp[2] = z;
    tmp[3] = w;
    return vld1q_f32(tmp);
}

LA_INLINE la_f32x4 la_f32x4_load(const float *p) {
    return vld1q_f32(p);
}

LA_INLINE la_f32x4 la_f32x4_loadu(const float *p) {
    return vld1q_f32(p);
}

LA_INLINE void la_f32x4_store(float *p, la_f32x4 v) {
    vst1q_f32(p, v);
}

LA_INLINE void la_f32x4_storeu(float *p, la_f32x4 v) {
    vst1q_f32(p, v);
}

LA_INLINE float la_f32x4_get_x(la_f32x4 v) { return vgetq_lane_f32(v, 0); }
LA_INLINE float la_f32x4_get_y(la_f32x4 v) { return vgetq_lane_f32(v, 1); }
LA_INLINE float la_f32x4_get_z(la_f32x4 v) { return vgetq_lane_f32(v, 2); }
LA_INLINE float la_f32x4_get_w(la_f32x4 v) { return vgetq_lane_f32(v, 3); }

#if defined(__aarch64__) || defined(_M_ARM64)
LA_INLINE la_f32x4 la_f32x4_splat_x(la_f32x4 v) { return vdupq_laneq_f32(v, 0); }
LA_INLINE la_f32x4 la_f32x4_splat_y(la_f32x4 v) { return vdupq_laneq_f32(v, 1); }
LA_INLINE la_f32x4 la_f32x4_splat_z(la_f32x4 v) { return vdupq_laneq_f32(v, 2); }
LA_INLINE la_f32x4 la_f32x4_splat_w(la_f32x4 v) { return vdupq_laneq_f32(v, 3); }
#else
LA_INLINE la_f32x4 la_f32x4_splat_x(la_f32x4 v) { return vdupq_lane_f32(vget_low_f32(v), 0); }
LA_INLINE la_f32x4 la_f32x4_splat_y(la_f32x4 v) { return vdupq_lane_f32(vget_low_f32(v), 1); }
LA_INLINE la_f32x4 la_f32x4_splat_z(la_f32x4 v) { return vdupq_lane_f32(vget_high_f32(v), 0); }
LA_INLINE la_f32x4 la_f32x4_splat_w(la_f32x4 v) { return vdupq_lane_f32(vget_high_f32(v), 1); }
#endif

LA_INLINE la_f32x4 la_f32x4_add(la_f32x4 a, la_f32x4 b) { return vaddq_f32(a, b); }
LA_INLINE la_f32x4 la_f32x4_sub(la_f32x4 a, la_f32x4 b) { return vsubq_f32(a, b); }
LA_INLINE la_f32x4 la_f32x4_mul(la_f32x4 a, la_f32x4 b) { return vmulq_f32(a, b); }

LA_INLINE la_f32x4 la_f32x4_rcp_fast(la_f32x4 v) {
    la_f32x4 est = vrecpeq_f32(v);
    est = vmulq_f32(est, vrecpsq_f32(v, est));
    est = vmulq_f32(est, vrecpsq_f32(v, est));
    return est;
}

LA_INLINE la_f32x4 la_f32x4_div(la_f32x4 a, la_f32x4 b) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vdivq_f32(a, b);
#else
    return vmulq_f32(a, la_f32x4_rcp_fast(b));
#endif
}

LA_INLINE la_f32x4 la_f32x4_neg(la_f32x4 v) { return vnegq_f32(v); }
LA_INLINE la_f32x4 la_f32x4_abs(la_f32x4 v) { return vabsq_f32(v); }

/* Computes a * b + c */
LA_INLINE la_f32x4 la_f32x4_fma(la_f32x4 a, la_f32x4 b, la_f32x4 c) {
#if LA_SIMD_HAS_FMA
    return vfmaq_f32(c, a, b);
#else
    return vmlaq_f32(c, a, b);
#endif
}

/* Computes c - a * b */
LA_INLINE la_f32x4 la_f32x4_fnma(la_f32x4 a, la_f32x4 b, la_f32x4 c) {
#if LA_SIMD_HAS_FMA
    return vfmsq_f32(c, a, b);
#else
    return vmlsq_f32(c, a, b);
#endif
}

LA_INLINE la_f32x4 la_f32x4_rsqrt_fast(la_f32x4 v) {
    la_f32x4 est = vrsqrteq_f32(v);
    est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(v, est), est));
    est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(v, est), est));
    return est;
}

LA_INLINE la_f32x4 la_f32x4_sqrt(la_f32x4 v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vsqrtq_f32(v);
#else
    la_u32x4 zero_mask = vceqq_f32(v, vdupq_n_f32(0.0f));
    la_f32x4 root = vmulq_f32(v, la_f32x4_rsqrt_fast(v));
    return vbslq_f32(zero_mask, vdupq_n_f32(0.0f), root);
#endif
}

LA_INLINE la_f32x4 la_f32x4_min(la_f32x4 a, la_f32x4 b) { return vminq_f32(a, b); }
LA_INLINE la_f32x4 la_f32x4_max(la_f32x4 a, la_f32x4 b) { return vmaxq_f32(a, b); }

LA_INLINE la_u32x4 la_f32x4_cmpeq(la_f32x4 a, la_f32x4 b) { return vceqq_f32(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmplt(la_f32x4 a, la_f32x4 b) { return vcltq_f32(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmple(la_f32x4 a, la_f32x4 b) { return vcleq_f32(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmpgt(la_f32x4 a, la_f32x4 b) { return vcgtq_f32(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmpge(la_f32x4 a, la_f32x4 b) { return vcgeq_f32(a, b); }

LA_INLINE la_u32x4 la_u32x4_and(la_u32x4 a, la_u32x4 b) { return vandq_u32(a, b); }
LA_INLINE la_u32x4 la_u32x4_or(la_u32x4 a, la_u32x4 b)  { return vorrq_u32(a, b); }
LA_INLINE la_u32x4 la_u32x4_not(la_u32x4 a)             { return vmvnq_u32(a); }

LA_INLINE la_f32x4 la_f32x4_select(la_u32x4 mask, la_f32x4 if_true, la_f32x4 if_false) {
    return vbslq_f32(mask, if_true, if_false);
}

LA_INLINE uint32_t la_u32x4_movemask(la_u32x4 m) {
    const uint32_t shifts[4] = { 1u, 2u, 4u, 8u };
    uint32x4_t bits = vandq_u32(m, vld1q_u32(shifts));
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_u32(bits);
#else
    uint32x2_t lo = vget_low_u32(bits);
    uint32x2_t hi = vget_high_u32(bits);
    uint32x2_t sum = vorr_u32(lo, hi);
    return vget_lane_u32(sum, 0) | vget_lane_u32(sum, 1);
#endif
}

LA_INLINE float la_f32x4_hadd(la_f32x4 v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_f32(v);
#else
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
#endif
}

LA_INLINE float la_f32x4_dot4(la_f32x4 a, la_f32x4 b) {
    return la_f32x4_hadd(vmulq_f32(a, b));
}

LA_INLINE float la_f32x4_dot3(la_f32x4 a, la_f32x4 b) {
    la_f32x4 prod = vmulq_f32(a, b);
    prod = vsetq_lane_f32(0.0f, prod, 3);
    return la_f32x4_hadd(prod);
}

LA_INLINE void la_f32x4_transpose4(la_f32x4 *c0, la_f32x4 *c1, la_f32x4 *c2, la_f32x4 *c3) {
#if defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t t0 = vtrn1q_f32(*c0, *c1);
    float32x4_t t1 = vtrn2q_f32(*c0, *c1);
    float32x4_t t2 = vtrn1q_f32(*c2, *c3);
    float32x4_t t3 = vtrn2q_f32(*c2, *c3);
    *c0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
    *c1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
    *c2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
    *c3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
#else
    float32x4x2_t t01 = vtrnq_f32(*c0, *c1);
    float32x4x2_t t23 = vtrnq_f32(*c2, *c3);
    *c0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    *c1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    *c2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    *c3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
#endif
}

/* Hardware 3-channel deinterleave/interleave (12 floats = 4 AoS vec3 -> 3 SoA f32x4) */
LA_INLINE void la_f32x4_load3_deinterleave(const float *p, la_f32x4 *out_x, la_f32x4 *out_y, la_f32x4 *out_z) {
    float32x4x3_t v = vld3q_f32(p);
    *out_x = v.val[0];
    *out_y = v.val[1];
    *out_z = v.val[2];
}

LA_INLINE void la_f32x4_store3_interleave(float *p, la_f32x4 x, la_f32x4 y, la_f32x4 z) {
    float32x4x3_t v;
    v.val[0] = x;
    v.val[1] = y;
    v.val[2] = z;
    vst3q_f32(p, v);
}

/* ============================================================================
 * Backend 2: x86-64 SSE2 / SSE4.1 / AVX2+FMA
 * ============================================================================ */
#elif defined(LA_SIMD_SSE2)

typedef __m128 la_f32x4;
typedef __m128 la_u32x4;

LA_INLINE la_f32x4 la_f32x4_zero(void) {
    return _mm_setzero_ps();
}

LA_INLINE la_f32x4 la_f32x4_splat(float s) {
    return _mm_set1_ps(s);
}

LA_INLINE la_f32x4 la_f32x4_set(float x, float y, float z, float w) {
    return _mm_setr_ps(x, y, z, w);
}

LA_INLINE la_f32x4 la_f32x4_load(const float *p) {
    return _mm_load_ps(p);
}

LA_INLINE la_f32x4 la_f32x4_loadu(const float *p) {
    return _mm_loadu_ps(p);
}

LA_INLINE void la_f32x4_store(float *p, la_f32x4 v) {
    _mm_store_ps(p, v);
}

LA_INLINE void la_f32x4_storeu(float *p, la_f32x4 v) {
    _mm_storeu_ps(p, v);
}

LA_INLINE float la_f32x4_get_x(la_f32x4 v) { return _mm_cvtss_f32(v); }
LA_INLINE float la_f32x4_get_y(la_f32x4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1))); }
LA_INLINE float la_f32x4_get_z(la_f32x4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2))); }
LA_INLINE float la_f32x4_get_w(la_f32x4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3))); }

LA_INLINE la_f32x4 la_f32x4_splat_x(la_f32x4 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)); }
LA_INLINE la_f32x4 la_f32x4_splat_y(la_f32x4 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)); }
LA_INLINE la_f32x4 la_f32x4_splat_z(la_f32x4 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)); }
LA_INLINE la_f32x4 la_f32x4_splat_w(la_f32x4 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)); }

LA_INLINE la_f32x4 la_f32x4_add(la_f32x4 a, la_f32x4 b) { return _mm_add_ps(a, b); }
LA_INLINE la_f32x4 la_f32x4_sub(la_f32x4 a, la_f32x4 b) { return _mm_sub_ps(a, b); }
LA_INLINE la_f32x4 la_f32x4_mul(la_f32x4 a, la_f32x4 b) { return _mm_mul_ps(a, b); }
LA_INLINE la_f32x4 la_f32x4_div(la_f32x4 a, la_f32x4 b) { return _mm_div_ps(a, b); }

LA_INLINE la_f32x4 la_f32x4_neg(la_f32x4 v) {
    return _mm_xor_ps(v, _mm_set1_ps(-0.0f));
}

LA_INLINE la_f32x4 la_f32x4_abs(la_f32x4 v) {
    return _mm_andnot_ps(_mm_set1_ps(-0.0f), v);
}

LA_INLINE la_f32x4 la_f32x4_fma(la_f32x4 a, la_f32x4 b, la_f32x4 c) {
#if LA_SIMD_HAS_FMA
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

LA_INLINE la_f32x4 la_f32x4_fnma(la_f32x4 a, la_f32x4 b, la_f32x4 c) {
#if LA_SIMD_HAS_FMA
    return _mm_fnmadd_ps(a, b, c);
#else
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
#endif
}

LA_INLINE la_f32x4 la_f32x4_rcp_fast(la_f32x4 v) {
    la_f32x4 est = _mm_rcp_ps(v);
    la_f32x4 two = _mm_set1_ps(2.0f);
    est = _mm_mul_ps(est, _mm_sub_ps(two, _mm_mul_ps(v, est)));
    return est;
}

LA_INLINE la_f32x4 la_f32x4_rsqrt_fast(la_f32x4 v) {
    la_f32x4 est = _mm_rsqrt_ps(v);
    la_f32x4 half = _mm_set1_ps(0.5f);
    la_f32x4 three = _mm_set1_ps(3.0f);
    est = _mm_mul_ps(_mm_mul_ps(half, est), _mm_sub_ps(three, _mm_mul_ps(_mm_mul_ps(v, est), est)));
    return est;
}

LA_INLINE la_f32x4 la_f32x4_sqrt(la_f32x4 v) { return _mm_sqrt_ps(v); }
LA_INLINE la_f32x4 la_f32x4_min(la_f32x4 a, la_f32x4 b) { return _mm_min_ps(a, b); }
LA_INLINE la_f32x4 la_f32x4_max(la_f32x4 a, la_f32x4 b) { return _mm_max_ps(a, b); }

LA_INLINE la_u32x4 la_f32x4_cmpeq(la_f32x4 a, la_f32x4 b) { return _mm_cmpeq_ps(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmplt(la_f32x4 a, la_f32x4 b) { return _mm_cmplt_ps(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmple(la_f32x4 a, la_f32x4 b) { return _mm_cmple_ps(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmpgt(la_f32x4 a, la_f32x4 b) { return _mm_cmpgt_ps(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmpge(la_f32x4 a, la_f32x4 b) { return _mm_cmpge_ps(a, b); }

LA_INLINE la_u32x4 la_u32x4_and(la_u32x4 a, la_u32x4 b) { return _mm_and_ps(a, b); }
LA_INLINE la_u32x4 la_u32x4_or(la_u32x4 a, la_u32x4 b)  { return _mm_or_ps(a, b); }
LA_INLINE la_u32x4 la_u32x4_not(la_u32x4 a) {
    return _mm_xor_ps(a, _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()));
}

LA_INLINE la_f32x4 la_f32x4_select(la_u32x4 mask, la_f32x4 if_true, la_f32x4 if_false) {
#if defined(LA_SIMD_SSE41)
    return _mm_blendv_ps(if_false, if_true, mask);
#else
    return _mm_or_ps(_mm_and_ps(mask, if_true), _mm_andnot_ps(mask, if_false));
#endif
}

LA_INLINE uint32_t la_u32x4_movemask(la_u32x4 m) {
    return (uint32_t)_mm_movemask_ps(m);
}

LA_INLINE float la_f32x4_hadd(la_f32x4 v) {
    la_f32x4 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    la_f32x4 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

LA_INLINE float la_f32x4_dot4(la_f32x4 a, la_f32x4 b) {
    return la_f32x4_hadd(_mm_mul_ps(a, b));
}

LA_INLINE float la_f32x4_dot3(la_f32x4 a, la_f32x4 b) {
#if defined(LA_SIMD_SSE41)
    return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x71));
#else
    la_f32x4 p = _mm_mul_ps(a, b);
    return la_f32x4_get_x(p) + la_f32x4_get_y(p) + la_f32x4_get_z(p);
#endif
}

LA_INLINE void la_f32x4_transpose4(la_f32x4 *c0, la_f32x4 *c1, la_f32x4 *c2, la_f32x4 *c3) {
    _MM_TRANSPOSE4_PS(*c0, *c1, *c2, *c3);
}

LA_INLINE void la_f32x4_load3_deinterleave(const float *p, la_f32x4 *out_x, la_f32x4 *out_y, la_f32x4 *out_z) {
    *out_x = la_f32x4_set(p[0], p[3], p[6], p[9]);
    *out_y = la_f32x4_set(p[1], p[4], p[7], p[10]);
    *out_z = la_f32x4_set(p[2], p[5], p[8], p[11]);
}

LA_INLINE void la_f32x4_store3_interleave(float *p, la_f32x4 x, la_f32x4 y, la_f32x4 z) {
    LA_ALIGN(16) float xs[4], ys[4], zs[4];
    la_f32x4_store(xs, x);
    la_f32x4_store(ys, y);
    la_f32x4_store(zs, z);
    p[0] = xs[0]; p[1] = ys[0]; p[2]  = zs[0];
    p[3] = xs[1]; p[4] = ys[1]; p[5]  = zs[1];
    p[6] = xs[2]; p[7] = ys[2]; p[8]  = zs[2];
    p[9] = xs[3]; p[10] = ys[3]; p[11] = zs[3];
}

/* ============================================================================
 * Backend 3: WebAssembly SIMD128
 * ============================================================================ */
#elif defined(LA_SIMD_WASM)

typedef v128_t la_f32x4;
typedef v128_t la_u32x4;

LA_INLINE la_f32x4 la_f32x4_zero(void) { return wasm_f32x4_splat(0.0f); }
LA_INLINE la_f32x4 la_f32x4_splat(float s) { return wasm_f32x4_splat(s); }
LA_INLINE la_f32x4 la_f32x4_set(float x, float y, float z, float w) { return wasm_f32x4_make(x, y, z, w); }
LA_INLINE la_f32x4 la_f32x4_load(const float *p) { return wasm_v128_load(p); }
LA_INLINE la_f32x4 la_f32x4_loadu(const float *p) { return wasm_v128_load(p); }
LA_INLINE void la_f32x4_store(float *p, la_f32x4 v) { wasm_v128_store(p, v); }
LA_INLINE void la_f32x4_storeu(float *p, la_f32x4 v) { wasm_v128_store(p, v); }

LA_INLINE float la_f32x4_get_x(la_f32x4 v) { return wasm_f32x4_extract_lane(v, 0); }
LA_INLINE float la_f32x4_get_y(la_f32x4 v) { return wasm_f32x4_extract_lane(v, 1); }
LA_INLINE float la_f32x4_get_z(la_f32x4 v) { return wasm_f32x4_extract_lane(v, 2); }
LA_INLINE float la_f32x4_get_w(la_f32x4 v) { return wasm_f32x4_extract_lane(v, 3); }

LA_INLINE la_f32x4 la_f32x4_splat_x(la_f32x4 v) { return wasm_i32x4_shuffle(v, v, 0, 0, 0, 0); }
LA_INLINE la_f32x4 la_f32x4_splat_y(la_f32x4 v) { return wasm_i32x4_shuffle(v, v, 1, 1, 1, 1); }
LA_INLINE la_f32x4 la_f32x4_splat_z(la_f32x4 v) { return wasm_i32x4_shuffle(v, v, 2, 2, 2, 2); }
LA_INLINE la_f32x4 la_f32x4_splat_w(la_f32x4 v) { return wasm_i32x4_shuffle(v, v, 3, 3, 3, 3); }

LA_INLINE la_f32x4 la_f32x4_add(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_add(a, b); }
LA_INLINE la_f32x4 la_f32x4_sub(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_sub(a, b); }
LA_INLINE la_f32x4 la_f32x4_mul(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_mul(a, b); }
LA_INLINE la_f32x4 la_f32x4_div(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_div(a, b); }
LA_INLINE la_f32x4 la_f32x4_neg(la_f32x4 v) { return wasm_f32x4_neg(v); }
LA_INLINE la_f32x4 la_f32x4_abs(la_f32x4 v) { return wasm_f32x4_abs(v); }
LA_INLINE la_f32x4 la_f32x4_fma(la_f32x4 a, la_f32x4 b, la_f32x4 c) { return wasm_f32x4_add(wasm_f32x4_mul(a, b), c); }
LA_INLINE la_f32x4 la_f32x4_fnma(la_f32x4 a, la_f32x4 b, la_f32x4 c) { return wasm_f32x4_sub(c, wasm_f32x4_mul(a, b)); }
LA_INLINE la_f32x4 la_f32x4_rcp_fast(la_f32x4 v) { return wasm_f32x4_div(wasm_f32x4_splat(1.0f), v); }
LA_INLINE la_f32x4 la_f32x4_sqrt(la_f32x4 v) { return wasm_f32x4_sqrt(v); }
LA_INLINE la_f32x4 la_f32x4_rsqrt_fast(la_f32x4 v) { return wasm_f32x4_div(wasm_f32x4_splat(1.0f), wasm_f32x4_sqrt(v)); }
LA_INLINE la_f32x4 la_f32x4_min(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_min(a, b); }
LA_INLINE la_f32x4 la_f32x4_max(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_max(a, b); }

LA_INLINE la_u32x4 la_f32x4_cmpeq(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_eq(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmplt(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_lt(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmple(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_le(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmpgt(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_gt(a, b); }
LA_INLINE la_u32x4 la_f32x4_cmpge(la_f32x4 a, la_f32x4 b) { return wasm_f32x4_ge(a, b); }
LA_INLINE la_u32x4 la_u32x4_and(la_u32x4 a, la_u32x4 b) { return wasm_v128_and(a, b); }
LA_INLINE la_u32x4 la_u32x4_or(la_u32x4 a, la_u32x4 b)  { return wasm_v128_or(a, b); }
LA_INLINE la_u32x4 la_u32x4_not(la_u32x4 a)             { return wasm_v128_not(a); }
LA_INLINE la_f32x4 la_f32x4_select(la_u32x4 mask, la_f32x4 t, la_f32x4 f) { return wasm_v128_bitselect(t, f, mask); }
LA_INLINE uint32_t la_u32x4_movemask(la_u32x4 m) { return (uint32_t)wasm_i32x4_bitmask(m); }

LA_INLINE float la_f32x4_hadd(la_f32x4 v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) +
           wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

LA_INLINE float la_f32x4_dot4(la_f32x4 a, la_f32x4 b) { return la_f32x4_hadd(wasm_f32x4_mul(a, b)); }
LA_INLINE float la_f32x4_dot3(la_f32x4 a, la_f32x4 b) {
    la_f32x4 p = wasm_f32x4_mul(a, b);
    return wasm_f32x4_extract_lane(p, 0) + wasm_f32x4_extract_lane(p, 1) + wasm_f32x4_extract_lane(p, 2);
}

LA_INLINE void la_f32x4_transpose4(la_f32x4 *c0, la_f32x4 *c1, la_f32x4 *c2, la_f32x4 *c3) {
    v128_t t0 = wasm_i32x4_shuffle(*c0, *c1, 0, 4, 1, 5);
    v128_t t1 = wasm_i32x4_shuffle(*c0, *c1, 2, 6, 3, 7);
    v128_t t2 = wasm_i32x4_shuffle(*c2, *c3, 0, 4, 1, 5);
    v128_t t3 = wasm_i32x4_shuffle(*c2, *c3, 2, 6, 3, 7);
    *c0 = wasm_i32x4_shuffle(t0, t2, 0, 1, 4, 5);
    *c1 = wasm_i32x4_shuffle(t0, t2, 2, 3, 6, 7);
    *c2 = wasm_i32x4_shuffle(t1, t3, 0, 1, 4, 5);
    *c3 = wasm_i32x4_shuffle(t1, t3, 2, 3, 6, 7);
}

LA_INLINE void la_f32x4_load3_deinterleave(const float *p, la_f32x4 *out_x, la_f32x4 *out_y, la_f32x4 *out_z) {
    *out_x = la_f32x4_set(p[0], p[3], p[6], p[9]);
    *out_y = la_f32x4_set(p[1], p[4], p[7], p[10]);
    *out_z = la_f32x4_set(p[2], p[5], p[8], p[11]);
}

LA_INLINE void la_f32x4_store3_interleave(float *p, la_f32x4 x, la_f32x4 y, la_f32x4 z) {
    LA_ALIGN(16) float xs[4], ys[4], zs[4];
    la_f32x4_store(xs, x); la_f32x4_store(ys, y); la_f32x4_store(zs, z);
    p[0] = xs[0]; p[1] = ys[0]; p[2]  = zs[0];
    p[3] = xs[1]; p[4] = ys[1]; p[5]  = zs[1];
    p[6] = xs[2]; p[7] = ys[2]; p[8]  = zs[2];
    p[9] = xs[3]; p[10] = ys[3]; p[11] = zs[3];
}

/* ============================================================================
 * Backend 4: Pure C99 Scalar Fallback (Portable Reference Implementation)
 * ============================================================================ */
#else

typedef struct LA_ALIGN(16) la_f32x4 {
    float v[4];
} la_f32x4;

typedef struct LA_ALIGN(16) la_u32x4 {
    uint32_t v[4];
} la_u32x4;

LA_INLINE la_f32x4 la_f32x4_zero(void) {
    la_f32x4 r;
    r.v[0] = 0.0f; r.v[1] = 0.0f; r.v[2] = 0.0f; r.v[3] = 0.0f;
    return r;
}

LA_INLINE la_f32x4 la_f32x4_splat(float s) {
    la_f32x4 r;
    r.v[0] = s; r.v[1] = s; r.v[2] = s; r.v[3] = s;
    return r;
}

LA_INLINE la_f32x4 la_f32x4_set(float x, float y, float z, float w) {
    la_f32x4 r;
    r.v[0] = x; r.v[1] = y; r.v[2] = z; r.v[3] = w;
    return r;
}

LA_INLINE la_f32x4 la_f32x4_load(const float *p) {
    la_f32x4 r;
    memcpy(r.v, p, sizeof(r.v));
    return r;
}

LA_INLINE la_f32x4 la_f32x4_loadu(const float *p) {
    la_f32x4 r;
    memcpy(r.v, p, sizeof(r.v));
    return r;
}

LA_INLINE void la_f32x4_store(float *p, la_f32x4 v) {
    memcpy(p, v.v, sizeof(v.v));
}

LA_INLINE void la_f32x4_storeu(float *p, la_f32x4 v) {
    memcpy(p, v.v, sizeof(v.v));
}

LA_INLINE float la_f32x4_get_x(la_f32x4 v) { return v.v[0]; }
LA_INLINE float la_f32x4_get_y(la_f32x4 v) { return v.v[1]; }
LA_INLINE float la_f32x4_get_z(la_f32x4 v) { return v.v[2]; }
LA_INLINE float la_f32x4_get_w(la_f32x4 v) { return v.v[3]; }

LA_INLINE la_f32x4 la_f32x4_splat_x(la_f32x4 v) { return la_f32x4_splat(v.v[0]); }
LA_INLINE la_f32x4 la_f32x4_splat_y(la_f32x4 v) { return la_f32x4_splat(v.v[1]); }
LA_INLINE la_f32x4 la_f32x4_splat_z(la_f32x4 v) { return la_f32x4_splat(v.v[2]); }
LA_INLINE la_f32x4 la_f32x4_splat_w(la_f32x4 v) { return la_f32x4_splat(v.v[3]); }

LA_INLINE la_f32x4 la_f32x4_add(la_f32x4 a, la_f32x4 b) {
    return la_f32x4_set(a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2], a.v[3] + b.v[3]);
}

LA_INLINE la_f32x4 la_f32x4_sub(la_f32x4 a, la_f32x4 b) {
    return la_f32x4_set(a.v[0] - b.v[0], a.v[1] - b.v[1], a.v[2] - b.v[2], a.v[3] - b.v[3]);
}

LA_INLINE la_f32x4 la_f32x4_mul(la_f32x4 a, la_f32x4 b) {
    return la_f32x4_set(a.v[0] * b.v[0], a.v[1] * b.v[1], a.v[2] * b.v[2], a.v[3] * b.v[3]);
}

LA_INLINE la_f32x4 la_f32x4_div(la_f32x4 a, la_f32x4 b) {
    return la_f32x4_set(a.v[0] / b.v[0], a.v[1] / b.v[1], a.v[2] / b.v[2], a.v[3] / b.v[3]);
}

LA_INLINE la_f32x4 la_f32x4_neg(la_f32x4 v) {
    return la_f32x4_set(-v.v[0], -v.v[1], -v.v[2], -v.v[3]);
}

LA_INLINE la_f32x4 la_f32x4_abs(la_f32x4 v) {
    return la_f32x4_set(la_absf(v.v[0]), la_absf(v.v[1]), la_absf(v.v[2]), la_absf(v.v[3]));
}

LA_INLINE la_f32x4 la_f32x4_fma(la_f32x4 a, la_f32x4 b, la_f32x4 c) {
    return la_f32x4_set(
        la_fmaf(a.v[0], b.v[0], c.v[0]),
        la_fmaf(a.v[1], b.v[1], c.v[1]),
        la_fmaf(a.v[2], b.v[2], c.v[2]),
        la_fmaf(a.v[3], b.v[3], c.v[3])
    );
}

LA_INLINE la_f32x4 la_f32x4_fnma(la_f32x4 a, la_f32x4 b, la_f32x4 c) {
    return la_f32x4_set(
        la_fmaf(-a.v[0], b.v[0], c.v[0]),
        la_fmaf(-a.v[1], b.v[1], c.v[1]),
        la_fmaf(-a.v[2], b.v[2], c.v[2]),
        la_fmaf(-a.v[3], b.v[3], c.v[3])
    );
}

LA_INLINE la_f32x4 la_f32x4_rcp_fast(la_f32x4 v) {
    return la_f32x4_set(1.0f / v.v[0], 1.0f / v.v[1], 1.0f / v.v[2], 1.0f / v.v[3]);
}

LA_INLINE la_f32x4 la_f32x4_rsqrt_fast(la_f32x4 v) {
    return la_f32x4_set(
        la_rsqrtf_fast(v.v[0]),
        la_rsqrtf_fast(v.v[1]),
        la_rsqrtf_fast(v.v[2]),
        la_rsqrtf_fast(v.v[3])
    );
}

LA_INLINE la_f32x4 la_f32x4_sqrt(la_f32x4 v) {
    return la_f32x4_set(sqrtf(v.v[0]), sqrtf(v.v[1]), sqrtf(v.v[2]), sqrtf(v.v[3]));
}

LA_INLINE la_f32x4 la_f32x4_min(la_f32x4 a, la_f32x4 b) {
    return la_f32x4_set(la_minf(a.v[0], b.v[0]), la_minf(a.v[1], b.v[1]),
                        la_minf(a.v[2], b.v[2]), la_minf(a.v[3], b.v[3]));
}

LA_INLINE la_f32x4 la_f32x4_max(la_f32x4 a, la_f32x4 b) {
    return la_f32x4_set(la_maxf(a.v[0], b.v[0]), la_maxf(a.v[1], b.v[1]),
                        la_maxf(a.v[2], b.v[2]), la_maxf(a.v[3], b.v[3]));
}

LA_INLINE la_u32x4 la_f32x4_cmpeq(la_f32x4 a, la_f32x4 b) {
    la_u32x4 m;
    m.v[0] = (a.v[0] == b.v[0]) ? 0xFFFFFFFFu : 0u;
    m.v[1] = (a.v[1] == b.v[1]) ? 0xFFFFFFFFu : 0u;
    m.v[2] = (a.v[2] == b.v[2]) ? 0xFFFFFFFFu : 0u;
    m.v[3] = (a.v[3] == b.v[3]) ? 0xFFFFFFFFu : 0u;
    return m;
}

LA_INLINE la_u32x4 la_f32x4_cmplt(la_f32x4 a, la_f32x4 b) {
    la_u32x4 m;
    m.v[0] = (a.v[0] < b.v[0]) ? 0xFFFFFFFFu : 0u;
    m.v[1] = (a.v[1] < b.v[1]) ? 0xFFFFFFFFu : 0u;
    m.v[2] = (a.v[2] < b.v[2]) ? 0xFFFFFFFFu : 0u;
    m.v[3] = (a.v[3] < b.v[3]) ? 0xFFFFFFFFu : 0u;
    return m;
}

LA_INLINE la_u32x4 la_f32x4_cmple(la_f32x4 a, la_f32x4 b) {
    la_u32x4 m;
    m.v[0] = (a.v[0] <= b.v[0]) ? 0xFFFFFFFFu : 0u;
    m.v[1] = (a.v[1] <= b.v[1]) ? 0xFFFFFFFFu : 0u;
    m.v[2] = (a.v[2] <= b.v[2]) ? 0xFFFFFFFFu : 0u;
    m.v[3] = (a.v[3] <= b.v[3]) ? 0xFFFFFFFFu : 0u;
    return m;
}

LA_INLINE la_u32x4 la_f32x4_cmpgt(la_f32x4 a, la_f32x4 b) {
    la_u32x4 m;
    m.v[0] = (a.v[0] > b.v[0]) ? 0xFFFFFFFFu : 0u;
    m.v[1] = (a.v[1] > b.v[1]) ? 0xFFFFFFFFu : 0u;
    m.v[2] = (a.v[2] > b.v[2]) ? 0xFFFFFFFFu : 0u;
    m.v[3] = (a.v[3] > b.v[3]) ? 0xFFFFFFFFu : 0u;
    return m;
}

LA_INLINE la_u32x4 la_f32x4_cmpge(la_f32x4 a, la_f32x4 b) {
    la_u32x4 m;
    m.v[0] = (a.v[0] >= b.v[0]) ? 0xFFFFFFFFu : 0u;
    m.v[1] = (a.v[1] >= b.v[1]) ? 0xFFFFFFFFu : 0u;
    m.v[2] = (a.v[2] >= b.v[2]) ? 0xFFFFFFFFu : 0u;
    m.v[3] = (a.v[3] >= b.v[3]) ? 0xFFFFFFFFu : 0u;
    return m;
}

LA_INLINE la_u32x4 la_u32x4_and(la_u32x4 a, la_u32x4 b) {
    la_u32x4 m;
    m.v[0] = a.v[0] & b.v[0]; m.v[1] = a.v[1] & b.v[1];
    m.v[2] = a.v[2] & b.v[2]; m.v[3] = a.v[3] & b.v[3];
    return m;
}

LA_INLINE la_u32x4 la_u32x4_or(la_u32x4 a, la_u32x4 b) {
    la_u32x4 m;
    m.v[0] = a.v[0] | b.v[0]; m.v[1] = a.v[1] | b.v[1];
    m.v[2] = a.v[2] | b.v[2]; m.v[3] = a.v[3] | b.v[3];
    return m;
}

LA_INLINE la_u32x4 la_u32x4_not(la_u32x4 a) {
    la_u32x4 m;
    m.v[0] = ~a.v[0]; m.v[1] = ~a.v[1];
    m.v[2] = ~a.v[2]; m.v[3] = ~a.v[3];
    return m;
}

LA_INLINE la_f32x4 la_f32x4_select(la_u32x4 mask, la_f32x4 if_true, la_f32x4 if_false) {
    return la_f32x4_set(
        mask.v[0] ? if_true.v[0] : if_false.v[0],
        mask.v[1] ? if_true.v[1] : if_false.v[1],
        mask.v[2] ? if_true.v[2] : if_false.v[2],
        mask.v[3] ? if_true.v[3] : if_false.v[3]
    );
}

LA_INLINE uint32_t la_u32x4_movemask(la_u32x4 m) {
    return ((m.v[0] ? 1u : 0u)) | ((m.v[1] ? 2u : 0u)) |
           ((m.v[2] ? 4u : 0u)) | ((m.v[3] ? 8u : 0u));
}

LA_INLINE float la_f32x4_hadd(la_f32x4 v) {
    return (v.v[0] + v.v[1]) + (v.v[2] + v.v[3]);
}

LA_INLINE float la_f32x4_dot4(la_f32x4 a, la_f32x4 b) {
    return la_fmaf(a.v[0], b.v[0], la_fmaf(a.v[1], b.v[1], la_fmaf(a.v[2], b.v[2], a.v[3] * b.v[3])));
}

LA_INLINE float la_f32x4_dot3(la_f32x4 a, la_f32x4 b) {
    return la_fmaf(a.v[0], b.v[0], la_fmaf(a.v[1], b.v[1], a.v[2] * b.v[2]));
}

LA_INLINE void la_f32x4_transpose4(la_f32x4 *c0, la_f32x4 *c1, la_f32x4 *c2, la_f32x4 *c3) {
    la_f32x4 r0 = la_f32x4_set(c0->v[0], c1->v[0], c2->v[0], c3->v[0]);
    la_f32x4 r1 = la_f32x4_set(c0->v[1], c1->v[1], c2->v[1], c3->v[1]);
    la_f32x4 r2 = la_f32x4_set(c0->v[2], c1->v[2], c2->v[2], c3->v[2]);
    la_f32x4 r3 = la_f32x4_set(c0->v[3], c1->v[3], c2->v[3], c3->v[3]);
    *c0 = r0; *c1 = r1; *c2 = r2; *c3 = r3;
}

LA_INLINE void la_f32x4_load3_deinterleave(const float *p, la_f32x4 *out_x, la_f32x4 *out_y, la_f32x4 *out_z) {
    *out_x = la_f32x4_set(p[0], p[3], p[6], p[9]);
    *out_y = la_f32x4_set(p[1], p[4], p[7], p[10]);
    *out_z = la_f32x4_set(p[2], p[5], p[8], p[11]);
}

LA_INLINE void la_f32x4_store3_interleave(float *p, la_f32x4 x, la_f32x4 y, la_f32x4 z) {
    p[0] = x.v[0]; p[1] = y.v[0]; p[2]  = z.v[0];
    p[3] = x.v[1]; p[4] = y.v[1]; p[5]  = z.v[1];
    p[6] = x.v[2]; p[7] = y.v[2]; p[8]  = z.v[2];
    p[9] = x.v[3]; p[10] = y.v[3]; p[11] = z.v[3];
}

#endif /* SIMD backend selection */

/* ============================================================================
 * Backend-Agnostic Higher-Level SIMD Helpers & Strict-Aliasing-Safe Conversions
 * ============================================================================ */

LA_INLINE la_f32x4 la_f32x4_clamp(la_f32x4 v, la_f32x4 lo, la_f32x4 hi) {
    return la_f32x4_min(la_f32x4_max(v, lo), hi);
}

LA_INLINE la_f32x4 la_f32x4_lerp(la_f32x4 a, la_f32x4 b, la_f32x4 t) {
    return la_f32x4_fma(t, b, la_f32x4_fnma(t, a, a));
}

/* Safe POD struct <-> SIMD register conversions (compiles to 1 load/store, zero UB) */
LA_INLINE la_f32x4 la_vec4_to_simd(la_vec4 v) {
    LA_ALIGN(16) float tmp[4];
    memcpy(tmp, &v, sizeof(tmp));
    return la_f32x4_load(tmp);
}

LA_INLINE la_vec4 la_simd_to_vec4(la_f32x4 s) {
    LA_ALIGN(16) float tmp[4];
    la_vec4 out;
    la_f32x4_store(tmp, s);
    memcpy(&out, tmp, sizeof(out));
    return out;
}

LA_INLINE la_f32x4 la_vec3a_to_simd(la_vec3a v) {
    LA_ALIGN(16) float tmp[4];
    memcpy(tmp, &v, sizeof(tmp));
    return la_f32x4_load(tmp);
}

LA_INLINE la_vec3a la_simd_to_vec3a(la_f32x4 s) {
    LA_ALIGN(16) float tmp[4];
    la_vec3a out;
    la_f32x4_store(tmp, s);
    tmp[3] = 0.0f;
    memcpy(&out, tmp, sizeof(out));
    return out;
}

LA_INLINE la_f32x4 la_quat_to_simd(la_quat q) {
    LA_ALIGN(16) float tmp[4];
    memcpy(tmp, &q, sizeof(tmp));
    return la_f32x4_load(tmp);
}

LA_INLINE la_quat la_simd_to_quat(la_f32x4 s) {
    LA_ALIGN(16) float tmp[4];
    la_quat out;
    la_f32x4_store(tmp, s);
    memcpy(&out, tmp, sizeof(out));
    return out;
}

/* ============================================================================
 * Optional 8-wide AVX2 SIMD Abstraction (used in la_batch.h when AVX2 is present)
 * ============================================================================ */
#if defined(LA_SIMD_AVX2)
typedef __m256 la_f32x8;
LA_INLINE la_f32x8 la_f32x8_splat(float s) { return _mm256_set1_ps(s); }
LA_INLINE la_f32x8 la_f32x8_load(const float *p) { return _mm256_load_ps(p); }
LA_INLINE la_f32x8 la_f32x8_loadu(const float *p) { return _mm256_loadu_ps(p); }
LA_INLINE void la_f32x8_store(float *p, la_f32x8 v) { _mm256_store_ps(p, v); }
LA_INLINE void la_f32x8_storeu(float *p, la_f32x8 v) { _mm256_storeu_ps(p, v); }
LA_INLINE la_f32x8 la_f32x8_add(la_f32x8 a, la_f32x8 b) { return _mm256_add_ps(a, b); }
LA_INLINE la_f32x8 la_f32x8_sub(la_f32x8 a, la_f32x8 b) { return _mm256_sub_ps(a, b); }
LA_INLINE la_f32x8 la_f32x8_mul(la_f32x8 a, la_f32x8 b) { return _mm256_mul_ps(a, b); }
LA_INLINE la_f32x8 la_f32x8_div(la_f32x8 a, la_f32x8 b) { return _mm256_div_ps(a, b); }
LA_INLINE la_f32x8 la_f32x8_sqrt(la_f32x8 v) { return _mm256_sqrt_ps(v); }
LA_INLINE la_f32x8 la_f32x8_min(la_f32x8 a, la_f32x8 b) { return _mm256_min_ps(a, b); }
LA_INLINE la_f32x8 la_f32x8_max(la_f32x8 a, la_f32x8 b) { return _mm256_max_ps(a, b); }
LA_INLINE la_f32x8 la_f32x8_fma(la_f32x8 a, la_f32x8 b, la_f32x8 c) {
#if LA_SIMD_HAS_FMA
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
#endif /* LA_SIMD_AVX2 */

LA_EXTERN_C_END

#endif /* LA_SIMD_H */
