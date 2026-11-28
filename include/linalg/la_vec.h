/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_vec.h
 * Purpose: 2D, 3D (packed & 16-byte SIMD aligned), and 4D vector operations.
 */

#ifndef LA_VEC_H
#define LA_VEC_H

#include "la_simd.h"

LA_EXTERN_C_BEGIN

/* ============================================================================
 * la_vec2 (8 bytes)
 * ============================================================================ */

LA_INLINE la_vec2 la_vec2_make(float x, float y) {
    la_vec2 v; v.x = x; v.y = y; return v;
}

LA_INLINE la_vec2 la_vec2_splat(float s) {
    return la_vec2_make(s, s);
}

LA_INLINE la_vec2 la_vec2_zero(void) {
    return la_vec2_make(0.0f, 0.0f);
}

LA_INLINE la_vec2 la_vec2_add(la_vec2 a, la_vec2 b) {
    return la_vec2_make(a.x + b.x, a.y + b.y);
}

LA_INLINE la_vec2 la_vec2_sub(la_vec2 a, la_vec2 b) {
    return la_vec2_make(a.x - b.x, a.y - b.y);
}

LA_INLINE la_vec2 la_vec2_mul(la_vec2 a, la_vec2 b) {
    return la_vec2_make(a.x * b.x, a.y * b.y);
}

LA_INLINE la_vec2 la_vec2_div(la_vec2 a, la_vec2 b) {
    return la_vec2_make(a.x / b.x, a.y / b.y);
}

LA_INLINE la_vec2 la_vec2_scale(la_vec2 v, float s) {
    return la_vec2_make(v.x * s, v.y * s);
}

/* Computes a + s * b */
LA_INLINE la_vec2 la_vec2_fma(la_vec2 a, la_vec2 b, float s) {
    return la_vec2_make(la_fmaf(b.x, s, a.x), la_fmaf(b.y, s, a.y));
}

LA_INLINE la_vec2 la_vec2_neg(la_vec2 v) {
    return la_vec2_make(-v.x, -v.y);
}

LA_INLINE la_vec2 la_vec2_abs(la_vec2 v) {
    return la_vec2_make(la_absf(v.x), la_absf(v.y));
}

LA_INLINE la_vec2 la_vec2_min(la_vec2 a, la_vec2 b) {
    return la_vec2_make(la_minf(a.x, b.x), la_minf(a.y, b.y));
}

LA_INLINE la_vec2 la_vec2_max(la_vec2 a, la_vec2 b) {
    return la_vec2_make(la_maxf(a.x, b.x), la_maxf(a.y, b.y));
}

LA_INLINE la_vec2 la_vec2_clamp(la_vec2 v, la_vec2 lo, la_vec2 hi) {
    return la_vec2_min(la_vec2_max(v, lo), hi);
}

LA_INLINE la_vec2 la_vec2_lerp(la_vec2 a, la_vec2 b, float t) {
    return la_vec2_make(la_lerpf(a.x, b.x, t), la_lerpf(a.y, b.y, t));
}

LA_INLINE float la_vec2_dot(la_vec2 a, la_vec2 b) {
    return la_fmaf(a.x, b.x, a.y * b.y);
}

/* 2D cross product (signed parallelogram area): a.x*b.y - a.y*b.x */
LA_INLINE float la_vec2_cross(la_vec2 a, la_vec2 b) {
    return la_fmaf(a.x, b.y, -a.y * b.x);
}

/* 90-degree counter-clockwise perpendicular vector */
LA_INLINE la_vec2 la_vec2_perp(la_vec2 v) {
    return la_vec2_make(-v.y, v.x);
}

LA_INLINE float la_vec2_len_sq(la_vec2 v) {
    return la_vec2_dot(v, v);
}

LA_INLINE float la_vec2_len(la_vec2 v) {
    return sqrtf(la_vec2_len_sq(v));
}

LA_INLINE float la_vec2_dist_sq(la_vec2 a, la_vec2 b) {
    return la_vec2_len_sq(la_vec2_sub(a, b));
}

LA_INLINE float la_vec2_dist(la_vec2 a, la_vec2 b) {
    return sqrtf(la_vec2_dist_sq(a, b));
}

LA_INLINE la_vec2 la_vec2_norm(la_vec2 v) {
    return la_vec2_scale(v, la_rsqrtf(la_vec2_len_sq(v)));
}

LA_INLINE la_vec2 la_vec2_norm_safe(la_vec2 v) {
    float lsq = la_vec2_len_sq(v);
    if (LA_UNLIKELY(lsq <= LA_EPSILON_SQ)) {
        return la_vec2_zero();
    }
    return la_vec2_scale(v, la_rsqrtf(lsq));
}

LA_INLINE la_vec2 la_vec2_reflect(la_vec2 v, la_vec2 unit_n) {
    return la_vec2_fma(v, unit_n, -2.0f * la_vec2_dot(v, unit_n));
}

LA_INLINE bool la_vec2_approx_eq(la_vec2 a, la_vec2 b, float eps) {
    return la_approx_eqf(a.x, b.x, eps) && la_approx_eqf(a.y, b.y, eps);
}

/* ============================================================================
 * la_vec3 (12 bytes, tightly packed storage)
 * ============================================================================ */

LA_INLINE la_vec3 la_vec3_make(float x, float y, float z) {
    la_vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

LA_INLINE la_vec3 la_vec3_splat(float s) {
    return la_vec3_make(s, s, s);
}

LA_INLINE la_vec3 la_vec3_zero(void) {
    return la_vec3_make(0.0f, 0.0f, 0.0f);
}

LA_INLINE la_vec3a la_vec3_to_vec3a(la_vec3 v) {
    la_vec3a r; r.x = v.x; r.y = v.y; r.z = v.z; r._pad = 0.0f; return r;
}

LA_INLINE la_vec3 la_vec3a_to_vec3(la_vec3a v) {
    return la_vec3_make(v.x, v.y, v.z);
}

LA_INLINE la_vec3 la_vec3_add(la_vec3 a, la_vec3 b) {
    return la_vec3_make(a.x + b.x, a.y + b.y, a.z + b.z);
}

LA_INLINE la_vec3 la_vec3_sub(la_vec3 a, la_vec3 b) {
    return la_vec3_make(a.x - b.x, a.y - b.y, a.z - b.z);
}

LA_INLINE la_vec3 la_vec3_mul(la_vec3 a, la_vec3 b) {
    return la_vec3_make(a.x * b.x, a.y * b.y, a.z * b.z);
}

LA_INLINE la_vec3 la_vec3_div(la_vec3 a, la_vec3 b) {
    return la_vec3_make(a.x / b.x, a.y / b.y, a.z / b.z);
}

LA_INLINE la_vec3 la_vec3_scale(la_vec3 v, float s) {
    return la_vec3_make(v.x * s, v.y * s, v.z * s);
}

/* Computes a + s * b */
LA_INLINE la_vec3 la_vec3_fma(la_vec3 a, la_vec3 b, float s) {
    return la_vec3_make(
        la_fmaf(b.x, s, a.x),
        la_fmaf(b.y, s, a.y),
        la_fmaf(b.z, s, a.z)
    );
}

LA_INLINE la_vec3 la_vec3_neg(la_vec3 v) {
    return la_vec3_make(-v.x, -v.y, -v.z);
}

LA_INLINE la_vec3 la_vec3_abs(la_vec3 v) {
    return la_vec3_make(la_absf(v.x), la_absf(v.y), la_absf(v.z));
}

LA_INLINE la_vec3 la_vec3_min(la_vec3 a, la_vec3 b) {
    return la_vec3_make(la_minf(a.x, b.x), la_minf(a.y, b.y), la_minf(a.z, b.z));
}

LA_INLINE la_vec3 la_vec3_max(la_vec3 a, la_vec3 b) {
    return la_vec3_make(la_maxf(a.x, b.x), la_maxf(a.y, b.y), la_maxf(a.z, b.z));
}

LA_INLINE la_vec3 la_vec3_clamp(la_vec3 v, la_vec3 lo, la_vec3 hi) {
    return la_vec3_min(la_vec3_max(v, lo), hi);
}

LA_INLINE la_vec3 la_vec3_lerp(la_vec3 a, la_vec3 b, float t) {
    return la_vec3_make(
        la_lerpf(a.x, b.x, t),
        la_lerpf(a.y, b.y, t),
        la_lerpf(a.z, b.z, t)
    );
}

LA_INLINE float la_vec3_dot(la_vec3 a, la_vec3 b) {
    return la_fmaf(a.x, b.x, la_fmaf(a.y, b.y, a.z * b.z));
}

LA_INLINE la_vec3 la_vec3_cross(la_vec3 a, la_vec3 b) {
    return la_vec3_make(
        la_fmaf(a.y, b.z, -a.z * b.y),
        la_fmaf(a.z, b.x, -a.x * b.z),
        la_fmaf(a.x, b.y, -a.y * b.x)
    );
}

LA_INLINE float la_vec3_len_sq(la_vec3 v) {
    return la_vec3_dot(v, v);
}

LA_INLINE float la_vec3_len(la_vec3 v) {
    return sqrtf(la_vec3_len_sq(v));
}

LA_INLINE float la_vec3_dist_sq(la_vec3 a, la_vec3 b) {
    return la_vec3_len_sq(la_vec3_sub(a, b));
}

LA_INLINE float la_vec3_dist(la_vec3 a, la_vec3 b) {
    return sqrtf(la_vec3_dist_sq(a, b));
}

LA_INLINE la_vec3 la_vec3_norm(la_vec3 v) {
    return la_vec3_scale(v, la_rsqrtf(la_vec3_len_sq(v)));
}

LA_INLINE la_vec3 la_vec3_norm_fast(la_vec3 v) {
    return la_vec3_scale(v, la_rsqrtf_fast(la_vec3_len_sq(v)));
}

LA_INLINE la_vec3 la_vec3_norm_safe(la_vec3 v) {
    float lsq = la_vec3_len_sq(v);
    if (LA_UNLIKELY(lsq <= LA_EPSILON_SQ)) {
        return la_vec3_zero();
    }
    return la_vec3_scale(v, la_rsqrtf(lsq));
}

LA_INLINE la_vec3 la_vec3_reflect(la_vec3 v, la_vec3 unit_n) {
    return la_vec3_fma(v, unit_n, -2.0f * la_vec3_dot(v, unit_n));
}

/* Refract unit incident vector i across unit normal n with index ratio eta = n1/n2. */
LA_INLINE la_vec3 la_vec3_refract(la_vec3 unit_i, la_vec3 unit_n, float eta) {
    float ndoti = la_vec3_dot(unit_n, unit_i);
    float k = 1.0f - eta * eta * (1.0f - ndoti * ndoti);
    if (k < 0.0f) {
        return la_vec3_zero(); /* Total internal reflection */
    }
    return la_vec3_sub(la_vec3_scale(unit_i, eta),
                       la_vec3_scale(unit_n, eta * ndoti + sqrtf(k)));
}

LA_INLINE la_vec3 la_vec3_project(la_vec3 v, la_vec3 onto) {
    float denom = la_vec3_len_sq(onto);
    if (LA_UNLIKELY(denom <= LA_EPSILON_SQ)) {
        return la_vec3_zero();
    }
    return la_vec3_scale(onto, la_vec3_dot(v, onto) / denom);
}

LA_INLINE la_vec3 la_vec3_reject(la_vec3 v, la_vec3 onto) {
    return la_vec3_sub(v, la_vec3_project(v, onto));
}

/*
 * Branchless orthonormal basis from a unit normal (Duff et al., JCGT 2017).
 * Ideal for physics contact tangent frames and shader TBN construction.
 */
LA_INLINE void la_vec3_orthonormal_basis(la_vec3 unit_n,
                                         la_vec3 *LA_RESTRICT out_t,
                                         la_vec3 *LA_RESTRICT out_b) {
    float s = la_copysignf(1.0f, unit_n.z);
    float a = -1.0f / (s + unit_n.z);
    float b = unit_n.x * unit_n.y * a;
    *out_t = la_vec3_make(1.0f + s * unit_n.x * unit_n.x * a, s * b, -s * unit_n.x);
    *out_b = la_vec3_make(b, s + unit_n.y * unit_n.y * a, -unit_n.y);
}

LA_INLINE bool la_vec3_approx_eq(la_vec3 a, la_vec3 b, float eps) {
    return la_approx_eqf(a.x, b.x, eps) &&
           la_approx_eqf(a.y, b.y, eps) &&
           la_approx_eqf(a.z, b.z, eps);
}

/* ============================================================================
 * la_vec3a (16 bytes, 16-byte SIMD aligned, _pad == 0.0f invariant)
 * ============================================================================ */

LA_INLINE la_vec3a la_vec3a_make(float x, float y, float z) {
    la_vec3a v; v.x = x; v.y = y; v.z = z; v._pad = 0.0f; return v;
}

LA_INLINE la_vec3a la_vec3a_splat(float s) {
    return la_vec3a_make(s, s, s);
}

LA_INLINE la_vec3a la_vec3a_zero(void) {
    return la_vec3a_make(0.0f, 0.0f, 0.0f);
}

LA_INLINE la_vec3a la_vec3a_add(la_vec3a a, la_vec3a b) {
    return la_simd_to_vec3a(la_f32x4_add(la_vec3a_to_simd(a), la_vec3a_to_simd(b)));
}

LA_INLINE la_vec3a la_vec3a_sub(la_vec3a a, la_vec3a b) {
    return la_simd_to_vec3a(la_f32x4_sub(la_vec3a_to_simd(a), la_vec3a_to_simd(b)));
}

LA_INLINE la_vec3a la_vec3a_mul(la_vec3a a, la_vec3a b) {
    return la_simd_to_vec3a(la_f32x4_mul(la_vec3a_to_simd(a), la_vec3a_to_simd(b)));
}

LA_INLINE la_vec3a la_vec3a_div(la_vec3a a, la_vec3a b) {
    /* Guard _pad lane to 1.0f to avoid 0/0 NaN */
    la_f32x4 sb = la_f32x4_set(b.x, b.y, b.z, 1.0f);
    return la_simd_to_vec3a(la_f32x4_div(la_vec3a_to_simd(a), sb));
}

LA_INLINE la_vec3a la_vec3a_scale(la_vec3a v, float s) {
    return la_simd_to_vec3a(la_f32x4_mul(la_vec3a_to_simd(v), la_f32x4_splat(s)));
}

/* Computes a + s * b */
LA_INLINE la_vec3a la_vec3a_fma(la_vec3a a, la_vec3a b, float s) {
    return la_simd_to_vec3a(
        la_f32x4_fma(la_vec3a_to_simd(b), la_f32x4_splat(s), la_vec3a_to_simd(a))
    );
}

LA_INLINE la_vec3a la_vec3a_neg(la_vec3a v) {
    return la_simd_to_vec3a(la_f32x4_neg(la_vec3a_to_simd(v)));
}

LA_INLINE la_vec3a la_vec3a_abs(la_vec3a v) {
    return la_simd_to_vec3a(la_f32x4_abs(la_vec3a_to_simd(v)));
}

LA_INLINE la_vec3a la_vec3a_min(la_vec3a a, la_vec3a b) {
    return la_simd_to_vec3a(la_f32x4_min(la_vec3a_to_simd(a), la_vec3a_to_simd(b)));
}

LA_INLINE la_vec3a la_vec3a_max(la_vec3a a, la_vec3a b) {
    return la_simd_to_vec3a(la_f32x4_max(la_vec3a_to_simd(a), la_vec3a_to_simd(b)));
}

LA_INLINE la_vec3a la_vec3a_clamp(la_vec3a v, la_vec3a lo, la_vec3a hi) {
    return la_simd_to_vec3a(
        la_f32x4_clamp(la_vec3a_to_simd(v), la_vec3a_to_simd(lo), la_vec3a_to_simd(hi))
    );
}

LA_INLINE la_vec3a la_vec3a_lerp(la_vec3a a, la_vec3a b, float t) {
    return la_simd_to_vec3a(
        la_f32x4_lerp(la_vec3a_to_simd(a), la_vec3a_to_simd(b), la_f32x4_splat(t))
    );
}

LA_INLINE float la_vec3a_dot(la_vec3a a, la_vec3a b) {
    return la_f32x4_dot3(la_vec3a_to_simd(a), la_vec3a_to_simd(b));
}

LA_INLINE la_vec3a la_vec3a_cross(la_vec3a a, la_vec3a b) {
    return la_vec3a_make(
        la_fmaf(a.y, b.z, -a.z * b.y),
        la_fmaf(a.z, b.x, -a.x * b.z),
        la_fmaf(a.x, b.y, -a.y * b.x)
    );
}

LA_INLINE float la_vec3a_len_sq(la_vec3a v) {
    return la_vec3a_dot(v, v);
}

LA_INLINE float la_vec3a_len(la_vec3a v) {
    return sqrtf(la_vec3a_len_sq(v));
}

LA_INLINE float la_vec3a_dist_sq(la_vec3a a, la_vec3a b) {
    return la_vec3a_len_sq(la_vec3a_sub(a, b));
}

LA_INLINE float la_vec3a_dist(la_vec3a a, la_vec3a b) {
    return sqrtf(la_vec3a_dist_sq(a, b));
}

LA_INLINE la_vec3a la_vec3a_norm(la_vec3a v) {
    return la_vec3a_scale(v, la_rsqrtf(la_vec3a_len_sq(v)));
}

LA_INLINE la_vec3a la_vec3a_norm_fast(la_vec3a v) {
    return la_vec3a_scale(v, la_rsqrtf_fast(la_vec3a_len_sq(v)));
}

LA_INLINE la_vec3a la_vec3a_norm_safe(la_vec3a v) {
    float lsq = la_vec3a_len_sq(v);
    if (LA_UNLIKELY(lsq <= LA_EPSILON_SQ)) {
        return la_vec3a_zero();
    }
    return la_vec3a_scale(v, la_rsqrtf(lsq));
}

LA_INLINE la_vec3a la_vec3a_reflect(la_vec3a v, la_vec3a unit_n) {
    return la_vec3a_fma(v, unit_n, -2.0f * la_vec3a_dot(v, unit_n));
}

LA_INLINE bool la_vec3a_approx_eq(la_vec3a a, la_vec3a b, float eps) {
    return la_approx_eqf(a.x, b.x, eps) &&
           la_approx_eqf(a.y, b.y, eps) &&
           la_approx_eqf(a.z, b.z, eps);
}

/* ============================================================================
 * la_vec4 (16 bytes, 16-byte SIMD aligned)
 * ============================================================================ */

LA_INLINE la_vec4 la_vec4_make(float x, float y, float z, float w) {
    la_vec4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v;
}

LA_INLINE la_vec4 la_vec4_splat(float s) {
    return la_simd_to_vec4(la_f32x4_splat(s));
}

LA_INLINE la_vec4 la_vec4_zero(void) {
    return la_simd_to_vec4(la_f32x4_zero());
}

LA_INLINE la_vec4 la_vec4_from_vec3(la_vec3 v, float w) {
    return la_vec4_make(v.x, v.y, v.z, w);
}

LA_INLINE la_vec4 la_vec4_from_vec3a(la_vec3a v, float w) {
    return la_vec4_make(v.x, v.y, v.z, w);
}

LA_INLINE la_vec3 la_vec4_xyz(la_vec4 v) {
    return la_vec3_make(v.x, v.y, v.z);
}

LA_INLINE la_vec3a la_vec4_xyza(la_vec4 v) {
    return la_vec3a_make(v.x, v.y, v.z);
}

LA_INLINE la_vec4 la_vec4_add(la_vec4 a, la_vec4 b) {
    return la_simd_to_vec4(la_f32x4_add(la_vec4_to_simd(a), la_vec4_to_simd(b)));
}

LA_INLINE la_vec4 la_vec4_sub(la_vec4 a, la_vec4 b) {
    return la_simd_to_vec4(la_f32x4_sub(la_vec4_to_simd(a), la_vec4_to_simd(b)));
}

LA_INLINE la_vec4 la_vec4_mul(la_vec4 a, la_vec4 b) {
    return la_simd_to_vec4(la_f32x4_mul(la_vec4_to_simd(a), la_vec4_to_simd(b)));
}

LA_INLINE la_vec4 la_vec4_div(la_vec4 a, la_vec4 b) {
    return la_simd_to_vec4(la_f32x4_div(la_vec4_to_simd(a), la_vec4_to_simd(b)));
}

LA_INLINE la_vec4 la_vec4_scale(la_vec4 v, float s) {
    return la_simd_to_vec4(la_f32x4_mul(la_vec4_to_simd(v), la_f32x4_splat(s)));
}

/* Computes a + s * b */
LA_INLINE la_vec4 la_vec4_fma(la_vec4 a, la_vec4 b, float s) {
    return la_simd_to_vec4(
        la_f32x4_fma(la_vec4_to_simd(b), la_f32x4_splat(s), la_vec4_to_simd(a))
    );
}

LA_INLINE la_vec4 la_vec4_neg(la_vec4 v) {
    return la_simd_to_vec4(la_f32x4_neg(la_vec4_to_simd(v)));
}

LA_INLINE la_vec4 la_vec4_abs(la_vec4 v) {
    return la_simd_to_vec4(la_f32x4_abs(la_vec4_to_simd(v)));
}

LA_INLINE la_vec4 la_vec4_min(la_vec4 a, la_vec4 b) {
    return la_simd_to_vec4(la_f32x4_min(la_vec4_to_simd(a), la_vec4_to_simd(b)));
}

LA_INLINE la_vec4 la_vec4_max(la_vec4 a, la_vec4 b) {
    return la_simd_to_vec4(la_f32x4_max(la_vec4_to_simd(a), la_vec4_to_simd(b)));
}

LA_INLINE la_vec4 la_vec4_clamp(la_vec4 v, la_vec4 lo, la_vec4 hi) {
    return la_simd_to_vec4(
        la_f32x4_clamp(la_vec4_to_simd(v), la_vec4_to_simd(lo), la_vec4_to_simd(hi))
    );
}

LA_INLINE la_vec4 la_vec4_lerp(la_vec4 a, la_vec4 b, float t) {
    return la_simd_to_vec4(
        la_f32x4_lerp(la_vec4_to_simd(a), la_vec4_to_simd(b), la_f32x4_splat(t))
    );
}

LA_INLINE float la_vec4_dot(la_vec4 a, la_vec4 b) {
    return la_f32x4_dot4(la_vec4_to_simd(a), la_vec4_to_simd(b));
}

LA_INLINE float la_vec4_len_sq(la_vec4 v) {
    return la_vec4_dot(v, v);
}

LA_INLINE float la_vec4_len(la_vec4 v) {
    return sqrtf(la_vec4_len_sq(v));
}

LA_INLINE float la_vec4_dist_sq(la_vec4 a, la_vec4 b) {
    return la_vec4_len_sq(la_vec4_sub(a, b));
}

LA_INLINE float la_vec4_dist(la_vec4 a, la_vec4 b) {
    return sqrtf(la_vec4_dist_sq(a, b));
}

LA_INLINE la_vec4 la_vec4_norm(la_vec4 v) {
    return la_vec4_scale(v, la_rsqrtf(la_vec4_len_sq(v)));
}

LA_INLINE la_vec4 la_vec4_norm_fast(la_vec4 v) {
    return la_vec4_scale(v, la_rsqrtf_fast(la_vec4_len_sq(v)));
}

LA_INLINE la_vec4 la_vec4_norm_safe(la_vec4 v) {
    float lsq = la_vec4_len_sq(v);
    if (LA_UNLIKELY(lsq <= LA_EPSILON_SQ)) {
        return la_vec4_zero();
    }
    return la_vec4_scale(v, la_rsqrtf(lsq));
}

LA_INLINE bool la_vec4_approx_eq(la_vec4 a, la_vec4 b, float eps) {
    return la_approx_eqf(a.x, b.x, eps) &&
           la_approx_eqf(a.y, b.y, eps) &&
           la_approx_eqf(a.z, b.z, eps) &&
           la_approx_eqf(a.w, b.w, eps);
}

LA_EXTERN_C_END

#endif /* LA_VEC_H */
