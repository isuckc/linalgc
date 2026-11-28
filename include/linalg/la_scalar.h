/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_scalar.h
 * Purpose: Numerically stable scalar math helpers, FMA, fast/exact rsqrt,
 *          shader-style interpolation/shaping functions, and comparisons.
 */

#ifndef LA_SCALAR_H
#define LA_SCALAR_H

#include "la_config.h"

LA_EXTERN_C_BEGIN

LA_INLINE float la_absf(float x) {
    return fabsf(x);
}

LA_INLINE float la_minf(float a, float b) {
    return (a < b) ? a : b;
}

LA_INLINE float la_maxf(float a, float b) {
    return (a > b) ? a : b;
}

LA_INLINE float la_clampf(float x, float lo, float hi) {
    return la_minf(la_maxf(x, lo), hi);
}

LA_INLINE float la_saturatef(float x) {
    return la_clampf(x, 0.0f, 1.0f);
}

LA_INLINE float la_fmaf(float a, float b, float c) {
    return fmaf(a, b, c);
}

/* Exact endpoint-preserving monotonic linear interpolation. */
LA_INLINE float la_lerpf(float a, float b, float t) {
    return la_fmaf(t, b, la_fmaf(-t, a, a));
}

LA_INLINE float la_signf(float x) {
    return (float)((x > 0.0f) - (x < 0.0f));
}

LA_INLINE float la_copysignf(float mag, float sgn) {
    return copysignf(mag, sgn);
}

LA_INLINE float la_stepf(float edge, float x) {
    return (x < edge) ? 0.0f : 1.0f;
}

LA_INLINE float la_smoothstepf(float edge0, float edge1, float x) {
    float denom = edge1 - edge0;
    float t = (la_absf(denom) > LA_EPSILON) ? la_saturatef((x - edge0) / denom) : 0.0f;
    return t * t * (3.0f - 2.0f * t);
}

LA_INLINE float la_remapf(float x, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    float t = (la_absf(denom) > LA_EPSILON) ? ((x - in_min) / denom) : 0.0f;
    return la_lerpf(out_min, out_max, t);
}

LA_INLINE float la_deg2rad(float deg) {
    return deg * LA_DEG2RAD;
}

LA_INLINE float la_rad2deg(float rad) {
    return rad * LA_RAD2DEG;
}

LA_INLINE float la_sqrtf(float x) {
    return sqrtf(x);
}

LA_INLINE float la_sqrtf_safe(float x) {
    return (x > 0.0f) ? sqrtf(x) : 0.0f;
}

LA_INLINE float la_rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

/*
 * Fast reciprocal square root (~23-bit accuracy via 2 Newton-Raphson iterations).
 * Uses memcpy bit-cast for 100% strict-aliasing compliance in C99 & C++17.
 */
LA_INLINE float la_rsqrtf_fast(float x) {
    uint32_t bits = 0;
    float y = 0.0f;
    float half_x = 0.5f * x;
    memcpy(&bits, &x, sizeof(bits));
    bits = 0x5f375a86u - (bits >> 1);
    memcpy(&y, &bits, sizeof(y));
    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    return y;
}

LA_INLINE void la_sincosf(float angle, float *LA_RESTRICT out_sin, float *LA_RESTRICT out_cos) {
    *out_sin = sinf(angle);
    *out_cos = cosf(angle);
}

LA_INLINE bool la_is_finitef(float x) {
    return isfinite(x) != 0;
}

LA_INLINE bool la_approx_eqf(float a, float b, float eps) {
    float diff = la_absf(a - b);
    float scale = la_maxf(1.0f, la_maxf(la_absf(a), la_absf(b)));
    return diff <= eps * scale;
}

LA_EXTERN_C_END

#endif /* LA_SCALAR_H */
