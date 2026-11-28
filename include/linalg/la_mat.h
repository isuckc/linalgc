/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_mat.h
 * Purpose: Column-major mat2, mat3, mat3a, and mat4 operations, SIMD 4x4
 *          multiply/transpose/inverse, fast rigid/affine inverses, normal
 *          matrices, and explicit RH/LH + ZO/NO/reverse-Z shader projections.
 */

#ifndef LA_MAT_H
#define LA_MAT_H

#include "la_vec.h"

LA_EXTERN_C_BEGIN

/* ============================================================================
 * la_mat2 (16 bytes, column-major)
 * ============================================================================ */

LA_INLINE la_mat2 la_mat2_from_cols(la_vec2 c0, la_vec2 c1) {
    la_mat2 m; m.cols[0] = c0; m.cols[1] = c1; return m;
}

LA_INLINE la_mat2 la_mat2_zero(void) {
    return la_mat2_from_cols(la_vec2_zero(), la_vec2_zero());
}

LA_INLINE la_mat2 la_mat2_identity(void) {
    return la_mat2_from_cols(la_vec2_make(1.0f, 0.0f), la_vec2_make(0.0f, 1.0f));
}

LA_INLINE la_mat2 la_mat2_from_angle(float radians) {
    float s, c;
    la_sincosf(radians, &s, &c);
    return la_mat2_from_cols(la_vec2_make(c, s), la_vec2_make(-s, c));
}

LA_INLINE la_vec2 la_mat2_mul_vec2(la_mat2 m, la_vec2 v) {
    return la_vec2_make(
        la_fmaf(m.cols[0].x, v.x, m.cols[1].x * v.y),
        la_fmaf(m.cols[0].y, v.x, m.cols[1].y * v.y)
    );
}

LA_INLINE la_mat2 la_mat2_mul(la_mat2 a, la_mat2 b) {
    return la_mat2_from_cols(
        la_mat2_mul_vec2(a, b.cols[0]),
        la_mat2_mul_vec2(a, b.cols[1])
    );
}

LA_INLINE la_mat2 la_mat2_transpose(la_mat2 m) {
    return la_mat2_from_cols(
        la_vec2_make(m.cols[0].x, m.cols[1].x),
        la_vec2_make(m.cols[0].y, m.cols[1].y)
    );
}

LA_INLINE float la_mat2_det(la_mat2 m) {
    return la_fmaf(m.cols[0].x, m.cols[1].y, -m.cols[1].x * m.cols[0].y);
}

LA_INLINE bool la_mat2_inv_safe(la_mat2 m, la_mat2 *LA_RESTRICT out) {
    float d = la_mat2_det(m);
    if (LA_UNLIKELY(la_absf(d) <= LA_EPSILON_SQ)) {
        *out = la_mat2_zero();
        return false;
    }
    float inv_d = 1.0f / d;
    *out = la_mat2_from_cols(
        la_vec2_make(m.cols[1].y * inv_d, -m.cols[0].y * inv_d),
        la_vec2_make(-m.cols[1].x * inv_d, m.cols[0].x * inv_d)
    );
    return true;
}

LA_INLINE la_mat2 la_mat2_inv(la_mat2 m) {
    la_mat2 out;
    (void)la_mat2_inv_safe(m, &out);
    return out;
}

LA_INLINE bool la_mat2_approx_eq(la_mat2 a, la_mat2 b, float eps) {
    return la_vec2_approx_eq(a.cols[0], b.cols[0], eps) &&
           la_vec2_approx_eq(a.cols[1], b.cols[1], eps);
}

/* ============================================================================
 * la_mat3 (36 bytes, tightly packed column-major)
 * ============================================================================ */

LA_INLINE la_mat3 la_mat3_from_cols(la_vec3 c0, la_vec3 c1, la_vec3 c2) {
    la_mat3 m; m.cols[0] = c0; m.cols[1] = c1; m.cols[2] = c2; return m;
}

LA_INLINE la_mat3 la_mat3_zero(void) {
    return la_mat3_from_cols(la_vec3_zero(), la_vec3_zero(), la_vec3_zero());
}

LA_INLINE la_mat3 la_mat3_identity(void) {
    return la_mat3_from_cols(
        la_vec3_make(1.0f, 0.0f, 0.0f),
        la_vec3_make(0.0f, 1.0f, 0.0f),
        la_vec3_make(0.0f, 0.0f, 1.0f)
    );
}

LA_INLINE la_mat3 la_mat3_from_diag(la_vec3 d) {
    return la_mat3_from_cols(
        la_vec3_make(d.x, 0.0f, 0.0f),
        la_vec3_make(0.0f, d.y, 0.0f),
        la_vec3_make(0.0f, 0.0f, d.z)
    );
}

/* Skew-symmetric cross-product matrix: la_mat3_mul_vec3(skew(v), u) == cross(v, u). */
LA_INLINE la_mat3 la_mat3_skew(la_vec3 v) {
    return la_mat3_from_cols(
        la_vec3_make(0.0f,  v.z, -v.y),
        la_vec3_make(-v.z, 0.0f,  v.x),
        la_vec3_make(v.y,  -v.x, 0.0f)
    );
}

LA_INLINE la_mat3 la_mat3_from_mat4(la_mat4 m) {
    return la_mat3_from_cols(
        la_vec4_xyz(m.cols[0]),
        la_vec4_xyz(m.cols[1]),
        la_vec4_xyz(m.cols[2])
    );
}

LA_INLINE la_mat3a la_mat3_to_mat3a(la_mat3 m) {
    la_mat3a r;
    r.cols[0] = la_vec3_to_vec3a(m.cols[0]);
    r.cols[1] = la_vec3_to_vec3a(m.cols[1]);
    r.cols[2] = la_vec3_to_vec3a(m.cols[2]);
    return r;
}

LA_INLINE la_mat3 la_mat3a_to_mat3(la_mat3a m) {
    return la_mat3_from_cols(
        la_vec3a_to_vec3(m.cols[0]),
        la_vec3a_to_vec3(m.cols[1]),
        la_vec3a_to_vec3(m.cols[2])
    );
}

LA_INLINE la_mat3 la_mat3_add(la_mat3 a, la_mat3 b) {
    return la_mat3_from_cols(
        la_vec3_add(a.cols[0], b.cols[0]),
        la_vec3_add(a.cols[1], b.cols[1]),
        la_vec3_add(a.cols[2], b.cols[2])
    );
}

LA_INLINE la_mat3 la_mat3_sub(la_mat3 a, la_mat3 b) {
    return la_mat3_from_cols(
        la_vec3_sub(a.cols[0], b.cols[0]),
        la_vec3_sub(a.cols[1], b.cols[1]),
        la_vec3_sub(a.cols[2], b.cols[2])
    );
}

LA_INLINE la_mat3 la_mat3_scale(la_mat3 m, float s) {
    return la_mat3_from_cols(
        la_vec3_scale(m.cols[0], s),
        la_vec3_scale(m.cols[1], s),
        la_vec3_scale(m.cols[2], s)
    );
}

LA_INLINE la_vec3 la_mat3_mul_vec3(la_mat3 m, la_vec3 v) {
    la_vec3 r = la_vec3_scale(m.cols[0], v.x);
    r = la_vec3_fma(r, m.cols[1], v.y);
    r = la_vec3_fma(r, m.cols[2], v.z);
    return r;
}

LA_INLINE la_mat3 la_mat3_mul(la_mat3 a, la_mat3 b) {
    return la_mat3_from_cols(
        la_mat3_mul_vec3(a, b.cols[0]),
        la_mat3_mul_vec3(a, b.cols[1]),
        la_mat3_mul_vec3(a, b.cols[2])
    );
}

LA_INLINE la_mat3 la_mat3_transpose(la_mat3 m) {
    return la_mat3_from_cols(
        la_vec3_make(m.cols[0].x, m.cols[1].x, m.cols[2].x),
        la_vec3_make(m.cols[0].y, m.cols[1].y, m.cols[2].y),
        la_vec3_make(m.cols[0].z, m.cols[1].z, m.cols[2].z)
    );
}

LA_INLINE float la_mat3_trace(la_mat3 m) {
    return m.cols[0].x + m.cols[1].y + m.cols[2].z;
}

LA_INLINE float la_mat3_det(la_mat3 m) {
    return la_vec3_dot(m.cols[0], la_vec3_cross(m.cols[1], m.cols[2]));
}

/*
 * Cofactor matrix of 3x3 M: satisfies M^T * cofactor(M) = det(M) * I.
 * Useful directly as a shader normal matrix (sign-preserving even under reflection).
 */
LA_INLINE la_mat3 la_mat3_cofactor(la_mat3 m) {
    return la_mat3_from_cols(
        la_vec3_cross(m.cols[1], m.cols[2]),
        la_vec3_cross(m.cols[2], m.cols[0]),
        la_vec3_cross(m.cols[0], m.cols[1])
    );
}

LA_INLINE bool la_mat3_inv_safe(la_mat3 m, la_mat3 *LA_RESTRICT out) {
    la_vec3 r0 = la_vec3_cross(m.cols[1], m.cols[2]);
    la_vec3 r1 = la_vec3_cross(m.cols[2], m.cols[0]);
    la_vec3 r2 = la_vec3_cross(m.cols[0], m.cols[1]);
    float det = la_vec3_dot(m.cols[0], r0);
    if (LA_UNLIKELY(la_absf(det) <= LA_EPSILON_SQ)) {
        *out = la_mat3_zero();
        return false;
    }
    float inv_det = 1.0f / det;
    *out = la_mat3_transpose(la_mat3_from_cols(
        la_vec3_scale(r0, inv_det),
        la_vec3_scale(r1, inv_det),
        la_vec3_scale(r2, inv_det)
    ));
    return true;
}

LA_INLINE la_mat3 la_mat3_inv(la_mat3 m) {
    la_mat3 out;
    (void)la_mat3_inv_safe(m, &out);
    return out;
}

LA_INLINE la_mat3 la_mat3_from_axis_angle(la_vec3 unit_axis, float radians) {
    float s, c;
    la_sincosf(radians, &s, &c);
    float omc = 1.0f - c;
    float x = unit_axis.x, y = unit_axis.y, z = unit_axis.z;
    float xy = x * y * omc, xz = x * z * omc, yz = y * z * omc;
    float xs = x * s, ys = y * s, zs = z * s;
    return la_mat3_from_cols(
        la_vec3_make(c + x * x * omc, xy + zs,         xz - ys),
        la_vec3_make(xy - zs,         c + y * y * omc, yz + xs),
        la_vec3_make(xz + ys,         yz - xs,         c + z * z * omc)
    );
}

/* Modified Gram-Schmidt orthonormalization (preserves right-handedness via c2 = c0 x c1). */
LA_INLINE la_mat3 la_mat3_orthonormalize(la_mat3 m) {
    la_vec3 c0 = la_vec3_norm_safe(m.cols[0]);
    la_vec3 c1 = la_vec3_norm_safe(la_vec3_fma(m.cols[1], c0, -la_vec3_dot(m.cols[1], c0)));
    la_vec3 c2 = la_vec3_cross(c0, c1);
    return la_mat3_from_cols(c0, c1, c2);
}

LA_INLINE bool la_mat3_approx_eq(la_mat3 a, la_mat3 b, float eps) {
    return la_vec3_approx_eq(a.cols[0], b.cols[0], eps) &&
           la_vec3_approx_eq(a.cols[1], b.cols[1], eps) &&
           la_vec3_approx_eq(a.cols[2], b.cols[2], eps);
}

/* ============================================================================
 * la_mat3a (48 bytes, 16-byte SIMD aligned, GLSL std140 mat3 / Metal float3x3)
 * ============================================================================ */

LA_INLINE la_mat3a la_mat3a_from_cols(la_vec3a c0, la_vec3a c1, la_vec3a c2) {
    la_mat3a m; m.cols[0] = c0; m.cols[1] = c1; m.cols[2] = c2; return m;
}

LA_INLINE la_mat3a la_mat3a_identity(void) {
    return la_mat3a_from_cols(
        la_vec3a_make(1.0f, 0.0f, 0.0f),
        la_vec3a_make(0.0f, 1.0f, 0.0f),
        la_vec3a_make(0.0f, 0.0f, 1.0f)
    );
}

LA_INLINE la_vec3a la_mat3a_mul_vec3a(la_mat3a m, la_vec3a v) {
    la_f32x4 sv = la_vec3a_to_simd(v);
    la_f32x4 r = la_f32x4_mul(la_vec3a_to_simd(m.cols[0]), la_f32x4_splat_x(sv));
    r = la_f32x4_fma(la_vec3a_to_simd(m.cols[1]), la_f32x4_splat_y(sv), r);
    r = la_f32x4_fma(la_vec3a_to_simd(m.cols[2]), la_f32x4_splat_z(sv), r);
    return la_simd_to_vec3a(r);
}

LA_INLINE la_mat3a la_mat3a_mul(la_mat3a a, la_mat3a b) {
    return la_mat3a_from_cols(
        la_mat3a_mul_vec3a(a, b.cols[0]),
        la_mat3a_mul_vec3a(a, b.cols[1]),
        la_mat3a_mul_vec3a(a, b.cols[2])
    );
}

/* Shader normal matrix (inverse-transpose of upper 3x3, or cofactor if near-singular). */
LA_INLINE la_mat3a la_mat3a_normal_from_mat4(la_mat4 m) {
    la_mat3 m3 = la_mat3_from_mat4(m);
    la_mat3 cof = la_mat3_cofactor(m3);
    float det = la_vec3_dot(m3.cols[0], cof.cols[0]);
    if (LA_LIKELY(la_absf(det) > LA_EPSILON_SQ)) {
        cof = la_mat3_scale(cof, 1.0f / det);
    }
    return la_mat3_to_mat3a(cof);
}

/* ============================================================================
 * la_mat4 (64 bytes, 16-byte SIMD aligned, Column-Major)
 * ============================================================================ */

LA_INLINE la_mat4 la_mat4_from_cols(la_vec4 c0, la_vec4 c1, la_vec4 c2, la_vec4 c3) {
    la_mat4 m;
    m.cols[0] = c0;
    m.cols[1] = c1;
    m.cols[2] = c2;
    m.cols[3] = c3;
    return m;
}

LA_INLINE la_mat4 la_mat4_zero(void) {
    la_vec4 z = la_vec4_zero();
    return la_mat4_from_cols(z, z, z, z);
}

LA_INLINE la_mat4 la_mat4_identity(void) {
    return la_mat4_from_cols(
        la_vec4_make(1.0f, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 1.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, 1.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_from_mat3(la_mat3 m) {
    return la_mat4_from_cols(
        la_vec4_from_vec3(m.cols[0], 0.0f),
        la_vec4_from_vec3(m.cols[1], 0.0f),
        la_vec4_from_vec3(m.cols[2], 0.0f),
        la_vec4_make(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

LA_INLINE float la_mat4_at(la_mat4 m, size_t row, size_t col) {
    const la_vec4 c = m.cols[col];
    switch (row) {
        case 0: return c.x;
        case 1: return c.y;
        case 2: return c.z;
        default: return c.w;
    }
}

/* SIMD column-major mat4 * vec4 (4 FMAs, zero horizontal reductions). */
LA_INLINE la_vec4 la_mat4_mul_vec4(la_mat4 m, la_vec4 v) {
    la_f32x4 sv = la_vec4_to_simd(v);
    la_f32x4 r = la_f32x4_mul(la_vec4_to_simd(m.cols[0]), la_f32x4_splat_x(sv));
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[1]), la_f32x4_splat_y(sv), r);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[2]), la_f32x4_splat_z(sv), r);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[3]), la_f32x4_splat_w(sv), r);
    return la_simd_to_vec4(r);
}

/* Affine point transform (w = 1): ignores perspective row. */
LA_INLINE la_vec3 la_mat4_mul_point3(la_mat4 m, la_vec3 p) {
    la_f32x4 r = la_vec4_to_simd(m.cols[3]);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[0]), la_f32x4_splat(p.x), r);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[1]), la_f32x4_splat(p.y), r);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[2]), la_f32x4_splat(p.z), r);
    return la_vec4_xyz(la_simd_to_vec4(r));
}

LA_INLINE la_vec3a la_mat4_mul_point3a(la_mat4 m, la_vec3a p) {
    la_f32x4 sp = la_vec3a_to_simd(p);
    la_f32x4 r = la_vec4_to_simd(m.cols[3]);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[0]), la_f32x4_splat_x(sp), r);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[1]), la_f32x4_splat_y(sp), r);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[2]), la_f32x4_splat_z(sp), r);
    return la_simd_to_vec3a(r);
}

/* Direction transform (w = 0): ignores translation. */
LA_INLINE la_vec3 la_mat4_mul_dir3(la_mat4 m, la_vec3 d) {
    la_f32x4 r = la_f32x4_mul(la_vec4_to_simd(m.cols[0]), la_f32x4_splat(d.x));
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[1]), la_f32x4_splat(d.y), r);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[2]), la_f32x4_splat(d.z), r);
    return la_vec4_xyz(la_simd_to_vec4(r));
}

LA_INLINE la_vec3a la_mat4_mul_dir3a(la_mat4 m, la_vec3a d) {
    la_f32x4 sd = la_vec3a_to_simd(d);
    la_f32x4 r = la_f32x4_mul(la_vec4_to_simd(m.cols[0]), la_f32x4_splat_x(sd));
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[1]), la_f32x4_splat_y(sd), r);
    r = la_f32x4_fma(la_vec4_to_simd(m.cols[2]), la_f32x4_splat_z(sd), r);
    return la_simd_to_vec3a(r);
}

/* Projective transform with perspective divide (p -> (M * [p, 1]).xyz / w). */
LA_INLINE la_vec3 la_mat4_project_point3(la_mat4 m, la_vec3 p) {
    la_vec4 clip = la_mat4_mul_vec4(m, la_vec4_from_vec3(p, 1.0f));
    float inv_w = (la_absf(clip.w) > LA_EPSILON) ? (1.0f / clip.w) : 0.0f;
    return la_vec3_make(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
}

/* SIMD 4x4 matrix multiplication. */
LA_INLINE la_mat4 la_mat4_mul(la_mat4 a, la_mat4 b) {
    la_f32x4 a0 = la_vec4_to_simd(a.cols[0]);
    la_f32x4 a1 = la_vec4_to_simd(a.cols[1]);
    la_f32x4 a2 = la_vec4_to_simd(a.cols[2]);
    la_f32x4 a3 = la_vec4_to_simd(a.cols[3]);

    la_mat4 out;
    for (int i = 0; i < 4; ++i) {
        la_f32x4 bi = la_vec4_to_simd(b.cols[i]);
        la_f32x4 r = la_f32x4_mul(a0, la_f32x4_splat_x(bi));
        r = la_f32x4_fma(a1, la_f32x4_splat_y(bi), r);
        r = la_f32x4_fma(a2, la_f32x4_splat_z(bi), r);
        r = la_f32x4_fma(a3, la_f32x4_splat_w(bi), r);
        out.cols[i] = la_simd_to_vec4(r);
    }
    return out;
}

LA_INLINE la_mat4 la_mat4_transpose(la_mat4 m) {
    la_f32x4 c0 = la_vec4_to_simd(m.cols[0]);
    la_f32x4 c1 = la_vec4_to_simd(m.cols[1]);
    la_f32x4 c2 = la_vec4_to_simd(m.cols[2]);
    la_f32x4 c3 = la_vec4_to_simd(m.cols[3]);
    la_f32x4_transpose4(&c0, &c1, &c2, &c3);
    return la_mat4_from_cols(
        la_simd_to_vec4(c0), la_simd_to_vec4(c1),
        la_simd_to_vec4(c2), la_simd_to_vec4(c3)
    );
}

/*
 * Fast inverse for rigid transforms (orthonormal 3x3 rotation R + translation t):
 *   inv([R t; 0 1]) = [R^T  -R^T * t; 0 1]
 */
LA_INLINE la_mat4 la_mat4_inv_rigid(la_mat4 m) {
    la_f32x4 c0 = la_vec4_to_simd(m.cols[0]);
    la_f32x4 c1 = la_vec4_to_simd(m.cols[1]);
    la_f32x4 c2 = la_vec4_to_simd(m.cols[2]);
    la_f32x4 c3 = la_vec4_to_simd(m.cols[3]);

    float tx = -la_f32x4_dot3(c0, c3);
    float ty = -la_f32x4_dot3(c1, c3);
    float tz = -la_f32x4_dot3(c2, c3);

    la_f32x4 z = la_f32x4_zero();
    la_f32x4_transpose4(&c0, &c1, &c2, &z);

    return la_mat4_from_cols(
        la_vec4_make(la_f32x4_get_x(c0), la_f32x4_get_y(c0), la_f32x4_get_z(c0), 0.0f),
        la_vec4_make(la_f32x4_get_x(c1), la_f32x4_get_y(c1), la_f32x4_get_z(c1), 0.0f),
        la_vec4_make(la_f32x4_get_x(c2), la_f32x4_get_y(c2), la_f32x4_get_z(c2), 0.0f),
        la_vec4_make(tx, ty, tz, 1.0f)
    );
}

/* Fast inverse for general affine transforms (arbitrary invertible 3x3 + translation). */
LA_INLINE bool la_mat4_inv_affine_safe(la_mat4 m, la_mat4 *LA_RESTRICT out) {
    la_mat3 m3 = la_mat3_from_mat4(m);
    la_mat3 inv3;
    if (LA_UNLIKELY(!la_mat3_inv_safe(m3, &inv3))) {
        *out = la_mat4_zero();
        return false;
    }
    la_vec3 t = la_vec4_xyz(m.cols[3]);
    la_vec3 inv_t = la_vec3_neg(la_mat3_mul_vec3(inv3, t));
    *out = la_mat4_from_cols(
        la_vec4_from_vec3(inv3.cols[0], 0.0f),
        la_vec4_from_vec3(inv3.cols[1], 0.0f),
        la_vec4_from_vec3(inv3.cols[2], 0.0f),
        la_vec4_from_vec3(inv_t, 1.0f)
    );
    return true;
}

LA_INLINE la_mat4 la_mat4_inv_affine(la_mat4 m) {
    la_mat4 out;
    (void)la_mat4_inv_affine_safe(m, &out);
    return out;
}

/* General 4x4 determinant via 2x2 Laplace expansion. */
LA_INLINE float la_mat4_det(la_mat4 m) {
    float c0x = m.cols[0].x, c0y = m.cols[0].y, c0z = m.cols[0].z, c0w = m.cols[0].w;
    float c1x = m.cols[1].x, c1y = m.cols[1].y, c1z = m.cols[1].z, c1w = m.cols[1].w;
    float c2x = m.cols[2].x, c2y = m.cols[2].y, c2z = m.cols[2].z, c2w = m.cols[2].w;
    float c3x = m.cols[3].x, c3y = m.cols[3].y, c3z = m.cols[3].z, c3w = m.cols[3].w;

    float s0 = c0x * c1y - c1x * c0y;
    float s1 = c0x * c1z - c1x * c0z;
    float s2 = c0x * c1w - c1x * c0w;
    float s3 = c0y * c1z - c1y * c0z;
    float s4 = c0y * c1w - c1y * c0w;
    float s5 = c0z * c1w - c1z * c0w;

    float d5 = c2z * c3w - c3z * c2w;
    float d4 = c2y * c3w - c3y * c2w;
    float d3 = c2y * c3z - c3y * c2z;
    float d2 = c2x * c3w - c3x * c2w;
    float d1 = c2x * c3z - c3x * c2z;
    float d0 = c2x * c3y - c3x * c2y;

    return s0 * d5 - s1 * d4 + s2 * d3 + s3 * d2 - s4 * d1 + s5 * d0;
}

/* General 4x4 inverse via 2x2 sub-determinant cofactors (handles projections & arbitrary 4x4). */
LA_INLINE bool la_mat4_inv_safe(la_mat4 m, la_mat4 *LA_RESTRICT out) {
    float c0x = m.cols[0].x, c0y = m.cols[0].y, c0z = m.cols[0].z, c0w = m.cols[0].w;
    float c1x = m.cols[1].x, c1y = m.cols[1].y, c1z = m.cols[1].z, c1w = m.cols[1].w;
    float c2x = m.cols[2].x, c2y = m.cols[2].y, c2z = m.cols[2].z, c2w = m.cols[2].w;
    float c3x = m.cols[3].x, c3y = m.cols[3].y, c3z = m.cols[3].z, c3w = m.cols[3].w;

    float s0 = c0x * c1y - c1x * c0y;
    float s1 = c0x * c1z - c1x * c0z;
    float s2 = c0x * c1w - c1x * c0w;
    float s3 = c0y * c1z - c1y * c0z;
    float s4 = c0y * c1w - c1y * c0w;
    float s5 = c0z * c1w - c1z * c0w;

    float d5 = c2z * c3w - c3z * c2w;
    float d4 = c2y * c3w - c3y * c2w;
    float d3 = c2y * c3z - c3y * c2z;
    float d2 = c2x * c3w - c3x * c2w;
    float d1 = c2x * c3z - c3x * c2z;
    float d0 = c2x * c3y - c3x * c2y;

    float det = s0 * d5 - s1 * d4 + s2 * d3 + s3 * d2 - s4 * d1 + s5 * d0;
    if (LA_UNLIKELY(la_absf(det) <= LA_EPSILON_SQ)) {
        *out = la_mat4_zero();
        return false;
    }

    la_f32x4 inv_det = la_f32x4_splat(1.0f / det);

    la_f32x4 r0 = la_f32x4_set(
         c1y * d5 - c1z * d4 + c1w * d3,
        -c0y * d5 + c0z * d4 - c0w * d3,
         c3y * s5 - c3z * s4 + c3w * s3,
        -c2y * s5 + c2z * s4 - c2w * s3
    );
    la_f32x4 r1 = la_f32x4_set(
        -c1x * d5 + c1z * d2 - c1w * d1,
         c0x * d5 - c0z * d2 + c0w * d1,
        -c3x * s5 + c3z * s2 - c3w * s1,
         c2x * s5 - c2z * s2 + c2w * s1
    );
    la_f32x4 r2 = la_f32x4_set(
         c1x * d4 - c1y * d2 + c1w * d0,
        -c0x * d4 + c0y * d2 - c0w * d0,
         c3x * s4 - c3y * s2 + c3w * s0,
        -c2x * s4 + c2y * s2 - c2w * s0
    );
    la_f32x4 r3 = la_f32x4_set(
        -c1x * d3 + c1y * d1 - c1z * d0,
         c0x * d3 - c0y * d1 + c0z * d0,
        -c3x * s3 + c3y * s1 - c3z * s0,
         c2x * s3 - c2y * s1 + c2z * s0
    );

    *out = la_mat4_from_cols(
        la_simd_to_vec4(la_f32x4_mul(r0, inv_det)),
        la_simd_to_vec4(la_f32x4_mul(r1, inv_det)),
        la_simd_to_vec4(la_f32x4_mul(r2, inv_det)),
        la_simd_to_vec4(la_f32x4_mul(r3, inv_det))
    );
    return true;
}

LA_INLINE la_mat4 la_mat4_inv(la_mat4 m) {
    la_mat4 out;
    (void)la_mat4_inv_safe(m, &out);
    return out;
}

/* ============================================================================
 * 3D Affine Transform Builders
 * ============================================================================ */

LA_INLINE la_mat4 la_mat4_translate(la_vec3 t) {
    return la_mat4_from_cols(
        la_vec4_make(1.0f, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 1.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, 1.0f, 0.0f),
        la_vec4_make(t.x,  t.y,  t.z,  1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_scale(la_vec3 s) {
    return la_mat4_from_cols(
        la_vec4_make(s.x,  0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, s.y,  0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, s.z,  0.0f),
        la_vec4_make(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_rotate_x(float radians) {
    float s, c;
    la_sincosf(radians, &s, &c);
    return la_mat4_from_cols(
        la_vec4_make(1.0f, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f,  c,    s,   0.0f),
        la_vec4_make(0.0f, -s,    c,   0.0f),
        la_vec4_make(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_rotate_y(float radians) {
    float s, c;
    la_sincosf(radians, &s, &c);
    return la_mat4_from_cols(
        la_vec4_make( c,   0.0f, -s,   0.0f),
        la_vec4_make(0.0f, 1.0f, 0.0f, 0.0f),
        la_vec4_make( s,   0.0f,  c,   0.0f),
        la_vec4_make(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_rotate_z(float radians) {
    float s, c;
    la_sincosf(radians, &s, &c);
    return la_mat4_from_cols(
        la_vec4_make( c,    s,   0.0f, 0.0f),
        la_vec4_make(-s,    c,   0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, 1.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_rotate_axis(la_vec3 unit_axis, float radians) {
    return la_mat4_from_mat3(la_mat3_from_axis_angle(unit_axis, radians));
}

/* ============================================================================
 * Camera & Shader Projection Matrices (Explicit Handedness & Clip Depth Range)
 *   RH = Right-Handed (-Z forward), LH = Left-Handed (+Z forward)
 *   ZO = Zero-to-One depth [0, 1] (Vulkan / Metal / WGSL / DirectX)
 *   NO = Negative-One-to-One depth [-1, 1] (OpenGL)
 * ============================================================================ */

LA_INLINE la_mat4 la_mat4_look_at_rh(la_vec3 eye, la_vec3 target, la_vec3 up) {
    la_vec3 f = la_vec3_norm(la_vec3_sub(target, eye));
    la_vec3 s = la_vec3_norm(la_vec3_cross(f, up));
    la_vec3 u = la_vec3_cross(s, f);
    return la_mat4_from_cols(
        la_vec4_make(s.x, u.x, -f.x, 0.0f),
        la_vec4_make(s.y, u.y, -f.y, 0.0f),
        la_vec4_make(s.z, u.z, -f.z, 0.0f),
        la_vec4_make(-la_vec3_dot(s, eye), -la_vec3_dot(u, eye), la_vec3_dot(f, eye), 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_look_at_lh(la_vec3 eye, la_vec3 target, la_vec3 up) {
    la_vec3 f = la_vec3_norm(la_vec3_sub(target, eye));
    la_vec3 s = la_vec3_norm(la_vec3_cross(up, f));
    la_vec3 u = la_vec3_cross(f, s);
    return la_mat4_from_cols(
        la_vec4_make(s.x, u.x, f.x, 0.0f),
        la_vec4_make(s.y, u.y, f.y, 0.0f),
        la_vec4_make(s.z, u.z, f.z, 0.0f),
        la_vec4_make(-la_vec3_dot(s, eye), -la_vec3_dot(u, eye), -la_vec3_dot(f, eye), 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_perspective_rh_zo(float fovy_rad, float aspect, float z_near, float z_far) {
    float f = 1.0f / tanf(0.5f * fovy_rad);
    float inv_nf = 1.0f / (z_far - z_near);
    return la_mat4_from_cols(
        la_vec4_make(f / aspect, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, -z_far * inv_nf, -1.0f),
        la_vec4_make(0.0f, 0.0f, -(z_far * z_near) * inv_nf, 0.0f)
    );
}

LA_INLINE la_mat4 la_mat4_perspective_rh_no(float fovy_rad, float aspect, float z_near, float z_far) {
    float f = 1.0f / tanf(0.5f * fovy_rad);
    float inv_nf = 1.0f / (z_far - z_near);
    return la_mat4_from_cols(
        la_vec4_make(f / aspect, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, -(z_far + z_near) * inv_nf, -1.0f),
        la_vec4_make(0.0f, 0.0f, -(2.0f * z_far * z_near) * inv_nf, 0.0f)
    );
}

LA_INLINE la_mat4 la_mat4_perspective_lh_zo(float fovy_rad, float aspect, float z_near, float z_far) {
    float f = 1.0f / tanf(0.5f * fovy_rad);
    float inv_nf = 1.0f / (z_far - z_near);
    return la_mat4_from_cols(
        la_vec4_make(f / aspect, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, z_far * inv_nf, 1.0f),
        la_vec4_make(0.0f, 0.0f, -(z_far * z_near) * inv_nf, 0.0f)
    );
}

LA_INLINE la_mat4 la_mat4_perspective_lh_no(float fovy_rad, float aspect, float z_near, float z_far) {
    float f = 1.0f / tanf(0.5f * fovy_rad);
    float inv_nf = 1.0f / (z_far - z_near);
    return la_mat4_from_cols(
        la_vec4_make(f / aspect, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, (z_far + z_near) * inv_nf, 1.0f),
        la_vec4_make(0.0f, 0.0f, -(2.0f * z_far * z_near) * inv_nf, 0.0f)
    );
}

/* Infinite reverse-Z right-handed perspective: maps z = -z_near -> 1.0, z = -inf -> 0.0. */
LA_INLINE la_mat4 la_mat4_perspective_inf_revz_rh(float fovy_rad, float aspect, float z_near) {
    float f = 1.0f / tanf(0.5f * fovy_rad);
    return la_mat4_from_cols(
        la_vec4_make(f / aspect, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, 0.0f, -1.0f),
        la_vec4_make(0.0f, 0.0f, z_near, 0.0f)
    );
}

LA_INLINE la_mat4 la_mat4_ortho_rh_zo(float left, float right, float bottom, float top, float z_near, float z_far) {
    float inv_rl = 1.0f / (right - left);
    float inv_tb = 1.0f / (top - bottom);
    float inv_fn = 1.0f / (z_far - z_near);
    return la_mat4_from_cols(
        la_vec4_make(2.0f * inv_rl, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 2.0f * inv_tb, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, -inv_fn, 0.0f),
        la_vec4_make(-(right + left) * inv_rl, -(top + bottom) * inv_tb, -z_near * inv_fn, 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_ortho_rh_no(float left, float right, float bottom, float top, float z_near, float z_far) {
    float inv_rl = 1.0f / (right - left);
    float inv_tb = 1.0f / (top - bottom);
    float inv_fn = 1.0f / (z_far - z_near);
    return la_mat4_from_cols(
        la_vec4_make(2.0f * inv_rl, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 2.0f * inv_tb, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, -2.0f * inv_fn, 0.0f),
        la_vec4_make(-(right + left) * inv_rl, -(top + bottom) * inv_tb, -(z_far + z_near) * inv_fn, 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_ortho_lh_zo(float left, float right, float bottom, float top, float z_near, float z_far) {
    float inv_rl = 1.0f / (right - left);
    float inv_tb = 1.0f / (top - bottom);
    float inv_fn = 1.0f / (z_far - z_near);
    return la_mat4_from_cols(
        la_vec4_make(2.0f * inv_rl, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 2.0f * inv_tb, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, inv_fn, 0.0f),
        la_vec4_make(-(right + left) * inv_rl, -(top + bottom) * inv_tb, -z_near * inv_fn, 1.0f)
    );
}

LA_INLINE la_mat4 la_mat4_ortho_lh_no(float left, float right, float bottom, float top, float z_near, float z_far) {
    float inv_rl = 1.0f / (right - left);
    float inv_tb = 1.0f / (top - bottom);
    float inv_fn = 1.0f / (z_far - z_near);
    return la_mat4_from_cols(
        la_vec4_make(2.0f * inv_rl, 0.0f, 0.0f, 0.0f),
        la_vec4_make(0.0f, 2.0f * inv_tb, 0.0f, 0.0f),
        la_vec4_make(0.0f, 0.0f, 2.0f * inv_fn, 0.0f),
        la_vec4_make(-(right + left) * inv_rl, -(top + bottom) * inv_tb, -(z_far + z_near) * inv_fn, 1.0f)
    );
}

LA_INLINE bool la_mat4_approx_eq(la_mat4 a, la_mat4 b, float eps) {
    return la_vec4_approx_eq(a.cols[0], b.cols[0], eps) &&
           la_vec4_approx_eq(a.cols[1], b.cols[1], eps) &&
           la_vec4_approx_eq(a.cols[2], b.cols[2], eps) &&
           la_vec4_approx_eq(a.cols[3], b.cols[3], eps);
}

LA_EXTERN_C_END

#endif /* LA_MAT_H */
