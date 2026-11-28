/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_quat.h
 * Purpose: Scalar-last quaternions (x, y, z, w), rotation conversions,
 *          shortest-arc construction, fast vector rotation, NLERP/SLERP,
 *          and rigid-body angular velocity integration.
 */

#ifndef LA_QUAT_H
#define LA_QUAT_H

#include "la_mat.h"

LA_EXTERN_C_BEGIN

LA_INLINE la_quat la_quat_make(float x, float y, float z, float w) {
    la_quat q; q.x = x; q.y = y; q.z = z; q.w = w; return q;
}

LA_INLINE la_quat la_quat_identity(void) {
    return la_quat_make(0.0f, 0.0f, 0.0f, 1.0f);
}

LA_INLINE float la_quat_dot(la_quat a, la_quat b) {
    return la_f32x4_dot4(la_quat_to_simd(a), la_quat_to_simd(b));
}

LA_INLINE float la_quat_len_sq(la_quat q) {
    return la_quat_dot(q, q);
}

LA_INLINE float la_quat_len(la_quat q) {
    return sqrtf(la_quat_len_sq(q));
}

LA_INLINE la_quat la_quat_scale(la_quat q, float s) {
    return la_simd_to_quat(la_f32x4_mul(la_quat_to_simd(q), la_f32x4_splat(s)));
}

LA_INLINE la_quat la_quat_neg(la_quat q) {
    return la_simd_to_quat(la_f32x4_neg(la_quat_to_simd(q)));
}

LA_INLINE la_quat la_quat_conj(la_quat q) {
    return la_quat_make(-q.x, -q.y, -q.z, q.w);
}

LA_INLINE la_quat la_quat_norm(la_quat q) {
    return la_quat_scale(q, la_rsqrtf(la_quat_len_sq(q)));
}

LA_INLINE la_quat la_quat_norm_fast(la_quat q) {
    return la_quat_scale(q, la_rsqrtf_fast(la_quat_len_sq(q)));
}

LA_INLINE la_quat la_quat_norm_safe(la_quat q) {
    float lsq = la_quat_len_sq(q);
    if (LA_UNLIKELY(lsq <= LA_EPSILON_SQ)) {
        return la_quat_identity();
    }
    return la_quat_scale(q, la_rsqrtf(lsq));
}

LA_INLINE la_quat la_quat_inv(la_quat q) {
    float lsq = la_quat_len_sq(q);
    if (LA_UNLIKELY(lsq <= LA_EPSILON_SQ)) {
        return la_quat_identity();
    }
    return la_quat_scale(la_quat_conj(q), 1.0f / lsq);
}

/* Hamilton product a * b (applies rotation b, then rotation a). */
LA_INLINE la_quat la_quat_mul(la_quat a, la_quat b) {
    return la_quat_make(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

/* Fast unit-quaternion rotation of 3D vector: v + 2 * q.xyz x (q.xyz x v + q.w * v). */
LA_INLINE la_vec3 la_quat_rotate_vec3(la_quat unit_q, la_vec3 v) {
    la_vec3 qv = la_vec3_make(unit_q.x, unit_q.y, unit_q.z);
    la_vec3 t = la_vec3_scale(la_vec3_cross(qv, v), 2.0f);
    return la_vec3_add(la_vec3_fma(v, t, unit_q.w), la_vec3_cross(qv, t));
}

LA_INLINE la_vec3a la_quat_rotate_vec3a(la_quat unit_q, la_vec3a v) {
    return la_vec3_to_vec3a(la_quat_rotate_vec3(unit_q, la_vec3a_to_vec3(v)));
}

/* Rotate vector by inverse of unit quaternion (conjugate rotation). */
LA_INLINE la_vec3 la_quat_unrotate_vec3(la_quat unit_q, la_vec3 v) {
    return la_quat_rotate_vec3(la_quat_conj(unit_q), v);
}

LA_INLINE la_quat la_quat_from_axis_angle(la_vec3 unit_axis, float radians) {
    float s, c;
    la_sincosf(0.5f * radians, &s, &c);
    return la_quat_make(unit_axis.x * s, unit_axis.y * s, unit_axis.z * s, c);
}

LA_INLINE void la_quat_to_axis_angle(la_quat unit_q,
                                     la_vec3 *LA_RESTRICT out_axis,
                                     float *LA_RESTRICT out_radians) {
    la_quat q = (unit_q.w < 0.0f) ? la_quat_neg(unit_q) : unit_q;
    float sin_half_sq = q.x * q.x + q.y * q.y + q.z * q.z;
    if (sin_half_sq <= LA_EPSILON_SQ) {
        *out_axis = la_vec3_make(1.0f, 0.0f, 0.0f);
        *out_radians = 0.0f;
        return;
    }
    float sin_half = sqrtf(sin_half_sq);
    float inv_sin = 1.0f / sin_half;
    *out_axis = la_vec3_make(q.x * inv_sin, q.y * inv_sin, q.z * inv_sin);
    *out_radians = 2.0f * atan2f(sin_half, q.w);
}

/* Extrinsic XYZ (roll_z * pitch_y * yaw_x = Qz * Qy * Qx). */
LA_INLINE la_quat la_quat_from_euler_xyz(float rx, float ry, float rz) {
    float sx, cx, sy, cy, sz, cz;
    la_sincosf(0.5f * rx, &sx, &cx);
    la_sincosf(0.5f * ry, &sy, &cy);
    la_sincosf(0.5f * rz, &sz, &cz);
    return la_quat_make(
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz
    );
}

/* Camera/FPS YXZ (Qy * Qx * Qz: yaw around Y, pitch around X, roll around Z). */
LA_INLINE la_quat la_quat_from_euler_yxz(float yaw_y, float pitch_x, float roll_z) {
    float sx, cx, sy, cy, sz, cz;
    la_sincosf(0.5f * pitch_x, &sx, &cx);
    la_sincosf(0.5f * yaw_y, &sy, &cy);
    la_sincosf(0.5f * roll_z, &sz, &cz);
    return la_quat_make(
        cy * sx * cz + sy * cx * sz,
        sy * cx * cz - cy * sx * sz,
        cy * cx * sz - sy * sx * cz,
        cy * cx * cz + sy * sx * sz
    );
}

/* Shortest-arc rotation mapping unit_from -> unit_to (handles 180-deg antiparallel case). */
LA_INLINE la_quat la_quat_from_unit_vecs(la_vec3 unit_from, la_vec3 unit_to) {
    float d = la_vec3_dot(unit_from, unit_to);
    if (LA_UNLIKELY(d < -1.0f + LA_EPSILON)) {
        la_vec3 t, b;
        la_vec3_orthonormal_basis(unit_from, &t, &b);
        return la_quat_make(t.x, t.y, t.z, 0.0f);
    }
    la_vec3 c = la_vec3_cross(unit_from, unit_to);
    return la_quat_norm_safe(la_quat_make(c.x, c.y, c.z, 1.0f + d));
}

LA_INLINE la_mat3 la_quat_to_mat3(la_quat q) {
    float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2;
    float yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
    float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;
    return la_mat3_from_cols(
        la_vec3_make(1.0f - (yy + zz), xy + wz,          xz - wy),
        la_vec3_make(xy - wz,          1.0f - (xx + zz), yz + wx),
        la_vec3_make(xz + wy,          yz - wx,          1.0f - (xx + yy))
    );
}

LA_INLINE la_mat4 la_quat_to_mat4(la_quat q) {
    return la_mat4_from_mat3(la_quat_to_mat3(q));
}

/* Shepperd's numerically stable rotation matrix -> quaternion conversion. */
LA_INLINE la_quat la_quat_from_mat3(la_mat3 m) {
    float m00 = m.cols[0].x, m10 = m.cols[0].y, m20 = m.cols[0].z;
    float m01 = m.cols[1].x, m11 = m.cols[1].y, m21 = m.cols[1].z;
    float m02 = m.cols[2].x, m12 = m.cols[2].y, m22 = m.cols[2].z;
    float trace = m00 + m11 + m22;
    la_quat q;

    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        float inv_s = 1.0f / s;
        q = la_quat_make((m21 - m12) * inv_s, (m02 - m20) * inv_s, (m10 - m01) * inv_s, 0.25f * s);
    } else if (m00 > m11 && m00 > m22) {
        float s = sqrtf(1.0f + m00 - m11 - m22) * 2.0f;
        float inv_s = 1.0f / s;
        q = la_quat_make(0.25f * s, (m01 + m10) * inv_s, (m02 + m20) * inv_s, (m21 - m12) * inv_s);
    } else if (m11 > m22) {
        float s = sqrtf(1.0f + m11 - m00 - m22) * 2.0f;
        float inv_s = 1.0f / s;
        q = la_quat_make((m01 + m10) * inv_s, 0.25f * s, (m12 + m21) * inv_s, (m02 - m20) * inv_s);
    } else {
        float s = sqrtf(1.0f + m22 - m00 - m11) * 2.0f;
        float inv_s = 1.0f / s;
        q = la_quat_make((m02 + m20) * inv_s, (m12 + m21) * inv_s, 0.25f * s, (m10 - m01) * inv_s);
    }
    return la_quat_norm_safe(q);
}

/* Shortest-path normalized linear interpolation (fast, constant-path, non-constant velocity). */
LA_INLINE la_quat la_quat_nlerp(la_quat a, la_quat b, float t) {
    if (la_quat_dot(a, b) < 0.0f) {
        b = la_quat_neg(b);
    }
    la_f32x4 r = la_f32x4_lerp(la_quat_to_simd(a), la_quat_to_simd(b), la_f32x4_splat(t));
    return la_quat_norm_safe(la_simd_to_quat(r));
}

/* Shortest-path spherical linear interpolation (constant angular velocity). */
LA_INLINE la_quat la_quat_slerp(la_quat a, la_quat b, float t) {
    float cos_theta = la_quat_dot(a, b);
    if (cos_theta < 0.0f) {
        b = la_quat_neg(b);
        cos_theta = -cos_theta;
    }
    if (cos_theta > 0.9995f) {
        return la_quat_nlerp(a, b, t);
    }
    float theta = acosf(la_clampf(cos_theta, -1.0f, 1.0f));
    float inv_sin = 1.0f / sinf(theta);
    float w0 = sinf((1.0f - t) * theta) * inv_sin;
    float w1 = sinf(t * theta) * inv_sin;
    la_f32x4 r = la_f32x4_fma(
        la_quat_to_simd(b), la_f32x4_splat(w1),
        la_f32x4_mul(la_quat_to_simd(a), la_f32x4_splat(w0))
    );
    return la_quat_norm_safe(la_simd_to_quat(r));
}

/* First-order symplectic quaternion integration: q' = norm(q + 0.5*dt * [omega, 0] * q). */
LA_INLINE la_quat la_quat_integrate_angvel(la_quat q, la_vec3 omega, float dt) {
    la_quat wq = la_quat_mul(la_quat_make(omega.x, omega.y, omega.z, 0.0f), q);
    la_f32x4 r = la_f32x4_fma(la_quat_to_simd(wq), la_f32x4_splat(0.5f * dt), la_quat_to_simd(q));
    return la_quat_norm_safe(la_simd_to_quat(r));
}

/* Exact exponential-map quaternion integration (accurate even for high angular speeds). */
LA_INLINE la_quat la_quat_integrate_angvel_exp(la_quat q, la_vec3 omega, float dt) {
    float angle = la_vec3_len(omega) * dt;
    if (angle <= LA_EPSILON) {
        return la_quat_integrate_angvel(q, omega, dt);
    }
    la_vec3 axis = la_vec3_scale(omega, dt / angle);
    return la_quat_norm_safe(la_quat_mul(la_quat_from_axis_angle(axis, angle), q));
}

LA_INLINE bool la_quat_approx_eq(la_quat a, la_quat b, float eps) {
    return la_approx_eqf(a.x, b.x, eps) &&
           la_approx_eqf(a.y, b.y, eps) &&
           la_approx_eqf(a.z, b.z, eps) &&
           la_approx_eqf(a.w, b.w, eps);
}

/* Checks rotation equivalence (handles double cover q == -q). */
LA_INLINE bool la_quat_equiv_eq(la_quat a, la_quat b, float eps) {
    return la_quat_approx_eq(a, b, eps) || la_quat_approx_eq(a, la_quat_neg(b), eps);
}

LA_EXTERN_C_END

#endif /* LA_QUAT_H */
