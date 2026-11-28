/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_batch.h
 * Purpose: Data-Oriented SoA & AoSoA SIMD batch kernels for physics integration,
 *          bulk matrix/quaternion transforms, broadphase AABB queries, and
 *          skeletal linear blend skinning.
 *
 * Linkage: Controlled by LA_API / LA_SEPARATE_IMPL / LINALG_IMPLEMENTATION.
 */

#ifndef LA_BATCH_H
#define LA_BATCH_H

#include "la_geom.h"

LA_EXTERN_C_BEGIN

/* ============================================================================
 * Public Declarations
 * ============================================================================ */

LA_API void la_batch_vec3_aos_to_soa(la_vec3_soa dst, const la_vec3 *LA_RESTRICT src, size_t count);
LA_API void la_batch_vec3_soa_to_aos(la_vec3 *LA_RESTRICT dst, la_vec3_soa_const src, size_t count);

LA_API void la_batch_vec3_aos_to_aosoa4(la_vec3_aosoa4 *LA_RESTRICT dst, const la_vec3 *LA_RESTRICT src, size_t count);
LA_API void la_batch_vec3_aosoa4_to_aos(la_vec3 *LA_RESTRICT dst, const la_vec3_aosoa4 *LA_RESTRICT src, size_t count);

LA_API void la_batch_vec3_add_soa(la_vec3_soa dst, la_vec3_soa_const a, la_vec3_soa_const b, size_t count);
LA_API void la_batch_vec3_sub_soa(la_vec3_soa dst, la_vec3_soa_const a, la_vec3_soa_const b, size_t count);
LA_API void la_batch_vec3_scale_soa(la_vec3_soa dst, la_vec3_soa_const a, float s, size_t count);
LA_API void la_batch_vec3_fma_soa(la_vec3_soa dst, la_vec3_soa_const a, la_vec3_soa_const b, float s, size_t count);
LA_API void la_batch_vec3_dot_soa(float *LA_RESTRICT dst, la_vec3_soa_const a, la_vec3_soa_const b, size_t count);
LA_API void la_batch_vec3_cross_soa(la_vec3_soa dst, la_vec3_soa_const a, la_vec3_soa_const b, size_t count);
LA_API void la_batch_vec3_norm_safe_soa(la_vec3_soa dst, la_vec3_soa_const src, size_t count);

LA_API void la_batch_mat4_mul_points_soa(la_vec3_soa dst, la_vec3_soa_const src, la_mat4 m, size_t count);
LA_API void la_batch_mat4_mul_dirs_soa(la_vec3_soa dst, la_vec3_soa_const src, la_mat4 m, size_t count);
LA_API void la_batch_mat4_mul_points_aosoa4(la_vec3_aosoa4 *LA_RESTRICT dst,
                                            const la_vec3_aosoa4 *LA_RESTRICT src,
                                            la_mat4 m, size_t block_count);

LA_API void la_batch_quat_rotate_soa(la_vec3_soa dst, la_quat_soa_const q, la_vec3_soa_const v, size_t count);
LA_API void la_batch_quat_integrate_soa(la_quat_soa q, la_vec3_soa_const omega, float dt, size_t count);

LA_API void la_batch_integrate_symplectic_euler(la_vec3_soa pos, la_vec3_soa vel,
                                                la_vec3_soa_const accel, float dt, size_t count);

LA_API size_t la_batch_aabb_overlap_query(la_aabb query, la_aabb_soa_const boxes, size_t count,
                                          uint32_t *LA_RESTRICT out_indices, size_t max_out);

LA_API void la_batch_skin_lbs4(la_vec3 *LA_RESTRICT dst_pos,
                               la_vec3 *LA_RESTRICT dst_norm,
                               const la_vec3 *LA_RESTRICT src_pos,
                               const la_vec3 *LA_RESTRICT src_norm,
                               const uint16_t (*LA_RESTRICT bone_indices)[4],
                               const float (*LA_RESTRICT bone_weights)[4],
                               const la_mat4 *LA_RESTRICT bone_palette,
                               size_t vertex_count);

/* ============================================================================
 * Implementations
 * ============================================================================ */
#if LA_IMPL_ENABLED

LA_API void la_batch_vec3_aos_to_soa(la_vec3_soa dst, const la_vec3 *LA_RESTRICT src, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float tmp[12];
        memcpy(tmp, &src[i], sizeof(tmp));
        la_f32x4 vx, vy, vz;
        la_f32x4_load3_deinterleave(tmp, &vx, &vy, &vz);
        la_f32x4_storeu(&dst.x[i], vx);
        la_f32x4_storeu(&dst.y[i], vy);
        la_f32x4_storeu(&dst.z[i], vz);
    }
    for (; i < count; ++i) {
        dst.x[i] = src[i].x;
        dst.y[i] = src[i].y;
        dst.z[i] = src[i].z;
    }
}

LA_API void la_batch_vec3_soa_to_aos(la_vec3 *LA_RESTRICT dst, la_vec3_soa_const src, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        la_f32x4 vx = la_f32x4_loadu(&src.x[i]);
        la_f32x4 vy = la_f32x4_loadu(&src.y[i]);
        la_f32x4 vz = la_f32x4_loadu(&src.z[i]);
        float tmp[12];
        la_f32x4_store3_interleave(tmp, vx, vy, vz);
        memcpy(&dst[i], tmp, sizeof(tmp));
    }
    for (; i < count; ++i) {
        dst[i].x = src.x[i];
        dst[i].y = src.y[i];
        dst[i].z = src.z[i];
    }
}

LA_API void la_batch_vec3_aos_to_aosoa4(la_vec3_aosoa4 *LA_RESTRICT dst,
                                        const la_vec3 *LA_RESTRICT src, size_t count) {
    size_t b = 0;
    size_t i = 0;
    for (; i + 4 <= count; i += 4, ++b) {
        float tmp[12];
        memcpy(tmp, &src[i], sizeof(tmp));
        la_f32x4 vx, vy, vz;
        la_f32x4_load3_deinterleave(tmp, &vx, &vy, &vz);
        la_f32x4_store(dst[b].x, vx);
        la_f32x4_store(dst[b].y, vy);
        la_f32x4_store(dst[b].z, vz);
    }
    if (i < count) {
        memset(&dst[b], 0, sizeof(la_vec3_aosoa4));
        for (size_t lane = 0; i < count; ++i, ++lane) {
            dst[b].x[lane] = src[i].x;
            dst[b].y[lane] = src[i].y;
            dst[b].z[lane] = src[i].z;
        }
    }
}

LA_API void la_batch_vec3_aosoa4_to_aos(la_vec3 *LA_RESTRICT dst,
                                        const la_vec3_aosoa4 *LA_RESTRICT src, size_t count) {
    size_t b = 0;
    size_t i = 0;
    for (; i + 4 <= count; i += 4, ++b) {
        la_f32x4 vx = la_f32x4_load(src[b].x);
        la_f32x4 vy = la_f32x4_load(src[b].y);
        la_f32x4 vz = la_f32x4_load(src[b].z);
        float tmp[12];
        la_f32x4_store3_interleave(tmp, vx, vy, vz);
        memcpy(&dst[i], tmp, sizeof(tmp));
    }
    for (size_t lane = 0; i < count; ++i, ++lane) {
        dst[i].x = src[b].x[lane];
        dst[i].y = src[b].y[lane];
        dst[i].z = src[b].z[lane];
    }
}

LA_API void la_batch_vec3_add_soa(la_vec3_soa dst, la_vec3_soa_const a,
                                  la_vec3_soa_const b, size_t count) {
    size_t i = 0;
#if defined(LA_SIMD_AVX2)
    for (; i + 8 <= count; i += 8) {
        la_f32x8_storeu(&dst.x[i], la_f32x8_add(la_f32x8_loadu(&a.x[i]), la_f32x8_loadu(&b.x[i])));
        la_f32x8_storeu(&dst.y[i], la_f32x8_add(la_f32x8_loadu(&a.y[i]), la_f32x8_loadu(&b.y[i])));
        la_f32x8_storeu(&dst.z[i], la_f32x8_add(la_f32x8_loadu(&a.z[i]), la_f32x8_loadu(&b.z[i])));
    }
#endif
    for (; i + 4 <= count; i += 4) {
        la_f32x4_storeu(&dst.x[i], la_f32x4_add(la_f32x4_loadu(&a.x[i]), la_f32x4_loadu(&b.x[i])));
        la_f32x4_storeu(&dst.y[i], la_f32x4_add(la_f32x4_loadu(&a.y[i]), la_f32x4_loadu(&b.y[i])));
        la_f32x4_storeu(&dst.z[i], la_f32x4_add(la_f32x4_loadu(&a.z[i]), la_f32x4_loadu(&b.z[i])));
    }
    for (; i < count; ++i) {
        dst.x[i] = a.x[i] + b.x[i];
        dst.y[i] = a.y[i] + b.y[i];
        dst.z[i] = a.z[i] + b.z[i];
    }
}

LA_API void la_batch_vec3_sub_soa(la_vec3_soa dst, la_vec3_soa_const a,
                                  la_vec3_soa_const b, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        la_f32x4_storeu(&dst.x[i], la_f32x4_sub(la_f32x4_loadu(&a.x[i]), la_f32x4_loadu(&b.x[i])));
        la_f32x4_storeu(&dst.y[i], la_f32x4_sub(la_f32x4_loadu(&a.y[i]), la_f32x4_loadu(&b.y[i])));
        la_f32x4_storeu(&dst.z[i], la_f32x4_sub(la_f32x4_loadu(&a.z[i]), la_f32x4_loadu(&b.z[i])));
    }
    for (; i < count; ++i) {
        dst.x[i] = a.x[i] - b.x[i];
        dst.y[i] = a.y[i] - b.y[i];
        dst.z[i] = a.z[i] - b.z[i];
    }
}

LA_API void la_batch_vec3_scale_soa(la_vec3_soa dst, la_vec3_soa_const a, float s, size_t count) {
    size_t i = 0;
    la_f32x4 vs = la_f32x4_splat(s);
    for (; i + 4 <= count; i += 4) {
        la_f32x4_storeu(&dst.x[i], la_f32x4_mul(la_f32x4_loadu(&a.x[i]), vs));
        la_f32x4_storeu(&dst.y[i], la_f32x4_mul(la_f32x4_loadu(&a.y[i]), vs));
        la_f32x4_storeu(&dst.z[i], la_f32x4_mul(la_f32x4_loadu(&a.z[i]), vs));
    }
    for (; i < count; ++i) {
        dst.x[i] = a.x[i] * s;
        dst.y[i] = a.y[i] * s;
        dst.z[i] = a.z[i] * s;
    }
}

/* Computes dst[i] = a[i] + s * b[i] */
LA_API void la_batch_vec3_fma_soa(la_vec3_soa dst, la_vec3_soa_const a,
                                  la_vec3_soa_const b, float s, size_t count) {
    size_t i = 0;
#if defined(LA_SIMD_AVX2)
    la_f32x8 vs8 = la_f32x8_splat(s);
    for (; i + 8 <= count; i += 8) {
        la_f32x8_storeu(&dst.x[i], la_f32x8_fma(la_f32x8_loadu(&b.x[i]), vs8, la_f32x8_loadu(&a.x[i])));
        la_f32x8_storeu(&dst.y[i], la_f32x8_fma(la_f32x8_loadu(&b.y[i]), vs8, la_f32x8_loadu(&a.y[i])));
        la_f32x8_storeu(&dst.z[i], la_f32x8_fma(la_f32x8_loadu(&b.z[i]), vs8, la_f32x8_loadu(&a.z[i])));
    }
#endif
    la_f32x4 vs = la_f32x4_splat(s);
    for (; i + 4 <= count; i += 4) {
        la_f32x4_storeu(&dst.x[i], la_f32x4_fma(la_f32x4_loadu(&b.x[i]), vs, la_f32x4_loadu(&a.x[i])));
        la_f32x4_storeu(&dst.y[i], la_f32x4_fma(la_f32x4_loadu(&b.y[i]), vs, la_f32x4_loadu(&a.y[i])));
        la_f32x4_storeu(&dst.z[i], la_f32x4_fma(la_f32x4_loadu(&b.z[i]), vs, la_f32x4_loadu(&a.z[i])));
    }
    for (; i < count; ++i) {
        dst.x[i] = la_fmaf(b.x[i], s, a.x[i]);
        dst.y[i] = la_fmaf(b.y[i], s, a.y[i]);
        dst.z[i] = la_fmaf(b.z[i], s, a.z[i]);
    }
}

LA_API void la_batch_vec3_dot_soa(float *LA_RESTRICT dst, la_vec3_soa_const a,
                                  la_vec3_soa_const b, size_t count) {
    size_t i = 0;
#if defined(LA_SIMD_AVX2)
    for (; i + 8 <= count; i += 8) {
        la_f32x8 d = la_f32x8_mul(la_f32x8_loadu(&a.x[i]), la_f32x8_loadu(&b.x[i]));
        d = la_f32x8_fma(la_f32x8_loadu(&a.y[i]), la_f32x8_loadu(&b.y[i]), d);
        d = la_f32x8_fma(la_f32x8_loadu(&a.z[i]), la_f32x8_loadu(&b.z[i]), d);
        la_f32x8_storeu(&dst[i], d);
    }
#endif
    for (; i + 4 <= count; i += 4) {
        la_f32x4 d = la_f32x4_mul(la_f32x4_loadu(&a.x[i]), la_f32x4_loadu(&b.x[i]));
        d = la_f32x4_fma(la_f32x4_loadu(&a.y[i]), la_f32x4_loadu(&b.y[i]), d);
        d = la_f32x4_fma(la_f32x4_loadu(&a.z[i]), la_f32x4_loadu(&b.z[i]), d);
        la_f32x4_storeu(&dst[i], d);
    }
    for (; i < count; ++i) {
        dst[i] = la_fmaf(a.x[i], b.x[i], la_fmaf(a.y[i], b.y[i], a.z[i] * b.z[i]));
    }
}

LA_API void la_batch_vec3_cross_soa(la_vec3_soa dst, la_vec3_soa_const a,
                                    la_vec3_soa_const b, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        la_f32x4 ax = la_f32x4_loadu(&a.x[i]);
        la_f32x4 ay = la_f32x4_loadu(&a.y[i]);
        la_f32x4 az = la_f32x4_loadu(&a.z[i]);
        la_f32x4 bx = la_f32x4_loadu(&b.x[i]);
        la_f32x4 by = la_f32x4_loadu(&b.y[i]);
        la_f32x4 bz = la_f32x4_loadu(&b.z[i]);

        la_f32x4 rx = la_f32x4_fnma(az, by, la_f32x4_mul(ay, bz));
        la_f32x4 ry = la_f32x4_fnma(ax, bz, la_f32x4_mul(az, bx));
        la_f32x4 rz = la_f32x4_fnma(ay, bx, la_f32x4_mul(ax, by));

        la_f32x4_storeu(&dst.x[i], rx);
        la_f32x4_storeu(&dst.y[i], ry);
        la_f32x4_storeu(&dst.z[i], rz);
    }
    for (; i < count; ++i) {
        float ax = a.x[i], ay = a.y[i], az = a.z[i];
        float bx = b.x[i], by = b.y[i], bz = b.z[i];
        dst.x[i] = la_fmaf(ay, bz, -az * by);
        dst.y[i] = la_fmaf(az, bx, -ax * bz);
        dst.z[i] = la_fmaf(ax, by, -ay * bx);
    }
}

/* Branchless SIMD normalization with zero-length guard (never produces NaN). */
LA_API void la_batch_vec3_norm_safe_soa(la_vec3_soa dst, la_vec3_soa_const src, size_t count) {
    size_t i = 0;
    la_f32x4 eps = la_f32x4_splat(LA_EPSILON_SQ);
    la_f32x4 one = la_f32x4_splat(1.0f);
    la_f32x4 zero = la_f32x4_zero();
    for (; i + 4 <= count; i += 4) {
        la_f32x4 vx = la_f32x4_loadu(&src.x[i]);
        la_f32x4 vy = la_f32x4_loadu(&src.y[i]);
        la_f32x4 vz = la_f32x4_loadu(&src.z[i]);
        la_f32x4 lsq = la_f32x4_fma(vz, vz, la_f32x4_fma(vy, vy, la_f32x4_mul(vx, vx)));
        la_u32x4 valid = la_f32x4_cmpgt(lsq, eps);
        la_f32x4 safe_lsq = la_f32x4_select(valid, lsq, one);
        la_f32x4 inv_len = la_f32x4_select(valid, la_f32x4_div(one, la_f32x4_sqrt(safe_lsq)), zero);
        la_f32x4_storeu(&dst.x[i], la_f32x4_mul(vx, inv_len));
        la_f32x4_storeu(&dst.y[i], la_f32x4_mul(vy, inv_len));
        la_f32x4_storeu(&dst.z[i], la_f32x4_mul(vz, inv_len));
    }
    for (; i < count; ++i) {
        la_vec3 v = la_vec3_norm_safe(la_vec3_make(src.x[i], src.y[i], src.z[i]));
        dst.x[i] = v.x; dst.y[i] = v.y; dst.z[i] = v.z;
    }
}

/* Broadcast single column-major mat4 across N SoA 3D points (w = 1). */
LA_API void la_batch_mat4_mul_points_soa(la_vec3_soa dst, la_vec3_soa_const src,
                                         la_mat4 m, size_t count) {
    size_t i = 0;
#if defined(LA_SIMD_AVX2)
    la_f32x8 m00_8 = la_f32x8_splat(m.cols[0].x), m01_8 = la_f32x8_splat(m.cols[1].x), m02_8 = la_f32x8_splat(m.cols[2].x), m03_8 = la_f32x8_splat(m.cols[3].x);
    la_f32x8 m10_8 = la_f32x8_splat(m.cols[0].y), m11_8 = la_f32x8_splat(m.cols[1].y), m12_8 = la_f32x8_splat(m.cols[2].y), m13_8 = la_f32x8_splat(m.cols[3].y);
    la_f32x8 m20_8 = la_f32x8_splat(m.cols[0].z), m21_8 = la_f32x8_splat(m.cols[1].z), m22_8 = la_f32x8_splat(m.cols[2].z), m23_8 = la_f32x8_splat(m.cols[3].z);
    for (; i + 8 <= count; i += 8) {
        la_f32x8 x = la_f32x8_loadu(&src.x[i]);
        la_f32x8 y = la_f32x8_loadu(&src.y[i]);
        la_f32x8 z = la_f32x8_loadu(&src.z[i]);
        la_f32x8_storeu(&dst.x[i], la_f32x8_fma(m02_8, z, la_f32x8_fma(m01_8, y, la_f32x8_fma(m00_8, x, m03_8))));
        la_f32x8_storeu(&dst.y[i], la_f32x8_fma(m12_8, z, la_f32x8_fma(m11_8, y, la_f32x8_fma(m10_8, x, m13_8))));
        la_f32x8_storeu(&dst.z[i], la_f32x8_fma(m22_8, z, la_f32x8_fma(m21_8, y, la_f32x8_fma(m20_8, x, m23_8))));
    }
#endif
    la_f32x4 m00 = la_f32x4_splat(m.cols[0].x), m01 = la_f32x4_splat(m.cols[1].x), m02 = la_f32x4_splat(m.cols[2].x), m03 = la_f32x4_splat(m.cols[3].x);
    la_f32x4 m10 = la_f32x4_splat(m.cols[0].y), m11 = la_f32x4_splat(m.cols[1].y), m12 = la_f32x4_splat(m.cols[2].y), m13 = la_f32x4_splat(m.cols[3].y);
    la_f32x4 m20 = la_f32x4_splat(m.cols[0].z), m21 = la_f32x4_splat(m.cols[1].z), m22 = la_f32x4_splat(m.cols[2].z), m23 = la_f32x4_splat(m.cols[3].z);

    for (; i + 4 <= count; i += 4) {
        la_f32x4 x = la_f32x4_loadu(&src.x[i]);
        la_f32x4 y = la_f32x4_loadu(&src.y[i]);
        la_f32x4 z = la_f32x4_loadu(&src.z[i]);
        la_f32x4 ox = la_f32x4_fma(m02, z, la_f32x4_fma(m01, y, la_f32x4_fma(m00, x, m03)));
        la_f32x4 oy = la_f32x4_fma(m12, z, la_f32x4_fma(m11, y, la_f32x4_fma(m10, x, m13)));
        la_f32x4 oz = la_f32x4_fma(m22, z, la_f32x4_fma(m21, y, la_f32x4_fma(m20, x, m23)));
        la_f32x4_storeu(&dst.x[i], ox);
        la_f32x4_storeu(&dst.y[i], oy);
        la_f32x4_storeu(&dst.z[i], oz);
    }
    for (; i < count; ++i) {
        la_vec3 p = la_mat4_mul_point3(m, la_vec3_make(src.x[i], src.y[i], src.z[i]));
        dst.x[i] = p.x; dst.y[i] = p.y; dst.z[i] = p.z;
    }
}

LA_API void la_batch_mat4_mul_dirs_soa(la_vec3_soa dst, la_vec3_soa_const src,
                                       la_mat4 m, size_t count) {
    size_t i = 0;
    la_f32x4 m00 = la_f32x4_splat(m.cols[0].x), m01 = la_f32x4_splat(m.cols[1].x), m02 = la_f32x4_splat(m.cols[2].x);
    la_f32x4 m10 = la_f32x4_splat(m.cols[0].y), m11 = la_f32x4_splat(m.cols[1].y), m12 = la_f32x4_splat(m.cols[2].y);
    la_f32x4 m20 = la_f32x4_splat(m.cols[0].z), m21 = la_f32x4_splat(m.cols[1].z), m22 = la_f32x4_splat(m.cols[2].z);

    for (; i + 4 <= count; i += 4) {
        la_f32x4 x = la_f32x4_loadu(&src.x[i]);
        la_f32x4 y = la_f32x4_loadu(&src.y[i]);
        la_f32x4 z = la_f32x4_loadu(&src.z[i]);
        la_f32x4 ox = la_f32x4_fma(m02, z, la_f32x4_fma(m01, y, la_f32x4_mul(m00, x)));
        la_f32x4 oy = la_f32x4_fma(m12, z, la_f32x4_fma(m11, y, la_f32x4_mul(m10, x)));
        la_f32x4 oz = la_f32x4_fma(m22, z, la_f32x4_fma(m21, y, la_f32x4_mul(m20, x)));
        la_f32x4_storeu(&dst.x[i], ox);
        la_f32x4_storeu(&dst.y[i], oy);
        la_f32x4_storeu(&dst.z[i], oz);
    }
    for (; i < count; ++i) {
        la_vec3 d = la_mat4_mul_dir3(m, la_vec3_make(src.x[i], src.y[i], src.z[i]));
        dst.x[i] = d.x; dst.y[i] = d.y; dst.z[i] = d.z;
    }
}

/* Aligned AoSoA4 point transform: 100% SIMD lane utilization + single-cache-line locality. */
LA_API void la_batch_mat4_mul_points_aosoa4(la_vec3_aosoa4 *LA_RESTRICT dst,
                                            const la_vec3_aosoa4 *LA_RESTRICT src,
                                            la_mat4 m, size_t block_count) {
    la_f32x4 m00 = la_f32x4_splat(m.cols[0].x), m01 = la_f32x4_splat(m.cols[1].x), m02 = la_f32x4_splat(m.cols[2].x), m03 = la_f32x4_splat(m.cols[3].x);
    la_f32x4 m10 = la_f32x4_splat(m.cols[0].y), m11 = la_f32x4_splat(m.cols[1].y), m12 = la_f32x4_splat(m.cols[2].y), m13 = la_f32x4_splat(m.cols[3].y);
    la_f32x4 m20 = la_f32x4_splat(m.cols[0].z), m21 = la_f32x4_splat(m.cols[1].z), m22 = la_f32x4_splat(m.cols[2].z), m23 = la_f32x4_splat(m.cols[3].z);

    for (size_t b = 0; b < block_count; ++b) {
        la_f32x4 x = la_f32x4_load(src[b].x);
        la_f32x4 y = la_f32x4_load(src[b].y);
        la_f32x4 z = la_f32x4_load(src[b].z);
        la_f32x4_store(dst[b].x, la_f32x4_fma(m02, z, la_f32x4_fma(m01, y, la_f32x4_fma(m00, x, m03))));
        la_f32x4_store(dst[b].y, la_f32x4_fma(m12, z, la_f32x4_fma(m11, y, la_f32x4_fma(m10, x, m13))));
        la_f32x4_store(dst[b].z, la_f32x4_fma(m22, z, la_f32x4_fma(m21, y, la_f32x4_fma(m20, x, m23))));
    }
}

/* Vectorized SoA quaternion rotation: dst[i] = rotate(q[i], v[i]). */
LA_API void la_batch_quat_rotate_soa(la_vec3_soa dst, la_quat_soa_const q,
                                     la_vec3_soa_const v, size_t count) {
    size_t i = 0;
    la_f32x4 two = la_f32x4_splat(2.0f);
    for (; i + 4 <= count; i += 4) {
        la_f32x4 qx = la_f32x4_loadu(&q.x[i]);
        la_f32x4 qy = la_f32x4_loadu(&q.y[i]);
        la_f32x4 qz = la_f32x4_loadu(&q.z[i]);
        la_f32x4 qw = la_f32x4_loadu(&q.w[i]);

        la_f32x4 vx = la_f32x4_loadu(&v.x[i]);
        la_f32x4 vy = la_f32x4_loadu(&v.y[i]);
        la_f32x4 vz = la_f32x4_loadu(&v.z[i]);

        /* t = 2 * cross(q.xyz, v) */
        la_f32x4 tx = la_f32x4_mul(two, la_f32x4_fnma(qz, vy, la_f32x4_mul(qy, vz)));
        la_f32x4 ty = la_f32x4_mul(two, la_f32x4_fnma(qx, vz, la_f32x4_mul(qz, vx)));
        la_f32x4 tz = la_f32x4_mul(two, la_f32x4_fnma(qy, vx, la_f32x4_mul(qx, vy)));

        /* out = v + qw * t + cross(q.xyz, t) */
        la_f32x4 ox = la_f32x4_add(la_f32x4_fma(qw, tx, vx), la_f32x4_fnma(qz, ty, la_f32x4_mul(qy, tz)));
        la_f32x4 oy = la_f32x4_add(la_f32x4_fma(qw, ty, vy), la_f32x4_fnma(qx, tz, la_f32x4_mul(qz, tx)));
        la_f32x4 oz = la_f32x4_add(la_f32x4_fma(qw, tz, vz), la_f32x4_fnma(qy, tx, la_f32x4_mul(qx, ty)));

        la_f32x4_storeu(&dst.x[i], ox);
        la_f32x4_storeu(&dst.y[i], oy);
        la_f32x4_storeu(&dst.z[i], oz);
    }
    for (; i < count; ++i) {
        la_quat qi = la_quat_make(q.x[i], q.y[i], q.z[i], q.w[i]);
        la_vec3 vi = la_vec3_make(v.x[i], v.y[i], v.z[i]);
        la_vec3 r = la_quat_rotate_vec3(qi, vi);
        dst.x[i] = r.x; dst.y[i] = r.y; dst.z[i] = r.z;
    }
}

/* Batched first-order symplectic quaternion angular integration + branchless normalization. */
LA_API void la_batch_quat_integrate_soa(la_quat_soa q, la_vec3_soa_const omega,
                                        float dt, size_t count) {
    size_t i = 0;
    la_f32x4 hdt = la_f32x4_splat(0.5f * dt);
    la_f32x4 eps = la_f32x4_splat(LA_EPSILON_SQ);
    la_f32x4 one = la_f32x4_splat(1.0f);
    la_f32x4 zero = la_f32x4_zero();

    for (; i + 4 <= count; i += 4) {
        la_f32x4 qx = la_f32x4_loadu(&q.x[i]);
        la_f32x4 qy = la_f32x4_loadu(&q.y[i]);
        la_f32x4 qz = la_f32x4_loadu(&q.z[i]);
        la_f32x4 qw = la_f32x4_loadu(&q.w[i]);

        la_f32x4 wx = la_f32x4_loadu(&omega.x[i]);
        la_f32x4 wy = la_f32x4_loadu(&omega.y[i]);
        la_f32x4 wz = la_f32x4_loadu(&omega.z[i]);

        /* dq = [w, 0] * q */
        la_f32x4 dqx = la_f32x4_fnma(wz, qy, la_f32x4_fma(wy, qz, la_f32x4_mul(wx, qw)));
        la_f32x4 dqy = la_f32x4_fma(wz, qx, la_f32x4_fnma(wx, qz, la_f32x4_mul(wy, qw)));
        la_f32x4 dqz = la_f32x4_fnma(wy, qx, la_f32x4_fma(wx, qy, la_f32x4_mul(wz, qw)));
        la_f32x4 dqw = la_f32x4_neg(la_f32x4_fma(wz, qz, la_f32x4_fma(wy, qy, la_f32x4_mul(wx, qx))));

        la_f32x4 nx = la_f32x4_fma(dqx, hdt, qx);
        la_f32x4 ny = la_f32x4_fma(dqy, hdt, qy);
        la_f32x4 nz = la_f32x4_fma(dqz, hdt, qz);
        la_f32x4 nw = la_f32x4_fma(dqw, hdt, qw);

        la_f32x4 lsq = la_f32x4_fma(nw, nw, la_f32x4_fma(nz, nz, la_f32x4_fma(ny, ny, la_f32x4_mul(nx, nx))));
        la_u32x4 valid = la_f32x4_cmpgt(lsq, eps);
        la_f32x4 inv_len = la_f32x4_div(one, la_f32x4_sqrt(la_f32x4_select(valid, lsq, one)));

        la_f32x4_storeu(&q.x[i], la_f32x4_select(valid, la_f32x4_mul(nx, inv_len), zero));
        la_f32x4_storeu(&q.y[i], la_f32x4_select(valid, la_f32x4_mul(ny, inv_len), zero));
        la_f32x4_storeu(&q.z[i], la_f32x4_select(valid, la_f32x4_mul(nz, inv_len), zero));
        la_f32x4_storeu(&q.w[i], la_f32x4_select(valid, la_f32x4_mul(nw, inv_len), one));
    }
    for (; i < count; ++i) {
        la_quat qi = la_quat_make(q.x[i], q.y[i], q.z[i], q.w[i]);
        la_vec3 wi = la_vec3_make(omega.x[i], omega.y[i], omega.z[i]);
        la_quat r = la_quat_integrate_angvel(qi, wi, dt);
        q.x[i] = r.x; q.y[i] = r.y; q.z[i] = r.z; q.w[i] = r.w;
    }
}

/* Fused single-pass Symplectic Euler integrator: vel += accel * dt; pos += vel * dt; */
LA_API void la_batch_integrate_symplectic_euler(la_vec3_soa pos, la_vec3_soa vel,
                                                la_vec3_soa_const accel, float dt, size_t count) {
    size_t i = 0;
#if defined(LA_SIMD_AVX2)
    la_f32x8 vdt8 = la_f32x8_splat(dt);
    for (; i + 8 <= count; i += 8) {
        la_f32x8 vx = la_f32x8_fma(la_f32x8_loadu(&accel.x[i]), vdt8, la_f32x8_loadu(&vel.x[i]));
        la_f32x8 vy = la_f32x8_fma(la_f32x8_loadu(&accel.y[i]), vdt8, la_f32x8_loadu(&vel.y[i]));
        la_f32x8 vz = la_f32x8_fma(la_f32x8_loadu(&accel.z[i]), vdt8, la_f32x8_loadu(&vel.z[i]));
        la_f32x8_storeu(&vel.x[i], vx);
        la_f32x8_storeu(&vel.y[i], vy);
        la_f32x8_storeu(&vel.z[i], vz);
        la_f32x8_storeu(&pos.x[i], la_f32x8_fma(vx, vdt8, la_f32x8_loadu(&pos.x[i])));
        la_f32x8_storeu(&pos.y[i], la_f32x8_fma(vy, vdt8, la_f32x8_loadu(&pos.y[i])));
        la_f32x8_storeu(&pos.z[i], la_f32x8_fma(vz, vdt8, la_f32x8_loadu(&pos.z[i])));
    }
#endif
    la_f32x4 vdt = la_f32x4_splat(dt);
    for (; i + 4 <= count; i += 4) {
        la_f32x4 vx = la_f32x4_fma(la_f32x4_loadu(&accel.x[i]), vdt, la_f32x4_loadu(&vel.x[i]));
        la_f32x4 vy = la_f32x4_fma(la_f32x4_loadu(&accel.y[i]), vdt, la_f32x4_loadu(&vel.y[i]));
        la_f32x4 vz = la_f32x4_fma(la_f32x4_loadu(&accel.z[i]), vdt, la_f32x4_loadu(&vel.z[i]));
        la_f32x4_storeu(&vel.x[i], vx);
        la_f32x4_storeu(&vel.y[i], vy);
        la_f32x4_storeu(&vel.z[i], vz);
        la_f32x4_storeu(&pos.x[i], la_f32x4_fma(vx, vdt, la_f32x4_loadu(&pos.x[i])));
        la_f32x4_storeu(&pos.y[i], la_f32x4_fma(vy, vdt, la_f32x4_loadu(&pos.y[i])));
        la_f32x4_storeu(&pos.z[i], la_f32x4_fma(vz, vdt, la_f32x4_loadu(&pos.z[i])));
    }
    for (; i < count; ++i) {
        float vx = la_fmaf(accel.x[i], dt, vel.x[i]);
        float vy = la_fmaf(accel.y[i], dt, vel.y[i]);
        float vz = la_fmaf(accel.z[i], dt, vel.z[i]);
        vel.x[i] = vx; vel.y[i] = vy; vel.z[i] = vz;
        pos.x[i] = la_fmaf(vx, dt, pos.x[i]);
        pos.y[i] = la_fmaf(vy, dt, pos.y[i]);
        pos.z[i] = la_fmaf(vz, dt, pos.z[i]);
    }
}

/* Broadphase SIMD 1-vs-N AABB overlap sweep: appends hit indices up to max_out. */
LA_API size_t la_batch_aabb_overlap_query(la_aabb query, la_aabb_soa_const boxes, size_t count,
                                          uint32_t *LA_RESTRICT out_indices, size_t max_out) {
    size_t hits = 0;
    size_t i = 0;

    la_f32x4 qmin_x = la_f32x4_splat(query.min.x);
    la_f32x4 qmin_y = la_f32x4_splat(query.min.y);
    la_f32x4 qmin_z = la_f32x4_splat(query.min.z);
    la_f32x4 qmax_x = la_f32x4_splat(query.max.x);
    la_f32x4 qmax_y = la_f32x4_splat(query.max.y);
    la_f32x4 qmax_z = la_f32x4_splat(query.max.z);

    for (; i + 4 <= count; i += 4) {
        la_u32x4 mx = la_u32x4_and(
            la_f32x4_cmple(qmin_x, la_f32x4_loadu(&boxes.max_x[i])),
            la_f32x4_cmpge(qmax_x, la_f32x4_loadu(&boxes.min_x[i]))
        );
        la_u32x4 my = la_u32x4_and(
            la_f32x4_cmple(qmin_y, la_f32x4_loadu(&boxes.max_y[i])),
            la_f32x4_cmpge(qmax_y, la_f32x4_loadu(&boxes.min_y[i]))
        );
        la_u32x4 mz = la_u32x4_and(
            la_f32x4_cmple(qmin_z, la_f32x4_loadu(&boxes.max_z[i])),
            la_f32x4_cmpge(qmax_z, la_f32x4_loadu(&boxes.min_z[i]))
        );
        uint32_t mask = la_u32x4_movemask(la_u32x4_and(la_u32x4_and(mx, my), mz));
        if (LA_UNLIKELY(mask != 0u)) {
            for (uint32_t lane = 0; lane < 4; ++lane) {
                if ((mask & (1u << lane)) && hits < max_out) {
                    out_indices[hits++] = (uint32_t)(i + lane);
                }
            }
        }
    }
    for (; i < count; ++i) {
        bool hit = (query.min.x <= boxes.max_x[i] && query.max.x >= boxes.min_x[i]) &&
                   (query.min.y <= boxes.max_y[i] && query.max.y >= boxes.min_y[i]) &&
                   (query.min.z <= boxes.max_z[i] && query.max.z >= boxes.min_z[i]);
        if (hit && hits < max_out) {
            out_indices[hits++] = (uint32_t)i;
        }
    }
    return hits;
}

/* 4-bone Linear Blend Skinning (LBS4) over mat4 palette (dst_norm/src_norm may be NULL). */
LA_API void la_batch_skin_lbs4(la_vec3 *LA_RESTRICT dst_pos,
                               la_vec3 *LA_RESTRICT dst_norm,
                               const la_vec3 *LA_RESTRICT src_pos,
                               const la_vec3 *LA_RESTRICT src_norm,
                               const uint16_t (*LA_RESTRICT bone_indices)[4],
                               const float (*LA_RESTRICT bone_weights)[4],
                               const la_mat4 *LA_RESTRICT bone_palette,
                               size_t vertex_count) {
    bool do_norm = (dst_norm != NULL && src_norm != NULL);
    for (size_t i = 0; i < vertex_count; ++i) {
        const uint16_t *idx = bone_indices[i];
        const float *w = bone_weights[i];

        la_f32x4 w0 = la_f32x4_splat(w[0]);
        la_f32x4 w1 = la_f32x4_splat(w[1]);
        la_f32x4 w2 = la_f32x4_splat(w[2]);
        la_f32x4 w3 = la_f32x4_splat(w[3]);

        const la_mat4 *m0 = &bone_palette[idx[0]];
        const la_mat4 *m1 = &bone_palette[idx[1]];
        const la_mat4 *m2 = &bone_palette[idx[2]];
        const la_mat4 *m3 = &bone_palette[idx[3]];

        la_f32x4 c0 = la_f32x4_fma(la_vec4_to_simd(m3->cols[0]), w3,
                      la_f32x4_fma(la_vec4_to_simd(m2->cols[0]), w2,
                      la_f32x4_fma(la_vec4_to_simd(m1->cols[0]), w1,
                                   la_f32x4_mul(la_vec4_to_simd(m0->cols[0]), w0))));
        la_f32x4 c1 = la_f32x4_fma(la_vec4_to_simd(m3->cols[1]), w3,
                      la_f32x4_fma(la_vec4_to_simd(m2->cols[1]), w2,
                      la_f32x4_fma(la_vec4_to_simd(m1->cols[1]), w1,
                                   la_f32x4_mul(la_vec4_to_simd(m0->cols[1]), w0))));
        la_f32x4 c2 = la_f32x4_fma(la_vec4_to_simd(m3->cols[2]), w3,
                      la_f32x4_fma(la_vec4_to_simd(m2->cols[2]), w2,
                      la_f32x4_fma(la_vec4_to_simd(m1->cols[2]), w1,
                                   la_f32x4_mul(la_vec4_to_simd(m0->cols[2]), w0))));
        la_f32x4 c3 = la_f32x4_fma(la_vec4_to_simd(m3->cols[3]), w3,
                      la_f32x4_fma(la_vec4_to_simd(m2->cols[3]), w2,
                      la_f32x4_fma(la_vec4_to_simd(m1->cols[3]), w1,
                                   la_f32x4_mul(la_vec4_to_simd(m0->cols[3]), w0))));

        la_vec3 p = src_pos[i];
        la_f32x4 rp = la_f32x4_fma(c2, la_f32x4_splat(p.z),
                      la_f32x4_fma(c1, la_f32x4_splat(p.y),
                      la_f32x4_fma(c0, la_f32x4_splat(p.x), c3)));
        dst_pos[i] = la_vec4_xyz(la_simd_to_vec4(rp));

        if (do_norm) {
            la_vec3 n = src_norm[i];
            la_f32x4 rn = la_f32x4_fma(c2, la_f32x4_splat(n.z),
                          la_f32x4_fma(c1, la_f32x4_splat(n.y),
                                       la_f32x4_mul(c0, la_f32x4_splat(n.x))));
            dst_norm[i] = la_vec3_norm_safe(la_vec4_xyz(la_simd_to_vec4(rn)));
        }
    }
}

#endif /* LA_IMPL_ENABLED */

LA_EXTERN_C_END

#endif /* LA_BATCH_H */
