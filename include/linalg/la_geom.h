/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_geom.h
 * Purpose: TRS transforms, AABBs (with Arvo SIMD transform), planes, rays
 *          (slab AABB, sphere, plane, Moller-Trumbore triangle), and view
 *          frustum extraction & culling (ZO and NO depth ranges).
 */

#ifndef LA_GEOM_H
#define LA_GEOM_H

#include "la_quat.h"
#include <float.h>

LA_EXTERN_C_BEGIN

/* ============================================================================
 * la_transform (TRS: rot, pos, scale)
 * ============================================================================ */

LA_INLINE la_transform la_transform_identity(void) {
    la_transform t;
    t.rot = la_quat_identity();
    t.pos = la_vec3a_zero();
    t.scale = la_vec3a_splat(1.0f);
    return t;
}

LA_INLINE la_transform la_transform_make(la_vec3 pos, la_quat rot, la_vec3 scale) {
    la_transform t;
    t.rot = rot;
    t.pos = la_vec3_to_vec3a(pos);
    t.scale = la_vec3_to_vec3a(scale);
    return t;
}

LA_INLINE la_mat4 la_transform_to_mat4(la_transform t) {
    la_mat3 r = la_quat_to_mat3(t.rot);
    return la_mat4_from_cols(
        la_vec4_from_vec3(la_vec3_scale(r.cols[0], t.scale.x), 0.0f),
        la_vec4_from_vec3(la_vec3_scale(r.cols[1], t.scale.y), 0.0f),
        la_vec4_from_vec3(la_vec3_scale(r.cols[2], t.scale.z), 0.0f),
        la_vec4_make(t.pos.x, t.pos.y, t.pos.z, 1.0f)
    );
}

/* Decompose affine TRS matrix (handles negative determinant reflections). */
LA_INLINE la_transform la_transform_from_mat4(la_mat4 m) {
    la_transform t;
    t.pos = la_vec4_xyza(m.cols[3]);

    la_vec3 c0 = la_vec4_xyz(m.cols[0]);
    la_vec3 c1 = la_vec4_xyz(m.cols[1]);
    la_vec3 c2 = la_vec4_xyz(m.cols[2]);

    float sx = la_vec3_len(c0);
    float sy = la_vec3_len(c1);
    float sz = la_vec3_len(c2);

    la_mat3 rot_m = la_mat3_from_cols(
        (sx > LA_EPSILON) ? la_vec3_scale(c0, 1.0f / sx) : la_vec3_make(1.0f, 0.0f, 0.0f),
        (sy > LA_EPSILON) ? la_vec3_scale(c1, 1.0f / sy) : la_vec3_make(0.0f, 1.0f, 0.0f),
        (sz > LA_EPSILON) ? la_vec3_scale(c2, 1.0f / sz) : la_vec3_make(0.0f, 0.0f, 1.0f)
    );

    if (la_mat3_det(rot_m) < 0.0f) {
        sx = -sx;
        rot_m.cols[0] = la_vec3_neg(rot_m.cols[0]);
    }

    t.scale = la_vec3a_make(sx, sy, sz);
    t.rot = la_quat_from_mat3(rot_m);
    return t;
}

LA_INLINE la_vec3 la_transform_point3(la_transform t, la_vec3 p) {
    la_vec3 scaled = la_vec3_mul(p, la_vec3a_to_vec3(t.scale));
    return la_vec3_add(la_quat_rotate_vec3(t.rot, scaled), la_vec3a_to_vec3(t.pos));
}

LA_INLINE la_vec3 la_transform_dir3(la_transform t, la_vec3 d) {
    la_vec3 scaled = la_vec3_mul(d, la_vec3a_to_vec3(t.scale));
    return la_quat_rotate_vec3(t.rot, scaled);
}

/* Compose transforms (exact for rigid / uniform-scale parent). */
LA_INLINE la_transform la_transform_mul(la_transform parent, la_transform child) {
    la_transform out;
    out.rot = la_quat_norm_safe(la_quat_mul(parent.rot, child.rot));
    out.scale = la_vec3a_mul(parent.scale, child.scale);
    la_vec3 scaled_child_pos = la_vec3_mul(la_vec3a_to_vec3(child.pos), la_vec3a_to_vec3(parent.scale));
    out.pos = la_vec3_to_vec3a(la_vec3_add(la_quat_rotate_vec3(parent.rot, scaled_child_pos),
                                           la_vec3a_to_vec3(parent.pos)));
    return out;
}

/* Inverse of a rigid transform (assumes scale == (1, 1, 1)). */
LA_INLINE la_transform la_transform_inv_rigid(la_transform t) {
    la_transform inv;
    inv.rot = la_quat_conj(t.rot);
    inv.scale = la_vec3a_splat(1.0f);
    inv.pos = la_vec3_to_vec3a(la_vec3_neg(la_quat_rotate_vec3(inv.rot, la_vec3a_to_vec3(t.pos))));
    return inv;
}

LA_INLINE la_transform la_transform_lerp(la_transform a, la_transform b, float t) {
    la_transform out;
    out.rot = la_quat_nlerp(a.rot, b.rot, t);
    out.pos = la_vec3a_lerp(a.pos, b.pos, t);
    out.scale = la_vec3a_lerp(a.scale, b.scale, t);
    return out;
}

/* ============================================================================
 * la_aabb (Axis-Aligned Bounding Box)
 * ============================================================================ */

LA_INLINE la_aabb la_aabb_empty(void) {
    la_aabb b;
    b.min = la_vec3a_splat(FLT_MAX);
    b.max = la_vec3a_splat(-FLT_MAX);
    return b;
}

LA_INLINE la_aabb la_aabb_from_minmax(la_vec3 min_p, la_vec3 max_p) {
    la_aabb b;
    b.min = la_vec3_to_vec3a(min_p);
    b.max = la_vec3_to_vec3a(max_p);
    return b;
}

LA_INLINE la_aabb la_aabb_from_center_extents(la_vec3 center, la_vec3 half_extents) {
    la_vec3a c = la_vec3_to_vec3a(center);
    la_vec3a e = la_vec3_to_vec3a(half_extents);
    la_aabb b;
    b.min = la_vec3a_sub(c, e);
    b.max = la_vec3a_add(c, e);
    return b;
}

LA_INLINE la_vec3a la_aabb_center(la_aabb b) {
    return la_vec3a_scale(la_vec3a_add(b.min, b.max), 0.5f);
}

LA_INLINE la_vec3a la_aabb_extents(la_aabb b) {
    return la_vec3a_scale(la_vec3a_sub(b.max, b.min), 0.5f);
}

LA_INLINE la_vec3a la_aabb_size(la_aabb b) {
    return la_vec3a_sub(b.max, b.min);
}

LA_INLINE float la_aabb_surface_area(la_aabb b) {
    la_vec3a d = la_aabb_size(b);
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

LA_INLINE float la_aabb_volume(la_aabb b) {
    la_vec3a d = la_aabb_size(b);
    return d.x * d.y * d.z;
}

LA_INLINE la_aabb la_aabb_expand_point(la_aabb b, la_vec3 p) {
    la_vec3a pa = la_vec3_to_vec3a(p);
    la_aabb out;
    out.min = la_vec3a_min(b.min, pa);
    out.max = la_vec3a_max(b.max, pa);
    return out;
}

LA_INLINE la_aabb la_aabb_merge(la_aabb a, la_aabb b) {
    la_aabb out;
    out.min = la_vec3a_min(a.min, b.min);
    out.max = la_vec3a_max(a.max, b.max);
    return out;
}

LA_INLINE bool la_aabb_contains_point(la_aabb b, la_vec3 p) {
    la_f32x4 sp = la_vec3a_to_simd(la_vec3_to_vec3a(p));
    la_u32x4 ge = la_f32x4_cmpge(sp, la_vec3a_to_simd(b.min));
    la_u32x4 le = la_f32x4_cmple(sp, la_vec3a_to_simd(b.max));
    return (la_u32x4_movemask(la_u32x4_and(ge, le)) & 0x7u) == 0x7u;
}

LA_INLINE bool la_aabb_overlaps(la_aabb a, la_aabb b) {
    la_u32x4 c1 = la_f32x4_cmple(la_vec3a_to_simd(a.min), la_vec3a_to_simd(b.max));
    la_u32x4 c2 = la_f32x4_cmpge(la_vec3a_to_simd(a.max), la_vec3a_to_simd(b.min));
    return (la_u32x4_movemask(la_u32x4_and(c1, c2)) & 0x7u) == 0x7u;
}

LA_INLINE la_vec3a la_aabb_closest_point(la_aabb b, la_vec3a p) {
    return la_vec3a_clamp(p, b.min, b.max);
}

/*
 * Fast SIMD AABB transformation by affine mat4 (Arvo's center/extents method):
 *   center' = M * center
 *   extents' = |M_3x3| * extents
 */
LA_INLINE la_aabb la_aabb_transform_mat4(la_aabb b, la_mat4 m) {
    la_vec3a c = la_aabb_center(b);
    la_vec3a e = la_aabb_extents(b);

    la_f32x4 sc = la_vec3a_to_simd(c);
    la_f32x4 se = la_vec3a_to_simd(e);

    la_f32x4 m0 = la_vec4_to_simd(m.cols[0]);
    la_f32x4 m1 = la_vec4_to_simd(m.cols[1]);
    la_f32x4 m2 = la_vec4_to_simd(m.cols[2]);
    la_f32x4 m3 = la_vec4_to_simd(m.cols[3]);

    la_f32x4 new_c = la_f32x4_fma(m0, la_f32x4_splat_x(sc), m3);
    new_c = la_f32x4_fma(m1, la_f32x4_splat_y(sc), new_c);
    new_c = la_f32x4_fma(m2, la_f32x4_splat_z(sc), new_c);

    la_f32x4 new_e = la_f32x4_mul(la_f32x4_abs(m0), la_f32x4_splat_x(se));
    new_e = la_f32x4_fma(la_f32x4_abs(m1), la_f32x4_splat_y(se), new_e);
    new_e = la_f32x4_fma(la_f32x4_abs(m2), la_f32x4_splat_z(se), new_e);

    la_aabb out;
    out.min = la_simd_to_vec3a(la_f32x4_sub(new_c, new_e));
    out.max = la_simd_to_vec3a(la_f32x4_add(new_c, new_e));
    return out;
}

/* ============================================================================
 * la_plane (nx*x + ny*y + nz*z + d = 0)
 * ============================================================================ */

LA_INLINE la_plane la_plane_make(float nx, float ny, float nz, float d) {
    la_plane p; p.nx = nx; p.ny = ny; p.nz = nz; p.d = d; return p;
}

LA_INLINE la_plane la_plane_from_point_normal(la_vec3 point, la_vec3 unit_normal) {
    return la_plane_make(unit_normal.x, unit_normal.y, unit_normal.z, -la_vec3_dot(unit_normal, point));
}

LA_INLINE la_plane la_plane_from_points(la_vec3 a, la_vec3 b, la_vec3 c) {
    la_vec3 n = la_vec3_norm_safe(la_vec3_cross(la_vec3_sub(b, a), la_vec3_sub(c, a)));
    return la_plane_from_point_normal(a, n);
}

LA_INLINE la_plane la_plane_norm(la_plane p) {
    float lsq = p.nx * p.nx + p.ny * p.ny + p.nz * p.nz;
    if (LA_UNLIKELY(lsq <= LA_EPSILON_SQ)) {
        return la_plane_make(0.0f, 1.0f, 0.0f, 0.0f);
    }
    float inv_l = la_rsqrtf(lsq);
    return la_plane_make(p.nx * inv_l, p.ny * inv_l, p.nz * inv_l, p.d * inv_l);
}

LA_INLINE float la_plane_signed_dist(la_plane p, la_vec3 point) {
    return la_fmaf(p.nx, point.x, la_fmaf(p.ny, point.y, la_fmaf(p.nz, point.z, p.d)));
}

LA_INLINE la_vec3 la_plane_project_point(la_plane unit_p, la_vec3 point) {
    float dist = la_plane_signed_dist(unit_p, point);
    return la_vec3_fma(point, la_vec3_make(unit_p.nx, unit_p.ny, unit_p.nz), -dist);
}

/* ============================================================================
 * la_ray & Intersection Queries
 * ============================================================================ */

LA_INLINE la_ray la_ray_make(la_vec3 origin, la_vec3 dir) {
    la_ray r;
    r.origin = la_vec3_to_vec3a(origin);
    r.dir = la_vec3_to_vec3a(dir);
    return r;
}

LA_INLINE la_vec3 la_ray_point_at(la_ray ray, float t) {
    return la_vec3a_to_vec3(la_vec3a_fma(ray.origin, ray.dir, t));
}

LA_INLINE bool la_ray_intersect_plane(la_ray ray, la_plane plane, float *LA_RESTRICT out_t) {
    float denom = la_fmaf(plane.nx, ray.dir.x, la_fmaf(plane.ny, ray.dir.y, plane.nz * ray.dir.z));
    if (LA_UNLIKELY(la_absf(denom) <= LA_EPSILON)) {
        return false;
    }
    float t = -(la_plane_signed_dist(plane, la_vec3a_to_vec3(ray.origin))) / denom;
    if (t < 0.0f) {
        return false;
    }
    if (out_t) {
        *out_t = t;
    }
    return true;
}

LA_INLINE bool la_ray_intersect_sphere(la_ray ray, la_vec3 center, float radius,
                                       float *LA_RESTRICT out_t) {
    la_vec3 oc = la_vec3_sub(la_vec3a_to_vec3(ray.origin), center);
    la_vec3 d = la_vec3a_to_vec3(ray.dir);
    float a = la_vec3_dot(d, d);
    float half_b = la_vec3_dot(oc, d);
    float c = la_vec3_dot(oc, oc) - radius * radius;
    float disc = half_b * half_b - a * c;
    if (disc < 0.0f) {
        return false;
    }
    float sqrt_d = sqrtf(disc);
    float t = (-half_b - sqrt_d) / a;
    if (t < 0.0f) {
        t = (-half_b + sqrt_d) / a;
        if (t < 0.0f) {
            return false;
        }
    }
    if (out_t) {
        *out_t = t;
    }
    return true;
}

/* Branchless SIMD slab method for Ray vs AABB intersection. */
LA_INLINE bool la_ray_intersect_aabb(la_ray ray, la_aabb box,
                                     float t_min, float t_max,
                                     float *LA_RESTRICT out_t) {
    float inv_x = (la_absf(ray.dir.x) > 1e-20f) ? (1.0f / ray.dir.x) : (ray.dir.x >= 0.0f ? 1e20f : -1e20f);
    float inv_y = (la_absf(ray.dir.y) > 1e-20f) ? (1.0f / ray.dir.y) : (ray.dir.y >= 0.0f ? 1e20f : -1e20f);
    float inv_z = (la_absf(ray.dir.z) > 1e-20f) ? (1.0f / ray.dir.z) : (ray.dir.z >= 0.0f ? 1e20f : -1e20f);

    la_f32x4 inv_d = la_f32x4_set(inv_x, inv_y, inv_z, 0.0f);
    la_f32x4 orig = la_vec3a_to_simd(ray.origin);
    la_f32x4 t0 = la_f32x4_mul(la_f32x4_sub(la_vec3a_to_simd(box.min), orig), inv_d);
    la_f32x4 t1 = la_f32x4_mul(la_f32x4_sub(la_vec3a_to_simd(box.max), orig), inv_d);

    la_f32x4 tsmall = la_f32x4_min(t0, t1);
    la_f32x4 tbig = la_f32x4_max(t0, t1);

    float t_enter = la_maxf(t_min, la_maxf(la_f32x4_get_x(tsmall),
                                   la_maxf(la_f32x4_get_y(tsmall), la_f32x4_get_z(tsmall))));
    float t_exit  = la_minf(t_max, la_minf(la_f32x4_get_x(tbig),
                                   la_minf(la_f32x4_get_y(tbig), la_f32x4_get_z(tbig))));

    if (t_exit < t_enter) {
        return false;
    }
    if (out_t) {
        *out_t = t_enter;
    }
    return true;
}

/* Moller-Trumbore Ray vs Triangle intersection (returns barycentric u, v and distance t). */
LA_INLINE bool la_ray_intersect_triangle(la_ray ray, la_vec3 v0, la_vec3 v1, la_vec3 v2,
                                         float *LA_RESTRICT out_t,
                                         float *LA_RESTRICT out_u,
                                         float *LA_RESTRICT out_v) {
    la_vec3 e1 = la_vec3_sub(v1, v0);
    la_vec3 e2 = la_vec3_sub(v2, v0);
    la_vec3 d = la_vec3a_to_vec3(ray.dir);
    la_vec3 h = la_vec3_cross(d, e2);
    float a = la_vec3_dot(e1, h);
    if (LA_UNLIKELY(la_absf(a) <= LA_EPSILON)) {
        return false;
    }
    float f = 1.0f / a;
    la_vec3 s = la_vec3_sub(la_vec3a_to_vec3(ray.origin), v0);
    float u = f * la_vec3_dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    la_vec3 q = la_vec3_cross(s, e1);
    float v = f * la_vec3_dot(d, q);
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }
    float t = f * la_vec3_dot(e2, q);
    if (t < 0.0f) {
        return false;
    }
    if (out_t) *out_t = t;
    if (out_u) *out_u = u;
    if (out_v) *out_v = v;
    return true;
}

/* ============================================================================
 * la_frustum (Gribb-Hartmann Extraction & Fast Culling)
 * Inward-pointing normalized planes: signed_dist(plane, p) >= 0 means inside.
 * ============================================================================ */

LA_INLINE la_plane la_frustum_plane_from_vec4(la_vec4 v) {
    return la_plane_norm(la_plane_make(v.x, v.y, v.z, v.w));
}

/* Extract frustum from View-Projection matrix with [0, 1] depth (Vulkan/Metal/WGSL/DX). */
LA_INLINE la_frustum la_frustum_from_vp_zo(la_mat4 vp) {
    la_mat4 t = la_mat4_transpose(vp);
    la_vec4 r0 = t.cols[0], r1 = t.cols[1], r2 = t.cols[2], r3 = t.cols[3];
    la_frustum f;
    f.planes[LA_FRUSTUM_LEFT]   = la_frustum_plane_from_vec4(la_vec4_add(r3, r0));
    f.planes[LA_FRUSTUM_RIGHT]  = la_frustum_plane_from_vec4(la_vec4_sub(r3, r0));
    f.planes[LA_FRUSTUM_BOTTOM] = la_frustum_plane_from_vec4(la_vec4_add(r3, r1));
    f.planes[LA_FRUSTUM_TOP]    = la_frustum_plane_from_vec4(la_vec4_sub(r3, r1));
    f.planes[LA_FRUSTUM_NEAR]   = la_frustum_plane_from_vec4(r2);
    f.planes[LA_FRUSTUM_FAR]    = la_frustum_plane_from_vec4(la_vec4_sub(r3, r2));
    return f;
}

/* Extract frustum from View-Projection matrix with [-1, 1] depth (OpenGL). */
LA_INLINE la_frustum la_frustum_from_vp_no(la_mat4 vp) {
    la_mat4 t = la_mat4_transpose(vp);
    la_vec4 r0 = t.cols[0], r1 = t.cols[1], r2 = t.cols[2], r3 = t.cols[3];
    la_frustum f;
    f.planes[LA_FRUSTUM_LEFT]   = la_frustum_plane_from_vec4(la_vec4_add(r3, r0));
    f.planes[LA_FRUSTUM_RIGHT]  = la_frustum_plane_from_vec4(la_vec4_sub(r3, r0));
    f.planes[LA_FRUSTUM_BOTTOM] = la_frustum_plane_from_vec4(la_vec4_add(r3, r1));
    f.planes[LA_FRUSTUM_TOP]    = la_frustum_plane_from_vec4(la_vec4_sub(r3, r1));
    f.planes[LA_FRUSTUM_NEAR]   = la_frustum_plane_from_vec4(la_vec4_add(r3, r2));
    f.planes[LA_FRUSTUM_FAR]    = la_frustum_plane_from_vec4(la_vec4_sub(r3, r2));
    return f;
}

LA_INLINE bool la_frustum_test_point(la_frustum f, la_vec3 p) {
    for (int i = 0; i < 6; ++i) {
        if (la_plane_signed_dist(f.planes[i], p) < 0.0f) {
            return false;
        }
    }
    return true;
}

LA_INLINE bool la_frustum_test_sphere(la_frustum f, la_vec3 center, float radius) {
    for (int i = 0; i < 6; ++i) {
        if (la_plane_signed_dist(f.planes[i], center) < -radius) {
            return false;
        }
    }
    return true;
}

/* Fast center/extents AABB frustum test: returns true if AABB intersects or is inside frustum. */
LA_INLINE bool la_frustum_test_aabb(la_frustum f, la_aabb box) {
    la_vec3a c = la_aabb_center(box);
    la_vec3a e = la_aabb_extents(box);
    for (int i = 0; i < 6; ++i) {
        la_plane pl = f.planes[i];
        float dist = la_fmaf(pl.nx, c.x, la_fmaf(pl.ny, c.y, la_fmaf(pl.nz, c.z, pl.d)));
        float radius = la_absf(pl.nx) * e.x + la_absf(pl.ny) * e.y + la_absf(pl.nz) * e.z;
        if (dist < -radius) {
            return false;
        }
    }
    return true;
}

LA_EXTERN_C_END

#endif /* LA_GEOM_H */
