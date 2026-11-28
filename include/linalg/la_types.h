/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_types.h
 * Purpose: Core POD vector, matrix, quaternion, geometry, and SoA/AoSoA types
 *          with explicit GPU/shader-compatible memory layouts and compile-time
 *          ABI verification.
 */

#ifndef LA_TYPES_H
#define LA_TYPES_H

#include "la_config.h"

LA_EXTERN_C_BEGIN

/* ============================================================================
 * Vector Types
 * ============================================================================ */

/* 2D vector: 8 bytes, 4-byte aligned (GLSL vec2 / Metal float2 compatible). */
typedef struct la_vec2 {
    float x;
    float y;
} la_vec2;

/* Tightly packed 3D vector: 12 bytes, 4-byte aligned (VBO / std430 storage). */
typedef struct la_vec3 {
    float x;
    float y;
    float z;
} la_vec3;

/*
 * SIMD-aligned 3D vector: 16 bytes, 16-byte aligned.
 * Invariant: _pad is maintained as 0.0f by library operations.
 * Matches GLSL std140 vec3 and Metal simd_float3.
 */
typedef struct LA_ALIGN(16) la_vec3a {
    float x;
    float y;
    float z;
    float _pad;
} la_vec3a;

/* 4D homogeneous vector: 16 bytes, 16-byte aligned (GLSL vec4 / Metal float4). */
typedef struct LA_ALIGN(16) la_vec4 {
    float x;
    float y;
    float z;
    float w;
} la_vec4;

/* ============================================================================
 * Quaternion Type (Scalar-Last: x, y, z, w)
 * Matches glTF, PhysX, Bullet, Jolt, and GLM memory convention.
 * ============================================================================ */

typedef struct LA_ALIGN(16) la_quat {
    float x;
    float y;
    float z;
    float w;
} la_quat;

/* ============================================================================
 * Matrix Types (Column-Major Storage)
 * Element (row, col) is stored in cols[col] at component row.
 * Matches GLSL, Vulkan, Metal, and WGSL with zero transpose on GPU upload.
 * ============================================================================ */

/* 2x2 column-major matrix: 16 bytes, 16-byte aligned (fits in one 128-bit reg). */
typedef struct LA_ALIGN(16) la_mat2 {
    la_vec2 cols[2];
} la_mat2;

/* Tightly packed 3x3 column-major matrix: 36 bytes, 4-byte aligned. */
typedef struct la_mat3 {
    la_vec3 cols[3];
} la_mat3;

/* SIMD-aligned 3x3 column-major matrix: 48 bytes, 16-byte aligned (GLSL std140 mat3 / Metal float3x3). */
typedef struct LA_ALIGN(16) la_mat3a {
    la_vec3a cols[3];
} la_mat3a;

/* 4x4 column-major matrix: 64 bytes (1 cache line), 16-byte aligned (GLSL/Metal mat4). */
typedef struct LA_ALIGN(16) la_mat4 {
    la_vec4 cols[4];
} la_mat4;

/* ============================================================================
 * Geometry & Physics Transform Primitives
 * ============================================================================ */

/* Drift-free TRS transform: 48 bytes, 16-byte aligned. */
typedef struct LA_ALIGN(16) la_transform {
    la_quat  rot;
    la_vec3a pos;
    la_vec3a scale;
} la_transform;

/* Axis-Aligned Bounding Box: 32 bytes, 16-byte aligned. */
typedef struct LA_ALIGN(16) la_aabb {
    la_vec3a min;
    la_vec3a max;
} la_aabb;

/* Plane equation n.x*px + n.y*py + n.z*pz + d = 0: 16 bytes, 16-byte aligned. */
typedef struct LA_ALIGN(16) la_plane {
    float nx;
    float ny;
    float nz;
    float d;
} la_plane;

/* Ray: 32 bytes, 16-byte aligned. */
typedef struct LA_ALIGN(16) la_ray {
    la_vec3a origin;
    la_vec3a dir;
} la_ray;

/* Frustum plane indices. */
typedef enum la_frustum_plane {
    LA_FRUSTUM_LEFT   = 0,
    LA_FRUSTUM_RIGHT  = 1,
    LA_FRUSTUM_BOTTOM = 2,
    LA_FRUSTUM_TOP    = 3,
    LA_FRUSTUM_NEAR   = 4,
    LA_FRUSTUM_FAR    = 5
} la_frustum_plane;

/* 6-plane view frustum: 96 bytes, 16-byte aligned. */
typedef struct LA_ALIGN(16) la_frustum {
    la_plane planes[6];
} la_frustum;

/* ============================================================================
 * Data-Oriented SoA & AoSoA Batch Types (Non-Owning Views & SIMD Blocks)
 * ============================================================================ */

typedef struct la_vec3_soa {
    float *LA_RESTRICT x;
    float *LA_RESTRICT y;
    float *LA_RESTRICT z;
} la_vec3_soa;

typedef struct la_vec3_soa_const {
    const float *LA_RESTRICT x;
    const float *LA_RESTRICT y;
    const float *LA_RESTRICT z;
} la_vec3_soa_const;

typedef struct la_vec4_soa {
    float *LA_RESTRICT x;
    float *LA_RESTRICT y;
    float *LA_RESTRICT z;
    float *LA_RESTRICT w;
} la_vec4_soa;

typedef struct la_vec4_soa_const {
    const float *LA_RESTRICT x;
    const float *LA_RESTRICT y;
    const float *LA_RESTRICT z;
    const float *LA_RESTRICT w;
} la_vec4_soa_const;

typedef struct la_quat_soa {
    float *LA_RESTRICT x;
    float *LA_RESTRICT y;
    float *LA_RESTRICT z;
    float *LA_RESTRICT w;
} la_quat_soa;

typedef struct la_quat_soa_const {
    const float *LA_RESTRICT x;
    const float *LA_RESTRICT y;
    const float *LA_RESTRICT z;
    const float *LA_RESTRICT w;
} la_quat_soa_const;

typedef struct la_aabb_soa {
    float *LA_RESTRICT min_x;
    float *LA_RESTRICT min_y;
    float *LA_RESTRICT min_z;
    float *LA_RESTRICT max_x;
    float *LA_RESTRICT max_y;
    float *LA_RESTRICT max_z;
} la_aabb_soa;

typedef struct la_aabb_soa_const {
    const float *LA_RESTRICT min_x;
    const float *LA_RESTRICT min_y;
    const float *LA_RESTRICT min_z;
    const float *LA_RESTRICT max_x;
    const float *LA_RESTRICT max_y;
    const float *LA_RESTRICT max_z;
} la_aabb_soa_const;

/* 4-wide AoSoA block (48 bytes, 16-byte aligned) for NEON / SSE / WASM SIMD. */
typedef struct LA_ALIGN(16) la_vec3_aosoa4 {
    float x[4];
    float y[4];
    float z[4];
} la_vec3_aosoa4;

/* 8-wide AoSoA block (96 bytes, 32-byte aligned) for AVX2 SIMD. */
typedef struct LA_ALIGN(32) la_vec3_aosoa8 {
    float x[8];
    float y[8];
    float z[8];
} la_vec3_aosoa8;

/* ============================================================================
 * Compile-Time ABI & Shader Layout Verification
 * ============================================================================ */

LA_STATIC_ASSERT(sizeof(la_vec2)        == 8,  la_vec2_size_8);
LA_STATIC_ASSERT(sizeof(la_vec3)        == 12, la_vec3_size_12);
LA_STATIC_ASSERT(sizeof(la_vec3a)       == 16, la_vec3a_size_16);
LA_STATIC_ASSERT(sizeof(la_vec4)        == 16, la_vec4_size_16);
LA_STATIC_ASSERT(sizeof(la_quat)        == 16, la_quat_size_16);
LA_STATIC_ASSERT(sizeof(la_mat2)        == 16, la_mat2_size_16);
LA_STATIC_ASSERT(sizeof(la_mat3)        == 36, la_mat3_size_36);
LA_STATIC_ASSERT(sizeof(la_mat3a)       == 48, la_mat3a_size_48);
LA_STATIC_ASSERT(sizeof(la_mat4)        == 64, la_mat4_size_64);
LA_STATIC_ASSERT(sizeof(la_transform)   == 48, la_transform_size_48);
LA_STATIC_ASSERT(sizeof(la_aabb)        == 32, la_aabb_size_32);
LA_STATIC_ASSERT(sizeof(la_plane)       == 16, la_plane_size_16);
LA_STATIC_ASSERT(sizeof(la_ray)         == 32, la_ray_size_32);
LA_STATIC_ASSERT(sizeof(la_frustum)     == 96, la_frustum_size_96);
LA_STATIC_ASSERT(sizeof(la_vec3_aosoa4) == 48, la_vec3_aosoa4_size_48);
LA_STATIC_ASSERT(sizeof(la_vec3_aosoa8) == 96, la_vec3_aosoa8_size_96);

LA_STATIC_ASSERT(LA_ALIGNOF(la_vec3a)       == 16, la_vec3a_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_vec4)        == 16, la_vec4_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_quat)        == 16, la_quat_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_mat2)        == 16, la_mat2_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_mat3a)       == 16, la_mat3a_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_mat4)        == 16, la_mat4_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_transform)   == 16, la_transform_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_aabb)        == 16, la_aabb_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_plane)       == 16, la_plane_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_ray)         == 16, la_ray_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_frustum)     == 16, la_frustum_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_vec3_aosoa4) == 16, la_vec3_aosoa4_align_16);
LA_STATIC_ASSERT(LA_ALIGNOF(la_vec3_aosoa8) == 32, la_vec3_aosoa8_align_32);

LA_EXTERN_C_END

#endif /* LA_TYPES_H */
