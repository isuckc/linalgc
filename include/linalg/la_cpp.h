/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_cpp.h
 * Purpose: Zero-cost C++17 ergonomics layer: namespace la type aliases (100%
 *          ABI-identical to C99 POD structs), ADL-safe arithmetic operators,
 *          overloaded math functions, and structured binding support.
 *
 * Configuration:
 *   - Define LA_NO_CPP to omit the C++17 layer entirely.
 *   - Define LA_NO_CPP_OPERATORS to omit global arithmetic operator overloads.
 */

#ifndef LA_CPP_H
#define LA_CPP_H

#include "la_batch.h"
#include "la_physics.h"

#if defined(__cplusplus) && !defined(LA_NO_CPP)

#include <tuple>
#include <type_traits>

/* ============================================================================
 * ADL-Safe Arithmetic Operators on Prefixed POD Types
 * ============================================================================ */
#if !defined(LA_NO_CPP_OPERATORS)

/* --- la_vec2 --- */
LA_INLINE la_vec2 operator+(la_vec2 a, la_vec2 b) { return la_vec2_add(a, b); }
LA_INLINE la_vec2 operator-(la_vec2 a, la_vec2 b) { return la_vec2_sub(a, b); }
LA_INLINE la_vec2 operator-(la_vec2 v)            { return la_vec2_neg(v); }
LA_INLINE la_vec2 operator*(la_vec2 a, la_vec2 b) { return la_vec2_mul(a, b); }
LA_INLINE la_vec2 operator*(la_vec2 v, float s)   { return la_vec2_scale(v, s); }
LA_INLINE la_vec2 operator*(float s, la_vec2 v)   { return la_vec2_scale(v, s); }
LA_INLINE la_vec2 operator/(la_vec2 a, la_vec2 b) { return la_vec2_div(a, b); }
LA_INLINE la_vec2 operator/(la_vec2 v, float s)   { return la_vec2_scale(v, 1.0f / s); }
LA_INLINE la_vec2 &operator+=(la_vec2 &a, la_vec2 b) { a = la_vec2_add(a, b); return a; }
LA_INLINE la_vec2 &operator-=(la_vec2 &a, la_vec2 b) { a = la_vec2_sub(a, b); return a; }
LA_INLINE la_vec2 &operator*=(la_vec2 &a, float s)   { a = la_vec2_scale(a, s); return a; }
LA_INLINE la_vec2 &operator/=(la_vec2 &a, float s)   { a = la_vec2_scale(a, 1.0f / s); return a; }
LA_INLINE bool operator==(la_vec2 a, la_vec2 b)      { return a.x == b.x && a.y == b.y; }
LA_INLINE bool operator!=(la_vec2 a, la_vec2 b)      { return !(a == b); }

/* --- la_vec3 --- */
LA_INLINE la_vec3 operator+(la_vec3 a, la_vec3 b) { return la_vec3_add(a, b); }
LA_INLINE la_vec3 operator-(la_vec3 a, la_vec3 b) { return la_vec3_sub(a, b); }
LA_INLINE la_vec3 operator-(la_vec3 v)            { return la_vec3_neg(v); }
LA_INLINE la_vec3 operator*(la_vec3 a, la_vec3 b) { return la_vec3_mul(a, b); }
LA_INLINE la_vec3 operator*(la_vec3 v, float s)   { return la_vec3_scale(v, s); }
LA_INLINE la_vec3 operator*(float s, la_vec3 v)   { return la_vec3_scale(v, s); }
LA_INLINE la_vec3 operator/(la_vec3 a, la_vec3 b) { return la_vec3_div(a, b); }
LA_INLINE la_vec3 operator/(la_vec3 v, float s)   { return la_vec3_scale(v, 1.0f / s); }
LA_INLINE la_vec3 &operator+=(la_vec3 &a, la_vec3 b) { a = la_vec3_add(a, b); return a; }
LA_INLINE la_vec3 &operator-=(la_vec3 &a, la_vec3 b) { a = la_vec3_sub(a, b); return a; }
LA_INLINE la_vec3 &operator*=(la_vec3 &a, float s)   { a = la_vec3_scale(a, s); return a; }
LA_INLINE la_vec3 &operator/=(la_vec3 &a, float s)   { a = la_vec3_scale(a, 1.0f / s); return a; }
LA_INLINE bool operator==(la_vec3 a, la_vec3 b)      { return a.x == b.x && a.y == b.y && a.z == b.z; }
LA_INLINE bool operator!=(la_vec3 a, la_vec3 b)      { return !(a == b); }

/* --- la_vec3a --- */
LA_INLINE la_vec3a operator+(la_vec3a a, la_vec3a b) { return la_vec3a_add(a, b); }
LA_INLINE la_vec3a operator-(la_vec3a a, la_vec3a b) { return la_vec3a_sub(a, b); }
LA_INLINE la_vec3a operator-(la_vec3a v)             { return la_vec3a_neg(v); }
LA_INLINE la_vec3a operator*(la_vec3a a, la_vec3a b) { return la_vec3a_mul(a, b); }
LA_INLINE la_vec3a operator*(la_vec3a v, float s)    { return la_vec3a_scale(v, s); }
LA_INLINE la_vec3a operator*(float s, la_vec3a v)    { return la_vec3a_scale(v, s); }
LA_INLINE la_vec3a operator/(la_vec3a a, la_vec3a b) { return la_vec3a_div(a, b); }
LA_INLINE la_vec3a operator/(la_vec3a v, float s)    { return la_vec3a_scale(v, 1.0f / s); }
LA_INLINE la_vec3a &operator+=(la_vec3a &a, la_vec3a b) { a = la_vec3a_add(a, b); return a; }
LA_INLINE la_vec3a &operator-=(la_vec3a &a, la_vec3a b) { a = la_vec3a_sub(a, b); return a; }
LA_INLINE la_vec3a &operator*=(la_vec3a &a, float s)    { a = la_vec3a_scale(a, s); return a; }
LA_INLINE la_vec3a &operator/=(la_vec3a &a, float s)    { a = la_vec3a_scale(a, 1.0f / s); return a; }
LA_INLINE bool operator==(la_vec3a a, la_vec3a b)       { return a.x == b.x && a.y == b.y && a.z == b.z; }
LA_INLINE bool operator!=(la_vec3a a, la_vec3a b)       { return !(a == b); }

/* --- la_vec4 --- */
LA_INLINE la_vec4 operator+(la_vec4 a, la_vec4 b) { return la_vec4_add(a, b); }
LA_INLINE la_vec4 operator-(la_vec4 a, la_vec4 b) { return la_vec4_sub(a, b); }
LA_INLINE la_vec4 operator-(la_vec4 v)            { return la_vec4_neg(v); }
LA_INLINE la_vec4 operator*(la_vec4 a, la_vec4 b) { return la_vec4_mul(a, b); }
LA_INLINE la_vec4 operator*(la_vec4 v, float s)   { return la_vec4_scale(v, s); }
LA_INLINE la_vec4 operator*(float s, la_vec4 v)   { return la_vec4_scale(v, s); }
LA_INLINE la_vec4 operator/(la_vec4 a, la_vec4 b) { return la_vec4_div(a, b); }
LA_INLINE la_vec4 operator/(la_vec4 v, float s)   { return la_vec4_scale(v, 1.0f / s); }
LA_INLINE la_vec4 &operator+=(la_vec4 &a, la_vec4 b) { a = la_vec4_add(a, b); return a; }
LA_INLINE la_vec4 &operator-=(la_vec4 &a, la_vec4 b) { a = la_vec4_sub(a, b); return a; }
LA_INLINE la_vec4 &operator*=(la_vec4 &a, float s)   { a = la_vec4_scale(a, s); return a; }
LA_INLINE la_vec4 &operator/=(la_vec4 &a, float s)   { a = la_vec4_scale(a, 1.0f / s); return a; }
LA_INLINE bool operator==(la_vec4 a, la_vec4 b)      { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
LA_INLINE bool operator!=(la_vec4 a, la_vec4 b)      { return !(a == b); }

/* --- la_quat --- */
LA_INLINE la_quat  operator-(la_quat q)              { return la_quat_neg(q); }
LA_INLINE la_quat  operator*(la_quat a, la_quat b)   { return la_quat_mul(a, b); }
LA_INLINE la_vec3  operator*(la_quat q, la_vec3 v)   { return la_quat_rotate_vec3(q, v); }
LA_INLINE la_vec3a operator*(la_quat q, la_vec3a v)  { return la_quat_rotate_vec3a(q, v); }
LA_INLINE la_quat &operator*=(la_quat &a, la_quat b) { a = la_quat_mul(a, b); return a; }
LA_INLINE bool operator==(la_quat a, la_quat b)      { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
LA_INLINE bool operator!=(la_quat a, la_quat b)      { return !(a == b); }

/* --- Matrices & Transforms --- */
LA_INLINE la_mat2  operator*(la_mat2 a, la_mat2 b)   { return la_mat2_mul(a, b); }
LA_INLINE la_vec2  operator*(la_mat2 m, la_vec2 v)   { return la_mat2_mul_vec2(m, v); }

LA_INLINE la_mat3  operator+(la_mat3 a, la_mat3 b)   { return la_mat3_add(a, b); }
LA_INLINE la_mat3  operator-(la_mat3 a, la_mat3 b)   { return la_mat3_sub(a, b); }
LA_INLINE la_mat3  operator*(la_mat3 a, la_mat3 b)   { return la_mat3_mul(a, b); }
LA_INLINE la_vec3  operator*(la_mat3 m, la_vec3 v)   { return la_mat3_mul_vec3(m, v); }
LA_INLINE la_mat3  operator*(la_mat3 m, float s)     { return la_mat3_scale(m, s); }
LA_INLINE la_mat3  operator*(float s, la_mat3 m)     { return la_mat3_scale(m, s); }

LA_INLINE la_mat3a operator*(la_mat3a a, la_mat3a b) { return la_mat3a_mul(a, b); }
LA_INLINE la_vec3a operator*(la_mat3a m, la_vec3a v) { return la_mat3a_mul_vec3a(m, v); }

LA_INLINE la_mat4  operator*(la_mat4 a, la_mat4 b)   { return la_mat4_mul(a, b); }
LA_INLINE la_vec4  operator*(la_mat4 m, la_vec4 v)   { return la_mat4_mul_vec4(m, v); }
LA_INLINE la_mat4 &operator*=(la_mat4 &a, la_mat4 b) { a = la_mat4_mul(a, b); return a; }

LA_INLINE la_transform operator*(la_transform p, la_transform c) { return la_transform_mul(p, c); }
LA_INLINE la_vec3      operator*(la_transform t, la_vec3 p)      { return la_transform_point3(t, p); }

#endif /* !LA_NO_CPP_OPERATORS */

/* ============================================================================
 * C++17 Structured Bindings for la_vec3a (binds [x, y, z], ignoring _pad)
 * ============================================================================ */

template <std::size_t I>
LA_INLINE float &get(la_vec3a &v) noexcept {
    if constexpr (I == 0) return v.x;
    else if constexpr (I == 1) return v.y;
    else return v.z;
}

template <std::size_t I>
LA_INLINE const float &get(const la_vec3a &v) noexcept {
    if constexpr (I == 0) return v.x;
    else if constexpr (I == 1) return v.y;
    else return v.z;
}

template <std::size_t I>
LA_INLINE float &&get(la_vec3a &&v) noexcept {
    if constexpr (I == 0) return static_cast<float &&>(v.x);
    else if constexpr (I == 1) return static_cast<float &&>(v.y);
    else return static_cast<float &&>(v.z);
}

namespace std {
template <> struct tuple_size<la_vec3a> : integral_constant<size_t, 3> {};
template <size_t I> struct tuple_element<I, la_vec3a> { using type = float; };
}

/* ============================================================================
 * namespace la: Idiomatic C++17 Aliases & Overloaded Free Functions
 * ============================================================================ */

namespace la {

using vec2            = ::la_vec2;
using vec3            = ::la_vec3;
using vec3a           = ::la_vec3a;
using vec4            = ::la_vec4;
using quat            = ::la_quat;
using mat2            = ::la_mat2;
using mat3            = ::la_mat3;
using mat3a           = ::la_mat3a;
using mat4            = ::la_mat4;
using transform       = ::la_transform;
using aabb            = ::la_aabb;
using plane           = ::la_plane;
using ray             = ::la_ray;
using frustum         = ::la_frustum;
using vec3_soa        = ::la_vec3_soa;
using vec3_soa_const  = ::la_vec3_soa_const;
using vec4_soa        = ::la_vec4_soa;
using vec4_soa_const  = ::la_vec4_soa_const;
using quat_soa        = ::la_quat_soa;
using quat_soa_const  = ::la_quat_soa_const;
using aabb_soa        = ::la_aabb_soa;
using aabb_soa_const  = ::la_aabb_soa_const;
using vec3_aosoa4     = ::la_vec3_aosoa4;
using vec3_aosoa8     = ::la_vec3_aosoa8;

/* Guarantee zero-cost C/C++ ABI identity */
static_assert(std::is_standard_layout_v<vec3> && std::is_trivially_copyable_v<vec3>, "la::vec3 must be POD");
static_assert(std::is_standard_layout_v<vec3a> && std::is_trivially_copyable_v<vec3a>, "la::vec3a must be POD");
static_assert(std::is_standard_layout_v<vec4> && std::is_trivially_copyable_v<vec4>, "la::vec4 must be POD");
static_assert(std::is_standard_layout_v<quat> && std::is_trivially_copyable_v<quat>, "la::quat must be POD");
static_assert(std::is_standard_layout_v<mat4> && std::is_trivially_copyable_v<mat4>, "la::mat4 must be POD");

/* Dot, cross, length, normalize */
LA_INLINE float dot(vec2 a, vec2 b)   { return la_vec2_dot(a, b); }
LA_INLINE float dot(vec3 a, vec3 b)   { return la_vec3_dot(a, b); }
LA_INLINE float dot(vec3a a, vec3a b) { return la_vec3a_dot(a, b); }
LA_INLINE float dot(vec4 a, vec4 b)   { return la_vec4_dot(a, b); }
LA_INLINE float dot(quat a, quat b)   { return la_quat_dot(a, b); }

LA_INLINE float cross(vec2 a, vec2 b)   { return la_vec2_cross(a, b); }
LA_INLINE vec3  cross(vec3 a, vec3 b)   { return la_vec3_cross(a, b); }
LA_INLINE vec3a cross(vec3a a, vec3a b) { return la_vec3a_cross(a, b); }

LA_INLINE float length(vec2 v)   { return la_vec2_len(v); }
LA_INLINE float length(vec3 v)   { return la_vec3_len(v); }
LA_INLINE float length(vec3a v)  { return la_vec3a_len(v); }
LA_INLINE float length(vec4 v)   { return la_vec4_len(v); }
LA_INLINE float length(quat q)   { return la_quat_len(q); }

LA_INLINE float length_sq(vec2 v)  { return la_vec2_len_sq(v); }
LA_INLINE float length_sq(vec3 v)  { return la_vec3_len_sq(v); }
LA_INLINE float length_sq(vec3a v) { return la_vec3a_len_sq(v); }
LA_INLINE float length_sq(vec4 v)  { return la_vec4_len_sq(v); }

LA_INLINE vec2  normalize(vec2 v)  { return la_vec2_norm(v); }
LA_INLINE vec3  normalize(vec3 v)  { return la_vec3_norm(v); }
LA_INLINE vec3a normalize(vec3a v) { return la_vec3a_norm(v); }
LA_INLINE vec4  normalize(vec4 v)  { return la_vec4_norm(v); }
LA_INLINE quat  normalize(quat q)  { return la_quat_norm(q); }

LA_INLINE vec2  normalize_safe(vec2 v)  { return la_vec2_norm_safe(v); }
LA_INLINE vec3  normalize_safe(vec3 v)  { return la_vec3_norm_safe(v); }
LA_INLINE vec3a normalize_safe(vec3a v) { return la_vec3a_norm_safe(v); }
LA_INLINE vec4  normalize_safe(vec4 v)  { return la_vec4_norm_safe(v); }
LA_INLINE quat  normalize_safe(quat q)  { return la_quat_norm_safe(q); }

LA_INLINE float lerp(float a, float b, float t) { return la_lerpf(a, b, t); }
LA_INLINE vec2  lerp(vec2 a, vec2 b, float t)   { return la_vec2_lerp(a, b, t); }
LA_INLINE vec3  lerp(vec3 a, vec3 b, float t)   { return la_vec3_lerp(a, b, t); }
LA_INLINE vec3a lerp(vec3a a, vec3a b, float t) { return la_vec3a_lerp(a, b, t); }
LA_INLINE vec4  lerp(vec4 a, vec4 b, float t)   { return la_vec4_lerp(a, b, t); }
LA_INLINE quat  nlerp(quat a, quat b, float t)  { return la_quat_nlerp(a, b, t); }
LA_INLINE quat  slerp(quat a, quat b, float t)  { return la_quat_slerp(a, b, t); }

/* Matrices */
LA_INLINE mat2  transpose(mat2 m)   { return la_mat2_transpose(m); }
LA_INLINE mat3  transpose(mat3 m)   { return la_mat3_transpose(m); }
LA_INLINE mat4  transpose(mat4 m)   { return la_mat4_transpose(m); }
LA_INLINE mat2  inverse(mat2 m)     { return la_mat2_inv(m); }
LA_INLINE mat3  inverse(mat3 m)     { return la_mat3_inv(m); }
LA_INLINE mat4  inverse(mat4 m)     { return la_mat4_inv(m); }
LA_INLINE mat4  inverse_rigid(mat4 m)  { return la_mat4_inv_rigid(m); }
LA_INLINE mat4  inverse_affine(mat4 m) { return la_mat4_inv_affine(m); }
LA_INLINE float determinant(mat2 m) { return la_mat2_det(m); }
LA_INLINE float determinant(mat3 m) { return la_mat3_det(m); }
LA_INLINE float determinant(mat4 m) { return la_mat4_det(m); }
LA_INLINE mat3a normal_matrix(mat4 m) { return la_mat3a_normal_from_mat4(m); }

/* Quaternions & Transforms */
LA_INLINE quat  conjugate(quat q) { return la_quat_conj(q); }
LA_INLINE quat  inverse(quat q)   { return la_quat_inv(q); }
LA_INLINE vec3  rotate(quat q, vec3 v)   { return la_quat_rotate_vec3(q, v); }
LA_INLINE vec3a rotate(quat q, vec3a v)  { return la_quat_rotate_vec3a(q, v); }
LA_INLINE mat3  to_mat3(quat q)          { return la_quat_to_mat3(q); }
LA_INLINE mat4  to_mat4(quat q)          { return la_quat_to_mat4(q); }
LA_INLINE mat4  to_mat4(transform t)     { return la_transform_to_mat4(t); }

/* Projections & Camera */
LA_INLINE mat4 translate(vec3 t) { return la_mat4_translate(t); }
LA_INLINE mat4 scale(vec3 s)     { return la_mat4_scale(s); }
LA_INLINE mat4 look_at_rh(vec3 eye, vec3 target, vec3 up) { return la_mat4_look_at_rh(eye, target, up); }
LA_INLINE mat4 look_at_lh(vec3 eye, vec3 target, vec3 up) { return la_mat4_look_at_lh(eye, target, up); }
LA_INLINE mat4 perspective_rh_zo(float fovy, float aspect, float zn, float zf) { return la_mat4_perspective_rh_zo(fovy, aspect, zn, zf); }
LA_INLINE mat4 perspective_rh_no(float fovy, float aspect, float zn, float zf) { return la_mat4_perspective_rh_no(fovy, aspect, zn, zf); }
LA_INLINE mat4 perspective_lh_zo(float fovy, float aspect, float zn, float zf) { return la_mat4_perspective_lh_zo(fovy, aspect, zn, zf); }
LA_INLINE mat4 perspective_lh_no(float fovy, float aspect, float zn, float zf) { return la_mat4_perspective_lh_no(fovy, aspect, zn, zf); }
LA_INLINE mat4 perspective_inf_revz_rh(float fovy, float aspect, float zn) { return la_mat4_perspective_inf_revz_rh(fovy, aspect, zn); }
LA_INLINE mat4 ortho_rh_zo(float l, float r, float b, float t, float zn, float zf) { return la_mat4_ortho_rh_zo(l, r, b, t, zn, zf); }
LA_INLINE mat4 ortho_rh_no(float l, float r, float b, float t, float zn, float zf) { return la_mat4_ortho_rh_no(l, r, b, t, zn, zf); }

/* Approximate equality */
LA_INLINE bool approx_eq(float a, float b, float eps = LA_EPSILON) { return la_approx_eqf(a, b, eps); }
LA_INLINE bool approx_eq(vec2 a, vec2 b, float eps = LA_EPSILON)   { return la_vec2_approx_eq(a, b, eps); }
LA_INLINE bool approx_eq(vec3 a, vec3 b, float eps = LA_EPSILON)   { return la_vec3_approx_eq(a, b, eps); }
LA_INLINE bool approx_eq(vec3a a, vec3a b, float eps = LA_EPSILON) { return la_vec3a_approx_eq(a, b, eps); }
LA_INLINE bool approx_eq(vec4 a, vec4 b, float eps = LA_EPSILON)   { return la_vec4_approx_eq(a, b, eps); }
LA_INLINE bool approx_eq(mat3 a, mat3 b, float eps = LA_EPSILON)   { return la_mat3_approx_eq(a, b, eps); }
LA_INLINE bool approx_eq(mat4 a, mat4 b, float eps = LA_EPSILON)   { return la_mat4_approx_eq(a, b, eps); }
LA_INLINE bool approx_eq(quat a, quat b, float eps = LA_EPSILON)   { return la_quat_equiv_eq(a, b, eps); }

} /* namespace la */

#endif /* defined(__cplusplus) && !defined(LA_NO_CPP) */

#endif /* LA_CPP_H */
