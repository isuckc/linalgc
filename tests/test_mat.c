#include "acutest.h"
#include "linalg/la_mat.h"

#define CHECK_NEAR(a, b, eps) \
    TEST_CHECK_(la_approx_eqf((a), (b), (eps)), \
        "Expected %.7f ~= %.7f", (double)(a), (double)(b))

static void test_mat2_and_mat3(void) {
    la_mat2 r = la_mat2_from_angle(LA_HALF_PI);
    la_vec2 v = la_mat2_mul_vec2(r, la_vec2_make(1.0f, 0.0f));
    TEST_CHECK(la_vec2_approx_eq(v, la_vec2_make(0.0f, 1.0f), 1e-5f));

    la_mat2 inv_r;
    TEST_CHECK(la_mat2_inv_safe(r, &inv_r));
    TEST_CHECK(la_mat2_approx_eq(la_mat2_mul(r, inv_r), la_mat2_identity(), 1e-5f));

    /* Skew matrix matches cross product */
    la_vec3 a = la_vec3_make(2.0f, -3.0f, 5.0f);
    la_vec3 b = la_vec3_make(-1.0f, 4.0f, 2.0f);
    TEST_CHECK(la_vec3_approx_eq(la_mat3_mul_vec3(la_mat3_skew(a), b), la_vec3_cross(a, b), 1e-5f));

    /* mat3 inverse */
    la_mat3 m3 = la_mat3_from_cols(
        la_vec3_make(2.0f, 1.0f, 0.0f),
        la_vec3_make(-1.0f, 3.0f, 2.0f),
        la_vec3_make(4.0f, 0.0f, 5.0f)
    );
    la_mat3 inv3;
    TEST_CHECK(la_mat3_inv_safe(m3, &inv3));
    TEST_CHECK(la_mat3_approx_eq(la_mat3_mul(m3, inv3), la_mat3_identity(), 1e-5f));

    /* Singular mat3 fails safely */
    la_mat3 sing = la_mat3_zero();
    TEST_CHECK(!la_mat3_inv_safe(sing, &inv3));

    /* Orthonormalize restores orthogonality */
    la_mat3 ortho = la_mat3_orthonormalize(m3);
    TEST_CHECK(la_mat3_approx_eq(la_mat3_mul(la_mat3_transpose(ortho), ortho), la_mat3_identity(), 1e-5f));
    CHECK_NEAR(la_mat3_det(ortho), 1.0f, 1e-5f);
}

static void test_mat4_transforms_and_inverses(void) {
    la_mat4 t = la_mat4_translate(la_vec3_make(3.0f, -4.0f, 5.0f));
    la_mat4 r = la_mat4_rotate_axis(la_vec3_norm(la_vec3_make(1.0f, 2.0f, 3.0f)), 0.85f);
    la_mat4 s = la_mat4_scale(la_vec3_make(2.0f, 0.5f, 3.0f));

    la_mat4 rigid = la_mat4_mul(t, r);
    la_mat4 rigid_inv = la_mat4_inv_rigid(rigid);
    TEST_CHECK(la_mat4_approx_eq(la_mat4_mul(rigid, rigid_inv), la_mat4_identity(), 1e-5f));

    la_mat4 affine = la_mat4_mul(rigid, s);
    la_mat4 affine_inv;
    TEST_CHECK(la_mat4_inv_affine_safe(affine, &affine_inv));
    TEST_CHECK(la_mat4_approx_eq(la_mat4_mul(affine, affine_inv), la_mat4_identity(), 1e-5f));

    /* General 4x4 inverse matches affine inverse and works on perspective matrices */
    la_mat4 gen_inv;
    TEST_CHECK(la_mat4_inv_safe(affine, &gen_inv));
    TEST_CHECK(la_mat4_approx_eq(gen_inv, affine_inv, 1e-5f));

    la_mat4 proj = la_mat4_perspective_rh_zo(la_deg2rad(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    la_mat4 proj_inv;
    TEST_CHECK(la_mat4_inv_safe(proj, &proj_inv));
    TEST_CHECK(la_mat4_approx_eq(la_mat4_mul(proj, proj_inv), la_mat4_identity(), 1e-4f));

    /* Normal matrix preserves orthogonality with transformed tangents */
    la_mat3a nmat = la_mat3a_normal_from_mat4(affine);
    la_vec3a n = la_vec3a_make(0.0f, 1.0f, 0.0f);
    la_vec3a tan = la_vec3a_make(1.0f, 0.0f, 0.0f);
    la_vec3a tn = la_mat3a_mul_vec3a(nmat, n);
    la_vec3a tt = la_mat4_mul_dir3a(affine, tan);
    CHECK_NEAR(la_vec3a_dot(tn, tt), 0.0f, 1e-5f);
}

static void test_shader_projections(void) {
    const float n = 0.5f, f = 100.0f;

    /* RH ZO: z = -n -> 0, z = -f -> 1 */
    la_mat4 p_rh_zo = la_mat4_perspective_rh_zo(LA_HALF_PI, 1.0f, n, f);
    CHECK_NEAR(la_mat4_project_point3(p_rh_zo, la_vec3_make(0.0f, 0.0f, -n)).z, 0.0f, 1e-5f);
    CHECK_NEAR(la_mat4_project_point3(p_rh_zo, la_vec3_make(0.0f, 0.0f, -f)).z, 1.0f, 1e-5f);

    /* RH NO: z = -n -> -1, z = -f -> +1 */
    la_mat4 p_rh_no = la_mat4_perspective_rh_no(LA_HALF_PI, 1.0f, n, f);
    CHECK_NEAR(la_mat4_project_point3(p_rh_no, la_vec3_make(0.0f, 0.0f, -n)).z, -1.0f, 1e-5f);
    CHECK_NEAR(la_mat4_project_point3(p_rh_no, la_vec3_make(0.0f, 0.0f, -f)).z,  1.0f, 1e-5f);

    /* LH ZO: z = +n -> 0, z = +f -> 1 */
    la_mat4 p_lh_zo = la_mat4_perspective_lh_zo(LA_HALF_PI, 1.0f, n, f);
    CHECK_NEAR(la_mat4_project_point3(p_lh_zo, la_vec3_make(0.0f, 0.0f, n)).z, 0.0f, 1e-5f);
    CHECK_NEAR(la_mat4_project_point3(p_lh_zo, la_vec3_make(0.0f, 0.0f, f)).z, 1.0f, 1e-5f);

    /* LH NO: z = +n -> -1, z = +f -> +1 */
    la_mat4 p_lh_no = la_mat4_perspective_lh_no(LA_HALF_PI, 1.0f, n, f);
    CHECK_NEAR(la_mat4_project_point3(p_lh_no, la_vec3_make(0.0f, 0.0f, n)).z, -1.0f, 1e-5f);
    CHECK_NEAR(la_mat4_project_point3(p_lh_no, la_vec3_make(0.0f, 0.0f, f)).z,  1.0f, 1e-5f);

    /* Infinite Reverse-Z RH: z = -n -> 1, large -z -> ~0 */
    la_mat4 p_revz = la_mat4_perspective_inf_revz_rh(LA_HALF_PI, 1.0f, n);
    CHECK_NEAR(la_mat4_project_point3(p_revz, la_vec3_make(0.0f, 0.0f, -n)).z, 1.0f, 1e-5f);
    CHECK_NEAR(la_mat4_project_point3(p_revz, la_vec3_make(0.0f, 0.0f, -100000.0f)).z, n / 100000.0f, 1e-5f);

    /* Ortho RH ZO & NO */
    la_mat4 o_zo = la_mat4_ortho_rh_zo(-10.0f, 10.0f, -5.0f, 5.0f, n, f);
    CHECK_NEAR(la_mat4_mul_point3(o_zo, la_vec3_make(0.0f, 0.0f, -n)).z, 0.0f, 1e-5f);
    CHECK_NEAR(la_mat4_mul_point3(o_zo, la_vec3_make(0.0f, 0.0f, -f)).z, 1.0f, 1e-5f);

    la_mat4 o_no = la_mat4_ortho_rh_no(-10.0f, 10.0f, -5.0f, 5.0f, n, f);
    CHECK_NEAR(la_mat4_mul_point3(o_no, la_vec3_make(0.0f, 0.0f, -n)).z, -1.0f, 1e-5f);
    CHECK_NEAR(la_mat4_mul_point3(o_no, la_vec3_make(0.0f, 0.0f, -f)).z,  1.0f, 1e-5f);

    /* Look-at RH maps target to -Z in view space */
    la_mat4 view = la_mat4_look_at_rh(la_vec3_make(0.0f, 0.0f, 5.0f), la_vec3_zero(), la_vec3_make(0.0f, 1.0f, 0.0f));
    la_vec3 target_vs = la_mat4_mul_point3(view, la_vec3_zero());
    TEST_CHECK(la_vec3_approx_eq(target_vs, la_vec3_make(0.0f, 0.0f, -5.0f), 1e-5f));
}

TEST_LIST = {
    { "mat2_and_mat3",               test_mat2_and_mat3 },
    { "mat4_transforms_and_inverses", test_mat4_transforms_and_inverses },
    { "shader_projections",          test_shader_projections },
    { NULL, NULL }
};
