#include "acutest.h"
#include "linalg/la_quat.h"

#define CHECK_NEAR(a, b, eps) \
    TEST_CHECK_(la_approx_eqf((a), (b), (eps)), \
        "Expected %.7f ~= %.7f", (double)(a), (double)(b))

static void test_quat_rotation_and_matrix_roundtrip(void) {
    la_vec3 axis = la_vec3_norm(la_vec3_make(1.0f, 2.0f, -1.5f));
    float angle = 1.23f;

    la_quat q = la_quat_from_axis_angle(axis, angle);
    CHECK_NEAR(la_quat_len(q), 1.0f, 1e-6f);

    /* Quaternion rotation matches mat3 rotation */
    la_mat3 m_q = la_quat_to_mat3(q);
    la_mat3 m_a = la_mat3_from_axis_angle(axis, angle);
    TEST_CHECK(la_mat3_approx_eq(m_q, m_a, 1e-5f));

    la_vec3 v = la_vec3_make(3.0f, -2.0f, 5.0f);
    la_vec3 rv_q = la_quat_rotate_vec3(q, v);
    la_vec3 rv_m = la_mat3_mul_vec3(m_a, v);
    TEST_CHECK(la_vec3_approx_eq(rv_q, rv_m, 1e-5f));

    /* Unrotate restores original vector */
    TEST_CHECK(la_vec3_approx_eq(la_quat_unrotate_vec3(q, rv_q), v, 1e-5f));

    /* Shepperd mat3 -> quat roundtrip across multiple axes/angles */
    const float angles[] = { 0.0f, 0.7f, LA_HALF_PI, 2.4f, LA_PI - 1e-3f };
    const la_vec3 axes[] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f },
        { 0.57735027f, -0.57735027f, 0.57735027f }
    };
    for (size_t i = 0; i < sizeof(axes) / sizeof(axes[0]); ++i) {
        for (size_t j = 0; j < sizeof(angles) / sizeof(angles[0]); ++j) {
            la_quat q0 = la_quat_from_axis_angle(axes[i], angles[j]);
            la_quat q1 = la_quat_from_mat3(la_quat_to_mat3(q0));
            TEST_CHECK(la_quat_equiv_eq(q0, q1, 1e-4f));
        }
    }

    /* Axis-angle roundtrip */
    la_vec3 out_axis;
    float out_angle;
    la_quat_to_axis_angle(q, &out_axis, &out_angle);
    TEST_CHECK(la_vec3_approx_eq(out_axis, axis, 1e-5f));
    CHECK_NEAR(out_angle, angle, 1e-5f);
}

static void test_quat_shortest_arc_and_interp(void) {
    la_vec3 a = la_vec3_norm(la_vec3_make(1.0f, 2.0f, 3.0f));
    la_vec3 b = la_vec3_norm(la_vec3_make(-2.0f, 1.0f, 0.5f));
    la_quat q_ab = la_quat_from_unit_vecs(a, b);
    TEST_CHECK(la_vec3_approx_eq(la_quat_rotate_vec3(q_ab, a), b, 1e-5f));

    /* Antiparallel 180-degree edge case */
    la_quat q_anti = la_quat_from_unit_vecs(a, la_vec3_neg(a));
    TEST_CHECK(la_vec3_approx_eq(la_quat_rotate_vec3(q_anti, a), la_vec3_neg(a), 1e-5f));

    /* SLERP midpoint matches half-angle rotation even with flipped sign (-q) */
    la_quat q0 = la_quat_identity();
    la_quat q1 = la_quat_neg(la_quat_from_axis_angle(la_vec3_make(0.0f, 1.0f, 0.0f), LA_HALF_PI));
    la_quat mid = la_quat_slerp(q0, q1, 0.5f);
    la_quat expected = la_quat_from_axis_angle(la_vec3_make(0.0f, 1.0f, 0.0f), LA_QUARTER_PI);
    TEST_CHECK(la_quat_equiv_eq(mid, expected, 1e-5f));

    la_quat nmid = la_quat_nlerp(q0, q1, 0.5f);
    TEST_CHECK(la_quat_equiv_eq(nmid, expected, 1e-5f));
}

static void test_quat_angular_integration(void) {
    /* Constant angular velocity around Z: omega = (0, 0, pi), dt = 0.5 -> 90 deg rotation */
    la_quat q = la_quat_identity();
    la_vec3 omega = la_vec3_make(0.0f, 0.0f, LA_PI);

    /* Exponential integration is exact in one large step */
    la_quat q_exp = la_quat_integrate_angvel_exp(q, omega, 0.5f);
    la_quat expected = la_quat_from_axis_angle(la_vec3_make(0.0f, 0.0f, 1.0f), LA_HALF_PI);
    TEST_CHECK(la_quat_equiv_eq(q_exp, expected, 1e-5f));

    /* Symplectic Euler converges accurately over small substeps */
    la_quat q_sym = q;
    const int steps = 100;
    const float dt = 0.5f / (float)steps;
    for (int i = 0; i < steps; ++i) {
        q_sym = la_quat_integrate_angvel(q_sym, omega, dt);
    }
    TEST_CHECK(la_quat_equiv_eq(q_sym, expected, 1e-3f));
}

TEST_LIST = {
    { "quat_rotation_and_matrix_roundtrip", test_quat_rotation_and_matrix_roundtrip },
    { "quat_shortest_arc_and_interp",       test_quat_shortest_arc_and_interp },
    { "quat_angular_integration",           test_quat_angular_integration },
    { NULL, NULL }
};
