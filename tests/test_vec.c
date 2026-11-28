#include "acutest.h"
#include "linalg/la_vec.h"

#define CHECK_NEAR(a, b, eps) \
    TEST_CHECK_(la_approx_eqf((a), (b), (eps)), \
        "Expected %.7f ~= %.7f", (double)(a), (double)(b))

static void test_vec2_ops(void) {
    la_vec2 a = la_vec2_make(3.0f, 4.0f);
    la_vec2 b = la_vec2_make(-1.0f, 2.0f);

    CHECK_NEAR(la_vec2_len(a), 5.0f, 1e-6f);
    CHECK_NEAR(la_vec2_dot(a, b), 5.0f, 1e-6f);
    CHECK_NEAR(la_vec2_cross(a, b), 10.0f, 1e-6f);

    la_vec2 p = la_vec2_perp(a);
    CHECK_NEAR(la_vec2_dot(a, p), 0.0f, 1e-6f);

    la_vec2 n = la_vec2_norm(a);
    CHECK_NEAR(la_vec2_len(n), 1.0f, 1e-6f);

    la_vec2 z = la_vec2_norm_safe(la_vec2_zero());
    TEST_CHECK(la_vec2_approx_eq(z, la_vec2_zero(), 1e-6f));

    la_vec2 refl = la_vec2_reflect(la_vec2_make(1.0f, -1.0f), la_vec2_make(0.0f, 1.0f));
    TEST_CHECK(la_vec2_approx_eq(refl, la_vec2_make(1.0f, 1.0f), 1e-6f));
}

static void test_vec3_ops(void) {
    la_vec3 x = la_vec3_make(1.0f, 0.0f, 0.0f);
    la_vec3 y = la_vec3_make(0.0f, 1.0f, 0.0f);
    la_vec3 z = la_vec3_cross(x, y);
    TEST_CHECK(la_vec3_approx_eq(z, la_vec3_make(0.0f, 0.0f, 1.0f), 1e-6f));

    la_vec3 a = la_vec3_make(1.0f, 2.0f, 3.0f);
    la_vec3 b = la_vec3_make(4.0f, -5.0f, 6.0f);
    CHECK_NEAR(la_vec3_dot(a, b), 12.0f, 1e-6f);

    la_vec3 fma = la_vec3_fma(a, b, 2.0f);
    TEST_CHECK(la_vec3_approx_eq(fma, la_vec3_make(9.0f, -8.0f, 15.0f), 1e-6f));

    /* Project + Reject reconstruct original vector */
    la_vec3 proj = la_vec3_project(a, b);
    la_vec3 rej = la_vec3_reject(a, b);
    TEST_CHECK(la_vec3_approx_eq(la_vec3_add(proj, rej), a, 1e-5f));
    CHECK_NEAR(la_vec3_dot(proj, rej), 0.0f, 1e-5f);

    /* Safe norm on zero avoids NaN */
    la_vec3 zero_n = la_vec3_norm_safe(la_vec3_zero());
    TEST_CHECK(la_is_finitef(zero_n.x) && la_is_finitef(zero_n.y) && la_is_finitef(zero_n.z));
    TEST_CHECK(la_vec3_approx_eq(zero_n, la_vec3_zero(), 1e-6f));

    /* Fast norm accuracy */
    la_vec3 nf = la_vec3_norm_fast(a);
    la_vec3 ne = la_vec3_norm(a);
    TEST_CHECK(la_vec3_approx_eq(nf, ne, 1e-4f));

    /* Orthonormal basis across poles and arbitrary directions */
    const la_vec3 normals[] = {
        { 0.0f, 0.0f, 1.0f },
        { 0.0f, 0.0f, -1.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.57735027f, -0.57735027f, 0.57735027f }
    };
    for (size_t i = 0; i < sizeof(normals) / sizeof(normals[0]); ++i) {
        la_vec3 t, bt;
        la_vec3_orthonormal_basis(normals[i], &t, &bt);
        CHECK_NEAR(la_vec3_len(t), 1.0f, 1e-5f);
        CHECK_NEAR(la_vec3_len(bt), 1.0f, 1e-5f);
        CHECK_NEAR(la_vec3_dot(normals[i], t), 0.0f, 1e-5f);
        CHECK_NEAR(la_vec3_dot(normals[i], bt), 0.0f, 1e-5f);
        CHECK_NEAR(la_vec3_dot(t, bt), 0.0f, 1e-5f);
    }

    /* Refraction (normal incidence + total internal reflection) */
    la_vec3 refr = la_vec3_refract(la_vec3_make(0.0f, -1.0f, 0.0f), la_vec3_make(0.0f, 1.0f, 0.0f), 0.75f);
    TEST_CHECK(la_vec3_approx_eq(refr, la_vec3_make(0.0f, -1.0f, 0.0f), 1e-5f));
    la_vec3 tir = la_vec3_refract(la_vec3_norm(la_vec3_make(0.95f, -0.312f, 0.0f)), la_vec3_make(0.0f, 1.0f, 0.0f), 1.5f);
    TEST_CHECK(la_vec3_approx_eq(tir, la_vec3_zero(), 1e-6f));
}

static void test_vec3a_and_vec4_ops(void) {
    la_vec3a a = la_vec3a_make(2.0f, -3.0f, 6.0f);
    la_vec3a b = la_vec3a_make(4.0f, 2.0f, -1.0f);

    CHECK_NEAR(la_vec3a_len(a), 7.0f, 1e-6f);
    CHECK_NEAR(la_vec3a_dot(a, b), -4.0f, 1e-6f);

    la_vec3a d = la_vec3a_div(a, b);
    CHECK_NEAR(d.x, 0.5f, 1e-6f);
    CHECK_NEAR(d.y, -1.5f, 1e-6f);
    CHECK_NEAR(d.z, -6.0f, 1e-6f);
    TEST_CHECK(d._pad == 0.0f);

    la_vec3a c = la_vec3a_cross(la_vec3a_make(1, 0, 0), la_vec3a_make(0, 1, 0));
    TEST_CHECK(la_vec3a_approx_eq(c, la_vec3a_make(0, 0, 1), 1e-6f));
    TEST_CHECK(c._pad == 0.0f);

    la_vec4 v4a = la_vec4_make(1.0f, 2.0f, 3.0f, 4.0f);
    la_vec4 v4b = la_vec4_make(5.0f, 6.0f, 7.0f, 8.0f);
    CHECK_NEAR(la_vec4_dot(v4a, v4b), 70.0f, 1e-5f);

    la_vec4 lerped = la_vec4_lerp(v4a, v4b, 0.5f);
    TEST_CHECK(la_vec4_approx_eq(lerped, la_vec4_make(3.0f, 4.0f, 5.0f, 6.0f), 1e-6f));

    la_vec4 n4 = la_vec4_norm(v4a);
    CHECK_NEAR(la_vec4_len(n4), 1.0f, 1e-6f);
    TEST_CHECK(la_vec4_approx_eq(la_vec4_norm_safe(la_vec4_zero()), la_vec4_zero(), 1e-6f));
}

TEST_LIST = {
    { "vec2_ops",           test_vec2_ops },
    { "vec3_ops",           test_vec3_ops },
    { "vec3a_and_vec4_ops", test_vec3a_and_vec4_ops },
    { NULL, NULL }
};
