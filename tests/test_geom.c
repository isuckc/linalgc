#include "acutest.h"
#include "linalg/la_geom.h"

#define CHECK_NEAR(a, b, eps) \
    TEST_CHECK_(la_approx_eqf((a), (b), (eps)), \
        "Expected %.7f ~= %.7f", (double)(a), (double)(b))

static void test_transform_trs_and_decompose(void) {
    la_transform tr = la_transform_make(
        la_vec3_make(3.0f, -2.0f, 5.0f),
        la_quat_from_axis_angle(la_vec3_norm(la_vec3_make(1.0f, 1.0f, 0.0f)), 0.9f),
        la_vec3_make(2.0f, 0.5f, 3.0f)
    );
    la_mat4 m = la_transform_to_mat4(tr);

    la_vec3 p = la_vec3_make(1.0f, -3.0f, 2.0f);
    TEST_CHECK(la_vec3_approx_eq(la_transform_point3(tr, p), la_mat4_mul_point3(m, p), 1e-5f));

    /* Decompose roundtrip */
    la_transform dec = la_transform_from_mat4(m);
    TEST_CHECK(la_vec3a_approx_eq(dec.pos, tr.pos, 1e-5f));
    TEST_CHECK(la_vec3a_approx_eq(dec.scale, tr.scale, 1e-5f));
    TEST_CHECK(la_quat_equiv_eq(dec.rot, tr.rot, 1e-5f));

    /* Rigid transform inverse */
    la_transform rigid = la_transform_make(
        la_vec3_make(4.0f, 5.0f, -6.0f),
        la_quat_from_axis_angle(la_vec3_make(0.0f, 1.0f, 0.0f), 1.1f),
        la_vec3_splat(1.0f)
    );
    la_transform inv = la_transform_inv_rigid(rigid);
    la_vec3 back = la_transform_point3(inv, la_transform_point3(rigid, p));
    TEST_CHECK(la_vec3_approx_eq(back, p, 1e-5f));
}

static void test_aabb_and_arvo_transform(void) {
    la_aabb box = la_aabb_from_minmax(la_vec3_make(-1.0f, -2.0f, -3.0f),
                                      la_vec3_make(1.0f, 2.0f, 3.0f));
    TEST_CHECK(la_aabb_contains_point(box, la_vec3_zero()));
    TEST_CHECK(!la_aabb_contains_point(box, la_vec3_make(2.0f, 0.0f, 0.0f)));
    CHECK_NEAR(la_aabb_volume(box), 48.0f, 1e-5f);

    /* Arvo's transform matches brute-force 8-corner transformation */
    la_mat4 m = la_mat4_mul(
        la_mat4_translate(la_vec3_make(10.0f, -5.0f, 2.0f)),
        la_mat4_rotate_axis(la_vec3_norm(la_vec3_make(1.0f, 2.0f, 3.0f)), 0.7f)
    );
    la_aabb arvo = la_aabb_transform_mat4(box, m);

    la_aabb brute = la_aabb_empty();
    for (int i = 0; i < 8; ++i) {
        la_vec3 corner = la_vec3_make(
            (i & 1) ? box.max.x : box.min.x,
            (i & 2) ? box.max.y : box.min.y,
            (i & 4) ? box.max.z : box.min.z
        );
        brute = la_aabb_expand_point(brute, la_mat4_mul_point3(m, corner));
    }
    TEST_CHECK(la_vec3a_approx_eq(arvo.min, brute.min, 1e-4f));
    TEST_CHECK(la_vec3a_approx_eq(arvo.max, brute.max, 1e-4f));
}

static void test_rays_and_frustum(void) {
    la_ray ray = la_ray_make(la_vec3_make(0.0f, 0.0f, 5.0f), la_vec3_make(0.0f, 0.0f, -1.0f));
    float t = 0.0f, u = 0.0f, v = 0.0f;

    /* Plane */
    la_plane pl = la_plane_from_point_normal(la_vec3_zero(), la_vec3_make(0.0f, 0.0f, 1.0f));
    TEST_CHECK(la_ray_intersect_plane(ray, pl, &t));
    CHECK_NEAR(t, 5.0f, 1e-5f);

    /* Sphere */
    TEST_CHECK(la_ray_intersect_sphere(ray, la_vec3_zero(), 1.0f, &t));
    CHECK_NEAR(t, 4.0f, 1e-5f);
    TEST_CHECK(!la_ray_intersect_sphere(ray, la_vec3_make(5.0f, 0.0f, 0.0f), 1.0f, &t));

    /* AABB slab */
    la_aabb box = la_aabb_from_minmax(la_vec3_splat(-1.0f), la_vec3_splat(1.0f));
    TEST_CHECK(la_ray_intersect_aabb(ray, box, 0.0f, 100.0f, &t));
    CHECK_NEAR(t, 4.0f, 1e-5f);
    la_aabb far_box = la_aabb_from_minmax(la_vec3_make(3.0f, 3.0f, -1.0f), la_vec3_make(4.0f, 4.0f, 1.0f));
    TEST_CHECK(!la_ray_intersect_aabb(ray, far_box, 0.0f, 100.0f, &t));

    /* Triangle (Moller-Trumbore) */
    TEST_CHECK(la_ray_intersect_triangle(
        ray,
        la_vec3_make(-1.0f, -1.0f, 0.0f),
        la_vec3_make( 1.0f, -1.0f, 0.0f),
        la_vec3_make( 0.0f,  1.0f, 0.0f),
        &t, &u, &v
    ));
    CHECK_NEAR(t, 5.0f, 1e-5f);
    TEST_CHECK(u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f);

    /* Frustum extraction (both ZO and NO depth ranges) */
    la_mat4 view = la_mat4_look_at_rh(la_vec3_make(0.0f, 0.0f, 10.0f), la_vec3_zero(), la_vec3_make(0.0f, 1.0f, 0.0f));
    la_mat4 proj_zo = la_mat4_perspective_rh_zo(LA_HALF_PI, 1.0f, 1.0f, 50.0f);
    la_mat4 proj_no = la_mat4_perspective_rh_no(LA_HALF_PI, 1.0f, 1.0f, 50.0f);

    la_frustum f_zo = la_frustum_from_vp_zo(la_mat4_mul(proj_zo, view));
    la_frustum f_no = la_frustum_from_vp_no(la_mat4_mul(proj_no, view));

    TEST_CHECK(la_frustum_test_point(f_zo, la_vec3_zero()));
    TEST_CHECK(la_frustum_test_point(f_no, la_vec3_zero()));
    TEST_CHECK(!la_frustum_test_point(f_zo, la_vec3_make(0.0f, 0.0f, 15.0f)));  /* behind camera */
    TEST_CHECK(!la_frustum_test_point(f_no, la_vec3_make(0.0f, 0.0f, -50.0f))); /* beyond far plane */

    TEST_CHECK(la_frustum_test_aabb(f_zo, box));
    TEST_CHECK(!la_frustum_test_aabb(f_zo, la_aabb_from_minmax(la_vec3_make(50.0f, 50.0f, 0.0f),
                                                               la_vec3_make(52.0f, 52.0f, 2.0f))));
}

TEST_LIST = {
    { "transform_trs_and_decompose", test_transform_trs_and_decompose },
    { "aabb_and_arvo_transform",     test_aabb_and_arvo_transform },
    { "rays_and_frustum",            test_rays_and_frustum },
    { NULL, NULL }
};
