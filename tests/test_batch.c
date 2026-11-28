#include "acutest.h"
#include "linalg/la_batch.h"

#include <stdlib.h>

#define CHECK_NEAR(a, b, eps) \
    TEST_CHECK_(la_approx_eqf((a), (b), (eps)), \
        "Expected %.7f ~= %.7f", (double)(a), (double)(b))

static float pseudo_rand(uint32_t *state) {
    *state = (*state * 1664525u + 1013904223u);
    return ((float)((*state >> 8) & 0xFFFFu) / 32767.5f) - 1.0f;
}

static void test_aos_soa_aosoa_roundtrips(void) {
    /* Odd and edge sizes to test SIMD main loop + scalar tail */
    const size_t counts[] = { 0, 1, 2, 3, 4, 5, 7, 8, 13, 65 };
    uint32_t rng = 0x12345678u;

    for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c) {
        size_t n = counts[c];
        size_t blocks = (n + 3) / 4;
        size_t alloc_n = (n == 0) ? 1 : n;
        size_t alloc_b = (blocks == 0) ? 1 : blocks;

        la_vec3 *aos_in = (la_vec3 *)malloc(alloc_n * sizeof(la_vec3));
        la_vec3 *aos_out = (la_vec3 *)malloc(alloc_n * sizeof(la_vec3));
        float *sx = (float *)malloc(alloc_n * sizeof(float));
        float *sy = (float *)malloc(alloc_n * sizeof(float));
        float *sz = (float *)malloc(alloc_n * sizeof(float));
        la_vec3_aosoa4 *aosoa = (la_vec3_aosoa4 *)malloc(alloc_b * sizeof(la_vec3_aosoa4));

        for (size_t i = 0; i < n; ++i) {
            aos_in[i] = la_vec3_make(pseudo_rand(&rng) * 10.0f,
                                     pseudo_rand(&rng) * 10.0f,
                                     pseudo_rand(&rng) * 10.0f);
        }

        la_vec3_soa soa = { sx, sy, sz };
        la_vec3_soa_const soa_c = { sx, sy, sz };

        la_batch_vec3_aos_to_soa(soa, aos_in, n);
        la_batch_vec3_soa_to_aos(aos_out, soa_c, n);
        for (size_t i = 0; i < n; ++i) {
            TEST_CHECK(la_vec3_approx_eq(aos_in[i], aos_out[i], 1e-6f));
        }

        la_batch_vec3_aos_to_aosoa4(aosoa, aos_in, n);
        la_batch_vec3_aosoa4_to_aos(aos_out, aosoa, n);
        for (size_t i = 0; i < n; ++i) {
            TEST_CHECK(la_vec3_approx_eq(aos_in[i], aos_out[i], 1e-6f));
        }

        free(aos_in); free(aos_out); free(sx); free(sy); free(sz); free(aosoa);
    }
}

static void test_soa_vector_and_transform_streams(void) {
    const size_t n = 37; /* non-multiple of 4 & 8 */
    float ax[37], ay[37], az[37];
    float bx[37], by[37], bz[37];
    float dx[37], dy[37], dz[37];
    float dots[37];

    uint32_t rng = 0xDEADBEEFu;
    for (size_t i = 0; i < n; ++i) {
        ax[i] = pseudo_rand(&rng) * 5.0f;
        ay[i] = pseudo_rand(&rng) * 5.0f;
        az[i] = pseudo_rand(&rng) * 5.0f;
        bx[i] = pseudo_rand(&rng) * 5.0f;
        by[i] = pseudo_rand(&rng) * 5.0f;
        bz[i] = pseudo_rand(&rng) * 5.0f;
    }
    /* Inject zero vector at index 5 to test branchless safe norm */
    ax[5] = ay[5] = az[5] = 0.0f;

    la_vec3_soa_const a = { ax, ay, az };
    la_vec3_soa_const b = { bx, by, bz };
    la_vec3_soa d = { dx, dy, dz };

    la_batch_vec3_dot_soa(dots, a, b, n);
    la_batch_vec3_cross_soa(d, a, b, n);
    for (size_t i = 0; i < n; ++i) {
        la_vec3 va = la_vec3_make(ax[i], ay[i], az[i]);
        la_vec3 vb = la_vec3_make(bx[i], by[i], bz[i]);
        CHECK_NEAR(dots[i], la_vec3_dot(va, vb), 1e-4f);
        TEST_CHECK(la_vec3_approx_eq(la_vec3_make(dx[i], dy[i], dz[i]), la_vec3_cross(va, vb), 1e-4f));
    }

    la_batch_vec3_norm_safe_soa(d, a, n);
    for (size_t i = 0; i < n; ++i) {
        la_vec3 ref = la_vec3_norm_safe(la_vec3_make(ax[i], ay[i], az[i]));
        TEST_CHECK(la_vec3_approx_eq(la_vec3_make(dx[i], dy[i], dz[i]), ref, 1e-5f));
    }

    /* mat4 point transform SoA & AoSoA4 match scalar reference */
    la_mat4 m = la_mat4_mul(
        la_mat4_translate(la_vec3_make(3.0f, -4.0f, 5.0f)),
        la_mat4_rotate_axis(la_vec3_norm(la_vec3_make(1.0f, -2.0f, 3.0f)), 1.2f)
    );
    la_batch_mat4_mul_points_soa(d, a, m, n);
    for (size_t i = 0; i < n; ++i) {
        la_vec3 ref = la_mat4_mul_point3(m, la_vec3_make(ax[i], ay[i], az[i]));
        TEST_CHECK(la_vec3_approx_eq(la_vec3_make(dx[i], dy[i], dz[i]), ref, 1e-4f));
    }

    la_vec3_aosoa4 blk_in[10], blk_out[10];
    la_vec3 aos_in[37], aos_out[37];
    la_batch_vec3_soa_to_aos(aos_in, a, n);
    la_batch_vec3_aos_to_aosoa4(blk_in, aos_in, n);
    la_batch_mat4_mul_points_aosoa4(blk_out, blk_in, m, 10);
    la_batch_vec3_aosoa4_to_aos(aos_out, blk_out, n);
    for (size_t i = 0; i < n; ++i) {
        la_vec3 ref = la_mat4_mul_point3(m, aos_in[i]);
        TEST_CHECK(la_vec3_approx_eq(aos_out[i], ref, 1e-4f));
    }
}

static void test_quat_physics_aabb_and_skinning(void) {
    const size_t n = 19;
    float qx[19], qy[19], qz[19], qw[19];
    float vx[19], vy[19], vz[19];
    float ox[19], oy[19], oz[19];
    float px[19], py[19], pz[19];
    float ax[19], ay[19], az[19];
    uint32_t rng = 0xCAFEBABEu;

    for (size_t i = 0; i < n; ++i) {
        la_vec3 axis = la_vec3_norm_safe(la_vec3_make(pseudo_rand(&rng), pseudo_rand(&rng), pseudo_rand(&rng) + 2.0f));
        la_quat q = la_quat_from_axis_angle(axis, pseudo_rand(&rng) * LA_PI);
        qx[i] = q.x; qy[i] = q.y; qz[i] = q.z; qw[i] = q.w;
        vx[i] = pseudo_rand(&rng) * 3.0f; vy[i] = pseudo_rand(&rng) * 3.0f; vz[i] = pseudo_rand(&rng) * 3.0f;
        px[i] = pseudo_rand(&rng); py[i] = pseudo_rand(&rng); pz[i] = pseudo_rand(&rng);
        ax[i] = 0.0f; ay[i] = -9.81f; az[i] = 0.0f;
    }

    la_quat_soa_const qc = { qx, qy, qz, qw };
    la_quat_soa qm = { qx, qy, qz, qw };
    la_vec3_soa_const vc = { vx, vy, vz };
    la_vec3_soa om = { ox, oy, oz };

    la_batch_quat_rotate_soa(om, qc, vc, n);
    for (size_t i = 0; i < n; ++i) {
        la_vec3 ref = la_quat_rotate_vec3(la_quat_make(qx[i], qy[i], qz[i], qw[i]),
                                          la_vec3_make(vx[i], vy[i], vz[i]));
        TEST_CHECK(la_vec3_approx_eq(la_vec3_make(ox[i], oy[i], oz[i]), ref, 1e-4f));
    }

    /* Quat integration matches scalar */
    la_quat ref_q[19];
    for (size_t i = 0; i < n; ++i) {
        ref_q[i] = la_quat_integrate_angvel(la_quat_make(qx[i], qy[i], qz[i], qw[i]),
                                            la_vec3_make(vx[i], vy[i], vz[i]), 0.016f);
    }
    la_batch_quat_integrate_soa(qm, vc, 0.016f, n);
    for (size_t i = 0; i < n; ++i) {
        TEST_CHECK(la_quat_approx_eq(la_quat_make(qx[i], qy[i], qz[i], qw[i]), ref_q[i], 1e-5f));
    }

    /* Symplectic Euler */
    float p0_y = py[7], v0_y = vy[7];
    la_vec3_soa pos = { px, py, pz };
    la_vec3_soa vel = { vx, vy, vz };
    la_vec3_soa_const acc = { ax, ay, az };
    la_batch_integrate_symplectic_euler(pos, vel, acc, 0.05f, n);
    float expected_v = v0_y - 9.81f * 0.05f;
    float expected_p = p0_y + expected_v * 0.05f;
    CHECK_NEAR(vy[7], expected_v, 1e-5f);
    CHECK_NEAR(py[7], expected_p, 1e-5f);

    /* Broadphase AABB overlap sweep */
    float minx[19], miny[19], minz[19], maxx[19], maxy[19], maxz[19];
    for (size_t i = 0; i < n; ++i) {
        minx[i] = (float)i * 2.0f; miny[i] = 0.0f; minz[i] = 0.0f;
        maxx[i] = (float)i * 2.0f + 1.0f; maxy[i] = 1.0f; maxz[i] = 1.0f;
    }
    la_aabb_soa_const boxes = { minx, miny, minz, maxx, maxy, maxz };
    la_aabb query = la_aabb_from_minmax(la_vec3_make(3.5f, 0.0f, 0.0f), la_vec3_make(8.5f, 1.0f, 1.0f));
    uint32_t hits[19];
    size_t hit_count = la_batch_aabb_overlap_query(query, boxes, n, hits, 19);
    /* Boxes overlapping [3.5, 8.5] are i=2 ([4,5]), i=3 ([6,7]), i=4 ([8,9]) */
    TEST_CHECK(hit_count == 3);
    TEST_CHECK(hits[0] == 2 && hits[1] == 3 && hits[2] == 4);

    /* 4-bone Linear Blend Skinning */
    la_mat4 palette[2] = {
        la_mat4_translate(la_vec3_make(10.0f, 0.0f, 0.0f)),
        la_mat4_translate(la_vec3_make(0.0f, 20.0f, 0.0f))
    };
    la_vec3 src_p[2] = { la_vec3_zero(), la_vec3_make(1.0f, 1.0f, 1.0f) };
    la_vec3 src_n[2] = { la_vec3_make(0.0f, 1.0f, 0.0f), la_vec3_make(0.0f, 0.0f, 1.0f) };
    uint16_t bidx[2][4] = { { 0, 1, 0, 0 }, { 1, 0, 0, 0 } };
    float bw[2][4] = { { 0.5f, 0.5f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f } };
    la_vec3 dst_p[2], dst_n[2];

    la_batch_skin_lbs4(dst_p, dst_n, src_p, src_n, bidx, bw, palette, 2);
    TEST_CHECK(la_vec3_approx_eq(dst_p[0], la_vec3_make(5.0f, 10.0f, 0.0f), 1e-5f));
    TEST_CHECK(la_vec3_approx_eq(dst_p[1], la_vec3_make(1.0f, 21.0f, 1.0f), 1e-5f));
    TEST_CHECK(la_vec3_approx_eq(dst_n[0], la_vec3_make(0.0f, 1.0f, 0.0f), 1e-5f));
}

TEST_LIST = {
    { "aos_soa_aosoa_roundtrips",         test_aos_soa_aosoa_roundtrips },
    { "soa_vector_and_transform_streams", test_soa_vector_and_transform_streams },
    { "quat_physics_aabb_and_skinning",   test_quat_physics_aabb_and_skinning },
    { NULL, NULL }
};
