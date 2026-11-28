#include <acutest.h>
#include "linalg.h"

static void test_amalgamation_end_to_end(void) {
    /* Core vec, mat, quat, geom */
    la_quat q = la_quat_from_axis_angle(la_vec3_make(0.0f, 1.0f, 0.0f), LA_HALF_PI);
    la_mat4 m = la_mat4_mul(la_mat4_translate(la_vec3_make(1.0f, 2.0f, 3.0f)), la_quat_to_mat4(q));
    la_mat4 inv = la_mat4_inv_rigid(m);
    TEST_CHECK(la_mat4_approx_eq(la_mat4_mul(m, inv), la_mat4_identity(), 1e-5f));

    /* Batch SoA kernel (exercises LA_API linkage in separate-impl mode!) */
    float px[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    float py[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float pz[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float vx[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    float vy[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float vz[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float ax[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float ay[8] = { -10, -10, -10, -10, -10, -10, -10, -10 };
    float az[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    la_vec3_soa pos = { px, py, pz };
    la_vec3_soa vel = { vx, vy, vz };
    la_vec3_soa_const acc = { ax, ay, az };
    la_batch_integrate_symplectic_euler(pos, vel, acc, 0.1f, 8);
    TEST_CHECK(la_approx_eqf(px[3], 3.1f, 1e-5f));
    TEST_CHECK(la_approx_eqf(py[3], -0.1f, 1e-5f));

    /* Physics solver (exercises LA_API linkage in separate-impl mode!) */
    const float A[4] = { 4.0f, 1.0f, 1.0f, 3.0f };
    const float b[2] = { 1.0f, 2.0f };
    float x[2], L[4];
    TEST_CHECK(la_solve_cholesky(A, b, x, L, 2));
    TEST_CHECK(la_approx_eqf(4.0f * x[0] + x[1], 1.0f, 1e-5f));
    TEST_CHECK(la_approx_eqf(x[0] + 3.0f * x[1], 2.0f, 1e-5f));
}

TEST_LIST = {
    { "amalgamation_end_to_end", test_amalgamation_end_to_end },
    { NULL, NULL }
};
