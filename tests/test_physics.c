#include "acutest.h"
#include "linalg/la_physics.h"

#define CHECK_NEAR(a, b, eps) \
    TEST_CHECK_(la_approx_eqf((a), (b), (eps)), \
        "Expected %.7f ~= %.7f", (double)(a), (double)(b))

static void test_inertia_helpers(void) {
    /* Unit cube of mass 12 -> I = diag(2, 2, 2) */
    la_vec3 d = la_inertia_box_diag(12.0f, la_vec3_splat(1.0f));
    TEST_CHECK(la_vec3_approx_eq(d, la_vec3_splat(2.0f), 1e-6f));

    /* Solid sphere mass 5, radius 2 -> I = 0.4 * 5 * 4 = 8 */
    la_mat3 sph = la_inertia_sphere(5.0f, 2.0f);
    CHECK_NEAR(sph.cols[0].x, 8.0f, 1e-5f);

    /* Capsule with zero cylinder height degenerates exactly to solid sphere */
    la_vec3 cap = la_inertia_capsule_y_diag(5.0f, 2.0f, 0.0f);
    TEST_CHECK(la_vec3_approx_eq(cap, la_vec3_splat(8.0f), 1e-4f));

    /* Parallel axis theorem along X by d=3: Ix unchanged, Iy/Iz increase by m*d^2 = 45 */
    la_mat3 shifted = la_inertia_parallel_axis(sph, 5.0f, la_vec3_make(3.0f, 0.0f, 0.0f));
    CHECK_NEAR(shifted.cols[0].x, 8.0f, 1e-5f);
    CHECK_NEAR(shifted.cols[1].y, 53.0f, 1e-5f);
    CHECK_NEAR(shifted.cols[2].z, 53.0f, 1e-5f);

    /* World inverse inertia diag fast path matches general R * I^-1 * R^T */
    la_vec3 inv_d = la_vec3_make(0.5f, 0.25f, 0.1f);
    la_quat q = la_quat_from_axis_angle(la_vec3_norm(la_vec3_make(1.0f, -2.0f, 3.0f)), 1.1f);
    la_mat3 w_gen = la_inertia_world_inv(la_mat3_from_diag(inv_d), q);
    la_mat3 w_fast = la_inertia_world_inv_diag(inv_d, q);
    TEST_CHECK(la_mat3_approx_eq(w_gen, w_fast, 1e-5f));
}

static void test_jacobi_eigen_and_polar_decomp(void) {
    /* Construct symmetric matrix from known eigenvalues and rotation */
    la_quat rot = la_quat_from_axis_angle(la_vec3_norm(la_vec3_make(1.0f, 2.0f, -1.0f)), 0.8f);
    la_mat3 R = la_quat_to_mat3(rot);
    la_vec3 true_eig = la_vec3_make(7.0f, 3.0f, 1.5f);
    la_mat3 A = la_mat3_mul(la_mat3_mul(R, la_mat3_from_diag(true_eig)), la_mat3_transpose(R));

    la_vec3 vals;
    la_mat3 vecs;
    la_mat3_eigen_sym(A, &vals, &vecs, 16);

    /* Reconstruct A = V * diag(vals) * V^T */
    la_mat3 recon = la_mat3_mul(la_mat3_mul(vecs, la_mat3_from_diag(vals)), la_mat3_transpose(vecs));
    TEST_CHECK(la_mat3_approx_eq(A, recon, 1e-4f));

    /* V is orthonormal with det == +1 */
    TEST_CHECK(la_mat3_approx_eq(la_mat3_mul(la_mat3_transpose(vecs), vecs), la_mat3_identity(), 1e-5f));
    CHECK_NEAR(la_mat3_det(vecs), 1.0f, 1e-5f);

    /* Polar decomposition extracts R from R * S (non-uniform stretch) */
    la_mat3 S = la_mat3_from_diag(la_vec3_make(1.5f, 0.8f, 2.2f));
    la_mat3 deformed = la_mat3_mul(R, S);
    la_quat extracted = la_mat3_extract_rotation(deformed, la_quat_identity(), 24);
    TEST_CHECK(la_quat_equiv_eq(extracted, rot, 1e-4f));
}

static void test_cholesky_ldlt_and_pgs_solvers(void) {
    /* 4x4 SPD matrix */
    const float A[16] = {
        10.0f,  2.0f, -1.0f,  0.5f,
         2.0f,  8.0f,  1.5f, -1.0f,
        -1.0f,  1.5f,  9.0f,  2.0f,
         0.5f, -1.0f,  2.0f,  7.0f
    };
    const float x_true[4] = { 1.5f, -2.0f, 0.75f, 3.0f };
    float b[4] = { 0 };
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            b[i] += A[i * 4 + j] * x_true[j];
        }
    }

    float x_chol[4], L[16], D[4];
    TEST_CHECK(la_solve_cholesky(A, b, x_chol, L, 4));
    for (int i = 0; i < 4; ++i) {
        CHECK_NEAR(x_chol[i], x_true[i], 1e-5f);
    }

    /* Non-SPD matrix fails Cholesky cleanly */
    const float bad[4] = { -1.0f, 0.0f, 0.0f, 2.0f };
    float xb[2], Lb[4];
    TEST_CHECK(!la_solve_cholesky(bad, b, xb, Lb, 2));

    /* LDL^T handles both SPD and indefinite symmetric matrices */
    float x_ldlt[4];
    TEST_CHECK(la_solve_ldlt(A, b, x_ldlt, L, D, 4));
    for (int i = 0; i < 4; ++i) {
        CHECK_NEAR(x_ldlt[i], x_true[i], 1e-5f);
    }

    /* Unconstrained PGS converges to exact solution */
    float x_pgs[4] = { 0, 0, 0, 0 };
    uint32_t iters = la_solve_pgs(A, b, NULL, NULL, x_pgs, 4, 100, 1e-6f);
    TEST_CHECK(iters < 100);
    for (int i = 0; i < 4; ++i) {
        CHECK_NEAR(x_pgs[i], x_true[i], 1e-4f);
    }

    /* Box-constrained PGS (contact impulse LCP: x >= 0) respects bounds & complementarity */
    const float lo[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const float hi[4] = { 100.0f, 100.0f, 100.0f, 100.0f };
    float x_lcp[4] = { 0, 0, 0, 0 };
    la_solve_pgs(A, b, lo, hi, x_lcp, 4, 200, 1e-6f);
    for (int i = 0; i < 4; ++i) {
        TEST_CHECK(x_lcp[i] >= -1e-6f);
        float res = -b[i];
        for (int j = 0; j < 4; ++j) res += A[i * 4 + j] * x_lcp[j];
        /* Complementarity: either x[i] == 0 and (Ax - b)[i] >= 0, or x[i] > 0 and (Ax - b)[i] == 0 */
        if (x_lcp[i] > 1e-5f) {
            CHECK_NEAR(res, 0.0f, 1e-4f);
        } else {
            TEST_CHECK(res >= -1e-4f);
        }
    }
}

TEST_LIST = {
    { "inertia_helpers",             test_inertia_helpers },
    { "jacobi_eigen_and_polar",      test_jacobi_eigen_and_polar_decomp },
    { "cholesky_ldlt_and_pgs",       test_cholesky_ldlt_and_pgs_solvers },
    { NULL, NULL }
};
