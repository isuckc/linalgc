/*
 * linalg.h - High-performance Data-Oriented Linear Algebra Library (C99+ / C++17+)
 *
 * Module: la_physics.h
 * Purpose: Rigid-body inertia tensors, world inverse inertia, 3x3 symmetric
 *          Jacobi eigensolver, Muller polar rotation extraction, and
 *          zero-allocation Cholesky, LDL^T, and Projected Gauss-Seidel (LCP) solvers.
 */

#ifndef LA_PHYSICS_H
#define LA_PHYSICS_H

#include "la_geom.h"

LA_EXTERN_C_BEGIN

/* ============================================================================
 * Rigid-Body Inertia Tensors
 * ============================================================================ */

/* Solid cuboid inertia tensor (diagonal), full side lengths (sx, sy, sz). */
LA_INLINE la_vec3 la_inertia_box_diag(float mass, la_vec3 full_size) {
    float k = mass / 12.0f;
    float x2 = full_size.x * full_size.x;
    float y2 = full_size.y * full_size.y;
    float z2 = full_size.z * full_size.z;
    return la_vec3_make(k * (y2 + z2), k * (x2 + z2), k * (x2 + y2));
}

LA_INLINE la_mat3 la_inertia_box(float mass, la_vec3 full_size) {
    return la_mat3_from_diag(la_inertia_box_diag(mass, full_size));
}

LA_INLINE la_mat3 la_inertia_sphere(float mass, float radius) {
    float i = 0.4f * mass * radius * radius;
    return la_mat3_from_diag(la_vec3_splat(i));
}

LA_INLINE la_mat3 la_inertia_hollow_sphere(float mass, float radius) {
    float i = (2.0f / 3.0f) * mass * radius * radius;
    return la_mat3_from_diag(la_vec3_splat(i));
}

/* Solid cylinder aligned along the Y axis. */
LA_INLINE la_vec3 la_inertia_cylinder_y_diag(float mass, float radius, float height) {
    float r2 = radius * radius;
    float iy = 0.5f * mass * r2;
    float ixz = (mass / 12.0f) * (3.0f * r2 + height * height);
    return la_vec3_make(ixz, iy, ixz);
}

LA_INLINE la_mat3 la_inertia_cylinder_y(float mass, float radius, float height) {
    return la_mat3_from_diag(la_inertia_cylinder_y_diag(mass, radius, height));
}

/* Solid capsule aligned along the Y axis (cylinder of length cylinder_height + 2 hemispheres). */
LA_INLINE la_vec3 la_inertia_capsule_y_diag(float mass, float radius, float cylinder_height) {
    float r2 = radius * radius;
    float r3 = r2 * radius;
    float vol_cyl = LA_PI * r2 * cylinder_height;
    float vol_sph = (4.0f / 3.0f) * LA_PI * r3;
    float total_vol = vol_cyl + vol_sph;
    if (LA_UNLIKELY(total_vol <= LA_EPSILON)) {
        return la_vec3_zero();
    }
    float m_cyl = mass * (vol_cyl / total_vol);
    float m_sph = mass * (vol_sph / total_vol);

    float iy = 0.5f * m_cyl * r2 + 0.4f * m_sph * r2;
    float ixz_cyl = m_cyl * (cylinder_height * cylinder_height / 12.0f + 0.25f * r2);
    float ixz_sph = m_sph * (0.4f * r2 + 0.25f * cylinder_height * cylinder_height +
                             0.375f * cylinder_height * radius);
    return la_vec3_make(ixz_cyl + ixz_sph, iy, ixz_cyl + ixz_sph);
}

LA_INLINE la_mat3 la_inertia_capsule_y(float mass, float radius, float cylinder_height) {
    return la_mat3_from_diag(la_inertia_capsule_y_diag(mass, radius, cylinder_height));
}

/* Parallel axis theorem: shifts center-of-mass inertia tensor by offset d. */
LA_INLINE la_mat3 la_inertia_parallel_axis(la_mat3 inertia_cm, float mass, la_vec3 offset) {
    float d2 = la_vec3_len_sq(offset);
    la_mat3 shift = la_mat3_from_cols(
        la_vec3_make(mass * (d2 - offset.x * offset.x), -mass * offset.y * offset.x, -mass * offset.z * offset.x),
        la_vec3_make(-mass * offset.x * offset.y, mass * (d2 - offset.y * offset.y), -mass * offset.z * offset.y),
        la_vec3_make(-mass * offset.x * offset.z, -mass * offset.y * offset.z, mass * (d2 - offset.z * offset.z))
    );
    return la_mat3_add(inertia_cm, shift);
}

/* World-space inverse inertia tensor: R * I_local^-1 * R^T. */
LA_INLINE la_mat3 la_inertia_world_inv(la_mat3 local_inv_inertia, la_quat rot) {
    la_mat3 r = la_quat_to_mat3(rot);
    return la_mat3_mul(la_mat3_mul(r, local_inv_inertia), la_mat3_transpose(r));
}

/* Fast world-space inverse inertia when local inverse inertia is diagonal (principal axes). */
LA_INLINE la_mat3 la_inertia_world_inv_diag(la_vec3 local_inv_diag, la_quat rot) {
    la_mat3 r = la_quat_to_mat3(rot);
    la_mat3 scaled = la_mat3_from_cols(
        la_vec3_scale(r.cols[0], local_inv_diag.x),
        la_vec3_scale(r.cols[1], local_inv_diag.y),
        la_vec3_scale(r.cols[2], local_inv_diag.z)
    );
    return la_mat3_mul(scaled, la_mat3_transpose(r));
}

/* ============================================================================
 * Public Solver Declarations (LA_API)
 * ============================================================================ */

/*
 * 3x3 symmetric Jacobi eigensolver: A = V * diag(eigenvalues) * V^T.
 * Returns a right-handed orthonormal V (det(V) == +1), suitable for la_quat_from_mat3.
 */
LA_API void la_mat3_eigen_sym(la_mat3 sym_a,
                              la_vec3 *LA_RESTRICT out_eigenvalues,
                              la_mat3 *LA_RESTRICT out_eigenvectors,
                              uint32_t max_sweeps);

/*
 * Fast polar decomposition rotation extraction (Muller et al., MIG 2016).
 * Extracts the closest rotation quaternion to 3x3 deformation matrix A.
 * Warm-start with previous frame's quaternion q (or la_quat_identity()).
 */
LA_API la_quat la_mat3_extract_rotation(la_mat3 a, la_quat q_init, uint32_t max_iters);

/*
 * Dense Cholesky solver: A * x = b for n x n symmetric positive-definite A (row-major).
 * L_scratch must hold at least n * n floats. Returns false if A is not SPD.
 */
LA_API bool la_solve_cholesky(const float *LA_RESTRICT A,
                              const float *LA_RESTRICT b,
                              float *LA_RESTRICT x,
                              float *LA_RESTRICT L_scratch,
                              size_t n);

/*
 * Dense LDL^T solver: A * x = b for n x n symmetric A (row-major), square-root free.
 * L_scratch must hold n * n floats, D_scratch must hold n floats.
 */
LA_API bool la_solve_ldlt(const float *LA_RESTRICT A,
                          const float *LA_RESTRICT b,
                          float *LA_RESTRICT x,
                          float *LA_RESTRICT L_scratch,
                          float *LA_RESTRICT D_scratch,
                          size_t n);

/*
 * Projected Gauss-Seidel (PGS) solver for box-constrained LCP:
 *   A * x = b subject to lo[i] <= x[i] <= hi[i] (warm-started from input x).
 * lo and/or hi may be NULL for unconstrained bounds. Returns iterations performed.
 */
LA_API uint32_t la_solve_pgs(const float *LA_RESTRICT A,
                             const float *LA_RESTRICT b,
                             const float *LA_RESTRICT lo,
                             const float *LA_RESTRICT hi,
                             float *LA_RESTRICT x,
                             size_t n,
                             uint32_t max_iters,
                             float tol);

/* ============================================================================
 * Solver Implementations
 * ============================================================================ */
#if LA_IMPL_ENABLED

LA_API void la_mat3_eigen_sym(la_mat3 sym_a,
                              la_vec3 *LA_RESTRICT out_eigenvalues,
                              la_mat3 *LA_RESTRICT out_eigenvectors,
                              uint32_t max_sweeps) {
    /* Work on symmetric 3x3 in row/col array a[r][c] and eigenvector columns v[r][c] */
    float a[3][3];
    float v[3][3] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f }
    };

    /* Symmetrize input defensively */
    a[0][0] = sym_a.cols[0].x;
    a[1][1] = sym_a.cols[1].y;
    a[2][2] = sym_a.cols[2].z;
    a[0][1] = a[1][0] = 0.5f * (sym_a.cols[1].x + sym_a.cols[0].y);
    a[0][2] = a[2][0] = 0.5f * (sym_a.cols[2].x + sym_a.cols[0].z);
    a[1][2] = a[2][1] = 0.5f * (sym_a.cols[2].y + sym_a.cols[1].z);

    if (max_sweeps == 0) {
        max_sweeps = 16;
    }

    static const int pairs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };

    for (uint32_t sweep = 0; sweep < max_sweeps; ++sweep) {
        float off = la_absf(a[0][1]) + la_absf(a[0][2]) + la_absf(a[1][2]);
        if (off <= 1e-7f) {
            break;
        }
        for (int k = 0; k < 3; ++k) {
            int p = pairs[k][0];
            int q = pairs[k][1];
            float apq = a[p][q];
            if (la_absf(apq) <= 1e-8f) {
                continue;
            }
            float tau = (a[q][q] - a[p][p]) / (2.0f * apq);
            float t = (tau >= 0.0f)
                ? (1.0f / (tau + sqrtf(1.0f + tau * tau)))
                : (-1.0f / (-tau + sqrtf(1.0f + tau * tau)));
            float c = 1.0f / sqrtf(1.0f + t * t);
            float s = t * c;

            float app = a[p][p];
            float aqq = a[q][q];
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0f;
            a[q][p] = 0.0f;

            int r = 3 - p - q;
            float arp = a[r][p];
            float arq = a[r][q];
            a[r][p] = a[p][r] = c * arp - s * arq;
            a[r][q] = a[q][r] = s * arp + c * arq;

            for (int row = 0; row < 3; ++row) {
                float vrp = v[row][p];
                float vrq = v[row][q];
                v[row][p] = c * vrp - s * vrq;
                v[row][q] = s * vrp + c * vrq;
            }
        }
    }

    la_mat3 eig_v = la_mat3_from_cols(
        la_vec3_make(v[0][0], v[1][0], v[2][0]),
        la_vec3_make(v[0][1], v[1][1], v[2][1]),
        la_vec3_make(v[0][2], v[1][2], v[2][2])
    );
    if (la_mat3_det(eig_v) < 0.0f) {
        eig_v.cols[2] = la_vec3_neg(eig_v.cols[2]);
    }

    if (out_eigenvalues) {
        *out_eigenvalues = la_vec3_make(a[0][0], a[1][1], a[2][2]);
    }
    if (out_eigenvectors) {
        *out_eigenvectors = eig_v;
    }
}

LA_API la_quat la_mat3_extract_rotation(la_mat3 a, la_quat q_init, uint32_t max_iters) {
    la_quat q = la_quat_norm_safe(q_init);
    if (max_iters == 0) {
        max_iters = 16;
    }
    for (uint32_t iter = 0; iter < max_iters; ++iter) {
        la_mat3 r = la_quat_to_mat3(q);
        la_vec3 omega_num = la_vec3_add(
            la_vec3_cross(r.cols[0], a.cols[0]),
            la_vec3_add(la_vec3_cross(r.cols[1], a.cols[1]),
                        la_vec3_cross(r.cols[2], a.cols[2]))
        );
        float denom = la_absf(la_vec3_dot(r.cols[0], a.cols[0]) +
                              la_vec3_dot(r.cols[1], a.cols[1]) +
                              la_vec3_dot(r.cols[2], a.cols[2])) + LA_EPSILON;
        la_vec3 omega = la_vec3_scale(omega_num, 1.0f / denom);
        float w = la_vec3_len(omega);
        if (w < 1e-6f) {
            break;
        }
        la_quat dq = la_quat_from_axis_angle(la_vec3_scale(omega, 1.0f / w), w);
        q = la_quat_norm_safe(la_quat_mul(dq, q));
    }
    return q;
}

LA_API bool la_solve_cholesky(const float *LA_RESTRICT A,
                              const float *LA_RESTRICT b,
                              float *LA_RESTRICT x,
                              float *LA_RESTRICT L,
                              size_t n) {
    if (n == 0) {
        return true;
    }
    memset(L, 0, n * n * sizeof(float));

    /* Factorization: A = L * L^T */
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            float sum = A[i * n + j];
            for (size_t k = 0; k < j; ++k) {
                sum = la_fmaf(-L[i * n + k], L[j * n + k], sum);
            }
            if (i == j) {
                if (LA_UNLIKELY(sum <= LA_EPSILON_SQ)) {
                    return false; /* Not positive definite */
                }
                L[i * n + j] = sqrtf(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }

    /* Forward substitution: L * y = b (stored in x) */
    for (size_t i = 0; i < n; ++i) {
        float sum = b[i];
        for (size_t k = 0; k < i; ++k) {
            sum = la_fmaf(-L[i * n + k], x[k], sum);
        }
        x[i] = sum / L[i * n + i];
    }

    /* Back substitution: L^T * x = y */
    for (size_t i = n; i-- > 0;) {
        float sum = x[i];
        for (size_t k = i + 1; k < n; ++k) {
            sum = la_fmaf(-L[k * n + i], x[k], sum);
        }
        x[i] = sum / L[i * n + i];
    }
    return true;
}

LA_API bool la_solve_ldlt(const float *LA_RESTRICT A,
                          const float *LA_RESTRICT b,
                          float *LA_RESTRICT x,
                          float *LA_RESTRICT L,
                          float *LA_RESTRICT D,
                          size_t n) {
    if (n == 0) {
        return true;
    }
    memset(L, 0, n * n * sizeof(float));

    /* Factorization: A = L * D * L^T (L has unit diagonal) */
    for (size_t j = 0; j < n; ++j) {
        float d = A[j * n + j];
        for (size_t k = 0; k < j; ++k) {
            float ljk = L[j * n + k];
            d = la_fmaf(-ljk * ljk, D[k], d);
        }
        if (LA_UNLIKELY(la_absf(d) <= LA_EPSILON_SQ)) {
            return false;
        }
        D[j] = d;
        L[j * n + j] = 1.0f;

        float inv_d = 1.0f / d;
        for (size_t i = j + 1; i < n; ++i) {
            float sum = A[i * n + j];
            for (size_t k = 0; k < j; ++k) {
                sum = la_fmaf(-L[i * n + k] * L[j * n + k], D[k], sum);
            }
            L[i * n + j] = sum * inv_d;
        }
    }

    /* L * y = b */
    for (size_t i = 0; i < n; ++i) {
        float sum = b[i];
        for (size_t k = 0; k < i; ++k) {
            sum = la_fmaf(-L[i * n + k], x[k], sum);
        }
        x[i] = sum;
    }

    /* D * z = y */
    for (size_t i = 0; i < n; ++i) {
        x[i] /= D[i];
    }

    /* L^T * x = z */
    for (size_t i = n; i-- > 0;) {
        float sum = x[i];
        for (size_t k = i + 1; k < n; ++k) {
            sum = la_fmaf(-L[k * n + i], x[k], sum);
        }
        x[i] = sum;
    }
    return true;
}

LA_API uint32_t la_solve_pgs(const float *LA_RESTRICT A,
                             const float *LA_RESTRICT b,
                             const float *LA_RESTRICT lo,
                             const float *LA_RESTRICT hi,
                             float *LA_RESTRICT x,
                             size_t n,
                             uint32_t max_iters,
                             float tol) {
    uint32_t iter = 0;
    for (; iter < max_iters; ++iter) {
        float max_delta = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diag = A[i * n + i];
            if (LA_UNLIKELY(la_absf(diag) <= LA_EPSILON_SQ)) {
                continue;
            }
            float sigma = 0.0f;
            const float *row = &A[i * n];
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    sigma = la_fmaf(row[j], x[j], sigma);
                }
            }
            float xi = (b[i] - sigma) / diag;
            if (lo && xi < lo[i]) xi = lo[i];
            if (hi && xi > hi[i]) xi = hi[i];
            float delta = la_absf(xi - x[i]);
            if (delta > max_delta) {
                max_delta = delta;
            }
            x[i] = xi;
        }
        if (max_delta <= tol) {
            return iter + 1;
        }
    }
    return iter;
}

#endif /* LA_IMPL_ENABLED */

LA_EXTERN_C_END

#endif /* LA_PHYSICS_H */
