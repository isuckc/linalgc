#define _POSIX_C_SOURCE 200809L

#include "linalg/la_batch.h"
#include "linalg/la_physics.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static volatile float g_sink = 0.0f;

static float *alloc_f32_aligned(size_t n) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, n * sizeof(float)) != 0) {
        abort();
    }
    return (float *)ptr;
}

static void *alloc_bytes_aligned(size_t bytes) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        abort();
    }
    return ptr;
}

int main(void) {
    printf("Backend: %s (FMA=%d)\n", LA_SIMD_NAME, LA_SIMD_HAS_FMA);
    printf("------------------------------------------------------------------------\n");
    printf("%-38s %10s %14s %10s\n", "Benchmark", "Time (ms)", "Throughput", "Metric");
    printf("------------------------------------------------------------------------\n");

    /* Prepare dense runtime matrices to prevent compile-time constant folding */
    const size_t M_COUNT = 256;
    la_mat4 *mats_a = (la_mat4 *)alloc_bytes_aligned(M_COUNT * sizeof(la_mat4));
    la_mat4 *mats_b = (la_mat4 *)alloc_bytes_aligned(M_COUNT * sizeof(la_mat4));
    la_mat4 *mats_out = (la_mat4 *)alloc_bytes_aligned(M_COUNT * sizeof(la_mat4));
    for (size_t i = 0; i < M_COUNT; ++i) {
        float ang = 0.1f + (float)i * 0.023f;
        la_vec3 axis = la_vec3_norm(la_vec3_make(1.0f + (float)(i & 3), 2.0f, -1.5f + (float)(i & 7) * 0.2f));
        la_vec3 tr = la_vec3_make((float)i * 0.1f, -(float)i * 0.05f, 3.0f);
        mats_a[i] = la_mat4_mul(la_mat4_translate(tr), la_mat4_rotate_axis(axis, ang));
        mats_b[i] = la_mat4_mul(la_mat4_translate(la_vec3_neg(tr)), la_mat4_rotate_axis(axis, -ang * 0.5f));
    }

    /* 1. Dense mat4 x mat4 throughput (independent stream) */
    {
        const size_t passes = 20000;
        double t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            for (size_t i = 0; i < M_COUNT; ++i) {
                mats_out[i] = la_mat4_mul(mats_a[i], mats_b[i]);
            }
        }
        double dt = now_sec() - t0;
        g_sink += mats_out[M_COUNT - 1].cols[0].x;

        double total = (double)passes * (double)M_COUNT;
        double ns_op = (dt * 1e9) / total;
        double gflops = (total * 112.0) / (dt * 1e9);
        printf("%-38s %10.2f %10.2f ns/op %7.2f GF/s\n", "mat4 * mat4 (dense stream)", dt * 1e3, ns_op, gflops);
    }

    /* 2. mat4 general inverse vs rigid inverse */
    {
        const size_t passes = 10000;
        double total = (double)passes * (double)M_COUNT;

        double t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            for (size_t i = 0; i < M_COUNT; ++i) {
                mats_out[i] = la_mat4_inv(mats_a[i]);
            }
        }
        double dt_gen = now_sec() - t0;
        g_sink += mats_out[M_COUNT - 1].cols[3].x;
        printf("%-38s %10.2f %10.2f ns/op\n", "mat4_inv (general 4x4)", dt_gen * 1e3, (dt_gen * 1e9) / total);

        t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            for (size_t i = 0; i < M_COUNT; ++i) {
                mats_out[i] = la_mat4_inv_rigid(mats_a[i]);
            }
        }
        double dt_rig = now_sec() - t0;
        g_sink += mats_out[M_COUNT - 1].cols[3].x;
        printf("%-38s %10.2f %10.2f ns/op\n", "mat4_inv_rigid (orthonormal R+T)", dt_rig * 1e3, (dt_rig * 1e9) / total);
    }

    /* 3. Batch transform 65,536 points: AoS vs SoA vs AoSoA4 */
    {
        const size_t N = 65536;
        const size_t blocks = N / 4;
        const size_t passes = 500;
        la_mat4 m = mats_a[17];

        la_vec3 *aos_src = (la_vec3 *)alloc_bytes_aligned(N * sizeof(la_vec3));
        la_vec3 *aos_dst = (la_vec3 *)alloc_bytes_aligned(N * sizeof(la_vec3));
        float *sx = alloc_f32_aligned(N), *sy = alloc_f32_aligned(N), *sz = alloc_f32_aligned(N);
        float *dx = alloc_f32_aligned(N), *dy = alloc_f32_aligned(N), *dz = alloc_f32_aligned(N);
        la_vec3_aosoa4 *blk_src = (la_vec3_aosoa4 *)alloc_bytes_aligned(blocks * sizeof(la_vec3_aosoa4));
        la_vec3_aosoa4 *blk_dst = (la_vec3_aosoa4 *)alloc_bytes_aligned(blocks * sizeof(la_vec3_aosoa4));

        for (size_t i = 0; i < N; ++i) {
            aos_src[i] = la_vec3_make((float)(i & 255) * 0.1f, (float)((i * 3) & 255) * 0.1f, (float)((i * 7) & 255) * 0.1f);
        }
        la_vec3_soa soa_src = { sx, sy, sz };
        la_vec3_soa_const soa_src_c = { sx, sy, sz };
        la_vec3_soa soa_dst = { dx, dy, dz };
        la_batch_vec3_aos_to_soa(soa_src, aos_src, N);
        la_batch_vec3_aos_to_aosoa4(blk_src, aos_src, N);

        double t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            for (size_t i = 0; i < N; ++i) {
                aos_dst[i] = la_mat4_mul_point3(m, aos_src[i]);
            }
        }
        double dt_aos = now_sec() - t0;
        g_sink += aos_dst[N - 1].x;
        printf("%-38s %10.2f %10.1f M/s   %7.2f GF/s\n", "transform 64K pts (AoS vec3)",
               dt_aos * 1e3, ((double)N * (double)passes) / (dt_aos * 1e6),
               ((double)N * (double)passes * 18.0) / (dt_aos * 1e9));

        t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            la_batch_mat4_mul_points_soa(soa_dst, soa_src_c, m, N);
        }
        double dt_soa = now_sec() - t0;
        g_sink += dx[N - 1];
        printf("%-38s %10.2f %10.1f M/s   %7.2f GF/s\n", "transform 64K pts (SoA SIMD)",
               dt_soa * 1e3, ((double)N * (double)passes) / (dt_soa * 1e6),
               ((double)N * (double)passes * 18.0) / (dt_soa * 1e9));

        t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            la_batch_mat4_mul_points_aosoa4(blk_dst, blk_src, m, blocks);
        }
        double dt_aosoa = now_sec() - t0;
        g_sink += blk_dst[blocks - 1].x[0];
        printf("%-38s %10.2f %10.1f M/s   %7.2f GF/s\n", "transform 64K pts (AoSoA4 SIMD)",
               dt_aosoa * 1e3, ((double)N * (double)passes) / (dt_aosoa * 1e6),
               ((double)N * (double)passes * 18.0) / (dt_aosoa * 1e9));

        /* 4. Symplectic Euler 64K bodies */
        float *vx = alloc_f32_aligned(N), *vy = alloc_f32_aligned(N), *vz = alloc_f32_aligned(N);
        float *ax = alloc_f32_aligned(N), *ay = alloc_f32_aligned(N), *az = alloc_f32_aligned(N);
        for (size_t i = 0; i < N; ++i) {
            vx[i] = 1.0f; vy[i] = 2.0f; vz[i] = -1.0f;
            ax[i] = 0.0f; ay[i] = -9.81f; az[i] = 0.0f;
        }
        la_vec3_soa vel = { vx, vy, vz };
        la_vec3_soa_const acc = { ax, ay, az };
        t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            la_batch_integrate_symplectic_euler(soa_dst, vel, acc, 0.016f, N);
        }
        double dt_eul = now_sec() - t0;
        g_sink += dx[N - 1] + vx[N - 1];
        printf("%-38s %10.2f %10.1f M/s\n", "symplectic Euler 64K bodies (SoA)",
               dt_eul * 1e3, ((double)N * (double)passes) / (dt_eul * 1e6));

        /* 5. Batch quat rotate 64K vectors */
        float *qx = alloc_f32_aligned(N), *qy = alloc_f32_aligned(N), *qz = alloc_f32_aligned(N), *qw = alloc_f32_aligned(N);
        la_quat q0 = la_quat_from_axis_angle(la_vec3_norm(la_vec3_make(1.0f, 2.0f, 3.0f)), 0.75f);
        for (size_t i = 0; i < N; ++i) {
            qx[i] = q0.x; qy[i] = q0.y; qz[i] = q0.z; qw[i] = q0.w;
        }
        la_quat_soa_const qsoa = { qx, qy, qz, qw };
        t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            la_batch_quat_rotate_soa(soa_dst, qsoa, soa_src_c, N);
        }
        double dt_qrot = now_sec() - t0;
        g_sink += dx[N - 1];
        printf("%-38s %10.2f %10.1f M/s\n", "quat rotate 64K vecs (SoA SIMD)",
               dt_qrot * 1e3, ((double)N * (double)passes) / (dt_qrot * 1e6));

        /* 6. Broadphase AABB overlap sweep 64K boxes */
        float *minx = alloc_f32_aligned(N), *miny = alloc_f32_aligned(N), *minz = alloc_f32_aligned(N);
        float *maxx = alloc_f32_aligned(N), *maxy = alloc_f32_aligned(N), *maxz = alloc_f32_aligned(N);
        uint32_t *hits = (uint32_t *)alloc_bytes_aligned(N * sizeof(uint32_t));
        for (size_t i = 0; i < N; ++i) {
            float base = (float)i * 2.0f;
            minx[i] = base; miny[i] = 0.0f; minz[i] = 0.0f;
            maxx[i] = base + 1.0f; maxy[i] = 1.0f; maxz[i] = 1.0f;
        }
        la_aabb_soa_const boxes = { minx, miny, minz, maxx, maxy, maxz };
        la_aabb query = la_aabb_from_minmax(la_vec3_make(100.0f, 0.0f, 0.0f), la_vec3_make(120.0f, 1.0f, 1.0f));
        size_t total_hits = 0;
        t0 = now_sec();
        for (size_t p = 0; p < passes; ++p) {
            total_hits += la_batch_aabb_overlap_query(query, boxes, N, hits, N);
        }
        double dt_aabb = now_sec() - t0;
        g_sink += (float)total_hits;
        printf("%-38s %10.2f %10.1f M/s\n", "AABB broadphase 64K boxes (SoA SIMD)",
               dt_aabb * 1e3, ((double)N * (double)passes) / (dt_aabb * 1e6));

        free(aos_src); free(aos_dst);
        free(sx); free(sy); free(sz); free(dx); free(dy); free(dz);
        free(blk_src); free(blk_dst);
        free(vx); free(vy); free(vz); free(ax); free(ay); free(az);
        free(qx); free(qy); free(qz); free(qw);
        free(minx); free(miny); free(minz); free(maxx); free(maxy); free(maxz);
        free(hits);
    }

    free(mats_a); free(mats_b); free(mats_out);
    printf("------------------------------------------------------------------------\n");
    return (g_sink == 1234567.0f) ? 1 : 0;
}
