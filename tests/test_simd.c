#include "acutest.h"
#include "linalg/la_simd.h"

#define CHECK_NEAR(a, b, eps) \
    TEST_CHECK_(la_approx_eqf((a), (b), (eps)), \
        "Expected %.7f ~= %.7f (diff=%.7e, eps=%.7e)", \
        (double)(a), (double)(b), (double)la_absf((a) - (b)), (double)(eps))

static void test_scalar_helpers(void) {
    CHECK_NEAR(la_lerpf(2.0f, 10.0f, 0.0f), 2.0f, 1e-6f);
    CHECK_NEAR(la_lerpf(2.0f, 10.0f, 1.0f), 10.0f, 1e-6f);
    CHECK_NEAR(la_lerpf(2.0f, 10.0f, 0.25f), 4.0f, 1e-6f);

    CHECK_NEAR(la_clampf(-5.0f, 0.0f, 1.0f), 0.0f, 1e-6f);
    CHECK_NEAR(la_clampf(5.0f, 0.0f, 1.0f), 1.0f, 1e-6f);
    CHECK_NEAR(la_smoothstepf(0.0f, 1.0f, 0.5f), 0.5f, 1e-6f);
    CHECK_NEAR(la_smoothstepf(0.0f, 1.0f, 0.0f), 0.0f, 1e-6f);
    CHECK_NEAR(la_smoothstepf(0.0f, 1.0f, 1.0f), 1.0f, 1e-6f);

    CHECK_NEAR(la_signf(3.0f), 1.0f, 1e-6f);
    CHECK_NEAR(la_signf(-3.0f), -1.0f, 1e-6f);
    CHECK_NEAR(la_signf(0.0f), 0.0f, 1e-6f);

    /* Fast rsqrt accuracy check over wide dynamic range */
    const float vals[] = { 0.01f, 0.25f, 1.0f, 2.0f, 4.0f, 17.3f, 1000.0f };
    for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); ++i) {
        float exact = 1.0f / sqrtf(vals[i]);
        float fast = la_rsqrtf_fast(vals[i]);
        CHECK_NEAR(fast, exact, 1e-5f);
    }

    CHECK_NEAR(la_sqrtf_safe(-1.0f), 0.0f, 1e-6f);
    CHECK_NEAR(la_deg2rad(180.0f), LA_PI, 1e-6f);
    CHECK_NEAR(la_rad2deg(LA_PI), 180.0f, 1e-5f);
}

static void test_simd_arithmetic_and_fma(void) {
    la_f32x4 a = la_f32x4_set(1.0f, 2.0f, 3.0f, 4.0f);
    la_f32x4 b = la_f32x4_set(5.0f, -6.0f, 7.0f, -8.0f);
    la_f32x4 c = la_f32x4_set(10.0f, 20.0f, 30.0f, 40.0f);

    la_f32x4 add = la_f32x4_add(a, b);
    CHECK_NEAR(la_f32x4_get_x(add), 6.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_y(add), -4.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_z(add), 10.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_w(add), -4.0f, 1e-6f);

    /* Verify FMA: a * b + c */
    la_f32x4 fma = la_f32x4_fma(a, b, c);
    CHECK_NEAR(la_f32x4_get_x(fma), 15.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_y(fma), 8.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_z(fma), 51.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_w(fma), 8.0f, 1e-6f);

    /* Verify FNMA: c - a * b */
    la_f32x4 fnma = la_f32x4_fnma(a, b, c);
    CHECK_NEAR(la_f32x4_get_x(fnma), 5.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_y(fnma), 32.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_z(fnma), 9.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_w(fnma), 72.0f, 1e-6f);

    /* Verify rsqrt_fast and rcp_fast */
    la_f32x4 pos = la_f32x4_set(1.0f, 4.0f, 9.0f, 16.0f);
    la_f32x4 rs = la_f32x4_rsqrt_fast(pos);
    CHECK_NEAR(la_f32x4_get_x(rs), 1.0f, 1e-4f);
    CHECK_NEAR(la_f32x4_get_y(rs), 0.5f, 1e-4f);
    CHECK_NEAR(la_f32x4_get_z(rs), 1.0f / 3.0f, 1e-4f);
    CHECK_NEAR(la_f32x4_get_w(rs), 0.25f, 1e-4f);

    la_f32x4 rcp = la_f32x4_rcp_fast(pos);
    CHECK_NEAR(la_f32x4_get_x(rcp), 1.0f, 1e-4f);
    CHECK_NEAR(la_f32x4_get_y(rcp), 0.25f, 1e-4f);
    CHECK_NEAR(la_f32x4_get_z(rcp), 1.0f / 9.0f, 1e-4f);
    CHECK_NEAR(la_f32x4_get_w(rcp), 0.0625f, 1e-4f);
}

static void test_simd_dot_masks_and_transpose(void) {
    la_f32x4 a = la_f32x4_set(1.0f, 2.0f, 3.0f, 100.0f);
    la_f32x4 b = la_f32x4_set(4.0f, 5.0f, 6.0f, -2.0f);

    CHECK_NEAR(la_f32x4_dot3(a, b), 32.0f, 1e-5f);
    CHECK_NEAR(la_f32x4_dot4(a, b), -168.0f, 1e-5f);

    la_u32x4 lt = la_f32x4_cmplt(a, b);
    TEST_CHECK(la_u32x4_movemask(lt) == 0x7u); /* lanes 0,1,2 are true; lane 3 is false */

    la_f32x4 sel = la_f32x4_select(lt, a, b);
    CHECK_NEAR(la_f32x4_get_x(sel), 1.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_y(sel), 2.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_z(sel), 3.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_w(sel), -2.0f, 1e-6f);

    /* 4x4 Transpose check */
    la_f32x4 c0 = la_f32x4_set(0, 1, 2, 3);
    la_f32x4 c1 = la_f32x4_set(4, 5, 6, 7);
    la_f32x4 c2 = la_f32x4_set(8, 9, 10, 11);
    la_f32x4 c3 = la_f32x4_set(12, 13, 14, 15);
    la_f32x4_transpose4(&c0, &c1, &c2, &c3);

    CHECK_NEAR(la_f32x4_get_x(c0), 0.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_y(c0), 4.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_z(c0), 8.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_w(c0), 12.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_w(c3), 15.0f, 1e-6f);
}

static void test_simd_deinterleave_and_pod_roundtrip(void) {
    float aos[12] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    la_f32x4 vx, vy, vz;
    la_f32x4_load3_deinterleave(aos, &vx, &vy, &vz);

    CHECK_NEAR(la_f32x4_get_x(vx), 1.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_y(vx), 4.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_z(vx), 7.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_w(vx), 10.0f, 1e-6f);
    CHECK_NEAR(la_f32x4_get_w(vz), 12.0f, 1e-6f);

    float out[12];
    la_f32x4_store3_interleave(out, vx, vy, vz);
    for (int i = 0; i < 12; ++i) {
        CHECK_NEAR(out[i], aos[i], 1e-6f);
    }

    la_vec4 v4 = { 1.5f, -2.5f, 3.5f, -4.5f };
    la_vec4 rt = la_simd_to_vec4(la_vec4_to_simd(v4));
    CHECK_NEAR(rt.x, v4.x, 1e-6f);
    CHECK_NEAR(rt.y, v4.y, 1e-6f);
    CHECK_NEAR(rt.z, v4.z, 1e-6f);
    CHECK_NEAR(rt.w, v4.w, 1e-6f);
}

TEST_LIST = {
    { "scalar_helpers",              test_scalar_helpers },
    { "simd_arithmetic_and_fma",     test_simd_arithmetic_and_fma },
    { "simd_dot_masks_transpose",    test_simd_dot_masks_and_transpose },
    { "simd_deinterleave_roundtrip", test_simd_deinterleave_and_pod_roundtrip },
    { NULL, NULL }
};
