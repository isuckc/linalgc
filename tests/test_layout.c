#include "acutest.h"
#include "linalg/la_types.h"

#include <stddef.h>

static void test_sizes_and_alignments(void) {
    TEST_CHECK(sizeof(la_vec2) == 8);
    TEST_CHECK(sizeof(la_vec3) == 12);
    TEST_CHECK(sizeof(la_vec3a) == 16);
    TEST_CHECK(sizeof(la_vec4) == 16);
    TEST_CHECK(sizeof(la_quat) == 16);
    TEST_CHECK(sizeof(la_mat2) == 16);
    TEST_CHECK(sizeof(la_mat3) == 36);
    TEST_CHECK(sizeof(la_mat3a) == 48);
    TEST_CHECK(sizeof(la_mat4) == 64);
    TEST_CHECK(sizeof(la_transform) == 48);
    TEST_CHECK(sizeof(la_aabb) == 32);
    TEST_CHECK(sizeof(la_plane) == 16);
    TEST_CHECK(sizeof(la_ray) == 32);
    TEST_CHECK(sizeof(la_frustum) == 96);
    TEST_CHECK(sizeof(la_vec3_aosoa4) == 48);
    TEST_CHECK(sizeof(la_vec3_aosoa8) == 96);

    TEST_CHECK(LA_ALIGNOF(la_vec2) == 4);
    TEST_CHECK(LA_ALIGNOF(la_vec3) == 4);
    TEST_CHECK(LA_ALIGNOF(la_vec3a) == 16);
    TEST_CHECK(LA_ALIGNOF(la_vec4) == 16);
    TEST_CHECK(LA_ALIGNOF(la_quat) == 16);
    TEST_CHECK(LA_ALIGNOF(la_mat2) == 16);
    TEST_CHECK(LA_ALIGNOF(la_mat3) == 4);
    TEST_CHECK(LA_ALIGNOF(la_mat3a) == 16);
    TEST_CHECK(LA_ALIGNOF(la_mat4) == 16);
    TEST_CHECK(LA_ALIGNOF(la_transform) == 16);
    TEST_CHECK(LA_ALIGNOF(la_aabb) == 16);
    TEST_CHECK(LA_ALIGNOF(la_plane) == 16);
    TEST_CHECK(LA_ALIGNOF(la_ray) == 16);
    TEST_CHECK(LA_ALIGNOF(la_frustum) == 16);
    TEST_CHECK(LA_ALIGNOF(la_vec3_aosoa4) == 16);
    TEST_CHECK(LA_ALIGNOF(la_vec3_aosoa8) == 32);
}

static void test_field_offsets(void) {
    /* Verify contiguous GPU-uploadable field offsets */
    TEST_CHECK(offsetof(la_vec4, x) == 0);
    TEST_CHECK(offsetof(la_vec4, y) == 4);
    TEST_CHECK(offsetof(la_vec4, z) == 8);
    TEST_CHECK(offsetof(la_vec4, w) == 12);

    TEST_CHECK(offsetof(la_mat4, cols) == 0);
    TEST_CHECK(offsetof(la_mat3a, cols) == 0);
    TEST_CHECK(offsetof(la_transform, rot) == 0);
    TEST_CHECK(offsetof(la_transform, pos) == 16);
    TEST_CHECK(offsetof(la_transform, scale) == 32);

    TEST_CHECK(offsetof(la_aabb, min) == 0);
    TEST_CHECK(offsetof(la_aabb, max) == 16);
}

static void test_backend_detection(void) {
    TEST_CHECK(LA_SIMD_NAME != NULL);
    TEST_MSG("Active SIMD backend: %s (FMA=%d)", LA_SIMD_NAME, LA_SIMD_HAS_FMA);
}

TEST_LIST = {
    { "sizes_and_alignments", test_sizes_and_alignments },
    { "field_offsets",        test_field_offsets },
    { "backend_detection",    test_backend_detection },
    { NULL, NULL }
};
