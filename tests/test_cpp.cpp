#include <acutest.h>

#if defined(LA_USE_AMALGAMATION)
#include "linalg.h"
#else
#include "linalg/la_cpp.h"
#endif

static void test_cpp_operators_and_namespace(void) {
    la::vec3 a{ 1.0f, 2.0f, 3.0f };
    la::vec3 b{ 4.0f, -1.0f, 2.0f };

    la::vec3 c = a + b * 2.0f;
    TEST_CHECK(la::approx_eq(c, la::vec3{ 9.0f, 0.0f, 7.0f }, 1e-6f));

    c -= a;
    TEST_CHECK(la::approx_eq(c, 2.0f * b, 1e-6f));

    TEST_CHECK(la::approx_eq(la::dot(a, b), 8.0f, 1e-6f));
    TEST_CHECK(la::approx_eq(la::cross(la::vec3{ 1, 0, 0 }, la::vec3{ 0, 1, 0 }),
                             la::vec3{ 0, 0, 1 }, 1e-6f));

    /* Matrix & quaternion operators */
    la::quat q = la_quat_from_axis_angle(la::vec3{ 0, 0, 1 }, LA_HALF_PI);
    la::vec3 rotated = q * la::vec3{ 1.0f, 0.0f, 0.0f };
    TEST_CHECK(la::approx_eq(rotated, la::vec3{ 0.0f, 1.0f, 0.0f }, 1e-5f));

    la::mat4 m = la::translate(la::vec3{ 1.0f, 2.0f, 3.0f }) * la::to_mat4(q);
    la::mat4 inv = la::inverse_rigid(m);
    TEST_CHECK(la::approx_eq(m * inv, la_mat4_identity(), 1e-5f));

    la::vec4 p{ 1.0f, 0.0f, 0.0f, 1.0f };
    la::vec4 tp = m * p;
    TEST_CHECK(la::approx_eq(tp, la::vec4{ 1.0f, 3.0f, 3.0f, 1.0f }, 1e-5f));
}

static void test_cpp_structured_bindings_and_abi(void) {
    /* Native aggregate structured bindings */
    la::vec3 v3{ 1.5f, 2.5f, 3.5f };
    auto [x, y, z] = v3;
    TEST_CHECK(x == 1.5f && y == 2.5f && z == 3.5f);

    /* Custom 3-element structured binding for 16-byte aligned la::vec3a */
    la::vec3a v3a = la_vec3a_make(7.0f, 8.0f, 9.0f);
    auto [ax, ay, az] = v3a;
    TEST_CHECK(ax == 7.0f && ay == 8.0f && az == 9.0f);

    /* ABI identity: C++ la:: types ARE the C99 la_* types */
    static_assert(std::is_same_v<la::vec3, ::la_vec3>);
    static_assert(std::is_same_v<la::vec3a, ::la_vec3a>);
    static_assert(std::is_same_v<la::vec4, ::la_vec4>);
    static_assert(std::is_same_v<la::mat4, ::la_mat4>);
    static_assert(sizeof(la::mat4) == 64 && alignof(la::mat4) == 16);
}

TEST_LIST = {
    { "cpp_operators_and_namespace",      test_cpp_operators_and_namespace },
    { "cpp_structured_bindings_and_abi",  test_cpp_structured_bindings_and_abi },
    { NULL, NULL }
};
