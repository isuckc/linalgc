# `linalgc` (`linalg.h`)

A **Data-Oriented, SIMD-accelerated linear algebra library** written in portable **C99+** with a zero-cost **C++17+** layer, designed specifically for **physics engines**, **real-time shaders (GLSL / Vulkan / Metal / WGSL)**, and high-throughput simulation workloads.

Distributed as both clean modular headers (`include/linalg/*.h`) and a deterministic single-header amalgamation (**[`linalg.h`](linalg.h)**).

---

## Highlights

- **Data-Oriented Design (DOD)**:
  - Distinct compact AoS (`la_vec3`), SIMD-aligned AoS (`la_vec3a`, `la_vec4`, `la_mat4`), **SoA** views (`la_vec3_soa`, `la_quat_soa`, `la_aabb_soa`), and **AoSoA4 / AoSoA8** cache-aligned blocks (`la_vec3_aosoa4`, `la_vec3_aosoa8`).
  - Zero heap allocations (`malloc`-free), caller-owned buffers, `restrict`-qualified batch kernels, and strict-aliasing-clean SIMD loads/stores (verified under ASan + UBSan).
- **Portable Multi-Backend SIMD**:
  - **ARM64 NEON** (with native FMA and `vld3q`/`vst3q` deinterleaving)
  - **x86-64 SSE2 / SSE4.1 / AVX2+FMA** (4-wide and 8-wide SoA batch paths)
  - **WebAssembly SIMD128**
  - **Pure C99 Scalar Fallback** (`#define LA_FORCE_SCALAR 1`)
- **Shader & GPU ABI Ready**:
  - **Column-major** matrix storage (`la_mat4` = 64 B, 16-byte aligned, matches GLSL/Vulkan/Metal/WGSL directly with zero transpose on upload).
  - Explicit projection builders for every coordinate & depth convention (`_rh_zo`, `_rh_no`, `_lh_zo`, `_lh_no`, plus infinite **Reverse-Z**).
- **Physics Engine Primitives & Solvers**:
  - Drift-free TRS `la_transform`, Arvo SIMD AABB transforms, branchless SIMD slab ray-AABB, Möller-Trumbore ray-triangle, Gribb-Hartmann frustum extraction (ZO & NO).
  - Rigid-body inertia tensors (box, sphere, hollow sphere, cylinder, capsule, parallel-axis theorem, fast diagonal world inverse inertia).
  - 3×3 symmetric **Jacobi eigensolver**, Müller iterative **polar rotation extraction**, and zero-allocation **Cholesky**, **LDLᵀ**, and **Projected Gauss-Seidel (box-constrained LCP)** solvers.
- **Ergonomic C99 & C++17 Dual API**:
  - Clean prefixed C99 API (`la_vec3_cross`, `la_mat4_mul`, `la_quat_slerp`).
  - Opt-out C++17 `namespace la` with zero-cost POD aliases (`std::is_same_v<la::mat4, ::la_mat4>`), ADL-safe operators (`+`, `-`, `*`, `/`), overloaded free functions, and structured bindings (`auto [x, y, z] = v;`).

---

## Quickstart

### Single-Header Drop-In (C99)

Copy [`linalg.h`](linalg.h) into your project and include it:

```c
#include "linalg.h"

int main(void) {
    la_mat4 view = la_mat4_look_at_rh(
        la_vec3_make(0.0f, 2.0f, 5.0f),
        la_vec3_zero(),
        la_vec3_make(0.0f, 1.0f, 0.0f)
    );
    la_mat4 proj = la_mat4_perspective_rh_zo(la_deg2rad(60.0f), 16.0f / 9.0f, 0.1f, 500.0f);
    la_mat4 vp   = la_mat4_mul(proj, view);

    la_frustum frustum = la_frustum_from_vp_zo(vp);
    la_aabb box = la_aabb_from_center_extents(la_vec3_zero(), la_vec3_splat(1.0f));
    bool visible = la_frustum_test_aabb(frustum, box);
    (void)visible;
    return 0;
}
```

### Multi-TU Separate Implementation Mode (Optional)

Hot math (`vec`, `mat`, `quat`, `geom`) is always `static inline`. For larger codebases that want batch kernels and solvers compiled once:

```c
// In most .c / .cpp files:
#define LA_SEPARATE_IMPL
#include "linalg.h"

// In exactly ONE .c file:
#define LA_SEPARATE_IMPL
#define LINALG_IMPLEMENTATION
#include "linalg.h"
```

### C++17 Ergonomics

```cpp
#include "linalg.h"

void step_camera() {
    la::quat q = la_quat_from_axis_angle(la::vec3{0.0f, 1.0f, 0.0f}, LA_HALF_PI);
    la::vec3 dir = q * la::vec3{1.0f, 0.0f, 0.0f};
    la::mat4 m = la::translate(la::vec3{0.0f, 2.0f, 5.0f}) * la::to_mat4(q);
    la::mat4 inv = la::inverse_rigid(m);

    la::vec3a p = la_vec3a_make(dir.x, dir.y, dir.z);
    auto [x, y, z] = p; // C++17 structured bindings (binds x, y, z; ignores _pad)
    (void)inv; (void)x; (void)y; (void)z;
}
```

### Configuration Macros

Define any of these before `#include "linalg.h"`:

| Macro | Effect |
| :--- | :--- |
| `LA_FORCE_SCALAR 1` | Force pure C99 scalar fallback (useful for cross-checking or exotic targets) |
| `LA_SEPARATE_IMPL` | Emit `extern` declarations for batch/solver kernels; pair with `LINALG_IMPLEMENTATION` in one TU |
| `LA_NO_CPP` | Disable C++17 `namespace la`, operators, and structured bindings |
| `LA_NO_CPP_OPERATORS` | Keep `namespace la` aliases/functions but disable global arithmetic operators |
| `LA_NO_FORCE_INLINE` | Use plain `static inline` instead of `always_inline` / `__forceinline` |

---

## Memory Layout & Shader ABI Contract

All types are verified at compile time with `LA_STATIC_ASSERT`.

| Type | Size | Align | GPU / Physics Role |
| :--- | ---: | ---: | :--- |
| `la_vec2` | 8 B | 4 B | GLSL `vec2`, Metal `float2`, UVs, 2D physics |
| `la_vec3` | 12 B | 4 B | Compact vertex buffers, `std430` storage |
| `la_vec3a` | 16 B | 16 B | SIMD 3D vector (`_pad == 0`), GLSL `std140` `vec3`, Metal `simd_float3` |
| `la_vec4` | 16 B | 16 B | GLSL `vec4`, Metal `float4`, single 128-bit SIMD register |
| `la_quat` | 16 B | 16 B | Scalar-last `(x, y, z, w)` rotation (glTF, PhysX, Bullet, Jolt, GLM) |
| `la_mat2` | 16 B | 16 B | Column-major 2×2 (fits in one 128-bit SIMD register) |
| `la_mat3` | 36 B | 4 B | Compact column-major 3×3 (inertia tensors, compact rotations) |
| `la_mat3a` | 48 B | 16 B | SIMD column-major 3×3 (GLSL `std140` `mat3`, Metal `float3x3`, normal matrices) |
| `la_mat4` | 64 B | 16 B | Exact 1 cache line, column-major GLSL / Vulkan / Metal / WGSL `mat4` |
| `la_transform` | 48 B | 16 B | Drift-free TRS (`rot`, `pos`, `scale`) |
| `la_aabb` | 32 B | 16 B | Broadphase bounding box (`min`, `max`) |
| `la_plane` | 16 B | 16 B | `nx*x + ny*y + nz*z + d = 0` |
| `la_ray` | 32 B | 16 B | `origin`, `dir` |
| `la_frustum` | 96 B | 16 B | 6 normalized inward-pointing culling planes (L, R, B, T, N, F) |
| `la_vec3_aosoa4` | 48 B | 16 B | 4-wide SIMD AoSoA block (`x[4], y[4], z[4]`) |
| `la_vec3_aosoa8` | 96 B | 32 B | 8-wide AVX2 AoSoA block (`x[8], y[8], z[8]`) |

---

## Projection & Coordinate Conventions

| Graphics API | Handedness | Clip Depth | Function |
| :--- | :--- | :--- | :--- |
| Vulkan / Metal / WebGPU / DX12 (RH) | Right-Handed (`-Z` forward) | `[0, 1]` (ZO) | `la_mat4_perspective_rh_zo` / `la_mat4_ortho_rh_zo` |
| Vulkan / Metal / WebGPU / DX12 (Rev-Z) | Right-Handed (`-Z` forward) | `[1, 0]` (∞ far) | `la_mat4_perspective_inf_revz_rh` |
| DirectX / Metal (LH) | Left-Handed (`+Z` forward) | `[0, 1]` (ZO) | `la_mat4_perspective_lh_zo` / `la_mat4_ortho_lh_zo` |
| OpenGL (RH) | Right-Handed (`-Z` forward) | `[-1, 1]` (NO) | `la_mat4_perspective_rh_no` / `la_mat4_ortho_rh_no` |
| OpenGL (LH) | Left-Handed (`+Z` forward) | `[-1, 1]` (NO) | `la_mat4_perspective_lh_no` / `la_mat4_ortho_lh_no` |

---

## Benchmarks (Apple Silicon ARM64, Single Core, `-O3`)

Measured with `make bench` (Apple Clang 21, `arm64-apple-darwin`, single thread):

| Benchmark | ARM NEON (`FMA=1`) | Scalar Fallback (`-DLA_FORCE_SCALAR=1`) | Speedup |
| :--- | ---: | ---: | ---: |
| `mat4 * mat4` (dense stream) | **3.10 ns/op** (36.18 GF/s) | 6.70 ns/op (16.72 GF/s) | **2.16×** |
| `mat4_inv_rigid` vs `mat4_inv` | **2.62 ns/op** vs 10.13 ns/op | 2.69 ns/op | **3.87×** (algorithmic) |
| Transform 64K pts (`AoSoA4 SIMD`) | **5,030 M/s** (**90.53 GF/s**) | 1,825 M/s (32.84 GF/s) | **2.76×** |
| AABB Broadphase 64K boxes (`SoA SIMD`) | **4,649 M/s** | 1,328 M/s | **3.50×** |
| Quat Rotate 64K vecs (`SoA SIMD`) | **2,166 M/s** | 2,173 M/s *(auto-vec)* | Memory-BW bound |
| Symplectic Euler 64K bodies (`SoA`) | **1,739 M/s** | 1,807 M/s *(auto-vec)* | Memory-BW bound |

> **Data-Oriented Takeaway**: Packing 3D point streams into cache-contiguous `la_vec3_aosoa4` blocks hits **90.5 GFLOP/s on a single core** — over **2.1× faster** than standard AoS `la_vec3` arrays (`42.8 GF/s`) due to 100% SIMD lane utilization and single-stream sequential prefetching.

---

## Building, Testing & Regenerating `linalg.h`

All builds compile cleanly with `-Wall -Wextra -Wpedantic -Werror`:

```sh
make all                # Native SIMD + Scalar fallback + ASan/UBSan test suites (C99)
make test-cpp           # C++17 operators, namespace la, structured bindings, ASan/UBSan
make amalgamate         # Regenerate standalone linalg.h from include/linalg/*.h
make test-amalgamation  # Verify linalg.h in C99 header-only, multi-TU separate impl, and C++17
make bench              # Run native SIMD vs forced scalar micro-benchmarks
```

---

## Project Layout

```text
├── linalg.h                  # Generated single-header amalgamation (C99+ / C++17+)
├── include/linalg/
│   ├── la_config.h           # Compiler/SIMD detection, alignment, LA_API, constants
│   ├── la_types.h            # POD types, SoA/AoSoA views, static ABI assertions
│   ├── la_scalar.h           # FMA, exact/fast rsqrt, smoothstep, comparisons
│   ├── la_simd.h             # Portable f32x4 / u32x4 / f32x8 SIMD abstraction
│   ├── la_vec.h              # vec2, vec3, vec3a, vec4 + Duff orthonormal basis
│   ├── la_mat.h              # mat2, mat3, mat3a, mat4, fast inverses, projections
│   ├── la_quat.h             # Quaternions, Shepperd conversion, SLERP, angvel integration
│   ├── la_geom.h             # TRS transform, Arvo AABB, plane, ray, frustum culling
│   ├── la_batch.h            # DOD SoA/AoSoA4 batch transforms, physics, broadphase, LBS4
│   ├── la_physics.h          # Inertia tensors, 3x3 Jacobi eigen, polar decomp, Cholesky/LDL^T/PGS
│   └── la_cpp.h              # C++17 namespace la, operators, structured bindings
├── scripts/amalgamate.py     # Deterministic amalgamation generator
├── tests/                    # acutest suites (layout, simd, vec, mat, quat, geom, batch, physics, cpp, amalgamation)
└── bench/bench_linalg.c      # SIMD vs Scalar & AoS vs SoA vs AoSoA4 benchmarks
```
