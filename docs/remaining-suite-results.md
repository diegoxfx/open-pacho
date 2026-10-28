# Map-reduce, preparation, and batched BLAS

Historical snapshot. The [current continuation](numerics-vulkan-results.md) adds
Vulkan executor optimization, register GEMM, and numerical stability/diagnostics.

Stage dated 2026-09-06, following [Vulkan execution and staging](vulkan-execution-results.md).
Local machine: RTX 5060 Ti 16 GB, Ryzen 7 5700X3D, Windows 11, driver 616.64.
Only this machine has physical validation. Specialization by hardware/OS family
remains a [secondary track](hardware-specialization.md).

## Implementations

1. **Vulkan map-reduce.** Fuses the elementwise producer with `sum` and the
   reductions forming `mean`/norms. Preserves producer dtype, requested
   accumulation, broadcasting, arbitrary axes, `keepdims`, booleans, and empty
   arrays. Uses compact partials for large reductions; does not create the full
   producer array. A producer shared between different consumers or returned
   as output is still materialized. GEMM/GEMV retain their boundary.
2. **Prepared compiled calls.** `compiled_function.prepare(*examples)` fixes the
   specialization and returns a callable with binding validation. Avoids general
   binding in the complete positional case and the LRU lookup. Dynamic scalars
   still change; new inputs must preserve shape, dtype, and context. It retains
   neither example arrays nor shared live outputs. It does not generate a native
   executor or a persistent GPU command graph.
3. **Prepared systems.** `solvers.prepare(A, method="cg" | "jacobi" | "sd")`
   takes an owned copy and validates the matrix once. Each `.solve(b)` retains
   b/x0 validation, tolerances, and the true residual at every iteration. Jacobi
   reuses the validated diagonal. Precision is fixed; a promotion required by b
   is rejected. Ordinary GPU solvers also prepare their compiled calls once per
   solve instead of repeating lookups at each iteration.
4. **Native batched CLBlast.** Uses S/DgemmStridedBatched for regular positive
   offsets and S/DgemmBatched for other available patterns. Broadcasting uses
   offsets into the original buffers. If optional symbols are missing, GEMM
   from the same provider is retained; a native error propagates. Existing
   protection for recycled GEMV output containing NaN/Inf continues.
5. **Tiled OpenCL GEMM.** Portable 16×16 path if the device supports the group
   and local memory; 8×8 with lower capabilities; previous kernel for small cases
   or insufficient capabilities. Preserves broadcasting and incomplete edge
   products. Allows multiply-add contraction like the previous GEMM; does not
   enable fast-math or change elementwise fusion.

Complete examples and limits are in [compilation](compilation.md) and
[execution and performance](performance.md). Users still choose the backend
and provider explicitly.

## Measurement method

The broad suite retains 319 configurations: 172 operations/steps, 63 full solvers,
and 84 scale/batching cases. Each mode has seven synchronized samples; resident
excludes upload/download and end-to-end includes them. Includes NumPy with
1/8/16 threads, OpenCL custom/CLBlast, Vulkan, CuPy, and direct cuBLAS/nvmath
references where supported. These are not 319 distinct workloads: the count
includes implementations, sizes, and dtypes. Neither PyTorch/JAX nor an OpenPacho
CUDA backend is included at this stage.

The new `benchmarks.prepared` control measures 48 method, size, dtype, and backend
configurations, with five modes: ordinary solve, prepared solve, preparation
separately, and both single-solve flows from the host. Each result is checked
against the ordinary solver, its history, and a residual computed in float64.
It uses rtol=1e-5/1e-8 for float32/64 and maxiter=1000; the broad suite uses
rtol=1e-4/1e-7 and maxiter=400. Each variant is compared within its own control,
without equating timings across different tolerances. NumPy in this control is
the OpenPacho backend with one thread; it must not be confused with native NumPy
in the broad suite.

Before/after profiles use the frozen wheel from the previous stage and current
source in separate processes. The energy profile distinguishes three paths:
compiled API, prepared API with validation, and private plan execution without
that validation. The latter is a diagnostic control only.

Historical medians alone do not prove causality: unmodified paths such as CuPy
also change. Clocks were not fixed and Windows driver/scheduler state was not
fully controlled. Unfavorable results and interleaved controls in the same
process are retained to separate some of that variation.

## Results and attribution

Medians in milliseconds. Local controls alternate variants within the same
process and retain eleven samples per variant after warm-up.

| Resident control | Previous path | New path | Interpretation |
| --- | ---: | ---: | --- |
| CLBlast, 64 matrices of 32×32 | 2.028 | 0.236 | Strided-batched, 8.6× less time than the GEMM loop |
| CLBlast, 16 matrices of 128×128 | 0.977 | 0.242 | Strided-batched, 4.0× |
| CLBlast, broadcasting (8,1,64,64) × (1,4,64,64) | 2.219 | 0.490 | Batched with offsets, 4.5× |
| OpenCL custom, GEMM 1024² float32 | 4.811 | 1.860 | Untiled versus 16×16 tile |
| OpenCL custom, GEMM 2048² float32 | 24.886 | 18.506 | 1.34×; still far from optimized BLAS |
| OpenCL custom, GEMM 1024² float64 | 8.526 | 8.590 | No meaningful improvement |

The initial tiling variant with contraction disabled gave 18.976 ms for float64
1024². That policy was discarded for GEMM: the previous path already allowed
contraction. The change does not affect the prohibition on contraction in fused
elementwise expressions. Controls are retained in `tile-control*.json`.

In the broad suite, custom float32 1024² GEMM changed from 5.544 to 1.794 ms
resident, but 2048² changed from 16.358 to 18.265 ms. CLBlast with irregular
broadcasting changed from 1.105 to 1.661 ms, although the interleaved control
favors the native path. These historical measurements are not presented as a
uniform improvement, nor is all variation attributed to the kernel change.

**Map-reduce does reduce verifiable work.** In the Vulkan 2048×32 energy profile,
dispatches and logical allocations drop from two to one. In profile CG (two
iterations), dispatches drop from 43 to 31 and `empty` calls from 51 to 32;
nine downloads and ten counted synchronizations remain, including the profile's
external synchronization. These are backend calls, not new native allocations:
the pool can reuse storage.

The separate Vulkan energy profile changes from 0.544 to 0.409 ms. In the broad
suite, energy 32768×32 changes from 2.390 to 0.840 ms resident, versus 3.413 ms
for single-threaded NumPy. However, with upload and download Vulkan takes
5.503 ms versus 3.325 ms for NumPy. Execution improvement does not eliminate
transfer cost.

**Preparing the executor does not resolve the main bottleneck.** For OpenCL
energy 2048×32, the compiled API gives 0.099 ms, the prepared API with validation
0.072 ms, and the private plan without validation 0.066 ms. Vulkan gives
0.409/0.441/0.391 ms: no measurable prepared-API gain in that round. The
interleaved CG control compares cached versus prepared calls, both validated:
OpenCL gives 2.318/2.327 ms and Vulkan 12.824/11.073 ms. This does not justify
rewriting the entire compiler before reducing backend execution cost.

**Prepared systems have a concrete, still modest range of usefulness.**
With 2048 float32 variables and rtol=1e-5:

| Backend / method | Ordinary resident | Prepared resident | Prepare A once |
| --- | ---: | ---: | ---: |
| OpenCL custom / CG | 20.362 | 18.173 | 0.824 |
| OpenCL custom / Jacobi | 23.756 | 23.938 | 1.073 |
| CLBlast / CG | 23.380 | 24.503 | 0.538 |
| Vulkan / CG | 65.042 | 62.144 | 1.079 |
| Vulkan / Jacobi | 88.671 | 85.958 | 2.421 |
| Vulkan / steepest descent | 141.201 | 141.728 | 1.226 |

No universal reuse count for amortization is claimed: when savings approach
observed variation, dividing by that difference yields an unstable estimate.
All five full modes are in `prepared.json`.

The broad suite's Jacobi 2048 float32, with its rtol=1e-4, still gives
**9.443 ms in eight-thread NumPy, 22.308 in custom OpenCL, and 76.424 in Vulkan**,
with resident data. Correctness and fewer dispatches do not mean this gap has
closed; there were also median regressions against the previous stage.

## Verification

- **625 tests pass, one skipped** because it needs two OpenCL devices.
  Includes Vulkan index limits, producer rounding, axes and empty arrays,
  prepared ownership/bindings, convergence history, tiled variants and fallback,
  CLBlast ABI and errors, and recycled output containing NaN.
- **319 comparison configurations and 48 preparation configurations**, all with
  seven samples per mode and full validation, with no recorded errors.
- Wheel built from sdist and installed without GPU dependencies: **208 tests
  pass, 401 skipped** because optional backends are absent.
- Ruff and formatting pass. Distribution in `.artifacts/dist-remaining-suite`;
  data, full table, figure, and manifest in `.artifacts/benchmarks/remaining-suite`.

## Remaining gaps

- Resident solvers still lose to NumPy in several cases. Preparing bindings does
  not eliminate driver calls or host convergence observations. The next execution
  work must measure Vulkan submission/recording and a small native executor,
  preserving in-flight resources and locks.
- Fusion of GEMV with updates/reductions is missing. CLBlast does not prevent
  adding custom paths, but this stage does not implement BLAS epilogues or a
  completely autonomous GPU solver.
- System preparation avoids repeated validation; it is neither a factorization
  nor a preconditioner and does not guarantee a timing improvement. It adds a copy of A.
- Transfers and launch overhead still dominate low-arithmetic-intensity cases.
  No change removes PCIe or makes every small GPU operation worthwhile.
  Resident execution losses are a separate gap.
- The norm still uses a sum of squares: underflow can cause false convergence;
  overflow and extreme coefficients also need a scaled path. Fusion and
  preparation do not resolve this numerical debt.
- AMD/Intel, devices without fp64, and other operating systems require physical
  validation. A CUDA backend and sparse matrices remain outside this stage.

## Reproduction

With OpenCL, CLBlast, Vulkan/glslang, and CUDA references configured:

```powershell
python -m benchmarks.execution --output .artifacts/benchmarks/remaining-suite/operations.json
python -m benchmarks.execution --solvers-only --output .artifacts/benchmarks/remaining-suite/solvers.json
python -m benchmarks.scale --output .artifacts/benchmarks/remaining-suite/scale.json
python -m benchmarks.prepared --output .artifacts/benchmarks/remaining-suite/prepared.json
python -m benchmarks.gaps --backends opencl,opencl-clblast,vulkan --output .artifacts/benchmarks/remaining-suite/after-profile.json
python -m benchmarks.gaps_tables --root .artifacts/benchmarks/remaining-suite --operations operations.json --solvers solvers.json --scale scale.json --output .artifacts/benchmarks/remaining-suite/report
```

Private JSON files retain samples, errors, and provenance; the CSV/Markdown report
retains all configurations. Initial tiling experiments are saved separately and
are not presented as the final state.
