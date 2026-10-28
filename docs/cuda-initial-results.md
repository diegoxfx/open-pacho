# Capability-based Vulkan and initial CUDA

Iteration from 2026-09-12 on RTX 5060 Ti, Ryzen 7 5700X3D, Windows 11, and driver
616.64. Continues [solvers and applications](solver-applications-results.md).
Numbers from previous sessions are retained as historical.

## Portable Vulkan optimization

The previous pool prevented old sizes from permanently blocking new
ones, but alternating nearby shapes still generated many allocations. It now
prefers exact capacity and, if none is available, takes the smallest compatible free buffer
up to `2*max(requested,256)`. It does not create larger buffers for this purpose.

The original VkBuffer capacity is retained when recycling; logical size
is used for descriptors and copies. Native allocated bytes retain
separate accounting. Locks, in-flight references, separate classes
for GPU/upload/readback, and limits of 128 entries/128 MiB are retained.

Four processes alternated: baseline, optimized, baseline, optimized. The
baseline loads a previous-runtime snapshot; compiler and shaders are identical.
Each process retains five matrices/transposes and alternates twenty-step ridge,
GEMV, and transposed GEMV, with 21 samples per case. Resident medians:

| float32 ridge shape | Baseline, two processes | Optimized, two processes |
| --- | ---: | ---: |
| 4096×128 | 20.54–21.72 ms | 7.28–7.78 ms |
| 4097×128 | 9.07–9.94 ms | 7.09–7.52 ms |
| 4096×129 | 9.43–11.65 ms | 6.92–7.79 ms |
| 4097×129 | 22.07–22.71 ms | 6.81–7.55 ms |
| 4095×127 | 36.70–36.92 ms | 7.16–7.34 ms |

A separate profile of ten mixed rounds went from **1,432 allocations and frees
to zero**, retaining 4,100 dispatches. Errors for all fifteen operations
matched exactly across processes; ridge reached relative L2 errors of 5.79e-8 to 8.48e-8
against the float64 trajectory.

The final cache retained 3.05 MiB versus 1.00 MiB. This is retained free memory, not
peak total memory; both versions retained 128 entries. Very different
shapes or a set exceeding limits may still cause allocations.
The result is an improvement to the portable runtime, without vendor/OS-specific tuning.

## First CUDA foundation

`_cuda.py` implements a retained primary context, dedicated stream, resident memory,
bounded pool, NVRTC, arithmetic/broadcasting, gather, floating-point reductions, and
2D vector/matrix products. The default provider is cuBLAS through
nvmath bindings; a direct custom kernel can also be selected. CuPy is not
a runtime dependency. There is no CUDA fusion, stable norms, solvers, or batching.

Selection and limits are in the [CUDA contract](cuda.md). `compile` can already
capture and reuse the plan, but CUDA rows are explicitly labeled
`eager-primitives`: they do not claim fused kernels.

Review added protection at important native boundaries: completing
upload before publishing the array to the nonblocking stream, removing every
freed pointer from the pool even if a later free fails, and synchronizing
module eviction before loading a new one. CPU tests reproduce
these conditions, and real tests check context ownership/restoration.

cuBLAS uses `PEDANTIC_MATH`; NVRTC disables FMA contraction, FTZ, and fast-math.
These are explicit decisions at this stage. Differences from CuPy are not attributed
to Tensor Cores or the hardware's physical limit without a corresponding control.

## Applications

The suite compares the same expressions, data, dtypes, preparation, and twenty
ridge steps. It includes NumPy with 1/8 threads, OpenCL custom/CLBlast, Vulkan, CuPy, and
both CUDA providers. OpenPacho appears in eager and compiled-plan mode; CUDA
still lacks fusion. There are 13 interleaved implementations, seven resident
and complete samples per case. Preparation and diagnostics remain visible.

For 16384×256 float32, medians of OpenPacho plans and controls are:

| Implementation | Resident/complete covariance | Covariance L2 error vs float64 | Resident/complete ridge |
| --- | ---: | ---: | ---: |
| NumPy 8 threads | 7.432 / 6.888 ms | 1.161e-7 | 5.330 / 13.516 ms |
| OpenCL custom | 4.216 / 7.073 ms | 1.907e-6 | 5.979 / 8.471 ms |
| OpenCL + CLBlast | 1.545 / 3.864 ms | 1.907e-6 | 12.303 / 14.920 ms |
| Vulkan | 1.898 / 4.902 ms | 1.906e-6 | 7.498 / 8.290 ms |
| CuPy | 0.861 / 4.072 ms | 1.125e-7 | 4.804 / 6.300 ms |
| CUDA + cuBLAS, without fusion | 0.859 / 3.471 ms | 1.288e-7 | 12.853 / 15.556 ms |
| CUDA custom, without fusion | 2.845 / 5.240 ms | 1.906e-6 | 19.633 / 22.905 ms |

CUDA/cuBLAS is close to CuPy in covariance; a two-microsecond difference
does not establish a victory. For ridge it takes 2.68× CuPy's time and more than OpenCL/Vulkan.
The initial custom product has no tiling and is not a generally fast alternative
to cuBLAS either. The backend name alone does not explain its performance.

CUDA/cuBLAS ridge has L2 error 6.802e-8 and CUDA custom 4.291e-7 against the same
float64 trajectory. These numerical differences are also retained in the
report. Twenty steps do not imply convergence. Modes are timed
in independent rounds; noise can make the complete median lower
than the resident median. JSON files contain samples and percentiles.

The mixed suite has more implementations than the previous one. Between-session
differences are not attributed to pool optimization: the alternating control
above is the evidence isolating that change.

Both applied rounds finished: **104 aligned and 104
irregular configurations**, seven samples per mode, zero errors/missing cases, and all
diagnostics finite. For ridge 4097×129 float32, resident medians were
2.753 ms NumPy 8 threads, 5.907 CuPy, 6.606 OpenCL custom, 6.423 CLBlast, 7.644
Vulkan, 14.959 CUDA/cuBLAS, and 12.140 CUDA custom. The small difference between
OpenCL custom and CLBlast is not general provider superiority.

## Adversarial accuracy and verification

The Decimal80 suite measured **158 configurations**: 42 each for OpenCL custom,
CLBlast, and Vulkan, and 16 for each initial CUDA provider. The 26 norm/solver
combinations per CUDA provider are recorded in `unsupported_cases`, not as
successful executions. There were zero exceptions/missing backends and outputs were finite,
but **40 numerical warnings**: the previous 28 and six for each CUDA provider.

The twelve CUDA warnings concern cancellation in sum, dot, and GEMM, in both
precisions. Relative L2 errors are around 0.9565–1.0 against the Decimal computation
on the actually represented inputs. A finite output with the correct dtype
can lose practically the entire result of that cancellation. Tree reduction
and pedantic cuBLAS do not guarantee compensated accumulation. Improving
those paths or allowing explicit higher-precision accumulation has its own headroom, cost, and
contract; this is not presented as an inevitable hardware boundary.

CUDA preserved subnormal summation in both dtypes in this control; Vulkan
float32 retains its previous warning. That result is not generalized to all
shaders, providers, or devices. The accuracy command returns code 1
for diagnostic warnings, not exceptions. Its thresholds are not a universal
API accuracy promise.

The complete suite passes **1,019 tests, with one skip** because two
OpenCL devices are required. It includes 38 CUDA cases, seven CPU regressions for native
boundaries, and nine new Vulkan reuse tests. OpenCL, Vulkan, and CUDA
were required; native CLBlast was enabled.

The wheel was built from sdist and installed without GPU dependencies in the
CPU environment: **311 passed, 672 skipped**, importing from site-packages outside `src`.
CUDA also ran a subprocess blocking CuPy. Ruff, formatting, and Python
3.10 syntax are checked; the runtime executed on Python 3.12.14. CUDA installation
on a second system and remote CI have not been verified.

## Next priority

1. Keep Vulkan as the main work: profile submission, scalar reads, and
   staging after reducing allocations; measure a bounded native executor if
   Python/CFFI dominates. Check a second vendor when hardware is available.
2. Initial CUDA: measure dedicated GEMV/DOT and per-call/context costs before
   extending operations. Then implement fused lowering and stable norms,
   with accuracy and lifetimes as requirements for enabling solvers.
3. Retain applied comparisons and irregular shapes; extend distances and
   batched transformations when those workloads guide a specific improvement.
4. Optimization by hardware/OS families remains a secondary track.
   Sparse operations and full NumPy compatibility remain outside this stage.

## Local evidence

Controls and snapshot: `.artifacts/cuda-initial`. Results and logs:
`.artifacts/benchmarks/cuda-initial`. The four files
`vulkan-capacity-{baseline,current}-{1,2}.json` isolate the pool. Final
applications are `applications.json` and `applications-irregular.json`.
The `applications-quick*.json` and `cuda-cli.json` files are exploratory checks
before native boundaries were finalized and do not support the table.
`accuracy.json` retains warnings and exclusions. Distributions are in
`.artifacts/dist-cuda-initial`; source/artifact hashes are in
`.artifacts/benchmarks/cuda-initial/manifest.json`.
