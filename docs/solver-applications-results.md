# Solvers, applications, and CUDA preparation

Continuation from 2026-09-12: [capability-based Vulkan and initial CUDA](cuda-initial-results.md).
The figures and CUDA status in this document correspond to the following historical iteration.

Iteration from 2026-09-08, RTX 5060 Ti / Ryzen 7 5700X3D / Windows 11 / driver
616.64. Continues [Vulkan, GEMM, and stability](numerics-vulkan-results.md).
Only this hardware has physical validation. Previous results remain
historical; they are not overwritten to attribute new code to them.

## How to interpret previous wins

In the previous round, OpenCL custom won by median in **8/9 resident cases
against the OpenPacho variant with CLBlast**, and 9/9 including transfers.
The resident exception was CG 1024 float64. Against CuPy, it won 8/9 resident
and 7/9 complete cases. These are comparisons of complete solvers, not proof
of superiority for every kernel. CLBlast provides BLAS products within our solver;
this is not a comparison with a CLBlast solver.

CuPy beat custom steepest descent at 2048 float32; custom won
at 512 float32 and 1024 float64. All three cases retained equal iteration counts.
Thus, it was neither a general SD defeat nor a universal CG victory.
The cases are densely stored SPD tridiagonal matrices, condition number ≤9;
that bounded family does not represent every dense application.

There was also a win in a specific operation: resident Vulkan custom GEMM 2048²
float32 took 3.047 ms versus 3.929 ms with CLBlast. OpenCL custom took
6.902 ms and CuPy 1.419 ms. These are different shapes, paths, and modes; they
cannot be summarized as all our kernels already outperforming CLBlast.

## Changes implemented now

- **Curvature without a Boolean kernel.** The host decision reads the existing scalar
  directly, checks finiteness/positivity, and retains its GPU value for an update
  in the same dtype. One dispatch per CG/SD iteration is removed; the host
  observation remains, and a candidate with invalid curvature is not accepted.
- **Plans between observations.** GEMV/DOT are grouped for curvature; SD includes
  its normalization. The advance groups the update, true residual, and scaled
  norm, returning the candidate and residual state. No tolerance
  is relaxed and no per-iteration checks are skipped.
- **One Vulkan command buffer per group.** Dependent dispatches from a compiled
  call share recording and retain barriers. Descriptor sets and
  buffers remain retained until the queue completes. The limit of 64 counts
  operations, not command buffers, to preserve the limit on resources in flight.
  Synchronization, cache pressure, or an exception closes the group safely.
- **Compiler separation by backend.** `_lowering.py` centralizes backend-specific
  code and execution; `_compiler.py` retains the graph and temporary lifetimes.
  This is the implemented foundation of the [CUDA path](cuda-backend-plan.md), not an
  available CUDA backend yet.
- **Vulkan reuse when changing shapes.** The pool evicts old size classes
  when a buffer arrives that needs space. Previously, the first 128
  free allocations could indefinitely block admission of new sizes.
  Eviction affects only completed buffers with no live references; it retains
  the limits of 128 buffers and 128 MiB and separates GPU, upload, and readback memory.

## Solvers: new complete round

63 configurations, seven resident and complete samples per configuration.
Same data, dtypes, and tolerances as the previous benchmark. Implementations
alternate between rounds, and diagnostics remain outside the timer.
This round and the phase controls below precede the final Vulkan pool
correction; they are retained with that identification.

Resident medians for 2048 float32, in milliseconds:

| Implementation | Jacobi (37 iter.) | SD (30 iter.) | CG (13 iter.) |
| --- | ---: | ---: | ---: |
| NumPy 8 threads | 9.067 | 14.720 | 8.312 |
| OpenCL custom | 9.056 | 14.491 | 6.812 |
| OpenCL + CLBlast | 14.045 | 20.950 | 8.715 |
| Vulkan | 20.586 | 33.226 | 15.593 |
| CuPy | 10.205 | 17.057 | 8.697 |

In the same order, custom's independent relative residuals are
8.907e-5 / 9.567e-5 / 7.086e-5; CuPy's are
8.907e-5 / 9.568e-5 / 7.086e-5. Relative solution error against the
generating solution is approximately 8.874e-5 / 3.490e-4 / 1.712e-4 in both.
The RHS is rounded when generating the system; that solution is not an exact oracle
for the rounded system. Speed was not improved by accepting a larger residual.

Custom SD now wins in this case in the new round. Vulkan improves over the
historical 41.966 ms for SD and 20.424 for CG, but still takes approximately 2.3×
custom's time for both. Vulkan Jacobi barely changes from the historical 21.533 ms.
Not every difference between sessions is treated as a causal gain.

## Execution controls and explanation of SD

A separate control alternates processes with the previous wheel and new code,
eleven samples per method, using the same 2048 float32 systems. In two rounds, Vulkan
SD goes from 34.64–41.05 to 24.62–24.63 ms; CG from 15.85–23.29 to 12.75–12.87 ms;
Jacobi from 19.61–24.80 to 16.41–16.71 ms. Histories and residuals are preserved.
There is considerable variability: these controls do not replace the mixed suite.
OpenCL SD goes from 15.84–23.39 to 13.09–13.35 ms, while CG/Jacobi have overlapping
ranges between versions. That mixed result is retained.

Isolated synchronized phases, 31 samples, help locate costs:

| float32 phase, 2048 | OpenCL custom | CLBlast | Vulkan | CuPy |
| --- | ---: | ---: | ---: | ---: |
| Normalization | 0.040 ms | 0.045 ms | 0.183 ms | 0.033 ms |
| Curvature + read | 0.185 ms | 0.256 ms | 0.343 ms | 0.224 ms |
| Advance + residual + norm + read | 0.175 ms | 0.535 ms | 0.408 ms | 0.341 ms |
| CG direction update | 0.045 ms | 0.043 ms | 0.225 ms | 0.086 ms |

These phases operate on a fixed state, with additional synchronization: **they are
not summed to reconstruct solve time and do not represent isolated GPU time**. They explain
verifiable opportunities, not an exact breakdown of historical latency.

By design, CG fuses a direction update that the CuPy comparator
expresses through several operations; SD only normalizes a vector at that point.
Thus, the benefit of our fusion depends on the method. SD also pays for GEMV, DOT,
curvature, and residual over more iterations than CG in these cases. This does not make
SD inherently slow within OpenPacho: grouping and validation changes
reduce cost without changing the recurrence. CuPy retains cuBLAS NRM2 as
a strong reference for the solver norm.

## Applications and gaps that should guide the next work

The [application suite](application-measurement-plan.md) adds sample covariance
and twenty ridge-regression steps with rectangular matrices. Preparation,
transfer, error, objective, and gradient norm are published.
It uses synthetic data and applied patterns; these are neither production datasets nor
ridge convergence tests. Complete results are in
`.artifacts/benchmarks/next-applications/applications-final.json`.

The final applied round, after correcting the pool, finished with **72
configurations**, seven samples per mode, and zero errors/missing cases. For
16384×256 float32:

| Implementation | Resident/complete covariance | Covariance L2 error vs float64 | Resident/complete ridge |
| --- | ---: | ---: | ---: |
| NumPy 8 threads | 6.711 / 6.389 ms | 1.161e-7 | 5.313 / 14.427 ms |
| OpenCL custom compiled | 3.708 / 5.994 ms | 1.907e-6 | 5.667 / 10.413 ms |
| OpenCL + CLBlast compiled | 1.529 / 3.825 ms | 1.907e-6 | 11.777 / 14.443 ms |
| Vulkan compiled | 1.741 / 4.579 ms | 1.906e-6 | 7.386 / 7.961 ms |
| CuPy | 0.876 / 3.925 ms | 1.125e-7 | 4.716 / 6.211 ms |

This reveals opportunities that square solvers did not show: CLBlast
outperforms our OpenCL custom by approximately 2.4× in resident covariance,
while custom wins in ridge. Vulkan takes 1.14× CLBlast's time in covariance and beats
NumPy; in ridge it takes 1.30× custom's and 1.57× CuPy's time in resident mode. The backend has no single performance
ratio that can be extrapolated to every application.

Maximum absolute covariance error is 2.269e-8 for custom/CLBlast,
2.223e-8 for Vulkan, and 1.116e-9 for CuPy. There is numerical headroom as well as timing
headroom: a result within tolerance does not mean identical accuracy. In ridge,
custom/Vulkan/CuPy L2 errors are around 6.6–6.9e-8; CLBlast reaches 4.401e-7.
The final objective is approximately 7712.505 and the gradient norm 998.429
on all paths: the twenty steps **have not converged**. The comparison measures
a common trajectory, not the time needed to solve ridge completely.

Preparing the ridge transpose costs 9.435 ms in NumPy with 8 threads,
0.412 in OpenCL custom, 0.495 in Vulkan, and 0.347 in CuPy. This is why it is published
separately: residency/preparation changes the practical value of each result.

Before the pool correction, this same suite recorded Vulkan ridge at
30.091 ms resident; the final round measures 7.386 ms. The between-session comparison
includes noise: the policy control below isolates the cause more precisely.

Vulkan still needs lower submission/scalar-read and large-staging costs.
Sharing command buffers reduces command management, but does not eliminate
Python/CFFI calls, barriers, descriptor updates, or host waits. Transfers were
not optimized in this iteration; their entire difference from OpenCL is not
attributed to PCIe. A small native executor and reusable staging remain candidates
that must be measured before incorporation.

For CuPy, the goal is to approach its performance with a comparable contract.
Its speed does not determine our physical limit: it uses the same device.
Optimized libraries and their access to the compiler/driver explain possible
software advantages, but profiling is still needed to assign causes. CuPy can
delegate operations to specialized libraries; not every operation follows the
same path. [CuPy performance guide](https://docs.cupy.dev/en/v14.0.0/user_guide/performance.html).
Its published advantage is not attributed to TF32, which is disabled in comparisons.

CUDA is the next proposed backend. HIP/ROCm, Metal, and SYCL have potential
value with their testing platforms, but are not implemented simultaneously.
The [backend decision](cuda-backend-plan.md) details stages, contract, and sources.

## Irregular dimensions and the Vulkan reuse problem

**144 tests** were added across NumPy, OpenCL custom, CLBlast, and Vulkan, with
float32/64: rectangular GEMM, odd dimensions around tiles,
even dimensions that are not powers of two, batch broadcasting, GEMV,
axis reductions, and square SPD solvers of sizes 31, 65, and 129. They check
accuracy, true residual, dtype, immutable inputs, and plan reuse.
Rectangular matrices are also exercised in covariance and ridge; Jacobi,
CG, and SD retain their square-system contract.

The first irregular round revealed Vulkan ridge 4097×129 was much slower than
4096×128. A profile showed hundreds of `vkAllocateMemory` and `vkFreeMemory` calls per
ten already-warmed executions. Changing shape could leave the pool full
of previous sizes: it discarded all new ones even when reusable.
This is a general runtime admission defect, not a boundary for odd
matrices or evidence of a hardware limitation.

The `pool_control.py` control keeps the same solver/runtime and replaces only
the recycling policy with the old one. It processes 4095×127, 4096×128, and 4097×129
in that order, with three warm-ups and eleven samples per shape, including
final-vector download. The allocation profile is taken outside the timer.

| float32 ridge shape | Old policy | New eviction | Allocations in ten executions, before → after |
| --- | ---: | ---: | ---: |
| 4095×127 | 6.539 ms | 7.526 ms | 0 → 0 |
| 4096×128 | 11.621 ms | 6.558 ms | 100 → 0 |
| 4097×129 | 47.911 ms | 6.417 ms | 810 → 0 |

Relative L2 errors are equal between policies: 8.476e-8, 6.474e-8, and 7.769e-8,
respectively, against the float64 trajectory. The first case does not improve;
the allocation reduction in subsequent cases does demonstrate elimination of
permanent exclusion of new sizes. No universal speedup is extrapolated.

**Pending limit:** alternating many shapes and retaining their arrays can exceed
the exact-match pool's capacity. An interleaved control with five shapes still
shows churn; eviction does not avoid reallocating what has just been evicted.
Capacity classes, suballocation, and fewer temporaries in flight must be
compared as next options. Increasing the limit without measuring memory would not
resolve the cause generally.

The final `applications-irregular-final.json` round completes another **72
configurations** with seven samples per mode and no errors/missing cases. For
4097×129 float32, ridge medians:

| Implementation | Resident | Complete | L2 error vs float64 trajectory |
| --- | ---: | ---: | ---: |
| NumPy 8 threads | 2.575 ms | 3.137 ms | 6.242e-8 |
| OpenCL custom compiled | 5.507 ms | 6.598 ms | 7.424e-8 |
| OpenCL + CLBlast compiled | 6.530 ms | 7.327 ms | 1.948e-7 |
| Vulkan compiled | 7.595 ms | 7.450 ms | 7.766e-8 |
| CuPy | 4.835 ms | 4.972 ms | 7.314e-8 |

Vulkan improves over the previous irregular round's 38.030 ms resident,
but NumPy beats every GPU on this problem. Modes are independent,
interleaved samples: noise can make the complete median lower
than the resident median; this does not mean data transfer has negative cost.
The JSON files retain samples and percentiles, including reference variations.

## Reproducible evidence

Phase/process controls live in `.artifacts/next-applications`; the
baseline was extracted from `.artifacts/dist-numerics-vulkan`. Current results
are in `.artifacts/benchmarks/next-applications`. `solver-tridiagonal-after-1.json`
is an intermediate version; `solver-tridiagonal-final-*.json` incorporates final
plans and commands, before the pool was modified. `applications*.json` without the
`final` suffix are rounds before that change; `applications*-final.json` includes it.
`pool-old.json` and `pool-new.json` isolate the recycling policy. `solver-before-1.json` is an exploratory control with another matrix and
is not used in the comparisons above.

The complete runtime and adapter suite passes **965 tests, with 1 skip**
because two OpenCL devices are required. It includes 144 irregular-shape cases,
four pool regressions, extreme cases, invalid curvature, rebinding,
backend-name-independent plans, and Vulkan resource lifetimes. OpenCL and
Vulkan are required, and native CLBlast is enabled. Ruff and formatting pass;
syntax is parsed as Python 3.10, without replacing execution on that version.

The final wheel, built from sdist and installed without PyOpenCL/Vulkan, passes
**294 tests, with 635 expected skips** for absent optional backends.
Import comes from site-packages outside `src`. Distributions are in
`.artifacts/dist-next-applications`; the source and result manifest is
in `.artifacts/benchmarks/next-applications/manifest.json`. Details in
[local verification](verification.md).

The adversarial suite was repeated for all three OpenPacho GPU paths: **126 cases,
three samples, 28 numerical flags, zero exceptions/missing cases**. The
flag set for that same previous subset is retained; this is not a reduction
from 68 to 28, because the unchanged CPU/CUDA comparators were not rerun this time.
Diagnostic thresholds are not universal forward-error guarantees.
