# Vulkan, OpenCL, and CUDA: shared execution — 2026-09-13

## Implementation

- **Vulkan:** detects/enables `VK_KHR_push_descriptor` and properties2. Within
  maxPushDescriptors, records a binding snapshot in the command buffer with one
  call instead of updating and binding a set. Beyond the limit or without the
  capability, retains descriptor sets. Reuses host records under the lock and
  retains in-flight GPU buffers. Shaders, barriers, dispatch/submission counts,
  and convergence checks are preserved.
- **OpenCL:** metadata LRUs for elementwise/gather and stride/offset calculations;
  they never store arrays or bound scalars. Indices use int32 when a conservative
  bound permits all intermediate accesses; int64 is retained for larger
  offsets/shapes. Stored dtype does not change. Public layouts remain
  materialized; this does not yet implement logical transposes or BLAS flags.
- **CUDA:** float32/64 vector/matrix and broadcast-batched matmul in eager and
  compile. cuBLAS uses strided-batched for regular batches with positive strides;
  general broadcasting uses per-batch GEMM, with reused pointers and independent
  outputs. Custom executes all batches in one launch. Eligible float32 GEMM
  shares the OpenCL/Vulkan tile/register generator; other shapes and float64
  retain the direct kernel. Stable norms, CUDA fusion, and dedicated cuBLAS
  vector primitives remain pending.

Optimizations are selected internally. No extras, public macros, example changes,
or new C/C++ executor were added. BLAS providers remain explicit, and errors
do not silently switch backends.

## Cross-API lessons and capabilities

Principles can be shared: reuse metadata without retaining values, keep buffers
resident, coalesced accesses, local memory, and register blocks. Float32 GEMM now
has a common generator for all three APIs. Sharing source/algorithm does not
guarantee the same machine code or time: compiler, dispatch, queue, barriers,
and waits still differ.

On the local machine, Vulkan advertises 32 push descriptors; OpenCL advertises
API 3.0 but OpenCL C 1.2 and does not advertise `cl_khr_subgroups`. The API
version number alone is insufficient to select subgroup instructions. Capability
evidence is used, not the GPU's commercial name.

`cl_nv_pragma_unroll` was tested in the OpenCL GEMM inner loop on four float32
shapes with 21 interleaved samples. Results were identical, but timings showed
no consistent gain: it remains an experiment and is not enabled in the runtime.
For 1025×513×769, median GPU events were approximately 0.329 ms in both variants;
for 1024³, 0.770/0.772 ms. This is a diagnostic with a profiled queue and event
wait, not a direct comparison against application wall time with queue.finish
or a physical ceiling.

Macros are useful for expressing a kernel variant; they do not replace querying/
enabling capabilities or choosing host calls. An FMA instruction can improve
error and speed while changing rounding; special matrix instructions can also
change accumulation or precision. They are not enabled as though equivalent
without establishing the corresponding contract. The
[hardware](hardware-specialization.md), mathematical-mode, and autotuning track continues.

Primary sources: [push descriptors](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_push_descriptor.html),
[cuBLAS contract](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmstridedbatched),
[unrolling hint](https://registry.khronos.org/OpenCL/extensions/nv/cl_nv_pragma_unroll.txt).

## Methodology and reproduction

Local hardware: RTX 5060 Ti, Windows 11, driver 616.64. Evidence does not certify
other vendors, other drivers, or Linux. The CUDA kernel pipeline retains
fmad=false and ftz=false; cuBLAS retains PEDANTIC_MATH. External comparisons
record NVIDIA_TF32_OVERRIDE=0 and actually imported versions.

`benchmarks.vulkan_dispatch --control descriptors` interleaves sets/push with
the same CFFI boundary, shaders, and data. It instruments a separate invocation;
API times include driver/waits, and GPU intervals include timestamp/stage effects.
One clock is not subtracted from another to call it Python time.

`benchmarks.backend_sharing` compares rectangular/odd operations, including
batches. It publishes first call, resident wall time, host-call duration, and
full wall time (upload/computation/download). Modes have independent samples;
their medians are not added. Host duration can include internal waits.
Destruction of public output is outside the timer; internal temporaries are part
of execution. Peak VRAM is not measured.

The `--control-wheel` control loads functions from an explicit historical wheel
and alternates those methods with current ones on the **same backend/context
and inputs**. OpenCL controls elementwise/gather; CUDA controls matmul. Methods
are assigned before timing. Other runtime methods are current. This is not
equivalent to executing a complete historical wheel. The JSON records wheel
hash, scope, and current-source hashes. The broad comparison in separate
processes does record the complete package actually imported; it provides
context for timing variation in unchanged operations.

```powershell
python -m benchmarks.vulkan_dispatch --control descriptors --repeats 21 --output .artifacts/push-new.json
python -m benchmarks.backend_sharing --backends opencl --control-wheel <previous-wheel> --repeats 21 --output .artifacts/opencl-control-new.json
python -m benchmarks.backend_sharing --backends cuda-custom --control-wheel <previous-wheel> --repeats 21 --output .artifacts/cuda-control-new.json
python -m benchmarks.backend_sharing --repeats 21 --output .artifacts/operations-new.json
```

Use new files, sequential GPU workloads, and repeat in another process.
Control seeds change order, not inputs. All comparisons publish numerical error.
For float64, these operations' NumPy reference has the same precision; the
additional adversarial suite uses Decimal80.

## Improvement controls

Two processes per control, 21 interleaved samples per mode. Tables show resident
wall-time medians in ms, not pure kernel time. Percentages are latency reduction;
the factor is previous/current time.

| OpenCL, float32 | Previous → current, round 1 | Previous → current, round 2 |
| --- | ---: | ---: |
| Elementwise chain, 33 elements | 0.449 → 0.345 | 0.445 → 0.292 |
| Chain, 1,048,577 elements | 0.557 → 0.328 | 0.496 → 0.374 |
| Transposition and copy, 4097×129 | 0.235 → 0.161 | 0.172 → 0.108 |
| Reversal and copy, 4097×129 | 0.148 → 0.099 | 0.175 → 0.112 |

Across the eight affected cases, counting both precisions and rounds, resident
reduction is **23–41%**. The control jointly measures metadata caching and narrow
indices; it does not attribute all savings to the shader or separate each
change's contribution. OpenCL GEMM did not change in this round: its variations,
including the small batch with a large difference, serve as a negative noise
control, not evidence of kernel improvement or regression.

| CUDA custom | Previous → current, round 1 | Previous → current, round 2 |
| --- | ---: | ---: |
| GEMM float32, 257×129×513 | 0.153 → 0.145 | 0.150 → 0.139 |
| GEMM float32, 1025×513×769 | 1.028 → 0.361 | 0.999 → 0.325 |
| GEMV float64, 2049×1023 | 0.218 → 0.220 | 0.214 → 0.214 |
| GEMM float64, 1025×513×769 | 4.750 → 4.750 | 4.746 → 4.746 |

The shared tile accelerates large float32 GEMM **2.8–3.1×**. It does not improve
float64, which keeps its direct kernel. The first batching version added host
work and index arithmetic to 2D calls; the control detected this, and a direct
path for unbatched vectors/matrices was restored. Final cuBLAS controls no longer
show the systematic penalty observed in small cases. CUDA batches have no
comparable previous time: they were previously unimplemented operations.
All executable pairs preserve recorded error metrics; that is not a bitwise
guarantee for every input.

| Vulkan, float32, sets → push | Round 1 | Round 2 |
| --- | ---: | ---: |
| DOT, 2049 | 0.238 → 0.234 | 0.238 → 0.233 |
| Ridge, 4097×129, 20 steps | 5.552 → 5.310 | 5.288 → 5.070 |
| Steepest descent, 513², same convergence | 23.403 → 22.341 | 23.491 → 22.255 |
| Sum, 1,048,577 elements | 0.311 → 0.377 | 0.321 → 0.351 |

**Vulkan improvement is modest and mixed.** There is one fewer host call per
dispatch, but not fewer dispatches, submissions, copies, or checks. Transfer
paths do not use descriptors; their differences are not push-caused gains either.
CG and several float64 cases still have no consistent improvement.

The large-sum regression required additional diagnosis: 51 interleaved samples
on **the same VkDevice, queue, and buffers**. Only pipelines/layouts compatible
with each mode alternate; a shader comment separates cache keys without changing
operations. There, sets/push give 0.221/0.220 ms for large float32 sum; ridge
5.079/4.958 ms and descent 22.954/22.601 ms. Outputs are identical. The regression
does not reproduce in this control; its cause in the two-logical-device control
has not yet been isolated. Both results are retained, and no size/GPU exception
is introduced based solely on that correlation. The capability enables a valid
path, not a universal speedup. A deployment A/B on another driver/hardware is
still needed.

## Precision and validation

The final application suite runs **104 configurations** of covariance and ridge
on 4095×127 and 4097×129 matrices, float32/64, with seven samples. Includes NumPy
at 1/8 threads, OpenCL custom/CLBlast, Vulkan, CUDA custom/cuBLAS, and CuPy.
There are no exceptions or absent providers. The first attempt had 96 runs and
CuPy unavailable because its default cache was outside the workspace; it was
repeated with `CUPY_CACHE_DIR` pointing to `.artifacts/cupy-cache`.

Maximum relative L2 error in those applications is **7.95e-7 for float32** and
**1.51e-15 for float64**, against NumPy float64. For float64 it is the same
precision. The Decimal80 adversarial suite runs **158 configurations**, without
exceptions, and retains the **40 earlier warnings**: nine per OpenCL provider,
ten for Vulkan, and six per CUDA provider. The flagged cases exactly match the
initial CUDA round. Cancellation, Vulkan float32 subnormals, and ill-conditioned
systems remain known issues. The 52 unsupported CUDA norm/solver combinations
are recorded separately. Diagnostic exit code 1 corresponds to warnings;
it is not an all-clear verification.

Recurrences in `examples/solvers.py` preserve their AST relative to the previous
sdist. Only the CUDA-scope description in metadata changes in
`benchmarks/applications.py`. Computations, tolerances, and checking frequency
were not modified to benefit a backend.

The full suite before the final CUDA adjustment passed **1,084 tests, one skipped**
(needs two OpenCL devices). After adding batch coverage and correcting the
unbatched path, all **75 CUDA tests** pass. Includes float32/64, empty/odd/
rectangular shapes, broadcasting, recycled outputs containing NaN/Inf, resource
retention, and cuBLAS strided-batched. Vulkan tests cover push and sets,
140 dispatches crossing the drain limit, and fallback at the binding limit.
OpenCL checks that caches rebind scalars/buffers and narrowing indices does not
narrow data, including gathers with negative strides.

The wheel built from sdist passes **326 tests, with 719 skips** in the CPU
environment without GPU runtimes. Import from the wheel, independent extras,
and example/solver separation are verified. NumPy remains the sole base
dependency. Ruff/formatting pass; Python 3.10 syntax is parsed and Python 3.12
is executed. This version has not yet been published.

## Gaps between providers

**144 configurations with a single provider per process** were also run: eight
sequential processes, same data, and 21 samples. They compare typical backend
use. They do not eliminate all frequency/OS-load drift. The additional comparison
interleaving all APIs in one process also validates 144 configurations, but
strongly perturbs some timings: large custom OpenCL GEMM changes from 0.401 ms
isolated to 3.312 ms interleaved, and custom CUDA from 0.322 to 3.021 ms.
The exact mechanism involving contexts, driver, and scheduling is not isolated.
Those interleaved numbers are not used as each API's ceiling or a measure of
our abstraction cost.

GEMM 1025×513×769, ms; each cell contains **resident / full**:

| Isolated provider | float32 | float64 |
| --- | ---: | ---: |
| NumPy, 1 thread | 7.271 / 7.306 | 14.816 / 14.964 |
| NumPy, 8 threads | 2.067 / 1.973 | 4.058 / 4.062 |
| OpenPacho OpenCL custom | 0.401 / 1.822 | 2.196 / 4.639 |
| OpenPacho OpenCL CLBlast | 0.336 / 1.808 | 2.168 / 4.573 |
| OpenPacho Vulkan | 0.338 / 2.121 | 2.267 / 5.500 |
| OpenPacho CUDA custom | 0.322 / 1.929 | 4.741 / 7.228 |
| OpenPacho CUDA cuBLAS | 0.230 / 1.673 | 2.729 / 5.192 |
| CuPy | 0.165 / 1.487 | 2.658 / 5.222 |

Common float32 GEMM is competitive across our APIs: Vulkan beats OpenCL in this
case, while OpenCL wins for small GEMM and small batches. Custom CUDA still
takes approximately 1.95× CuPy's time for large GEMM, and OpenPacho/cuBLAS 1.39×.
The latter difference includes call selection, binding, context, pool, and
execution; it is not a Python-only measurement. Float32 relative L2 errors for
large GEMM are 4.06–4.08e-7 in custom paths and 2.89e-7 in CUDA/cuBLAS and CuPy
paths. This is evidence of recorded precision, not a declaration of numerical
equivalence between providers.

Batches (7,33,17)@(7,17,65) still clearly favor CPU: around 0.020 ms resident,
versus 0.089 OpenCL, 0.126 custom CUDA, and 0.200 Vulkan. General custom CUDA
broadcasting uses one launch and gives 0.138 ms, while cuBLAS uses six GEMMs
and gives 0.371 ms for (2,1,33,17)@(1,3,17,65). No provider other than the selected
one is forced; a future pointer-batched path has room for improvement.

Resident data matter: the large OpenCL chain gives 0.332 ms, but
upload/computation/download gives 2.720 ms, versus 1.684/1.805 ms for
single-threaded NumPy. CuPy also loses in full execution of that chain: 2.345 ms.
Transfer savings and residency remain workload-specific, not a promise that
GPU always beats NumPy.

## Justified next optimizations

1. **Vulkan:** staging/mappings, submission, and waits, especially small reads;
   push reduces descriptor work but does not resolve these boundaries. Measure
   event/fence and executor grouping/reuse before adding C/C++.
2. **OpenCL:** separate generation/binding, execution, and queue waiting. The
   tested pragma did not close the GEMM gap; measure tiles/packing by shape,
   with CLBlast as a reference provider and controls isolated from context effects.
3. **CUDA:** dedicated cuBLAS GEMV/DOT, elementwise fusion, and float64 tiles
   when cases justify them. Avoid building a Python list per broadcast batch
   through a pointer-batched path, respecting those buffers' lifetimes.
4. **Shared:** logical layouts/transposes and an explicit precision policy.
   Capabilities enable variants; autotuning, hardware profiles, and fast modes
   need their own measurements and contracts. Abstraction and hardware do not
   justify attributing current gaps to an unmeasured physical limit.

Local artifacts in `.artifacts/backend-sharing`: `control-opencl-{1,2}.json`,
`final-control-cuda-{custom,cublas}-{1,2}.json`, `push-{1,2}.json`,
`push-same-device.json`, `isolated-*.json`, `operations-final.json`,
`applications-final.json`, `accuracy.json`, and `comparison.csv`. The before/after,
quick, and first CUDA controls are exploratory evidence, not replacements for
final controls. The manifest distinguishes later source changes in files outside
each control's backend.
