# CUDA argument preparation

Local round from 2026-09-13, Windows / Python 3.12 / RTX 5060 Ti, driver 616.64.
Previous fusion is retained and host work per launch is reduced.
Resident time improves in all 72 controlled comparisons: the median of
per-case percentage improvements is **13.9%**. For complete time that median is
**3.3%**, with nine worse results. This is not a universal speedup.

## Decision and change

The profile from the [previous stage](cuda-fusion-results.md) identified preparation and
submission as pending areas. An additional profile confirmed that `_run` still
spent time constructing NumPy arrays of scalars/pointers on every call.
The change is limited to that work, without modifying kernels or algorithms.

[`KernelArguments`](../src/openpacho/_cuda_arguments.py) creates the `void**` vector
and host storage for its values once. Each compiled kernel retains
its vector; values are updated on every launch under the backend lock.
No references to caller arrays or scalars are stored. GPU allocations
remain retained by `_pending` until the stream completes, independently
of argument storage. The 128-module LRU and `close()` also release
these templates.

ABI sizes, signed zeros, and previous type resolution are preserved. Device
pointers use 64 bits; host pointers in the vector use the native
size. An unexpected change in parameter count or type fails before
launch. Validation detected that NumPy 2.5.3 does not allow assigning `np.bool_` to
`ctypes.c_ubyte`: the final version uses `ctypes.c_bool`, with a verified one-byte
representation. This does not modify the measured float32/64 paths.

The driver copies values pointed to by `kernelParams` during launch,
so the next call can reuse that storage. The lifetime of the pointed-to GPU buffer
remains separate.
[CUDA parameter contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html),
[preparation in CUDA bindings](https://nvidia.github.io/cuda-python/cuda-bindings/13.3.2/overview.html).

Contexts, stream, synchronization frequency, copies,
math policy, and custom/cuBLAS selection are unchanged. Alpha/beta parameters
for cuBLAS calls retain their previous implementation. The public API and
installation extras remain the same.

## Measurement control

[`benchmarks/cuda_dispatch.py`](../benchmarks/cuda_dispatch.py) compares `_run`
from the previous wheel with the current version, in **the same instance/context and with the same
compiled function**. It verifies other CUDA methods by AST, except `_compile`,
whose change adds the storage field to the cache. Other runtime
modules, including generators and lowerer, are checked by hash against the wheel.
The historical package is not executed in full: the control isolates packing.

The nine fusion cases are reused in float32/64: chains of 33 and 1,048,577
elements, energy 1025×1023, row energy 257×129 and 17×4099, row centering
2049×33, squared loss 2049×1023, covariance 1025×65, and ridge 2049×129.
Ridge retains the same twenty steps; it does not measure time to convergence.

- Two rounds per provider, **31 alternating pairs** per case/scope. Process
  order is reversed in the second round, and variant order changes
  in every pair. First call and warm-up remain outside the samples.
- Resident: inputs/preparation already available, computation and synchronization.
  Complete: upload, preparation, computation, and download, with warm kernels/plans.
  Both are wall times with Python/driver/waits, not just GPU execution.
- Observed idle state before each process: at least 10 s, three readings with
  GPU/memory utilization ≤5%, temperature within 2 °C of the reference and
  ≤52 °C, with no observed thermal throttling. Then, warm-up of at least
  three calls and 0.25 s per mode/scope. Accepted windows end at
  46–48 °C after 10.4 s. This does not guarantee constant clocks during samples.
- **144 CUDA configurations and 92 references** completed. Current references
  are eight-thread NumPy and CuPy eager/fuse. OpenCL/Vulkan are checked
  functionally; their earlier timings are retained as historical.
- Instrumented diagnostics and events are separate from timing samples.
  All samples, p10/p90, first call, and errors are retained.

The first attempt failed wheel validation because of an incorrect runner
path, before measuring. It was corrected and its log remains in `control/`;
valid measurements are in `paired/`. The final Boolean ABI constant correction
was made after this exclusively floating-point batch; follow-up and
the full suite do use that correction. Original hashes are retained.

## Before and after results

Milliseconds, medians **round 1 / round 2**. The previous control already includes fusion.
Percentages must not be chained with other rounds taken at different times.

| Case | Previous resident | Current resident | Previous complete | Current complete |
| --- | ---: | ---: | ---: | ---: |
| cuBLAS, large chain f32 | 0.086 / 0.087 | 0.073 / 0.073 | 2.441 / 2.531 | 2.422 / 2.498 |
| cuBLAS, energy f32 | 0.162 / 0.179 | 0.140 / 0.151 | 1.653 / 1.645 | 1.636 / 1.625 |
| cuBLAS, row centering f32 | 0.225 / 0.229 | 0.187 / 0.189 | 0.723 / 0.568 | 0.683 / 0.536 |
| Custom, ridge f32 | 7.679 / 6.300 | 6.125 / 5.183 | 6.991 / 6.900 | 5.673 / 5.743 |
| Custom, ridge f64 | 6.946 / 8.097 | 5.149 / 6.896 | 7.594 / 9.369 | 6.365 / 8.034 |
| cuBLAS, ridge f32 | 6.961 / 5.891 | 6.105 / 5.410 | 7.111 / 6.675 | 8.455 / 6.128 |
| cuBLAS, ridge f64 | 5.981 / 6.170 | 5.371 / 5.504 | 6.772 / 6.982 | 6.269 / 6.453 |

The 72 resident improvements range from **7.7% to 31.3%**. Complete measurements
improve in 63 of 72 comparisons, with mixed results for large chains,
squared loss, f64 covariance, and f64 row energy. These differences are not
automatically attributed to PCIe or a kernel.

### Follow-up on the adverse cuBLAS ridge f32 result

The 18.9% complete-time loss in the first round motivated three additional blocks,
with 51 alternating pairs per scope, observed idle state, and the same algorithm:

| Block | Previous → current resident | Previous → current complete |
| --- | ---: | ---: |
| 1 | 6.124 → 5.660 ms | 6.757 → 6.260 ms |
| 2 | 6.311 → 5.861 ms | 6.759 → 6.181 ms |
| 3 | 6.123 → 5.599 ms | 6.893 → 6.347 ms |

The loss did not reproduce in those blocks; the initial datum is not discarded either.
Complete measurements show variation even in CuPy: ridge f32 took
4.806 / 7.320 ms. There is insufficient evidence for a workload-specific exception.

## What remains to optimize

A **CPU-only** control for four arguments, with 31 alternating batches of
2000 calls, falls from **6.0 to 0.70 μs** per packing operation. It includes neither driver,
compiler binding, allocation, kernel, nor synchronization. The cost of this
substep is already small; further refinement offers less headroom than other areas.

The large f32 chain takes 73 μs resident with CUDA/cuBLAS and **30 / 33 μs** with
CuPy fuse. Their complete times are close because of copies: CUDA 2.422 /
2.498 ms versus CuPy fuse 2.440 / 2.434 ms. For energy, CuPy eager remains
better than its fuse: 0.137 / 0.099 ms versus 0.401 / 0.406 ms resident.
The CuPy mode favoring the comparator is not the only one selected.

NumPy still wins ridge at this size: **1.111 / 1.093 ms complete**. This
round's goal is to remove abstraction cost, without presenting an improvement
relative to our own version as superiority over CPU or other libraries.

Diagnostics retain launch counts: custom ridge uses 80 custom
kernels; cuBLAS uses 40 custom kernels and the algorithm's forty products.
In cuBLAS ridge, **243 context push/pop pairs** appear in the instrumented
call, including synchronization and recycling. Reducing that repetition
warrants a separate control, with context restoration and concurrency tested.

Events for the large CUDA/cuBLAS chain delimit about 28 μs in the first
diagnostic, versus 73 μs uninstrumented wall time. **One figure is not subtracted
from the other to assert an exact Python cost**: event intervals may
include submission/queue gaps, and instrumentation changes execution. They are
indicators of remaining headroom, not a physical NVIDIA bound.
[CUDA event semantics](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html).

The next priority shifts away from argument packing: control
context activations/submission and plan binding; compare their possible savings
with GEMV/DOT, axis reductions, and layouts. Vulkan/OpenCL keep those areas
open. No path is changed out of inertia, nor is the contract weakened to win.

## Accuracy and verification

Both variants' results are exactly equal in all controlled
cases. Against NumPy computed in float64, maximum relative L2 errors are:

| Current path | float32 | float64 |
| --- | ---: | ---: |
| CUDA custom | 4.36×10⁻⁷ | 8.20×10⁻¹⁶ |
| CUDA cuBLAS | 8.37×10⁻⁸ | 3.19×10⁻¹⁶ |
| CuPy, both modes | 1.80×10⁻⁷ | 3.06×10⁻¹⁶ |

Float64 is compared with an equal-precision reference, not an exact
solution. Maximum absolute CUDA float32 error is 0.00489; errors are retained per
case alongside timings. They are not guarantees for different inputs.

The complete suite with OpenCL/Vulkan/CUDA required and native CLBlast has
**1200 passing tests and one skip** because two OpenCL devices are required.
New regressions cover ABI bytes/sizes, signed zeros, rebinding without
input retention, incorrect types/counts, failed launch and retry,
locking, release on module eviction, and Boolean results on the GPU.
Existing validation covers concurrency, adversarial accuracy, tails/rectangles,
RNG, buffer lifetimes, and context restoration. See [packaging and controls](verification.md).

## Artifacts and reproduction

`.artifacts/cuda-dispatch/summary.csv` collects the 236 main configurations;
`paired-gains.csv` records the 72 comparisons and paired differences.
`paired/` and `references/` retain JSON, samples, commands, and telemetry. The
follow-up is in `ridge-followup.json` (six additional configurations),
its script, and its gate; the CPU diagnostic is in `packing_profile.py/json`.
cProfile profiles are instrumented diagnostics, not GPU benchmarks.

The previous local wheel has SHA256
`013d623c9d2a0504d1a4e6177fa83e5e0953519dc18d418788cc9c95eeead97b`.
With extras installed, use new output directories:

```powershell
python -m benchmarks.isolated --worker cuda_dispatch --backends cuda-custom,cuda-cublas --rounds 2 --repeats 31 --max-temperature 52 --control-wheel .artifacts/cuda-dispatch/before.whl --output .artifacts/cuda-dispatch/paired
python -m benchmarks.isolated --worker cuda_dispatch --backends numpy-8t,cupy --rounds 2 --repeats 31 --max-temperature 52 --output .artifacts/cuda-dispatch/references
```

Automatic backend/telemetry UUID association remains pending. Observed hardware
is the same as in the previous round; Linux and other vendors were not
physically verified. The version remains 0.1.0a1, without publication.
