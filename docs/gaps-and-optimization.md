# Real gaps and OpenPacho optimization

**Snapshot before the next Vulkan execution improvement.** Results for cached
readback, reusable descriptor sets, and grouped submissions are in
[the continuation of the main priority](vulkan-execution-results.md).
The most recent stage implements [Vulkan, register GEMM, and numerical stability](numerics-vulkan-results.md).
The [intermediate iteration](remaining-suite-results.md) retains map-reduce,
preparation, and batched BLAS. Hardware/OS specializations form a
[lower-priority track](hardware-specialization.md).

Date: 2026-09-06. RTX 5060 Ti 16 GB, Ryzen 7 5700X3D, 32 GB RAM, Windows 11,
driver 616.64. This report continues the [previous snapshot](execution-results.md).

**There is substantial room within OpenPacho. Transfers explain some losses,
but do not explain slow solvers with resident data or Vulkan overhead.**
A portable backend also does not need to implement all of NumPy to offer value:
the goal remains chaining common computations over resident data, with known
semantics and explicit backend/provider selection.

## What was implemented

- Cooperative OpenCL GEMV, with contiguous reads across threads. Short DOT uses
  one launch; large DOT uses device partials. The threshold avoids paying for a
  second launch for a few thousand elements.
- `compile` fuses OpenCL map-reduce without storing the full transformed array.
  Contiguous rows of 32 or more terms are distributed across threads; serializing
  them per row caused a regression during this investigation and was corrected.
  Intermediate rounding, accumulator dtype, and broadcasting are preserved.
- `x*x` now counts as a single consumer of x. Producers shared by distinct
  consumers or returned as outputs retain a conservative boundary.
- GPU solvers reuse the squared residual they already computed and checked;
  finiteness and residual use compiled plans. CG removes a redundant observation.
  It preserves true residual, tolerances, history, and checks at **every iteration**.
- Vulkan: elementwise fusion, cooperative GEMV/DOT and reductions, matmul with
  8×8 tiles, a buffer pool, and command-buffer and descriptor-pool reuse.
  Immutable device capabilities are cached as Python values to avoid repeatedly
  reading CFFI structures. In-flight resources are not recycled.
- `matmul` supports batch broadcasting on all three backends. `dot` above rank
  2 remains outside the contract; its rules are not confused with matmul's.
- CLBlast uses native GEMV/DOT when available; GEMM remains an alternative from
  the same provider for builds missing those symbols. Batches are handled through
  offsets and a GEMM loop, still without native batched BLAS.

The CUDA backend and sparse matrices remain future work. No automatic CPU/CUDA
switch or BLAS choice hidden from users is added.

## Method: what each piece of evidence measures

The main suite retains NumPy at 1/8/16 BLAS threads, OpenCL custom/CLBlast and
Vulkan in eager/compile, CuPy, direct cuBLAS, and nvmath-python. GEMV/DOT controls
that exceed caches and three batched matmul shapes are added. Each mode has
seven rounds with randomized implementation order and dtype, shape, and value
validation. Full solvers also verify convergence and host residual.

Resident timings include Python dispatch, result allocation, and synchronization.
Upload+computation+download timings use fresh GPU inputs; NumPy retains its host
inputs without an artificial copy. Plans and programs are warm. The main suite
separately records first execution, upload, download, and preparation. It does
not measure process cold start or cold compilation of every shader.

Additional profiles use a frozen wheel of the previous state and current code
in separate processes. `gaps.py` retains the same compiled callable across
warm-up, measurement, and profiling; it does not recapture the function for every
sample. Its diagnostic CG converges in two iterations on a well-conditioned
system; **it does not have the same input as the full solvers in `execution.py`**.
CuPy executes the recurrence in `benchmarks.suite.native_solver`, with its
finiteness, curvature, and residual checks. An unchecked recurrence is not
presented as equivalent to the public solver.

`cProfile` and counters run separately from wall-time samples. Their times
include waits; **they are not CPU utilization percentages**, and nested inclusive
times are not added. OpenCL events are another instrumented control: they record
actual kernel/copy timestamps, exclude opaque CLBlast commands, and do not allow
automatically labeling `wall − kernel` as “Python.”

Before/after rounds are not a clock-locked experiment. CuPy controls also vary
between processes. Small differences and a single microbenchmark do not prove
causal improvement; all samples and quartiles are retained. Energy consumption,
peak memory, and sustained performance over hours were not measured.
CuPy/cuBLAS/nvmath share the same NVIDIA GPU; they are not evidence of
cross-vendor portability.

## Measured limits and how much control we have

### Quantified results

**319 complete configurations without validation errors:** 172 operations/steps,
63 full solvers, and 84 size/batch controls. Full tables
(`.artifacts/benchmarks/gaps-report/all-results.md`) and the CSV with quartiles
(`.artifacts/benchmarks/gaps-report/all-results.csv`) are undistributed local
artifacts; the following tables retain the essentials in this document.

Resident milliseconds, float32. “Before” is the `execution-results.md` snapshot;
NumPy/CuPy are final-round measurements, not their historical times.

| Case | OpenCL before → now | Vulkan before → now | NumPy now | CuPy now |
| --- | ---: | ---: | ---: | ---: |
| Full Jacobi, 2048 | 25.121 → 12.732 | 350.699 → 82.502 | 10.201 | 11.712 |
| Steepest descent, 2048 | 34.135 → 19.631 | 538.382 → 100.486 | 12.972 | 22.426 |
| Full CG, 2048 | 17.428 → 9.764 | 311.270 → 49.800 | 7.819 | 10.572 |
| 100 updates, 1,048,576 elements | 11.280 → 10.890 | 303.090 → 72.459 | 194.964 | 6.395 |
| Matmul 2048² | 16.402 → 16.416 | 20.447 → 11.905 | 29.083 | 1.478 |

**Solvers still lose to NumPy in these cases.** Vulkan's progress over its previous
implementation is large, but it remains far behind OpenCL and CuPy. For 2048²
matmul, final CLBlast gives 3.864 ms, cuBLAS 1.471 ms, and nvmath 1.578 ms:
there is a provider/kernel gap that fusing an isolated matmul does not resolve.

Final large controls; each cell is **resident / upload+computation+download**:

| Case | NumPy | OpenCL custom | CLBlast | Vulkan | CuPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMV 8192² float32 | 8.127 / 7.779 | 0.871 / 39.511 | 1.285 / 40.022 | 1.280 / 125.833 | 0.854 / 52.824 |
| GEMV 8192² float64 | 17.171 / 16.799 | 1.499 / 78.694 | 1.522 / 81.050 | 1.931 / 279.588 | 1.483 / 105.923 |
| DOT 16,777,216 float32 | 2.675 / 2.554 | 0.891 / 20.321 | 0.617 / 20.055 | 1.035 / 54.218 | 0.805 / 23.232 |
| Batch of 64 matrices, 32² float32 | 0.101 / 0.094 | 0.210 / 0.401 | 2.013 / 2.308 | 0.515 / 2.063 | 0.195 / 0.328 |
| Batch of 16 matrices, 128² float32 | 0.943 / 0.966 | 0.291 / 1.198 | 0.680 / 1.905 | 0.886 / 6.463 | 0.239 / 1.020 |

Large custom GEMV is close to CuPy; uploading 256 MiB every time destroys that
advantage. Vulkan transfer cost is much higher than OpenCL on the same link:
**that excess cannot be justified by PCIe alone**. CLBlast small batches clearly
expose the call-loop cost, even though its large GEMM is far better than custom.
Float64 GEMV traverses memory and lacks GEMM's arithmetic intensity: the FP64
limit does not affect both equally.

### Kernel, executor, and measurement sensitivity

In the OpenCL event control, GEMV2048 changed from 213 to 68 µs of kernel time,
while the instrumented call changed from 290 to 192 µs. That control's diagnostic
CG changed from 2.284 to 0.570 ms accumulated in kernels, but from 6.253 to
3.822 ms wall time. Kernel gain does not translate proportionally into algorithm
gain. Short DOT retains a mixed case: its kernel changed from 33.7 to 13.8 µs,
while instrumented wall time changed from 96.6 to 129.2 µs. That regression is
not hidden.

In `gaps.py`, compiled energy for 2048×32 takes 90.0 µs on OpenCL; invoking an
already-prepared plan with valid bindings takes 66.1 µs. On Vulkan: 617.3 versus
572.0 µs. This measures the binding/variant-lookup margin; **it does not demonstrate
the speed of a future native executor**. Profiled Vulkan CG retains 43 `_run`
calls; `_new` struct construction totals 6.7 ms inclusive in a 21.8 ms profiled
invocation. That time is not added to its child calls.

An apparent regression still requires caution: the broad suite gives 1.188 ms
for OpenCL32768×32 energy, versus a historical 0.311 ms. An additional control
with the same computation gives 128–131 µs measured in backend blocks and
201 µs alternating backends; CuPy gives 173–193 µs in that control. At 2048×32,
OpenCL changes from 101–103 to 162 µs when alternating. The 21 samples per mode
and return to blocked mode are in `gaps-final/scheduling.json`.

**Alternating backends changes results and explains some, but not all, of the
difference from the broad suite.** Contributions from driver state, clocks,
contexts, or preceding transfers have not been isolated. Unfavorable broad times
and isolated controls are retained; neither is replaced by the better result,
and this sensitivity is not declared solved. Small tuning decisions require
replication across processes and greater order/state control.

### Gap classification

| Gap | Evidence / interpretation | Control and action |
| --- | --- | --- |
| PCIe and large transfers | Observed link under load: Gen3 ×8. 64 MiB copies: 6.79–7.09 GB/s depending on direction and host-memory type. | Kernels cannot change this link's physical capacity. We can avoid copies, maintain residency, use reusable staging, and group transfers. |
| Pageable host memory | At 4 MiB, H2D: 876 µs pageable versus 631 µs from host memory allocated/mapped by OpenCL. At 64 MiB improvement is small. | Staging has room at intermediate sizes. This control excludes buffer creation/mapping; it does not promise that percentage improvement for the public API. |
| Dispatch latency | An empty OpenCL kernel: 33.7 µs wall time, 10.7 µs event time in this control. | Empirical floor for that path and machine, **not a physical constant or wholly unavoidable cost**. Includes bindings/driver/queue. Fusion and grouped submission amortize it. |
| Large resident GEMV/DOT | Controls exceed caches and compare against CuPy on the same GPU. | If already close to that reference, prioritizing more kernel tuning may offer less than residency and execution. Being close to CuPy does not prove reaching a physical ceiling. |
| Resident solvers | Differences persist even without per-iteration matrix uploads. There are reductions, GEMV, allocations, dispatch, and scalar observations. | Largely controllable. With Python control we must wait for its decisions. Moving control to an executor/device is an architectural extension; omitting checks would change the contract. |
| Vulkan Python/CFFI | Repeated capability reads and struct construction appear in cProfile. Caching limits reduced the first component. | Our implementation's cost. Struct/descriptor creation, recording, and one submit per operation remain; batching and a more compact submission path are missing. This is not a Vulkan limit. |
| Batched CLBlast matmul | The adapter emits one GEMM call per matrix; small-matrix batches pay for many calls. | Controllable integration: use batched/strided-batched GEMM and compare against custom. The BLAS boundary must not prevent epilogues or custom kernels. |
| FP64 on this GPU | Compute capability 12.0; scalar FP32:FP64 throughput of 64:1 according to NVIDIA. | A restriction of the chosen hardware, not Python or OpenCL. Improving a kernel does not turn this into a high-throughput FP64 GPU. Do not silently reduce precision. |
| Numerical range of norms | Sum of squares can underflow or overflow; false convergence with tiny values has been reproduced. | Controllable numerical gap, separate from performance. Requires robust norms and coefficients, not an arbitrary epsilon or changed tolerances. |
| AMD/Intel and other systems | Only the local NVIDIA GPU was used. | Missing evidence, not a demonstrated limitation. Portable kernels and APIs require real-hardware validation; PoCL or a software driver does not replace it. |

For PCIe Gen3 ×8, the 128/130 encoding calculation gives about 7.88 GB/s per
direction before protocol overhead; this is not guaranteed useful bandwidth.
The copy control is already near that scale. Encoding and per-lane rate are
documented by [Intel](https://www.intel.com/content/www/us/en/docs/programmable/683724/18-0/physical-layer.html).
D2D counts payload bytes once; that quotient is not compared directly with an
advertised figure counting reads and writes. A float32 2048² matrix occupies
16 MiB; this CPU has 100 MB of total cache published by
[AMD](https://www.amd.com/en/newsroom/press-releases/2024-1-8-amd-reveals-next-gen-desktop-processors-for-extrem.html).
The CUDA runtime reports 32 MiB of L2 on this GPU. The 8192² matrix occupies
256 MiB: large controls avoid concluding “DRAM bandwidth” from small cached problems.

The FP64 ratio applies to scalar arithmetic, not Tensor Cores. TF32 was not
enabled and float64 was not replaced by float32.
[NVIDIA compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
Khronos documents allocation and command-buffer creation/release costs, but
attribution to our runtime comes from local profiling, not just those guides.
[Vulkan allocation](https://docs.vulkan.org/guide/latest/memory_allocation.html),
[command-buffer reuse](https://docs.vulkan.org/samples/latest/samples/performance/command_buffer_usage/README.html).

## Correctness: a provider failure and numerical debt

CLBlast 1.7.0 GEMV produced NaN with finite inputs when a recycled destination
contained NaN/Inf, even with `beta=0`. This was isolated by disabling only native
GEMV: GEMM solved the same case; disabling DOT did not. The adapter now fills
the destination with zero through a GPU operation on the same queue. There is
no upload or host wait, but there is an additional command included in final
timings. Regressions check normal/transposed GEMV, float32/64, and repeated
CG2048 with a contaminated pool. This behavior is not attributed to all BLAS
providers or every CLBlast version.

Preexisting numerical debt is reproduced with `A=eye(2)`,
`b=[1e-30,-1e-30]` float32, `rtol=atol=0`: the squared residual underflows.
The float64 check gives a norm of approximately 1.414e-30 while the float32
sum of squares gives zero. A complete solution must also cover CG/SD coefficients;
changing only the norm can move the failure to DOT. This is explicitly documented
in [compatibility](compatibility.md). It was reproduced on all three local
backends: all declared convergence in zero iterations with reported residual
zero; the float64 check gave 1.414e-30.

## Architectural decision and next investment

**Eager execution with resident arrays + optional compile** is retained, with a
common IR and backend-specific lowering. Python assignments are not fusion
boundaries within a compiled function. The project need not become a global
lazy tracer or require 100% NumPy coverage to apply these optimizations.

1. **Vulkan execution and transfers.** Fewer Python structures per dispatch,
   safe descriptor-set reuse, recording/submitting operation groups, and better
   staging. A small native module can serve the executor without changing the
   array API. Preserve in-flight resource management and locks.
2. **Solvers and useful boundaries.** Fuse GEMV with updates/reductions where
   the provider permits, Vulkan map-reduce, and reusable preparation of immutable
   systems. The latter needs an explicit object/contract: input validation must
   not be removed based on a hidden assumption.
3. **Prepared plan executor.** The control calling the plan directly measures
   savings from omitting binding/specialization; it is not yet native code or a
   public API. If savings are small relative to `_run`, prioritize backend
   submission before rewriting the complete compiler.
4. **Contract-based BLAS with custom kernels.** Integrate native batching;
   optimize custom OpenCL GEMM (still untiled) and improve Vulkan tiles if the
   workload justifies it. Preserve dtypes, broadcasting, and errors when switching providers.
5. **Robustness and external evidence.** Scaled norms/coefficients with range
   tests; run on AMD and Intel, including a device without fp64. That hardware's
   absence remains pending rather than becoming a validated promise.

Reducing host overhead must preserve checks at every iteration. A future API
with lower frequency would need to be explicit and measured separately. Small
data remain supported; winning there does not drive the design.

## Artifacts and reproduction

Artifacts are stored in `.artifacts/benchmarks`, outside versioned code:
`gaps-before` (frozen wheel), `gaps-after` (intermediate stage), `gaps-final`,
`gaps-execution-final`, `gaps-solvers-final`, and `gaps-report`. The intermediate
stage exposes corrected regressions; it does not represent the final runtime.

```powershell
$env:PYOPENCL_NO_CACHE='1'
$env:PACHO_VULKAN_COMPILER='<path to glslang or glslc>'
$env:OPENPACHO_CLBLAST_LIBRARY='<path to CLBlast>'
$env:CUPY_CACHE_DIR='<private writable directory>'
python -m benchmarks.gaps --output .artifacts/benchmarks/gaps-final/profile.json
python -m benchmarks.opencl_events --output .artifacts/benchmarks/gaps-final/events.json
python -m benchmarks.transport_floor --output .artifacts/benchmarks/gaps-after/transport.json
python -m benchmarks.scale --output .artifacts/benchmarks/gaps-final/scale.json
python -m benchmarks.execution --output .artifacts/benchmarks/gaps-execution-final/raw.json
python -m benchmarks.execution --solvers-only --output .artifacts/benchmarks/gaps-solvers-final/raw.json
python -m benchmarks.gaps_tables
python -m benchmarks.scheduling_control
```

`transport_floor` and `opencl_events` are separate OpenCL controls; the former
also collects `nvidia-smi` on this machine. Do not run another GPU workload during
benchmarks. Historical comparison uses the frozen wheel through `PYTHONPATH`,
whose path is recorded in profiles. The verification report retains the
[test and packaging scope](verification.md).
