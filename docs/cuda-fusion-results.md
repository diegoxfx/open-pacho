# CUDA fusion and selection of the next improvement opportunity

Local measurement of 2026-09-13, Windows, Python 3.12 and RTX 5060 Ti, driver 616.64.
Fusion approximately halves execution of twenty ridge steps
compared with the previous CUDA plan, without changing the algorithm. It does not produce
uniform end-to-end improvement or general parity with CuPy.

## Why the priority shifted

The latest [Vulkan completion](routing-completion-results.md) optimization
gave small or mixed improvements. That does not exhaust Vulkan's opportunities: its
submission and transfer costs remain. However, CUDA retained a more directly
removable cost: each elementwise plan node launched a primitive and
materialized its result. Fusion already tested on other backends had greater
immediate potential than further tuning the same Vulkan wait.

The criteria are usage frequency, removable cost, resident/complete improvement,
repeatability, accuracy and effort. Focus rotates between optimization areas;
a backend is not declared finished after a marginal round. Correctness prevails
if a functional regression appears.

## Implementation

[`CUDALowering`](../src/openpacho/_cuda_lowering.py) implements the existing
`backend.create_lowering()` factory. Selection happens when preparing the
plan; the shared compiler adds no backend-name conditions.

- Elementwise groups become an NVRTC kernel. Broadcasting,
  scalars by value and each operation's rounding retain their contract.
- Map-reduce evaluates the expression inside the reduction and stores only partials.
  It retains the previous CUDA tree: 128 threads, blocks of up to 4096 terms and
  subsequent partial reduction when needed.
- The prepared payload retains code/geometry without retaining inputs or scalar
  values from previous calls. Dependency management, public outputs
  and shared temporaries remain in the shared compiler.
- Matmul continues using the explicit custom/cuBLAS provider. There is no fusion
  inside cuBLAS or CUDA GEMV epilogue introduced at this stage.

`@pacho.compile`, `pacho.compile(function)` and `pacho.fuse` access the lowerer.
Eager execution retains its behavior. Covariance/ridge algorithms
and external solvers are unchanged, as are optional dependencies.
CUDA still lacks scaled norms, bool accumulators and graphs.

`--fmad=false`, `--ftz=false`, precise division and square root are retained; fusion
does not authorize fast-math. Reduction barriers are reached by every
thread, including tails and empty reductions. Contracts consulted:
[NVRTC options](https://docs.nvidia.com/cuda/nvrtc/) and
[CUDA block synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

## Protocol and control

[`benchmarks/fusion.py`](../benchmarks/fusion.py) measures nine cases in float32/64:
chains of 33 and 1,048,577 elements; energy 1025×1023; row energy 257×129
and 17×4099; row centering 2049×33; squared loss 2049×1023; covariance
1025×65; and ridge 2049×129 over twenty identical steps. Common array operators and
methods are used, with the same inputs per seed.

The CUDA control alternates the order of two modes per case:

- **primitive-plan:** prepares without a lowerer, like the previous wheel.
- **fused-plan:** prepares with `CUDALowering`, in the same instance and context.

The runner verifies through AST that `_run`, `_compile`, `_activate`, `empty`,
`elementwise`, `reduce`, `matmul` and `synchronize` are equal to those in the previous
wheel. SHA256 of its CUDA module:
`3a87b365f08bc8dbd7dfd58fa3be5e95acc0d39ba628790d3aefb43fc6a3265f`.
This is a lowering control with the same primitives, not an integral execution
of the old package. The factory is restored even on an exception.

Processes are sequential, with randomized initial order and reverse order in the
second round. Each process requires at least ten seconds, three
consecutive readings with GPU/memory utilization ≤5 %, temperature within
2 °C of the reference and ≤52 °C, with no observed thermal throttling. This is followed
by warmup of at least three calls and 0.25 s per mode/scope, outside
the clocks. Synchronization occurs before and after every sample.

There are 21 samples per configuration and scope, two CUDA rounds and two per
GPU reference. **144 CUDA configurations and 218 reference configurations** completed
measurement. NumPy with one thread has only one round: the second did not reach the
required idle state within 120 s and was not run. The criterion was not relaxed. Accepted
windows ended at 43–47 °C after waits of 10.4–24.9 s. This does not guarantee
constant clocks or eliminate system activity during samples.

**Resident** scope excludes upload, input preparation and download;
ridge reuses the prepared transpose. **Complete** scope includes upload,
preparation, computation and download, with already warmed kernels/plans. Both
measure wall time, including Python, driver and waits. The first call
is separated in JSON files; it may reuse kernels compiled by other cases
in the process and does not always represent an independent cold start.

## Before and after

Milliseconds; each cell shows medians for **round 1 / round 2**. JSON
and CSV files retain all samples, p10/p90, first call and errors.

| CUDA custom, float32 | Previous resident | Fused resident | Previous complete | Fused complete |
| --- | ---: | ---: | ---: | ---: |
| Chain 1,048,577 | 0.292 / 0.290 | 0.098 / 0.086 | 2.650 / 2.639 | 2.432 / 2.426 |
| Energy 1025×1023 | 0.529 / 0.374 | 0.192 / 0.168 | 1.931 / 1.905 | 1.957 / 1.681 |
| Row energy 257×129 | 0.281 / 0.353 | 0.096 / 0.083 | 0.531 / 0.540 | 0.539 / 0.334 |
| Squared loss | 0.429 / 0.439 | 0.209 / 0.211 | 1.964 / 1.958 | 1.730 / 1.729 |
| Covariance | 0.559 / 0.550 | 0.708 / 0.442 | 0.733 / 0.743 | 0.850 / 0.850 |
| Ridge, twenty steps | 12.149 / 12.316 | 5.948 / 5.806 | 12.913 / 12.851 | 6.555 / 6.666 |

Custom float64 ridge goes from **12.133 / 13.135** to **6.016 / 6.116 ms** resident;
complete goes from 12.968 / 14.126 to 6.770 / 7.252 ms. With cuBLAS, float32 ridge
goes from **12.592 / 12.566** to **6.422 / 6.261 ms** resident. Its complete time is more
variable: 15.243 / 13.089 versus 10.374 / 6.858 ms. That variation is not hidden.

Energy and chains show that eliminating launches can save a lot
with resident data and little when copies dominate. The difference between
scopes is not interpreted as an isolated measurement of PCIe cost.

### Covariance follow-up

The complete-path custom float32 loss in both rounds motivated another control,
with the same context, 31 alternating pairs per scope and three blocks. It passed
the same idle criterion after 10.4 s, at 47 °C. It did not change the runtime:

| Block | Previous → fused resident | Previous → fused complete |
| --- | ---: | ---: |
| 1 | 0.561 → 0.440 ms | 0.747 → 0.633 ms |
| 2 | 0.963 → 0.703 ms | 0.761 → 0.636 ms |
| 3 | 0.555 → 0.444 ms | 0.749 → 0.635 ms |

The initial loss **did not reproduce in this alternating control**. Dispersion
remains visible, especially in the second block. Initial data
remain in the report; their cause is not attributed to a kernel and no
exception based on workload name is added. Custom float64 and cuBLAS covariance
also improved in the main control.

## References and remaining gaps

Resident milliseconds, float32, medians of both rounds. This table compares
the measured paths; it is not a universal ranking or a physical hardware bound.

| Implementation | Chain 1,048,577 | Energy 1025×1023 | Ridge 2049×129 |
| --- | ---: | ---: | ---: |
| NumPy, 8 threads | 1.813 / 1.869 | 1.616 / 1.585 | 0.789 / 0.797 |
| OpenCL custom, compiled | 0.133 / 0.130 | 0.114 / 0.112 | 5.464 / 5.290 |
| OpenCL CLBlast, compiled | 0.129 / 0.155 | 0.131 / 0.114 | 6.387 / 6.564 |
| Vulkan, compiled | 0.196 / 0.171 | 0.224 / 0.232 | 7.624 / 4.983 |
| CUDA custom, fused | 0.098 / 0.086 | 0.192 / 0.168 | 5.948 / 5.806 |
| CUDA cuBLAS, fused | 0.086 / 0.089 | 0.161 / 0.175 | 6.422 / 6.261 |
| CuPy eager | 0.085 / 0.082 | 0.100 / 0.104 | 5.211 / 4.700 |
| CuPy fuse | 0.029 / 0.046 | 0.411 / 0.413 | — |

[`cupy.fuse`](https://docs.cupy.dev/en/stable/user_guide/kernel.html#kernel-fusion)
is compared with the same elementwise/reduction functions supported by this suite.
It helps on chains; on this large energy case it is worse than CuPy eager. Both modes
are retained. Ridge and covariance use the common CuPy eager path; there is no claim
that they represent its best possible implementation.

In complete scope, eight-thread NumPy still wins ridge at this size:
**1.141 / 1.315 ms**, versus 6.555 / 6.666 ms on CUDA custom and 5.179 / 4.957 ms
on CuPy. Complete energy on OpenCL custom gives 1.565 / 1.571 ms and CuPy eager
1.509 / 1.502 ms; large resident gains do not transfer in full.
The chain of only 33 elements remains clearly a CPU case: approximately
2–3 μs, versus 85–86 μs on fused CUDA custom and 23 μs on CuPy fuse.

nvmath Matmul is not remeasured in this round: matmul implementation did not
change. It remains a reference in matrix reports. PyTorch/JAX were not added
and solvers were not modified to favor a backend.

## Memory, launches and numerical error

Diagnostics run outside timing samples. They count custom launches
and bytes requested through `empty`; they **do not** measure peak live memory,
new driver allocations or internal cuBLAS launches.

| CUDA custom float32 | Previous → fused launches | Previous → fused requested bytes |
| --- | ---: | ---: |
| Energy 1025×1023 | 4 → 2 | 8,389,628 → 1,028 |
| Covariance 1025×65 | 6 → 5 | 567,320 → 567,060 |
| Ridge, twenty steps | 140 → 80 | 379,440 → 348,480 |

Energy avoids materializing two complete matrices; 256 float32 partials
and one output remain. Ridge gains mainly from fewer operations/submissions,
not from eliminating large transfers. With cuBLAS, custom ridge kernels
go from 100 to 40, in addition to the forty BLAS products retained by the algorithm.

Comparison uses the NumPy expression calculated in float64. For float64 outputs,
that reference has **the same precision**, rather than being an exact solution. Maximum observed
relative L2 error over all cases of each group:

| Current path | float32 | float64 |
| --- | ---: | ---: |
| Fused CUDA custom | 4.36×10⁻⁷ | 8.20×10⁻¹⁶ |
| Fused CUDA cuBLAS | 8.37×10⁻⁸ | 3.19×10⁻¹⁶ |
| OpenCL custom | 4.34×10⁻⁷ | 8.37×10⁻¹⁶ |
| Vulkan | 4.31×10⁻⁷ | 8.27×10⁻¹⁶ |
| CuPy, both modes | 1.80×10⁻⁷ | 3.06×10⁻¹⁶ |

These maxima are not guarantees for other inputs. The CUDA float32 maximum
absolute error is 0.00489, corresponding to the scale of accumulated results; both
errors are retained per case. Additional tests check cancellation, subnormals,
NaN/Inf, signed zero, float64→float32→float64 narrowing, promotion before
each operation and absence of hidden FMA. Ordinary reduction is still not
a stable norm. The twenty ridge steps do not measure time to convergence.

## Next selection

1. **CUDA preparation/submission and axis reductions.** The fused chain still
   takes approximately 2–4 times as long as CuPy fuse. Separating argument
   preparation, context activation, launch and GPU will allow choosing the
   next change. Avoid assuming the entire gap is kernel time.
2. **Cost-based materialization.** Fusing a small operation with a
   large output through broadcasting may repeat work. Measure expressions and
   geometries; do not introduce rules by application name.
3. **Vulkan and matrix kernels.** Retain investigation of submission/copies,
   GEMV/DOT and layouts. Their priority rises when a profile shows more removable
   savings than the previous areas; do not keep microtuning a wait
   through inertia. Verified regressions are addressed before expanding scope.

## Reproduction and verification

From an environment with extras installed and local paths configured for
CLBlast/glslang, using a new output folder:

```powershell
python -m benchmarks.isolated --worker fusion --backends cuda-custom,cuda-cublas --rounds 2 --repeats 21 --max-temperature 52 --control-wheel .artifacts/cuda-fusion/before.whl --output .artifacts/cuda-fusion/control
python -m benchmarks.isolated --worker fusion --backends numpy-1t,numpy-8t,opencl,opencl-clblast,vulkan,cupy --rounds 2 --repeats 21 --max-temperature 52 --output .artifacts/cuda-fusion/references
```

The control wheel is an earlier local artifact, not a published version.
The package records source hashes; manifests preserve policy,
readings, commands and idle failures. Automatic association between backend
selection and telemetry UUID remains pending. The observed GPU was
`GPU-16858217-5c55-717b-c7dc-045e1345b3de`; no extrapolation is made to other devices.

Artifacts in `.artifacts/cuda-fusion`: `summary.csv` gathers all 362 configurations;
`control/` and `references/` retain original JSON files. The alternating follow-up
is separate in `covariance-control.json` and its script, with six additional
configurations. `environment.json` records versions and the check that application/solver
algorithms did not change from the previous sdist.
[Current verification](verification.md) records tests and the wheel.
