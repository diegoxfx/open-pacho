# Measuring applications as well as primitives

Plan and first suite dated 2026-09-08. The previous solvers use dense storage
for an SPD tridiagonal matrix with diagonal 2.5 and off-diagonals -1, condition
number ≤9. That control supports repeatable results and iteration counts, but
does not represent the diversity of dense problems, conditioning, or application
matrices. Nor is it evidence for prioritizing sparse support, which remains
out of scope.

## First implemented expansion

`benchmarks/applications.py` contains two applied patterns with deterministic
synthetic data. These are not presented as results on production datasets.

| Workload | Computation | What it reveals |
| --- | --- | --- |
| Sample covariance | Centering by column mean; `center.T @ center / (n-1)` | Axis reduction, broadcasting, transpose materialization, and rectangular GEMM. |
| Ridge regression | 20 steps of `w -= rate * (Xt @ (X @ w-y) + lambda*w)` | Reuse of a matrix and prepared transpose, two rectangular GEMVs per step, and a fused update. |

Full shapes: 4096×128 and 16384×256, float32/64. Seed 47. Ridge uses
`rate=0.25/n`, `lambda=0.1`, normal data scaled by `1/sqrt(d)`, and a noisy RHS.
The twenty steps are a **fixed-work control**, not a claim of convergence or of
the best optimizer for that problem. The final objective, gradient norm, and
error against the same float64 trajectory are published.

Each implementation receives the same inputs. Direct NumPy with 1/8 threads,
OpenCL custom/CLBlast, Vulkan, optional CUDA custom/cuBLAS, and CuPy are retained;
OpenPacho has eager and compiled controls. The ridge transpose is materialized
once and timed as preparation. Resident mode reuses that preparation; full mode
includes upload, preparation, computation, and download. In covariance, centering
and transposing the centered array are part of each computation. That cost is
not hidden.

```powershell
.venv/Scripts/python.exe -m benchmarks.applications --quick --backends "" --output .artifacts/benchmarks/applications-cpu.json
.venv/Scripts/python.exe -m benchmarks.applications --repeats 7 --output .artifacts/benchmarks/applications.json
.venv/Scripts/python.exe -m benchmarks.applications --irregular --repeats 7 --output .artifacts/benchmarks/applications-irregular.json
```

The report records first call/compilation, preparation, residency, transfers,
and diagnostics outside the timer. The float64 reference for float64 cases has
the same precision and is not an oracle. Adversarial accuracy metrics using
Decimal continue to complement these measurements at scale.

The `--irregular` profile uses 4095×127 and 4097×129 for both precisions and
workloads. It compares odd dimensions on either side of the powers of two in
the 4096×128 case. `tests/test_irregular_workloads.py` adds 144 combinations:
rectangular GEMM around tiles 16/32/64/128, even non-power-of-two sizes, GEMV,
reductions along both axes, and odd-batch broadcasting. Square solvers of sizes
31/65/129 use dense SPD matrices outside the tridiagonal family, compare ordinary
and prepared systems, and verify an independent residual. A rectangular matrix
is not passed to a solver whose API requires a square system.

Since 2026-09-12, `--backends` also supports `cuda-cublas,cuda-custom`.
These are paths in the initial CUDA backend; `compiled` captures the plan and
fuses supported elementwise/map-reduce groups through `CUDALowering`. Matmul and
BLAS remain separate. CUDA now also supports stable norms and the solver
examples; bool inputs can be reduced with an explicit floating accumulator,
but bool accumulation and graphs remain out of scope. The JSON files identify
plans as `compiler_lowering: backend-plan` and record native library versions.
The [CUDA fusion round](cuda-fusion-results.md) documents that scope; the
[initial CUDA round](cuda-initial-results.md) retains baseline measurements for
comparing timings and errors.

## Agreed new families

The current solvers live in `examples/solvers.py`, outside the package. New
domain algorithms will maintain that separation, with a readable example and
an independent timing runner; no neural or physics framework is added to the runtime.

| Proposed family | First case | What it reveals | Validation |
| --- | --- | --- | --- |
| Dense network, implemented | Two-layer inference with tanh and one SGD step with explicit gradients, in `examples/dense_network.py` | Rectangular GEMM, bias/broadcasting, activations, transposes, and temporaries | All gradients against finite differences in a small case; prediction, loss, gradient, and parameter errors; same update in all references |
| Coupled oscillators, implemented | Velocity-Verlet steps in a dense SPD harmonic system, with a reused stiffness matrix | Repeated GEMV, fusible updates, synchronization, and error accumulation | Analytical/modal trajectory for small cases, temporal error, and energy drift; same dt, steps, and initial conditions |

Both support odd/rectangular sizes in the appropriate operands and mostly use
existing operations. Dense simulation is a control for dense operations, not an
efficient way to represent all local physics; do not densify a sparse problem
to obtain a favorable benchmark.

Training does not imply adding autodiff. Start with direct NumPy, OpenCL
custom/CLBlast, Vulkan, and CuPy/CUDA where their primitives allow it. PyTorch/JAX
are additional controls if they implement the same work and precision; separate
compilation/capture and distinguish autodiff from explicit gradients. Retain
eager/compiled paths, errors, failures, transfers, and peak memory.
The [first dense network](dense-network-results.md) has an example, tests, and an
independent runner; it records bytes but does not yet instrument peak memory.
The [oscillator simulation](oscillators-results.md) adds a fixed-step run with a
single reused compiled step. It separates backend differences, modal temporal
error, and energy; it also does not yet measure peak memory.

## Next workloads and the decisions they must justify

1. **Blocked distances for neighbor search/clustering.** Compare fused
   differences against the Gram identity, including cancellation when subtracting
   large terms. Measure peak memory and blocking cost; a faster method that
   loses small distances does not meet the same objective.
2. **Batched transformations.** Many small matrices with broadcasting, irregular
   sizes, and different batch counts. Separate per-matrix dispatch from true
   batching; do not discard the workload because each matrix is small.
3. **Solvers on varied dense matrices.** SPD families generated with controlled
   spectra, regularized Gram matrices, and multiple RHSs. Measure time to
   tolerance, iterations, solution error, residual, and memory; do not infer
   these from twenty steps.
4. **Reused pipelines.** Centering/normalization, covariance, and projections
   with factors already available. Evaluate `mean`, transpose, GEMM, and
   temporaries before adding a new factorization. Do not advertise complete
   PCA without an eigensolver.

New function selection should follow these cases. `maximum`/clipping, efficient
axis reductions, and matmul with transposed operands may offer more immediate
value than indiscriminately expanding the NumPy surface.

## Protocol for deciding priorities

- Separate small problems, cache-resident data, and matrices larger than cache;
  record bytes and peak memory alongside time and error.
- Compare the same dtype, inputs, algorithm, and step count or stopping criterion.
  Treat an algorithmic improvement as an additional, identified comparison.
- Publish median and dispersion, first execution, and preparation cost;
  interleave implementations and retain mixed or failed cases.
- Measure GPU timestamps and sustained copy/compute rates to build a per-workload
  limits model. Do not call CuPy's speed a physical limit.
- When an application case reveals a regression, isolate its dominant operation
  before changing kernels or provider selection.

Results from this stage are collected in
[solvers, applications, and CUDA preparation](solver-applications-results.md).
