# First optimization stage — 2026-09-06

Elementwise fusion, an OpenCL memory pool, by-value scalars,
fused solver updates, and an optional CLBlast adapter were implemented.
Backend selection remains explicit. The API and its limits are in
[performance.md](performance.md).

## Final comparison

26 configurations, 390 series, and 7 rounds per series, with no execution or
numerical validation errors. NumPy with 1/8/16 BLAS threads,
the previous unmodified wheel, and the current version were measured in the same session.
The lowest median NumPy time was selected per case. Hardware: Ryzen 7 5700X3D and RTX 5060 Ti, Windows;
CLBlast 1.7.0, NumPy 2.5.3, and PyOpenCL 2026.1.4. Only this GPU was validated.

Medians in **milliseconds including upload, computation, and download**, warm runtime,
with inputs and validation outside the timer. CPU NumPy receives/returns host data
and is not burdened with an artificial copy. Every call synchronizes. These medians
alone do not establish statistical significance for small differences.

| Case | NumPy | Previous OpenPacho | Current custom | Current fused | Current CLBlast |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 updates, 1,048,576 elements, float32 | 197.50 | 231.58 | 34.08 | **26.18** | — |
| 100 updates, 1,048,576 elements, float64 | 378.19 | 258.35 | 35.43 | **28.20** | — |
| sin/exp/cos, 1,048,576 elements, float32 | 7.66 | 9.16 | 2.95 | **2.68** | — |
| Matmul 2048², float32 | 30.45 | 36.64 | 24.38 | — | **11.17** |
| Matmul 2048², float64 | 63.15 | 86.27 | 61.60 | — | 60.78 |
| Jacobi 4096 variables, float64 | 138.71 | 166.72 | **95.10** | — | 97.16 |
| Jacobi 1024 variables, float32 | **3.12** | 32.27 | 27.01 | — | 29.01 |
| CG 1024 variables, float32 | **2.47** | 22.08 | 24.22 | — | 24.80 |

For generic expressions, `custom` already includes reusable memory and by-value
scalars. `fused` adds `@fuse`. For solvers, both `custom` and `CLBlast`
already use fused updates. CLBlast changes floating-point products.

## Contributions and remaining work

- The fused float32 sequence is approximately **7.54× faster than NumPy**
  and **8.85× faster than the previous wheel** in this experiment. The change from 231.58 to 34.08 ms
  corresponds to the combined execution improvements; fusion provides the change
  from 34.08 to 26.18 ms. Pool and scalar effects were not isolated from each other.
- CLBlast provides **2.73× over NumPy** for 2048² float32 matmul. The
  small float64 difference should not be interpreted as an established advantage.
- CLBlast does not always beat the custom kernel. The initial adapter also uses GEMM
  for vectors. Dedicated GEMV/DOT and kernels combining products/updates
  remain opportunities for further work.
- Small solvers still lose. CG with 1024 variables even regresses
  against the previous wheel. Tracing and checks are profiling candidates;
  this experiment does not isolate the cause of that regression.
- Solvers were compared using equivalent recurrences and true-residual criteria,
  on SPD tridiagonal matrices **stored and processed densely**. This does not
  compare against a specialized sparse solver or cover every problem.
- The requested precision is used, without reducing float64 to float32, and
  residual checks are retained at every iteration. Rounding coefficients on
  the device can alter iteration counts; convergence and the solution are validated.
- The decorator traces Python on every call and fuses an expression, not a complete
  loop. It covers neither reductions nor matmul. The NumPy backend does not compile that graph.

## Evidence and reproduction

The complete local report
(`.artifacts/benchmarks/optimization-final-2026-09-06/REPORT.md`, not distributed)
includes every case and links samples, resident timings, and validations.
Runtime hashes remained unchanged during the final run. Snapshots
of the runner and suite were saved, and the previous wheel remains in
`.artifacts/benchmarks/baseline-0.1.0a1.whl`.

Earlier exploratory experiments were retained separately: six matmul configurations
initially had a benchmark dimension-construction error,
corrected before the final run. An overly broad disabling of FP_CONTRACT was also
corrected: it is now disabled only for fused expressions. The figures above come
exclusively from the complete final run, with zero errors.

Command: `python -m benchmarks.optimization --output NEW_DIRECTORY --clblast
--legacy-wheel .artifacts/benchmarks/baseline-0.1.0a1.whl --rounds 7`, with
`OPENPACHO_CLBLAST_LIBRARY` pointing to the native binary. See
[benchmarks/README.md](../benchmarks/README.md).

Implementation verification: 207 tests passed and 1 skipped in the
CPU/OpenCL suite; the CLBlast variant also passed the complete suite before the
FP_CONTRACT adjustment (which affects only custom kernels). The adapter also
has mock-ABI tests and 10 native product cases. Additional review of
broadcasting, buffer lifetimes, and residency found no new correctness
regressions. The wheel installed in the independent CPU environment
passed 102 tests, with 99 skips for cases requiring GPU/CLBlast.
Out-of-range floating-point-to-integer conversions remain
an earlier documented limitation.

Sparse matrices and preconditioners remain future debt.
