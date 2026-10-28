# Tracking numerical error

The [rectangular GEMM, cuBLAS, and layouts round](three-iterations-results.md)
retains errors alongside timings. New custom tiles preserve the bits
of the previous path in 84 measured groups; seven groups remain outside
float32 tolerance because of long accumulation/cancellation, including controls where
the optimization is not even activated. Tolerances were not raised, and speed
gains are not presented as accuracy gains. Classifying these warnings and
defining per-operation guarantees is part of [release preparation](release-readiness.md).
The isolated Vulkan confirmation also records one element outside the same
threshold in a product with K=8191; both controls and all five
candidate tiles reproduce it. CLBlast with flags and cuBLAS vector selection pass their
respective tolerances in the measured rounds.

Time and result quality are recorded separately. Numerical checks
run after stopping the timer and synchronizing the backend;
validation is not removed, nor are benchmark tolerances relaxed.

The [padding reassessment](padding-reassessment.md) from September 19
retains bitwise-identical results in 172 groups when enabling/disabling
padding. The main suite has three CUDA groups with one element outside
`rtol=5e-5, atol=5e-5` against float64: K=8191, with shapes/seeds documented
in the report. Both unpadded controls and padded mode fail
on the same elements. A CPU reproduction with float32 products and sequential
summation exactly matches the CUDA value of the investigated case. Improving
accumulation/cancellation remains pending; neither FMA nor tolerance is changed.
`benchmarks.padding_decision` distinguishes `complete` from `all_reference_close` and
retains finite violations while measuring the remaining modes.

The [staging/RNG round](staging-random-results.md) adds separate distribution
tracking: known Philox vectors, counter continuity, ranges,
moments, histograms/tails, and transformation errors against the same CPU dtype.
It neither replaces adversarial matrix tests nor resolves their cancellation warnings.
Matrix kernels and solver algorithms do not change in that round.

The [GEMM guards round](kernel-stability-results.md) remeasures 42 adversarial
OpenCL cases: the same nine warnings and zero exceptions. The 88 public-call
controls preserve their metrics between the previous/current generator;
the guard change does not modify accumulation. CUDA/Vulkan do not repeat the adversarial
suite in this round because they retain their default shaders.

The [shared execution round](backend-sharing-results.md) from September 13
repeats 158 configurations with Decimal80: zero exceptions and the same 40 warnings,
with no new or resolved cases. CUDA still declares 52 norm/solver combinations
outside scope. The 104 irregular applied configurations
have finite results, with maximum relative L2 error of 7.95e-7 in float32
and 1.51e-15 in float64 against NumPy float64; for float64 this is equal precision.
These ordinary inputs do not replace adversarial diagnostics.

The [Vulkan commands round](vulkan-dispatch-results.md) repeats its 42 adversarial
cases: the same ten warnings, with no new exceptions. The correction for
float32 powers with negative bases is also covered by eager,
compile, and map-reduce regressions; it does not resolve cancellation limitations in reductions.

The [initial CUDA round](cuda-initial-results.md) adds native providers
`cuda-cublas` and `cuda-custom` to `benchmarks.accuracy`. Their norm/solver exclusions
are recorded explicitly. This round has 158 measured configurations
and 40 warnings: six per CUDA provider, plus the previous 28 from OpenCL/Vulkan.
CUDA warnings concern cancellation in sum, dot, and GEMM; finiteness does not equal accuracy.

## Measurements at scale

`benchmarks.execution`, `benchmarks.scale`, and `benchmarks.prepared` record
maximum absolute error and relative L2 error, alongside dtype and reference.
For operations, a NumPy reference is recomputed in float64 from inputs
already rounded to the original dtype. For float64 results this is an
equal-precision comparison, **not a certification of exactness**. Changing
reduction order, BLAS provider, or FMA contraction may change result bits
without violating the numerical contract.

Solvers in `examples/solvers.py`, outside the package API, also record the true residual `b - A @ x`, evaluated with
scaling and float64, and normwise backward error:

`||b - A @ x||₂ / (||A||F * ||x||₂ + ||b||₂)`.

A's norm is Frobenius: it is stated explicitly to avoid confusing this
inexpensive metric with a spectral norm. Error against the generating solution
is also recorded when available. Because the RHS is generated through a
multiplication in the input dtype, that solution may differ slightly from
the exact solution of the rounded system. The reference label retains
this limitation. The prepared-system report separates metrics for each
mode to make accuracy or convergence changes visible.

Diagnostic norms use scaling; they do not square extreme values
directly. A zero reference produces zero relative error if the result is also
zero, and explicit infinity otherwise. Nonfinite results are counted and
flagged. JSON uses `null` or an explicit string when it cannot represent a
number; it does not write NaN or Infinity tokens incompatible with strict JSON.

`benchmarks.gaps_tables` exports error columns and a resident-time table
alongside accuracy. Historical reports without these metrics show an empty
cell; an error is not invented from an old validation flag.

## Adversarial cases with a more precise reference

```powershell
.venv\Scripts\python.exe -m benchmarks.accuracy --repeats 3 --output .artifacts/benchmarks/accuracy/raw.json
```

The command produces JSON and Markdown. For a run without GPU:

```powershell
.venv\Scripts\python.exe -m benchmarks.accuracy --backends numpy,pacho-numpy --output .artifacts/benchmarks/accuracy/cpu.json
```

The suite uses standard-library `decimal` with 80 digits. It converts
floating-point inputs exactly to Decimal and computes sum,
DOT, GEMV, GEMM, energy, and norm references. For small systems, it solves
the actually represented system by elimination with pivoting and also evaluates its
residual in Decimal. The energy reference corresponds to the real-valued expression;
the difference includes rounding of its intermediate operations.

`finfo(longdouble)` is inspected and its mantissa bits are saved: it is not assumed
that Windows offers a NumPy type more precise than float64. Decimal80 is a
high-precision reference for these bounded cases, not exact arithmetic for
any imaginable combination of ranges.

Cases cover cancellation, norms that previously suffered underflow/overflow,
subnormal arithmetic, irregular matrix sizes, fusion, and systems with small/large
RHS. Hilbert systems demonstrate that a small residual
can coexist with a large solution error. The 2-norm condition number estimated
in float64 is published, without treating it as a certified bound.

CuPy covers operations and solvers expressed through its API; cuBLAS and nvmath cover
GEMM through existing adapters. `numpy` identifies the direct comparator,
and `pacho-numpy` identifies OpenPacho's implementation on its CPU backend. Details
of the comparator solver algorithm must be read alongside results;
the OpenPacho path uses the external example `examples/solvers.py`. The final
round uses the same normalized recurrence in OpenPacho and comparators.
NumPy computes the norm by scaling first; the CuPy solver calls cuBLAS NRM2 to
avoid a fragile norm based on `sum(v*v)`. Its pointer mode is restored after
every call. The suite's `norm` operation cases still measure each comparator's
native function; those results do not describe cuBLAS NRM2.
Development records with previous recurrences are historical.

This suite's thresholds are indicators for investigation, **not a universal
relative-error promise**, nor do they replace test tolerances. Severe
cancellation or an ill-conditioned problem may exceed the threshold even
with a valid implementation. The process exits with a nonzero code if it
finds exceptions, unavailable requested backends, or flags outside the
threshold; it retains every row and does not interrupt remaining cases. Timings
of these small inputs provide diagnostic context, not a
ranking of backend performance at scale.
