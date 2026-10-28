# Float64 contraction: a bounded causal experiment

Rounds completed on 2026-10-22. The runtime and its default policies remain
unchanged.

The [dense network](dense-network-results.md) and the
[shared GEMM candidate](cuda-fp64-gemm-results.md) left a specific question: how
much of the CUDA/OpenCL difference comes from arithmetic contraction. This
iteration compares rounding policies using benchmark code kept separate from
the runtime. It changes neither defaults nor the public API.

## Design

CUDA compares the current direct GEMM and the shared 16×16 candidate separately.
Within each geometry, both compilations receive exactly the same source and
options except `--fmad=false` versus `--fmad=true`. Subnormals and division/square-root
precision are preserved. OpenCL captures its current GEMM source and adds only
`#pragma OPENCL FP_CONTRACT OFF` to the control without contraction.
Both OpenCL programs retain `-cl-std=CL1.2`; effective options and environment
variables that could alter them are also saved.

Each executable belongs to the experiment: it is unloaded after synchronization,
without installing it in the runtime cache. Hashes, cache identities, and flags
are checked during preparation and execution. Source, options, versions,
capabilities, kernel attributes, and PTX/binary are retained. Counts of
`fma.f64`, `mul.f64`, and `add.f64` are limited to the `compute` entry: this is
evidence of the compiler's static IR, not loaded machine code or the dynamic
operation count. An opaque binary, unresolved mix of instructions, or additional
mathematical flags prevents a strong causal conclusion.

The 19 small cases use M=N=8 and K=15/16/17/31/32/33, random inputs,
cancellation, rounded products, subnormals, signed zeros, overflow, and
nonfinites. Their 64 outputs are checked against two binary64 recurrences:
separately rounded product and sum, or product plus accumulator rounded once.
The second oracle uses exact fractions; it depends on neither BLAS nor hardware
FMA. Each recurrence starts at +0 and respects increasing K order. The +0×+0
products of the final OpenCL tile are modeled explicitly: they can affect the
sign of a final zero.

Conformance to a policy does not imply mathematical accuracy. The exact dot
product of the binary inputs and the exact sum of already-rounded products,
when finite, are retained separately. For large GEMMs, five coordinates are
checked with those oracles and the full output against NumPy float64, a
same-precision reference, with rtol=2e-10 and atol=2e-12.

Measured sizes, M×K×N, are 4096×128×256, 128×4096×256, and 256×4096×32:
products already used by the dense-network application. Each comparison contains
two controls with the same policy and independent buffers, plus the candidate.
All six permutations are used equally often, with 24 samples per mode, a
separate first call, and warm-up of at least three calls and 0.10 seconds.
Two processes per backend reverse CUDA/OpenCL order and require observed idle
conditions before each worker.

The timer measures synchronized wall time with resident inputs and a preallocated
destination; it includes Python dispatch. It measures neither an application
nor isolated GPU time. Each call receives a destination with alternating NaN/Inf,
outside the timer, and is checked in full against its first output. The control
also reproduces production-path finite bits and nonfinite classes. Different
policies have different references; an FMA is not required to reproduce separate
rounding. Validation and report writing occur between sample groups and can leave
the GPU idle. These are measurements of synchronized calls in bursts, not a
sustained-performance test; prior warm-up does not remove that interval.
Interleaved controls expose variation within that scope.

## Reproducible execution

```powershell
.venv/Scripts/python.exe -m benchmarks.fp64_contraction --backend cuda-custom --profile all --repeats 24 --output <new-json>
.venv/Scripts/python.exe -m benchmarks.fp64_contraction --backend opencl --profile all --repeats 24 --output <another-new-json>
```

Run sequentially with exclusive device use and external idle checks. Compiler
artifacts are saved next to the JSON in a new directory; existing files are
not overwritten. Execution status, numerical acceptance, and evidence sufficiency
are independent, and failures are retained.

## Results and decision

The four runs complete **132 rows, 264 variants, and 1,296 performance samples**,
with 5,971 accepted output-comparison records, including first calls. The 44,316
coordinate evaluations against the two policies include repeated data and do not
represent independent points. There are no execution, numerical-acceptance, or
cleanup errors. All 33 runtime files and benchmark sources are identical across
the four runs. Each case's data also retain their hashes.

Local hardware: NVIDIA GeForce RTX 5060 Ti, Windows, driver 617.14; Python
3.12.14, NumPy 2.5.3, PyOpenCL 2026.1.4, and NVRTC 13.3.33. Results do not
establish performance on other GPUs or drivers.

For all three large sizes, `compute` PTX contains eight float64 multiplications
and eight additions in uncontracted direct CUDA, versus eight FMAs in its
contracted variant. CUDA tile16 and OpenCL contain 16 multiplications plus 16
additions, or 16 FMAs, respectively. These are static body counts, not the GEMM's
dynamic total. Native OpenCL generates FMA on this driver; `FP_CONTRACT OFF`
removes it. Output checks agree with those policies.

The table uses the average of the two control medians when a policy has two
controls. The final column is separate time divided by FMA time; above one favors
FMA. R1/R2 are different processes.

| Path and M×K×N | Separate, ms R1 / R2 | FMA, ms R1 / R2 | Separate / FMA R1 / R2 |
| --- | ---: | ---: | ---: |
| Direct CUDA, 4096×128×256 | 1.938 / 1.704 | 1.089 / 0.832 | 1.779 / 2.047 |
| Direct CUDA, 128×4096×256 | 1.907 / 1.817 | 0.882 / 0.803 | 2.162 / 2.262 |
| Direct CUDA, 256×4096×32 | 0.484 / 0.487 | 0.277 / 0.261 | 1.746 / 1.867 |
| CUDA tile16, 4096×128×256 | 1.933 / 1.707 | 1.075 / 0.826 | 1.799 / 2.066 |
| CUDA tile16, 128×4096×256 | 1.888 / 1.826 | 0.935 / 0.849 | 2.019 / 2.149 |
| CUDA tile16, 256×4096×32 | 0.527 / 0.540 | 0.277 / 0.309 | 1.904 / 1.745 |
| Native OpenCL, 4096×128×256 | 1.927 / 1.709 | 1.069 / 0.842 | 1.803 / 2.030 |
| Native OpenCL, 128×4096×256 | 1.869 / 1.930 | 0.910 / 0.912 | 2.055 / 2.116 |
| Native OpenCL, 256×4096×32 | 0.529 / 0.532 | 0.284 / 0.299 | 1.862 / 1.776 |

The speedup repeats: **1.75–2.26× in CUDA** and **1.78–2.12× in OpenCL**, within
this microbenchmark. The largest ratio between the medians of two identical
controls is 1.087 (8.7% difference). Samples and dispersion are retained; small
differences between APIs or geometries are not turned into firm gains.

Contraction explains a substantial part of the observed GEMM difference between
default paths. The entire application difference is not attributed to FMA:
these times exclude allocation, transfers, and the model's other operations.
Order across geometries was not balanced either, so this iteration does not
select one over another. The [previous tile16 result](cuda-fp64-gemm-results.md)
still does not justify promotion. **The runtime does not change, and no public
mathematical mode is added.**

The resumed work retains the first CUDA round, which finished before the quota
stop. Its idle gate was recorded at 41 °C, but its subsequent telemetry was not.
A first resume attempt exhausted 300 seconds without launching a worker: the
idle GPU was around 44–45 °C, above the old 43 °C threshold. That attempt remains
in `session-resume.json`.

The new session establishes a 45 °C reference and retains the absolute 52 °C
limit, 2 °C tolerance, and maximum 5% utilization in the final three stable
readings, with a minimum ten-second wait. This does not require ten continuous
seconds without activity: the second OpenCL round's gate records 9% GPU
utilization at 4.265 seconds, before the three final accepted readings. Both
OpenCL rounds pass their checks in `session-resume-2.json`; that attempt's final
CUDA gate exhausts 300 seconds with activity and temperature above limits,
without launching its worker. After leaving the GPU idle, `session-resume-3.json`
retains the same thresholds and extends only the maximum wait to 900 seconds.
The last CUDA round passes the gate at 45 °C in 10.516 seconds; its readings and
subsequent state are in that third record. The pause and changed thermal condition
prevent treating the four runs as an identical continuous session; the main
comparison is between interleaved policies within each process and its repetition.

## Precision and limits of FMA

For small random cases, the first round's direct CUDA control has 384 distinct
coordinates: FMA has lower absolute error in 151, higher in 107, and equal in
126. For the 15 sampled coordinates of the three large GEMMs, counts are 7/6/2.
Comparison is against the exact dot product of binary inputs. Controls,
geometries, and rounds are not duplicated in these counts, and this fixed sample
is not interpreted as a statistical guarantee.

Structured cases explain why FMA should not be called an always more accurate
mode. Let `x=1+2^-52`, `r=1+2^-51`, `u=2^-1074`, and `M` be the largest finite
float64. The table describes the CUDA recurrence in increasing K order, with
initial accumulator +0; the indicated terms end the product.

| Terms, in order | Exact mathematical sum | Separate rounding | Sequential FMA |
| --- | --- | --- | --- |
| `-r`, `x*x` | `2^-104` | `+0` | `2^-104` |
| `x*x`, `-r` | `2^-104` | `+0` | `+0` |
| `2^54`, `1`, `-2^54` | `1` | `+0` | `+0` |
| `u`, `u*0.5` | `1.5*u` | `u` | `2*u` |
| `-u*0.5` | `-u/2` | `+0` | `-0` |
| `-M`, `M*2` | `M` | `+Inf` | `M` |
| `M*2`, `-M*2` | `0` | `NaN` | `+Inf` |

FMA avoids intermediate rounding and some overflows, but does not recover
information lost in earlier steps. At the subnormal tie `1.5*u`, both results
have the same absolute error; `2*u` is the result of rounding once to even.
The exact error fraction is retained even if converting it to float64 yields zero.

In `negative-underflow-zero`, CUDA FMA ends at -0. OpenCL adds the incomplete
tile's +0×+0 products and ends at +0, even with FMA. That difference in effective
order is modeled in the oracle; it is not an overwrite failure or proof of
general equivalence between backends.

## Criterion for the next iteration

The priority is to bring the FMA/separate comparison to the complete dense-network
flow in custom CUDA, first with experimental executables and explicit selection.
Native OpenCL already generates FMA on this machine. The comparison must use
the same geometry, inputs, dtype, and algorithm for inference and SGD, including
the existing small, regular, and odd cases. Measure preparation, first call,
resident time, and full time; check prediction, loss, gradients, and parameters.
The decision requires repeated improvement of those flows and an explicit error
budget. This microbenchmark does not demonstrate that gain.

If application results justify it, a public opt-in policy must define its scope
by operation/provider and participate in kernel and plan identity. It must not
enable reassociation, FTZ, or other changes under the FMA name, or reuse an
executable compiled with another policy. Current defaults remain unchanged;
a timing advantage does not make different policies bitwise equivalent.

As an independent direction, the [oscillator client](oscillators-results.md)
justifies measuring dispatch, allocations, and internal waits before optimizing
its prepared executor. That recommendation comes from its application timings
and CUDA retention code, not extrapolation from these GEMMs. The private draft
of that diagnosis is neither integrated nor executed in this iteration.

## Verification and evidence

Reports, sources, and binaries remain in
`.artifacts/iterations-2026-10-22/fp64-contraction/`. `summary.json` collects counts
and medians; all four `round-*.json` retain samples, references, options, and
hashes. `session*.json` records also preserve failed idle attempts and missing
post-run telemetry from the first CUDA round. The initial 60-row check remains
separate, without mixing it into these performance measurements.

**Independent QA: APPROVE.** All 86 targeted CPU tests in
`tests/test_fp64_contraction*.py` pass, as do Ruff check and formatting. The review
reconstructs all 22 input pairs, recalculates 2,191 reference points with explicit
integer rounding and exact rational sums, and reconciles artifacts, counts,
tables, and hashes. The idle description reflects the final three readings,
without claiming ten continuous seconds of inactivity. The verdict and its audit
are in `qa-fma-final.md`, `qa-fma-final-audit.json`, and `qa-fma-final.xml`, alongside
this iteration's directory. The global suite is not run, and the next iteration
is not started.
