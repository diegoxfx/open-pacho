# Binding, CUDA context, and vector products — 2026-09-18

This iteration compares three changes against the final wheel from the previous round.
The largest expected gain is in long custom DOT, which previously accumulated
in a single thread. Binding is a cross-cutting improvement; context management
has a smaller, plan-dependent effect. These are comparisons with the previous
OpenPacho version, not a claim of outperforming CuPy, cuBLAS, or NumPy.

## Implementation

1. **Compiler positional binding.** Parameter names and signature eligibility
   are resolved when constructing the compiled function. If all parameters arrive
   positionally, rebuilding `inspect.BoundArguments` is avoided. Keyword arguments,
   omitted defaults, and keyword-only parameters retain the general path.
   Validation, the LRU, shape/dtype/backend/static keys,
   scalar rebinding, and errors remain. It requires neither `prepare` nor algorithm changes.
   The first capture of a variant does bind the signature; hits avoid that work.
2. **Nested CUDA context and selective scope.** The lock protects activation
   depth. A nested call queries the thread's current context before
   omitting push/pop; if another library changed it, the appropriate context
   is activated/restored. It is also restored on exceptions. The compiler resolves
   `execution_scope_min_steps` once: CUDA requests at least two steps; Vulkan
   retains its behavior. No waits are added and the limit of
   64 pending operations is unchanged. Management follows the
   [Driver API contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
3. **Cooperative CUDA custom GEMV/DOT.** `_cuda_gemv.py` adapts the
   OpenCL algorithm: 128 threads read terms from a row and reduce in shared memory.
   It supports matrix rows and contiguous batches, float32/64, and irregular indices.
   DOT with more than 4096 terms uses partials of 1024 and a GPU reduction.
   The path is selected for a right-hand vector and K >= 256; smaller rows
   retain the previous kernel. CuBLAS retains its explicit provider.

The first experiment enabled cooperation from K=64 and scope on all plans.
Additional preparation did not pay off for several short operations. That
indiscriminate selection was discarded; K=256 and two steps are conservative
local criteria, neither a universal crossover nor device-specific autotuning.

## Method and traceability

Windows, Python 3.12, RTX 5060 Ti, driver 616.64. One backend runs per process,
with 31 samples per scope, rotated/reversed variant order, and two seeds.
Observed idle state is required between processes: at least ten seconds, three consecutive
readings, GPU/memory utilization <=5%, temperature <=52 °C, and within the
relative idle ceiling. Each variant warms up outside the timer for at
least 0.25 seconds and three calls. This does not guarantee thermal steady state.

`benchmarks.three_actions` measures baseline, isolated binding, isolated context,
isolated GEMV, and all changes together where applicable. Each variant has its own plan
so the scope bound during capture is correct. Control selection
is outside the timer. The control loads only the previous methods from a
local wheel, preserving current class/allocation identities and
checking other runtime modules through hashes.

The **resident** scope includes the Python call and synchronization, with inputs
already uploaded. **Complete** adds upload, preparation, and download. First call
is recorded separately. Ridge retains twenty iterations of the same domain
function; it does not represent time to convergence. Solvers are not changed.

Profiles with events and counters are separate instrumented diagnostics:
their durations are inclusive and are neither summed nor subtracted from total time to
deduce a supposed pure Python or GPU cost.

## Measured results

The eight valid runs contain **660 configurations and 492 comparisons
against baseline**, with 31 samples per scope. The following figures aggregate
percentage reductions of per-case and per-round medians without weighting;
they do not represent a universal application mix.

| Isolated change / backend | Less resident time | Less complete time |
| --- | ---: | ---: |
| Binding, OpenCL custom | 8.05% | 2.34% |
| Binding, Vulkan | 4.21% | 2.09% |
| Binding, CUDA/cuBLAS | 5.85% | 1.21% |
| Binding, CUDA custom | 7.96% | 2.25% |
| All changes, CUDA/cuBLAS | 6.32% | 1.32% |
| All changes, CUDA custom | 8.60% | 2.04% |

To see the matrix change without diluting it among operations that do not use GEMV:

| CUDA custom, all changes | Previous resident | Current resident | Previous complete | Current complete |
| --- | ---: | ---: | ---: | ---: |
| DOT 1,048,577, f32 | 15.96–16.69 ms | 0.198–0.201 ms | 12.70–12.75 ms | 1.674–1.686 ms |
| DOT 1,048,577, f64 | 54.89–55.28 ms | 0.188–0.190 ms | 32.96–33.05 ms | 2.923–2.939 ms |
| GEMV 129×8193, f32 | 0.205–0.207 ms | 0.077–0.078 ms | 1.049–1.053 ms | 0.923–0.925 ms |
| GEMV 129×8193, f64 | 0.597–0.598 ms | 0.086–0.101 ms | 1.983–2.046 ms | 1.562–1.601 ms |

Ranges are the two observed medians, not confidence intervals.
Scopes are warmed/measured separately: resident time must not be subtracted from
complete time to estimate transfer; clocks and device state may
differ between scopes. The large improvement ratios correspond to replacing
our previous serial DOT, not to a win over another library.

Isolated scope saves a median **2.41% custom / 2.62% cuBLAS** on
selected multi-step cases (ridge, loss, covariance, and centering). Across the full
aggregate it is marginal: −0.74% custom / +0.26% cuBLAS resident. For ridge
cuBLAS f32, diagnostics reduce push/pop pairs from **243 to 25** per call;
current-context queries and other executor costs remain.

The short K=256/257 boundaries still have mixed results: cooperative reduction
and its preparation can add a few microseconds. The threshold does not
guarantee a gain for every shape. Ridge does not improve uniformly either: in the
second custom f32 round, complete time went from 5.85 to 7.46 ms while
resident time fell from 4.97 to 4.88 ms; f64 resident went from 6.60 to 6.68 ms. These
data remain in the report even though they contradict the aggregate gain.
A ridge f32 follow-up with only baseline/final selection and 51
pairs per scope was attempted, but reached no measurement: memory activity exceeded the idle
gate throughout the allowed 300 seconds. `ridge-followup.json` retains the
telemetry and incomplete state. The specific regression still has no isolated
explanation; it is considered neither demonstrated noise nor a confirmed ridge gain.

## Accuracy and selection

Dtypes, `fmad=false`, `ftz=false`, precise division/square root, and cuBLAS mode
are unchanged. Binding and context retain results exactly equal to the control
in comparisons. GEMV/DOT change summation order: neither bitwise identity
nor a universal numerical improvement is promised.

The float32 129×8193 case with seed 941 exposed an earlier error: the serial
kernel exceeds rtol=5e-5/atol=5e-6 against NumPy and the float64 reference. The new
reduction passes the same threshold. A regression with those inputs was added; the runner
retains failure flags for every historical variant, without widening tolerances
or hiding them. For float64, NumPy float64 is an equal-precision comparison,
not a higher-precision oracle.

All **168 configurations with the final selection** pass the recorded thresholds
against both references. The three recorded failures are the same
previous serial case in baseline/binding/context, not three different problems.
Maximum relative L2 error of selected outputs is 4.36e-7 in f32 and
2.83e-14 in f64. This is not a guarantee for every input.

For GEMV 129×8193 f32, relative error with seed 941 falls from 1.59e-6 to
1.84e-7. There is a counterexample to “parallelizing always improves accuracy”: long
f64 DOT with seed 942 goes from 2.93e-16 to 2.11e-14 against NumPy float64;
both meet tolerance, but the new sum is farther from that reference.

## Limits and next priority

The reduction in context cost is marginal in the aggregate and is not a physical
ceiling. It does not warrant displacing kernels or copies with more headroom. The positional
path also does not accelerate named calls or eliminate per-invocation validation.
Short cases remain dominated by Python/driver; transfers still
limit complete benefit. There is no new pinned memory, overlap, or CUDA graph
support in this round.

The next matrix priority is cuBLAS-specific GEMV/DOT and applications
with reusable transposes; the current provider still uses GEMM for vector
cases. Short rows and cooperative geometry need more measurements
before autotuning. For Vulkan, the present improvement is shared binding; it is not
attributed to a new kernel, nor is executor headroom declared exhausted.

## Verification

The complete suite has **1,250 passing tests and one skip** because a
second OpenCL device is required, with OpenCL/Vulkan/CUDA required and native CLBlast.
Another **89 tests** validate benchmarks. The wheel built from sdist passes
**391 CPU tests, with 860 GPU skips**, loading the packaged runtime and
checking independence of optional imports. Ruff/formatting and Python
3.10 syntax pass. See [verification details](verification.md).

## Reproduction

Local data remain in `.artifacts/three-actions`, including incomplete
attempts. `paired/` was stopped by the baseline numerical comparison;
`paired-v2/` is exploratory and ends with a runner error when attempting to close
OpenCL, which does not expose `close()`. Cleanup now uses `synchronize()` in that case
and saves the report even if cleanup fails; both paths have CPU tests.

`final-paired/` contains the first selected CUDA custom run. The batch
stopped because idle state was not reached before the next backend; the gate was not relaxed.
`remaining/` and `custom-repeat/` complete measurements with a maximum wait of 300
seconds, preserving all idle criteria. A later formatting-only adjustment
in `_cuda_gemv.py` has checked, recorded AST equivalence.

To repeat from a compatible previous local wheel and a new directory:

```powershell
python -m benchmarks.isolated --worker three_actions --backends cuda-custom,cuda-cublas,opencl,vulkan --rounds 2 --repeats 31 --max-temperature 52 --max-wait-seconds 300 --control-wheel .artifacts/three-actions/before.whl --output .artifacts/three-actions/reproduction
```

Idle threshold, samples, error, device, and hashes are retained
in JSON. Automatic backend/UUID association and other vendors/OSes remain
pending. The version remains 0.1.0a1 without publication.
