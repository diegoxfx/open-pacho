# Local verification of 0.1.0a1

## Current stage: float64 contraction — 2026-10-22

The [FMA/separate experiment](fp64-contraction-results.md) completes two CUDA
and two OpenCL rounds: 132 rows, 264 variants, 1,296 samples and 5,971 accepted
output checks. The 44,316 coordinate evaluations include repetitions;
they do not represent independent observations. Rounding policy,
native-control reproduction, poisoned destinations, caches and cleanup are verified.

The 33 runtime files remain identical; CUDA custom retains
`fmad=false` and no public selector is added. Kernel improvements
justify a future dense-network flow comparison, not automatic
promotion. Reports distinguish mathematical accuracy from equivalence,
missing telemetry in the first round and failed idle attempts.

All 86 targeted CPU tests and Ruff checks/formatting on the six new Python
files pass. Independent QA: **APPROVE**, with all 22 inputs reconstructed,
2,191 reference points recalculated using integer rounding and exact sums,
and reconciliation of artifacts, tables, hashes and idle conditions. The
full suite is not run and the wheel is not rebuilt.

## Previous stage: coupled oscillators — 2026-10-22

The [velocity-Verlet client](oscillators-results.md) reuses a compiled step
in a loop without host observations. 40 targeted tests pass: 21 CPU example
tests, eight native float32/64 tests and 11 benchmark tests. Coverage includes second-order
convergence, energy, reversibility, 65 GEMVs per trajectory, rebinding and immutable
inputs. The runtime is unchanged and the full suite is not run.

Fourteen processes complete 156 rows and 3,744 samples, all accepted. All
12 GPU processes pass observed idle gates. Backend accuracy, modal temporal
error, energy and complete costs are separated. At measured sizes, eight-thread
NumPy exceeds GPU paths; plans improve OpenPacho compared with eager. An exact
bottleneck is not inferred from wall times.
Independent QA: **APPROVE**, with 32 CPU tests rerun and verification of
156 cells, physical references, hashes and 12 idle gates; runtime intact.

## Previous stage: CUDA float64 GEMM experiment — 2026-10-22

The [diagnostic and candidate](cuda-fp64-gemm-results.md) do not modify the runtime.
GEMM accounts for 91.2–95.3% of recorded GPU events in four plans; the
shared tile preserves results but does not consistently improve
flows. The decision is not to promote it. Two rounds complete 60 cases, 8,640 samples and
9,360 accepted checks. Additional validation covers 54 launches
with poisoned destinations and nine vector/empty/K=0 paths.

26 targeted CPU tests of the generator and protocol pass. Exact numerical
cases separate equivalence from accuracy. The full suite is not run.
Independent QA: **APPROVE**, with all 26 CPU tests rerun, table/sample
reconciliation, 23 reproduced controls and 172 exact points checked.
The 33 runtime files remain identical to the previous state.

## Previous stage: dense network as a client — 2026-10-22

The [dense-network example](dense-network-results.md) adds inference and one SGD step
with explicit gradients, without runtime changes. 45 targeted tests pass:
24 CPU example tests, eight native tests and 13 benchmark protocol tests. Each
parameter's derivative is verified by finite differences, as is compiled rebinding.

Two rounds compare NumPy 1/8 threads, OpenCL custom/CLBlast, Vulkan, CUDA
custom/cuBLAS and CuPy. They retain 288 accepted rows and 6,912 resident/complete
samples, 24 unsupported Vulkan float64 rows and the 24 initial
CuPy cache failures. Two additional CuPy processes with private caches complete
their comparison. Limitations, variation and preparation/transfer costs
are separated. The full suite is not run and peak memory is not measured.
Independent QA: **APPROVE**, with 37 CPU tests rerun and reconciliation
of all 140 result cells, 23,328 diagnostics and 14 GPU gates.

## Previous stage: extrema preparation — 2026-10-22

The [plan comparison](extrema-planning-results.md) records six processes,
66 case groups and 11,880 samples with two independent controls, two rounds
and observed idle time. All outputs preserve control bits and meet the
reference. Materialized float32 min/max is promoted on OpenCL/CUDA; Vulkan
regressions prevent promotion there. Small gains or unmeasured dtypes are not generalized.

106 targeted tests pass and four are skipped for unsupported CUDA integers.
Coverage includes extrema, producer boundaries, rebinding, array retention,
lowerer extensions and the benchmark protocol; the full suite is not run.
The post-promotion check retains 1,188 additional samples with
forced controls. Independent QA: **APPROVE**, including 45 CPU tests and
reconciliation of measurements, tables and hashes.

## Previous stage: resident extrema — 2026-10-22

The [min/max iteration](extrema-results.md) adds extrema along axes on all
backends and in compile, with preserved dtype and explicit NaN/zero rules.
Softmax/logsumexp remain external examples. 267 targeted tests pass;
four are skipped for unsupported CUDA integers. Independent QA approved.
The full suite
was not run and no acceleration is claimed without measurement.

## Previous stage: GEMM accumulation investigation — 2026-10-22

The [accumulator comparison](gemm-accumulation-results.md) adds experimental
infrastructure without changing the runtime: six isolated processes, 78 case groups,
468 variant results and 4,320 resident samples. Neumaier passes severe
cancellation cases at high cost; blocked variants improve random
cases but retain cancellation losses. No candidate is promoted.
54 CPU tests and Ruff pass; independent QA approved. See protocol, limits
and evidence in the report.

## Previous stage: resident selection — 2026-10-22

The [where iteration](selection-results.md) adds selection with broadcasting,
NumPy 2 scalar promotion and fusion/map-reduce on every backend.
44 new tests and 53 targeted regressions pass; one int64 test is skipped
on CUDA because of its explicit scope. Independent QA approved. The full
suite was not run and no acceleration is claimed without measurement.

## Previous stage: stable CUDA norms — 2026-10-22

The [norm iteration](cuda-norm-results.md) shares the scaled algorithm
with CUDA, in eager and compile, and enables external solvers without
changing their recurrences or tolerances.

- **257 targeted tests pass**, with OpenCL/Vulkan/CUDA required;
  51 PyOpenCL warnings. Ruff checks and formatting pass on all eight modified Python
  files. The full suite was not run and the wheel was not rebuilt.
- **52 Decimal80 norm/solver diagnostics** across CUDA custom and cuBLAS,
  without exceptions; six forward-error alerts on Hilbert systems are
  retained. Extreme norms and ordinary systems pass.
- Windows, Python 3.12.14, RTX 5060 Ti, driver 617.14. Integrated tests use
  `--capture=sys`; file-descriptor capture reproduced a native
  NVRTC `0xc0000008` exception. Vulkan uses an accessible TEMP and explicit GLSL compiler.

Artifacts in `.artifacts/iterations-2026-10-22`; scope and limits are detailed in
the linked report. Subsequent stages are historical snapshots.

## Previous stage: three iterations — 2026-09-24

The [September round](three-iterations-results.md) adds custom rectangular GEMM
without copies, measurement-selected cuBLAS batched DOT/GEMV and private
transpose flags compiled with CLBlast. Vulkan retains its kernel after
two isolated confirmations unfavorable to the new tiles.

- **1,652 tests pass, one skipped**, running `tests` and `benchmarks` with
  OpenCL/Vulkan/CUDA required and native CLBlast; 277 PyOpenCL warnings. The skipped test
  requires a second physical OpenCL device.
- **38,880 public samples** across candidates and final selection, plus 2,352
  Vulkan confirmation samples with preallocated output. Twelve sequential
  processes, without subagent GPU tests; observed idle gates,
  10.5–229.2 s, without relaxing thresholds. Two controls per case, balanced order.
- All 84 rectangular GEMM groups preserve bits; seven remain outside tolerance
  equally in controls and variant. One Vulkan group also retains an
  identical violation in all modes. CLBlast layouts and vector cuBLAS pass
  their tolerances. Completion, performance and numerical validity remain separate.
- Wheel from sdist: **525 pass, 909 skipped** in a CPU environment without GPU bindings.
  Import from the wheel, smoke test with optional imports blocked, independent
  extras, NumPy as the only base dependency and identical wheel sources.
- Global Ruff checks/formatting and `pip check` pass. The CPU workflow adds
  benchmark infrastructure tests; remote execution remains pending.

Artifacts in `.artifacts/next-three`, with complete reports, samples,
telemetry, historical controls and hashes. Wheel SHA256:
`83b7c21a716da655aaf4ed7f6b66af703ba259e5a9213924937cec31da94a270`.
The version remains `0.1.0a1`; it is not published. See the
[criteria for the first stable release](release-readiness.md).

## Previous stage: padding reassessment — 2026-09-19

The [reassessment](padding-reassessment.md) retains explicit padding,
disabled by default: it gains on some rectangular shapes
but worsens most. The runtime and its numerical policy are unchanged.

- **30,960 public samples**, 172 shape/backend/seed groups, 22 distinct
  shapes and thirteen complete processes. OpenCL/Vulkan/CUDA in the main suite;
  OpenCL/CUDA in the nearby-size follow-up. Observed idle time before
  each process, with waits of 10.4–62.2 seconds; no overlapping GPU processes.
- Padding on/off retains bitwise identical results in all
  172 groups. Three CUDA groups in the main suite exceed tolerance
  by one element, equally in both controls and with padding. Failures
  and the initial incomplete attempt are retained; completing the benchmark is not
  presented as passing accuracy. OpenCL/Vulkan and the follow-up pass the threshold.
- **173 benchmark tests pass**. New tests cover order balance,
  independent plans, complete-path transfers, failure persistence,
  rebinding, numerical validation and CLI. Float32 literals in the experimental
  variant generator are corrected without changing package kernels.
- Ruff checks and formatting of `src`, `tests`, `benchmarks` and `examples` pass;
  Python 3.10 syntax checked in **128 Python files**.
- The **29 Python runtime files** match the previous stage's wheel
  byte for byte, and their hashes match across all thirteen measured processes.
  The full runtime suite is not repeated; its last run was the
  1,341 passed tests documented below.

Artifacts in `.artifacts/padding-decision`, with samples, errors, telemetry,
worker sources, report selection and provenance controls.
The reproducible benchmark is in `benchmarks.padding_decision`. This round's sdist
includes the updated benchmark and documentation; there is no new publication.
Local scope is retained: Windows, RTX 5060 Ti, driver 616.64.

## Previous stage: kernel and copy inspection — 2026-09-19

The [inspection](kernel-inspection-results.md) fixes accidental
float64 promotion in float32 GEMM loads, caches copy preparation without retaining
inputs and counts padding submissions to select CUDA scope.
The default math policy is unchanged.

- Runtime with OpenCL/Vulkan/CUDA required and native CLBlast: **1,341 pass,
  one skipped** because another OpenCL device is needed; 273 PyOpenCL warnings.
  The initial selection of affected tests gave **115 pass**.
- **141 benchmark tests pass**, including Vulkan capture/ABI, PTX parser,
  source controls, wheel isolation and restoration after errors.
- Ruff and formatting pass; Python 3.10 syntax checked in **126 Python files**.
  Actual execution on Python 3.12 / Windows / RTX 5060 Ti / driver 616.64.
- Compiler diagnostics: **72 configurations, 1,512 wall-time samples** and
  360 separate GPU diagnostics. Copy review: **144 configurations,
  4,464 public samples** including all copies; 36 groups of bitwise
  identical results. Two rounds per suite and observed idle time without relaxing controls.
  Each report retains numerical errors and clock scope.
- Wheel built from sdist: **449 pass, 893 skipped** in a CPU environment,
  importing the wheel and checking that it loads no GPU/example dependencies.
  NumPy remains the only base dependency. Wheel runtime identical to sources;
  sdist documentation and local links checked.

Artifacts in `.artifacts/kernel-inspection`. Wheel SHA256:
`a577d80ffde9c8be17ab1c2ca9c51278aaf34412e6be84bd0ecd24450fa66e3b`.
Inspection tools are not part of the installable package. Vulkan ISA was not
obtained: the driver exposed no internal representations. Validation on
other vendors/OSs or remote CI execution is not claimed. Version 0.1.0a1,
unpublished. Padded `mean` is documented as future work.

## Previous stage: optional padding — 2026-09-18

[Optional padding](padding.md) is added to `compile`/its decorator and `matmul`,
disabled by default. Prepared executors inherit the policy. Custom
float32 GEMM on OpenCL, Vulkan and CUDA may use buffers aligned to their
tiles; other cases retain their path and provider.
- Runtime with OpenCL/Vulkan/CUDA required and native CLBlast: **1,323 pass,
  one skipped** because another OpenCL device is required; 273 PyOpenCL compilation
  warnings. Uses `PYOPENCL_NO_CACHE=1` and the local Vulkan compiler.
- **96 benchmark tests pass**. New regressions cover the flag
  contract, capability selection, BLAS rejection, input rebinding/lifetime,
  independent results, memory/index limits, broadcast batches,
  logical dimensions before reductions and bitwise copies of subnormals,
  signed zeros, NaN and infinities. Preparation errors propagate.
- [Padding benchmark](padding-results.md): six processes, two rounds,
  72 configurations, **6,048 public samples** and 756 diagnostic samples with
  prepacked inputs. All idle controls passed without relaxation;
  actual pauses of 10.5–156.4 seconds. All 36 pairs produce
  exactly equal results with padding on/off within the same backend.
  Tests retain errors against a float64 reference, not just times.
- Ruff and formatting pass. Python 3.10 syntax checked in **120 Python files**;
  actual execution on Python 3.12, Windows, RTX 5060 Ti and driver 616.64.
- Wheel built from sdist: **431 pass, 893 skipped** in the CPU environment
  without GPU runtimes, importing from the wheel. NumPy is verified to remain
  the only base dependency and GPU/example extras are not imported when using
  CPU operations. Wheel runtime matches sources byte for byte.
- Sdist documentation and local links checked against the workspace.

Artifacts in `.artifacts/padding`: samples, telemetry, errors, plan
metadata, test logs/XML, scripts and distributions. Wheel SHA256:
`77f55a8f31d4bd915be9cc763be7f6143fbee3b7937f53815d453da82cd9a2f9`.
Figures do not validate other vendors/OSs or remote CI. Version 0.1.0a1,
unpublished.

## Previous stage: binding, context and CUDA GEMV — 2026-09-18

The [three-optimization round](three-actions-results.md) adds shared positional
binding, nested CUDA activation with selective scope and cooperative custom
GEMV/DOT. It does not change domain algorithms, extras or math flags.

- Runtime with OpenCL/Vulkan/CUDA required and native CLBlast: **1,250 pass,
  one skipped** because a second OpenCL device is required; 264 PyOpenCL compilation
  warnings. `PYOPENCL_NO_CACHE=1` and the local Vulkan compiler are configured.
- **89 benchmark tests pass**, including control selection, separate plans
  per variant and cleanup of backends without `close()` or with cleanup failure.
- New regressions cover binding/errors/defaults/statics and input lifetimes,
  foreign context, nesting, exceptions, concurrency, 255/256/257 boundaries,
  rectangular matrices and batches, long DOT, cancellation and nonfinite values.
  The 129×8193 input where the previous serial kernel failed tolerance is retained.
- **660 main configurations, 492 controlled comparisons**, 31 samples
  per scope and eight valid processes. All 168 final-selection cases pass
  recorded tolerances; numerical failures from previous controls remain
  visible. Observed idle time between processes, retaining one timeout without skipping the gate.
- Ruff and formatting pass. Python 3.10 syntax checked in 113 Python files;
  actual execution on Python 3.12 / Windows / RTX 5060 Ti, driver 616.64.
- Wheel built from sdist: **391 pass, 860 skipped** in a CPU environment,
  importing the runtime from the wheel and checking the boundary without optional
  GPU/example imports. NumPy remains the only base dependency.
- Wheel runtime identical to sources; sdist documentation checked
  against the workspace and local links checked. The additional ridge follow-up
  could not measure due to a 300-second idle timeout; it retains telemetry
  and does not replace mixed results from valid batches.

Artifacts in `.artifacts/three-actions`: per-variant results, samples,
telemetry, errors, diagnostics, test logs/XML and reproduction scripts.
Exploratory and failed trials are retained separately from final selection.
Other vendors/OSs and remote CI remain pending. Version 0.1.0a1, unpublished.

## Previous stage: CUDA launch arguments — 2026-09-13

The [CUDA dispatch round](cuda-dispatch-results.md) reuses per-kernel
host storage without modifying shaders, plans, algorithms, contexts or numerical
policy. Every call rebinds values under the existing lock.

- Runtime suite with OpenCL/Vulkan/CUDA required and native CLBlast:
  **1,200 pass, one skipped** because two OpenCL devices are needed; 264 PyOpenCL
  compilation warnings. `PYOPENCL_NO_CACHE=1` avoids the nonwritable persistent
  cache. Benchmark infrastructure tests add **80 passes**.
- Five new CPU controls cover ABI sizes/bytes, zero sign, rebinding
  without retaining inputs, incorrect types/count, lock, failed launch
  and retry, and argument release when evicting a completed module.
  A GPU regression checks two pending bool fills with different values.
- The initial test found an `np.bool_`/`ctypes.c_ubyte` incompatibility;
  the final implementation uses `c_bool`, with a verified one-byte ABI. The first
  broad RNG batch had OpenCL failures from missing cache configuration and
  Vulkan skips from a missing compiler path; the final full suite does
  configure both and passes. Initial logs are retained.
- **236 main configurations**, 31 samples per scope: 144 CUDA and 92
  NumPy/CuPy references, in eight valid sequential processes and two rounds.
  The first attempt failed in the runner before measuring; it is not used as a result.
- The cuBLAS f32 ridge follow-up adds **six configurations**, three blocks
  of 51 pairs per scope. It does not reproduce the initial complete-path loss; both
  sets are retained. Numerical errors and exact control results are
  recorded. The bool ABI was fixed after the exclusively floating-point
  controls; it does not change their paths. The follow-up already uses the final fix.
- Wheel built from sdist: **374 pass, 827 skipped** in a CPU environment,
  importing from the wheel and blocking GPU/example imports to verify
  independence. Extras and base dependencies are unchanged.
- Ruff/formatting, Python 3.10 syntax and local links checked. Actual runtime
  on Python 3.12 / Windows / RTX 5060 Ti. Other devices/OSs and remote CI pending.

Artifacts in `.artifacts/cuda-dispatch`: tests.xml/log, benchmark-tests.log,
wheel-tests.log, summary.csv, paired-gains.csv, profiles, JSON and telemetry.
The version remains 0.1.0a1, unpublished.

## Previous stage: CUDA fusion — 2026-09-13

The [CUDA fusion round](cuda-fusion-results.md) implements elementwise
and map-reduce lowering through the backend factory. It preserves primitives,
numerical policy and comparators' domain functions.

- Full suite with OpenCL/Vulkan/CUDA required and native CLBlast:
  **1,194 pass, one skipped** because two OpenCL devices are required; 264 PyOpenCL
  compilation warnings. Uses `PYOPENCL_NO_CACHE=1`.
- **127 focused CUDA tests** passed before the full suite. They cover
  both providers, tails/axes/empty inputs, broadcasting, scalar rebinding,
  shared outputs, buffer lifetimes, concurrency, partial limits,
  promotion/narrowing, no hidden FMA, cancellation and subnormals.
- The `examples/compiled_numpy.py --backend cuda` example validates row
  energy and shows a map-reduce plan with cache reuse.
- **362 measured configurations**, 21 samples per scope: 144 CUDA controls
  and 218 references. Four CUDA and eleven reference processes completed;
  the second one-thread NumPy round did not run due to idle timeout. All
  GPU comparators have two rounds. Samples/errors/failures are retained.
- A custom float32 covariance follow-up adds **six configurations**,
  31 alternating pairs per scope in three blocks. It did not reproduce the
  initial loss; both sets remain separate and variability is documented.
- Wheel built from sdist: **369 pass, 826 skipped** in a CPU environment.
  Import and computation/RNG from the wheel work with GPU and
  example imports blocked. NumPy remains the only base dependency, with independent extras.
- Ruff/formatting and Python 3.10 syntax pass; actual execution on Python 3.12,
  Windows and RTX 5060 Ti. A second vendor and remote CI were not verified.

Artifacts: `.artifacts/cuda-fusion`, including tests.xml/log, wheel-tests.log,
summary.csv, original controls and covariance follow-up. The version remains
0.1.0a1, unpublished. Earlier reports retain their snapshots.

## Previous stage: Vulkan routing and completion — 2026-09-13

The [routing/completion round](routing-completion-results.md) separates RNG
adapters, binds factories/scopes when preparing plans and adopts a reusable Vulkan fence,
with a fix for pool reentrancy from finalizers.

- Final suite with OpenCL/Vulkan/CUDA required and native CLBlast: **1,142 pass,
  one skipped** because two OpenCL devices are needed; 264 PyOpenCL compilation
  warnings. `PYOPENCL_NO_CACHE=1` avoids the environment's nonwritable persistent cache.
- The first full attempt uncovered a test counter that did not accept
  empty submits. It was corrected to respect count and verify one marker
  on reading; 136 focused tests and then the final full suite pass.
- Regressions: name-independent factories, errors without fallback,
  bound scopes and cleanup, rebound RNG scalars, sequences/counters,
  reentrant pool, pending resources and completion failures without recycling.
- Vulkan: **144 configurations** in three valid processes, 31 samples per mode;
  separate diagnostics, complete times, error and convergence. The first batch,
  incomplete due to the pool bug, is excluded and retained as evidence.
- RNG: **264 valid configurations and 12 capability exclusions** in six
  processes, 31 samples; exact before/after equality. Two additional interleaved
  controls investigate small OpenCL/CUDA latencies. Separate CPU preparation
  control; it is not equivalent to GPU dispatch time.
- Wheel built from sdist: **369 pass, 774 skipped** in a CPU environment.
  Import from the wheel and computation/RNG with GPU/example imports blocked
  are verified. NumPy remains the only base dependency and extras are independent.
- Ruff/formatting, Python 3.10 syntax and local links pass. Actual runtime on
  Python 3.12 / Windows / RTX 5060 Ti. A second vendor and remote CI were not verified.

Artifacts: `.artifacts/routing-completion`, including tests-final.xml/log,
wheel-tests.log, CSV with all measurements and original/derived JSON files. The
idle/download reference metadata correction preserves hash and original;
it does not change times or errors. The version remains 0.1.0a1, unpublished.

## Previous stage: read staging and RNG — 2026-09-13

The [staging/RNG round](staging-random-results.md) retains Vulkan readback
mappings, keeping previous uploads. It adds resident Philox with a limited
API and capabilities per distribution/dtype.

- Full suite with OpenCL/Vulkan/CUDA required and native CLBlast:
  **1,202 pass, one skipped** because two OpenCL devices are needed, 264 PyOpenCL
  compilation warnings. This run precedes limiting persistence to readback.
- After that selection and RNG parameter adjustment: **72 focused tests**
  pass (staging and RNG on all four backends) and **198 Vulkan tests** pass.
  Coverage includes mappings, pool, commands, pending resources, fusion and operations.
  The CPU regression for negative scales rounding to zero and enormous integers
  also passes; it does not consume invalid counters.
- **30 CPU tests** of RNG metrics/validation/runner pass after correcting
  the accuracy label for same-dtype float32 references.
- **64 final configurations** compare readback with methods from the previous wheel,
  31 samples in two rounds. Another 64 measured the general-persistence variant
  rejected as the default. Matrix kernels/solvers are unchanged.
- RNG: **228 valid configurations, 12 capability exclusions** in ten
  processes; another final Vulkan control adds 36 valid cases and 12 exclusions.
  Error, moments, samples and observed idle time (up to 31.1 s) are retained.
- Wheel from sdist: **354 pass, 772 skipped** in a CPU environment. Import
  verified within the wheel and computation/RNG with GPU/example modules blocked.
  Separate extras, NumPy as the only base dependency; examples/benchmarks outside
  the wheel. The version is unchanged and unpublished.
- Ruff/formatting, Python 3.10 syntax and local links verified. Actual execution
  on Python 3.12 / Windows / RTX 5060 Ti. No second vendor or remote CI.

Artifacts in `.artifacts/staging-random`. Derived `corrected-random` JSON files
correct a same-precision reference label and preserve hash/original;
they do not alter times or numerical errors. The first suite without CLBlast
configuration remains as a historical log; the subsequent required run does include it.

## Previous stage: stability and GEMM guards — 2026-09-13

The [kernel round](kernel-stability-results.md) separates completion, observed
idle time and warmup; it compares ten variants and promotes only
OpenCL guard specialization on full tiles.

- **1,122 tests pass, one skipped**, running `tests` and `benchmarks`, with
  OpenCL/Vulkan/CUDA required and CLBlast available. The skip requires two
  OpenCL devices; 256 PyOpenCL compilation warnings were emitted.
- **300 kernel configurations** (three backends, two rounds), 21 samples,
  with correct accuracy and output sentinels. Observed idle time 10.4–16.6 s,
  accepted thermal windows of 40–42 °C. This does not imply constant clocks.
- **88 public configurations** of OpenCL GEMM controls, same instance and
  context, without errors and with equal numerical metrics between variants.
- **42 adversarial OpenCL cases**, zero exceptions and the same nine previous
  alerts. Other providers were not remeasured for adversarial accuracy;
  their default shaders were preserved.
- Wheel from sdist: **327 pass, 727 skipped** in the CPU environment. Import
  checked within the wheel, independent extras and examples/solvers outside
  the package. Only runtime change: `_gemm.py`; saved shaders from all
  12 controls remain identical to the previous wheel.
- CPU protocol tests: timeout, activity/temperature, sensor loss,
  identity, process order, worker failure, warmup and control restoration
  on exceptions. Their times are not used as GPU benchmarks.
  After the final telemetry-selector propagation adjustment to the worker,
  **25 focused tests** pass; the GPU suite is not repeated for that CLI change.
- Ruff and formatting pass (88 files). No dependency or public API changes;
  artifacts in `.artifacts/kernel-stability`. Other hardware/OSs and remote CI pending.

## Previous stage: shared execution — 2026-09-13

Capability-based Vulkan push descriptors and sets fallback, OpenCL metadata cache and
bounded indices, shared float32 tiles and CUDA batched matmul.
See [results, controls and gaps](backend-sharing-results.md).

- Full suite: **1,084 pass, one skipped**, with all three backends required
  and CLBlast available. The skip requires two OpenCL devices. After the
  final nonbatched-path adjustment and new coverage, **75 CUDA tests pass**.
  The earlier full run emitted 252 PyOpenCL compiler warnings.
- Final wheel built from sdist: **326 pass, 719 skipped** in the CPU
  environment, importing directly from the wheel without replacing its historical installation.
  NumPy is the only base dependency, extras are independent and examples,
  benchmarks and solvers remain outside the wheel; the sdist retains examples.
- Vulkan: two controls of 60 configurations and an additional five-case
  control on the same VkDevice. In-flight resources, limit fallback, outputs
  and convergence are validated. Time improvement is modest/mixed.
- OpenCL/CUDA: two rounds per provider with previous/current functions on
  the same context; 21 samples. Batches absent in the historical wheel are
  declared unsupported. The control detected and enabled removal of an
  initial CUDA penalty on nonbatched operations.
- **144 configurations** in isolated-provider processes and another **144** with
  interleaved APIs. All valid, without missing backends. Interleaving changes
  some times substantially; it is not used as a physical ceiling or pure abstraction cost.
- **104 irregular applied configurations**, seven samples, without exceptions
  or absences. The first attempt skipped CuPy due to its cache outside the workspace;
  the final run uses a local CUPY_CACHE_DIR.
- **158 adversarial Decimal80 configurations**, zero exceptions and the same
  **40 alerts** as the initial CUDA round; 52 CUDA norm/solver cases explicitly
  excluded. Numerical alerts remain pending.
- Solver AST identical to the previous sdist; applications change only
  CUDA-scope text in metadata. Ruff and formatting pass (81 files),
  Python 3.10 syntax analyzed; actual execution on Python 3.12.

Local evidence corresponds to RTX 5060 Ti, Windows 11 and driver 616.64. Other
hardware/OSs and remote CI remain pending. Artifacts, diagnostic scripts and
provenance in `.artifacts/backend-sharing`.

## Previous stage: Vulkan command boundary — 2026-09-12

The per-device CFFI table reduces conversions in frequent calls without
changing recurrences, tolerances, check frequency, measured workload kernels
or command counts. It includes separate host/GPU timing diagnostics
and an independent fix for float32 power with negative bases.
See [measurements and limits](vulkan-dispatch-results.md).

- **1,045 tests pass, one skipped**, with OpenCL/Vulkan/CUDA required and
  native CLBlast available. The skip requires two OpenCL devices. Includes
  regressions for ABI/VkResult, per-device isolation, in-flight resources,
  power and instrumentation cleanup. 251 PyOpenCL compilation warnings
  were emitted, without test failures.
- **324 pass, 677 skipped** loading the wheel directly in the CPU environment
  without GPU runtimes. Origin within the wheel, import/computation
  with GPU modules and examples blocked, and CUDA-independent extras were verified.
- Wheel built from sdist: excludes examples, benchmarks and solvers; the sdist
  retains examples. NumPy remains the only base dependency. No binary
  dependency was added and the historical CPU installation was not replaced.
- **Two controls of 60 configurations**, 21 samples per mode: interleaved
  wrapped/direct paths, 30 float32/64 cases per process. No errors; same
  outputs, numerical metrics, iterations, command counts and copied bytes.
  Instrumented diagnostics outside timed samples.
- **72 irregular applied configurations**, seven samples: NumPy, OpenCL
  custom/CLBlast, Vulkan and CuPy. No errors or missing backends. This does not replace
  the paired control for causally attributing an improvement to the executor.
- **42 adversarial Vulkan accuracy configurations**, three samples: ten alerts,
  zero exceptions. The alert set matches the Vulkan subset of
  the previous CUDA round. Diagnostic exit code 1 flags those findings; it does not
  mean cancellation or subnormal problems have been resolved.
- Ruff and formatting pass (78 files). Python 3.10 syntax analyzed; actual
  execution on Python 3.12. Other vendors/OSs and remote CI remain pending.

Artifacts: `.artifacts/vulkan-dispatch`, with `final-1.json`, `final-2.json`,
`comparison.csv`, `accuracy.json`, `applications.json`, logs, distributions and
manifest. Runtime and runner sources match hashes recorded
in both final controls. This is not yet a C/C++ executor or a measurement of
the physical limit of Vulkan or PCIe.

## Previous stage: application separation — 2026-09-12

Solvers and prepared systems moved to `examples/solvers.py`, outside
`src/openpacho` and the installable API/wheel. Solver capability was removed from
the CUDA runtime. `get_backend(array)` allows examples and clients to recover
the existing instance through the public API without copying or reconfiguring their data.
See [agreed direction](architecture-evolution.md) and [examples](../examples/README.md).

- **1,024 tests pass, one skipped**, with OpenCL/Vulkan/CUDA required and
  native CLBlast available. The skip still requires two OpenCL devices.
- **316 pass, 672 skipped** when loading the wheel from a CPU environment without GPU
  runtimes; the import path within the wheel was checked. It was loaded directly
  without replacing the CPU environment's historical installation.
- Wheel built from sdist: excludes `openpacho.solvers`, examples and benchmarks.
  The sdist includes examples. Metadata: NumPy as the only base dependency;
  OpenCL/Vulkan extras independent of CUDA. CPU import and computation with
  GPU modules and `examples` blocked; no new dependencies were downloaded.
- AST comparison against the previous CUDA wheel: all thirteen computation,
  preparation and entry functions retain their structure except for backend access through
  the public API. Recurrences, tolerances and convergence checks were unchanged.
- Ruff and formatting pass (73 files). CPU examples executed. After closing the
  documentation/metadata audit, **28 additional focused tests** pass.
- The first OpenCL run encountered a nonwritable SQLite cache outside the
  workspace. It was repeated with `PYOPENCL_NO_CACHE=1`; this is test configuration,
  not a runtime modification or performance measurement.
- A wheel subprocess test initially loaded the environment's historical package;
  it was corrected to use the same package origin as its parent process.
- Performance and adversarial accuracy benchmarks were not repeated for this separation.
  Subsequent results correspond to their snapshots; no new speedups are announced.
  Other hardware/OSs and remote CI remain pending.

Artifacts: `.artifacts/architecture-scope`, with logs, wheel/sdist and source
manifest. Math modes, autotuning, layouts and the native executor described in
the architectural direction remain pending designs.

## Previous stage: capability-based Vulkan and initial CUDA — 2026-09-12

The CUDA implementation and new Vulkan reuse policy are documented
in [current results](cuda-initial-results.md) and [CUDA scope](cuda.md).
OpenCL/Vulkan remain independent options. CUDA does not yet include
fusion, stable norms, solvers or batched matmul.

- **1,019 tests pass, one skipped**, with OpenCL/Vulkan/CUDA required and
  native CLBlast. The skipped test requires two OpenCL devices.
- Wheel from sdist installed without GPU dependencies: **311 pass, 672 skipped**.
  Import from site-packages; CUDA runtime also tested without CuPy.
- **208 applied configurations**, seven samples per mode, zero errors or
  absences. CUDA distinguishes an unfused plan from fused lowering.
- **158 adversarial accuracy configurations**, zero exceptions, 40 numerical
  alerts. 52 norm/solver cases outside initial CUDA scope are recorded.
- Alternating Vulkan control in four processes: 1,432 allocations/releases down to
  zero, same dispatches/errors; retained cache increases from 1 to 3 MiB.
- Ruff and formatting pass. Python 3.10 syntax analyzed, Python 3.12.14 runtime.
  Wheel/sdist: `.artifacts/dist-cuda-initial`. Other hardware/OSs and remote CI pending.

## Previous stage: solvers, applications and CUDA preparation — 2026-09-08

- **965 pass, 1 skipped** in the full runtime and adapter suite,
  including 144 irregular-shape tests and four Vulkan pool regressions. The skipped test requires two OpenCL devices.
- Final wheel built from sdist and installed without PyOpenCL/Vulkan: **294
  pass, 635 skipped** for missing optional backends. Import from
  site-packages, outside `src`, verified. Distributions in `.artifacts/dist-next-applications`.
- **63 solver configurations**, seven samples per mode; zero execution
  errors or missing backends. Data, tolerances and iterations preserved.
- **126 adversarial OpenPacho GPU cases**, three samples: 28 indicators, without
  exceptions/absences. These are the same indicators as the previous subset.
- Lowering separated by backend, Vulkan groups in one command buffer and solver
  plans between observations. CUDA is not yet an available backend.
- **144 final applied configurations** of covariance/ridge (72 aligned
  and 72 irregular), seven samples per mode, without errors or missing backends;
  preparation and accuracy visible. The pool control isolates its policy through
  allocation measurements and numerical errors.
- [Current results](solver-applications-results.md),
  [applied suite](application-measurement-plan.md) and [CUDA path](cuda-backend-plan.md).

## Previous stage: Vulkan, GEMM and stability — 2026-09-07

- **809 pass, 1 skipped**, with OpenCL/Vulkan required and native CLBlast.
  The skipped test requires two OpenCL devices.
- Wheel built from sdist, CPU installation without GPU backends and import
  from site-packages: **249 pass, 527 skipped**.
- **319 performance configurations and 48 preparation configurations**, seven samples
  per mode, without validation errors or missing backends.
- **264 accuracy configurations**, three samples: 68 diagnostic indicators,
  zero exceptions. Includes numerical limitations of references; they are not
  automatically 68 bugs and are not hidden under the label of successful execution.
- Ruff and formatting pass. Distributions: `.artifacts/dist-numerics-vulkan`.
- [Current results and gaps](numerics-vulkan-results.md),
  [accuracy](numerical-accuracy.md) and [CLBlast investigation](clblast-techniques.md).
  AMD/Intel and remote CI remain unvalidated in this session.

## Previous stage: map-reduce, preparation and batched BLAS — 2026-09-06

- **625 pass, 1 skipped**, with OpenCL/Vulkan required and native CLBlast.
  The skipped test needs two OpenCL devices.
- Wheel from sdist and installed without GPU dependencies: **208 pass, 401 skipped**.
- **319 comparative configurations and 48 preparation configurations**: seven samples per
  mode, without validation errors. Additional interleaved controls for tiling,
  batching and solver bindings; before/after profiles with code origin.
- Ruff and formatting pass. Wheel/sdist: `.artifacts/dist-remaining-suite`.
- [Implementations, mixed results and remaining gaps](remaining-suite-results.md).
  AMD/Intel hardware is not validated and the norm for extreme values is not yet resolved.

## Previous stage: Vulkan execution and staging — 2026-09-06

- CPU/OpenCL/Vulkan suite with GPU backends required: **530 pass, 1 skipped**
  because two OpenCL devices are required. Includes 118 Vulkan tests.
- Wheel built from sdist and installed without GPU dependencies: **194 pass,
  320 skipped** for missing optional backends.
- **319 configurations** repeated, seven rounds per mode, without validation
  errors. Vulkan transfer controls with eleven samples were added.
- Ruff and formatting pass. Distribution artifacts in `.artifacts/dist-vulkan-execution`.
- See [changes, results and limits](vulkan-execution-results.md). Hardware/OS is
  a secondary track; the implemented path is portable and remains validated only on
  the local NVIDIA GPU and Windows.

Subsequent stages retain their historical results.

## Previous stage: gaps and optimization — 2026-09-06

- **518 tests pass, 1 skipped**, with OpenCL and Vulkan required, native
  CLBlast and benchmark adapters. The skipped test requires two OpenCL devices.
- **37 additional tests pass** selecting CLBlast in the matrix suite:
  batch broadcasting, irregular products and CG residual.
- Wheel built from sdist and installed without GPU dependencies: **193 pass,
  309 skipped** for the expected absence of optional backends.
- Ruff and formatting pass. Wheel/sdist in `.artifacts/dist-gaps`.
- **319 performance configurations**, seven rounds per mode, without
  validation errors; also separate profiles, OpenCL events, transfer
  controls and sensitivity to execution order.
- CLBlast GEMV is tested with recycled NaN/Inf destinations; solvers retain
  their check frequency. The preexisting numerical norm-underflow gap
  was reproduced and documented separately.

See [gaps, results and limits](gaps-and-optimization.md). AMD/Intel and the remote
CI matrix remain pending. Subsequent sections are historical snapshots.

## First implementation

Date: 2026-09-06. Windows 11, Python 3.12.14, NumPy 2.5.3,
PyOpenCL 2026.1.4. OpenCL device: NVIDIA GeForce RTX 5060 Ti,
driver 616.64, double precision available.

| Verification | Result |
| --- | --- |
| Full suite with OpenCL required | **140 tests pass; 1 skipped** because it requires two physical OpenCL devices. |
| Clean installation from wheel, without PyOpenCL | **73 tests pass; 68 skipped** for the expected absence of OpenCL. Import was verified to come from the installed package, not `src`. |
| Migration example with NumPy and GPU | Correct; distance result `[2.2360678, 7.071068]`. |
| GPU conjugate-gradient example | Convergence in 2 iterations; solution `[0.09090909, 0.6363636]`, measured residual 0. |
| Data residency | Tests prevent downloads during compatible arithmetic; solvers download only scalars. |
| Build | Wheel and sdist generated; wheel built from sdist. Their contents were inspected. |
| Editable installation | Correct in `.venv`, with development and OpenCL dependencies. |
| Ruff | Lint and formatting pass. |
| Dependencies | `pip check` without incompatibilities. |
| Python 3.10 syntax | Successful parsing; this does not replace running tests on Python 3.10. |

The NVIDIA driver compiler emits a warning about the `noinline` attribute on
successfully compiled kernels. The suite does not suppress those warnings.
`PYOPENCL_NO_CACHE=1` was used during most tests due to runtime-environment
restrictions; execution with the normal cache enabled was also verified.

## Illustrative measurement

256×256 matrices, float32, seed 47, 5 repetitions, median wall time.
OpenCL time includes Python dispatch and synchronization, with already resident inputs;
it excludes initial compilation and transfers. All three cases were validated against NumPy.

| Operation | Resident OpenCL | Direct NumPy |
| --- | ---: | ---: |
| Elementwise sum | 0.137 ms | 0.0155 ms |
| Matrix product | 0.129 ms | 0.191 ms |
| Sum reduction | 0.136 ms | 0.0242 ms |

Initial transfer of both inputs took approximately 0.154 ms. The first
execution of each operation, including compilation, took between 61 and 112 ms. These
results from one local run imply no universal advantage and do not reproduce
the paper's experiments. Local JSON and JUnit files were retained in
`.artifacts`, excluded only from this checkout.

## Still to be checked

- AMD and Intel GPUs, a device without fp64 and systems other than Windows.
- Remote execution of the CI matrix and PoCL: the workflow is prepared, not run here.
- Sustained performance, larger sizes and adversarial numerical conditions.
- Compatibility of complete NumPy applications outside the documented subset.
- Vulkan is verified on the local GPU; other vendors remain. There is no PyPI publication.


## Compiler and Vulkan — 2026-09-06

After incorporating compile as a function/decorator, cached plans and Vulkan:

- CPU/OpenCL/Vulkan suite and benchmark adapters: **318 passed, 1 skipped**.
  OpenCL and Vulkan required, native CLBlast enabled. The skipped test requires
  two OpenCL devices; this machine exposes one.
- 65 specific Vulkan tests, including public API, capture and three float32/64
  solvers; 42 CPU/OpenCL compiler cases (including 0-D scalars).
- Wheel installed without GPU dependencies: **122 passed, 186 skipped**.
  The `examples/compiled_numpy.py` example validated on CPU, OpenCL and Vulkan; Vulkan
  discovery CLI verified.
- Ruff and formatting pass; wheel and sdist built with `python -m build --no-isolation`.
- 235 benchmark configurations, 7 resident rounds and 7 with transfers,
  without errors. Includes cuBLAS, CuPy, nvmath-python and solvers to convergence.
  See [results and methodology](execution-results.md).

The architecture does not promise full NumPy; the limits of
[compile](compilation.md) and [Vulkan](vulkan.md) are part of the contract.
Earlier historical figures correspond to earlier runtime stages.
