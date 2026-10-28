# Three iterations: rectangular GEMM, CUDA vectors, and compiled layouts

Date: 2026-09-24. Validation hardware: NVIDIA RTX 5060 Ti, Windows,
driver 616.64. This evidence does not certify other vendors or represent a
physical hardware limit.

## Changes and selection

1. **Custom GEMM with few rows.** `_gemm_rect.py` preserves sequential
   accumulation in K and uses 64 threads per group. OpenCL computes 16×32 tiles
   with K=16; CUDA computes 16×16 with K=32. The usual path computes 32×32.
   Reducing the tile can distribute these matrices better, but worsens other
   sizes: it is neither applied universally nor blindly shared across backends.
2. **cuBLAS vector products.** DOT writes the result directly on device.
   GEMV supports batched matrices with a shared vector, transpose flags, and
   contiguous offsets. Final policy is decided through synchronized public calls,
   not by the native primitive's name.
3. **Compiled private transposes.** The compiler queries an optional backend
   hook, and CLBlast receives flags and leading dimensions. A complete 2D
   single-consumer transpose can be omitted if it is not a public output.
   Eager execution, shared transposes, other providers, and unsupported layouts
   retain copies. Users' NumPy program syntax does not change.

Rectangular selection requires float32, existing register-GEMM capabilities,
two 2D matrices, `32 <= M < 96`, `256 <= N <= 4096`, and a previous grid of at
most 224 tiles. OpenCL requires K≥4081 and CUDA K≥2048. Fully aligned cases
(M/N multiples of 32 and K of 16), batches, and larger regions retain their path.
Explicit padding keeps its separate plan. These are conservative boundaries
from this measurement, not an autotuner or universal guarantees of superiority.

Kernels use only existing portable instructions. Both variants keep shared
memory within the already-required 4 KiB, guards for irregular boundaries, and
the previous mathematical policy. Users do not need to tune tiles, create private
buffers, or change algorithms.

## Protocol

Sequential GPU processes directed by the main agent. Subagents did not run GPU
tests. Before each process: at least 10 s idle, three consecutive readings with
GPU/memory utilization ≤5%, temperature ≤52 °C, and at most 2 °C above the idle
reference. A timeout stops the experiment; it does not authorize bypassing the
condition. Compilation, warm-up, NumPy references, and validation are outside
the timer.

Each case has two independent previous controls and one variant. All six order
permutations are interleaved equally, 30 samples per mode and scope, in two
rounds with different seeds and reversed backend order. `prepared`/`resident`
includes the prepared API, output allocation, and synchronization; `complete`
also includes uploading all inputs and downloading the result. Control selection
occurs outside the timer. Speedup divides the geometric mean of both controls'
medians by the variant median.

Rectangular controls disable only the new tile configuration. Layout controls
disable the hook only when preparing independent plans. cuBLAS controls extract
only `CUDABackend.matmul` from the local historical `0.1.0a1` wheel, SHA256
`a577d80ffde9c8be17ab1c2ca9c51278aaf34412e6be84bd0ecd24450fa66e3b`;
the rest of the runtime is current. The OpenCL `0.0f` literal was already fixed
in that baseline: no gain here compares against that promotion bug.

Reports contain samples, source hashes, telemetry, plans, paths, errors, and
tolerance failures. A complete run does not imply numerical approval. Local
artifacts are in `.artifacts/next-three/candidates` and
`.artifacts/next-three/confirmation`, also retaining pilots and discarded
candidates. Final formatting does not modify the rectangular generator; its AST
was checked as identical, and the selector's equivalent condition union was
cross-checked over 4,374 combinations (`format-equivalence.json`).

Total: **38,880 public samples**, including the vector candidate later narrowed,
plus **2,352 Vulkan confirmation samples** with preallocated output. Twelve
confirmation/measurement processes completed successfully. All their idle gates
recorded `observed-idle`, with waits of 10.5 to 229.2 s. Exploratory pilots remain
separate and are not added to those counts.

## Rectangular GEMM: two public rounds

21 shapes per backend and round, including regions that must retain the previous
kernel. No selected case worsened resident time in these two rounds: gains were
1.14–1.79× on OpenCL and 1.09–3.09× on custom CUDA. These ranges describe only
the measured selected shapes.

| Float32 product | Backend | Resident, control → new (ms, round 1) | Resident speedup, rounds 1 / 2 | Full, rounds 1 / 2 |
| --- | --- | ---: | ---: | ---: |
| 65×8191 @ 8191×2051 | OpenCL custom | 1.650 → 1.024 | 1.61× / 1.60× | 1.055× / 1.054× |
| 64×8191 @ 8191×2051 | OpenCL custom | 1.652 → 0.923 | 1.79× / 1.79× | 1.066× / 1.063× |
| 65×8191 @ 8191×257 | OpenCL custom | 0.945 → 0.549 | 1.72× / 1.70× | 1.173× / 1.170× |
| 65×8191 @ 8191×2051 | CUDA custom | 1.766 → 1.020 | 1.73× / 1.71× | 1.066× / 1.068× |
| 65×8191 @ 8191×257 | CUDA custom | 1.143 → 0.370 | 3.09× / 3.04× | 1.367× / 1.351× |

In selected groups, the two previous controls differed by at most 1.55% in
resident medians and 1.02% in full medians. Square controls, batches, and excluded
regions retain previous code. Savings come from a different work distribution,
not from eliminating transfers included in the control or omitting program checks.

### Precision this optimization does not resolve

All 84 rectangular shape/backend/seed groups preserved control bits. No tolerance
was widened: rtol=atol=5e-5 against a float64 reference from the same float32 inputs.
Seven groups with elements outside the threshold persist, identical in both
controls and the variant:

- K=16383: OpenCL had 7/15 elements and CUDA 6/14, depending on round.
- `64×8191 @ 8191×4096`, round 2: one element on each backend; this shape does
  not even activate the new tile.
- `65×8191 @ 8191×257`, CUDA, round 2: one element.

A timing improvement therefore does not make all these cases numerically
approved. Long float32 accumulation and cancellation still await an explicit
precision policy. The 3× CUDA case includes that warning.

## CLBlast: omitting private transposes

28 groups in two rounds: twelve products with left/right/both flags and two
centered covariances. All pass the float64 reference with rtol=atol=5e-5.

| Float32 operation | Resident, control → new (ms, round 1) | Resident speedup, rounds 1 / 2 | Full, rounds 1 / 2 |
| --- | ---: | ---: | ---: |
| A.T @ B.T, logical product 257×513×129 | 0.161 → 0.100 | 1.61× / 1.59× | 1.157× / 1.154× |
| A @ B.T, logical product 65×8191×2051 | 1.563 → 0.950 | 1.65× / 1.65× | 1.054× / 1.052× |
| A.T @ B.T, logical product 65×8191×2051 | 1.613 → 0.971 | 1.66× / 1.69× | 1.056× / 1.058× |
| Centered covariance of 4097×129 | 0.374 → 0.346 | 1.08× / 1.06× | 1.036× / 1.049× |
| Centered covariance of 8191×513 | 2.478 → 2.425 | 1.02× / 1.03× | 1.012× / 1.009× |

Small covariance had an 11.45% difference between resident controls in the first
round: its 1.08× is not a precise gain estimate. The second had much closer
controls. For large covariance the benefit is marginal; it does not justify
continuing to prioritize microadjustments to that pattern. Removing copies does
not guarantee a win if another layout leads to a less efficient BLAS kernel.

All 28 groups also matched their materialized controls bitwise, but that is a
suite observation, not an API guarantee. Maximum absolute error was 1.36e-4 and
maximum relative L2 error 1.63e-6; tolerance combines absolute and relative parts
per element. Maximum absolute error is not assumed to have to fall below the
absolute term of a mixed tolerance.

## cuBLAS: selecting after measurement

The first candidate used GEMV for every product with a vector. In two rounds it
lost approximately 4–15% of resident throughput in many unbatched cases. The
integrated version keeps GEMM for those products and uses DOT for two vectors
and GEMV for matrices with explicit batches. Pointer-mode changes stay under a
lock, with explicit recovery if restoring HOST fails; submitted buffers remain
retained.

The two final rounds add singleton batches in both orientations. Previous cases,
both dtypes, and a GEMM control are retained. Batched improvement includes removing
general Python index computation: the experiment does not separate that saving
from the native GEMV kernel.

| Operation | Resident speedup, final rounds 1 / 2 | Full, rounds 1 / 2 |
| --- | ---: | ---: |
| DOT 257, float32 | 1.12× / 1.11× | 1.126× / 1.025× |
| DOT 8191, float64 | 1.12× / 1.10× | 0.993× / 1.047× |
| DOT 1048573, float32 | 1.10× / 1.11× | 1.016× / 1.003× |
| (4,257,511) @ (511,), float32 | 1.13× / 1.08× | 1.012× / 1.006× |
| (511,) @ (4,511,257), float64 | 1.14× / 1.10× | 1.017× / 1.021× |

Long float64 DOT was mixed (0.998× / 1.117× resident); no universal 10% improvement
is claimed. Singleton batches varied approximately 1.00× to 1.03× resident and
the full flow was mixed. The unbatched path restores previous behavior, with
residual downward differences of up to ~2.3% in this suite, around 0.4–1.9 µs.
New selection/recovery also adds host work: that cost is not presented as eliminated.

The negative GEMM control had an apparent full gain of 6.8% in one round without
changing its kernel; another A/A control differed 9.8% resident. This limits the
interpretation of small percentages. Full CUDA gains are modest and noisy
relative to resident savings. Samples and controls are retained.

All 56 final groups pass existing vector tolerances (rtol=5e-5/atol=5e-6 for
float32; 1e-11/1e-12 for float64). The float64 reference is higher precision only
for float32. Changing primitives can change the summation tree; bitwise equality
with GEMM is not required. The report's `native` label means the final policy,
which retains GEMM where appropriate.

## Vulkan: isolated confirmation of a rejected direction

The pilot already ran OpenCL, CUDA, and Vulkan sequentially, without subagent
benchmarks. Vulkan was repeated in two additional isolated processes: six shapes,
five candidate tiles, and two controls of the current kernel, 28 samples per
mode. Cyclic orders and their reverses balance positions of all seven modes.
Observed idle conditions before each process; no other project GPU benchmark or
test process ran simultaneously. This controls our work, not an absolute absence
of desktop/driver activity.

| Product | Control A / control B, round 1 (ms) | Best new candidate, round 1 / 2 (ms) | Control A / B, round 2 (ms) |
| --- | ---: | ---: | ---: |
| 65×8191 @ 8191×2051 | 0.758 / 0.759 | 0.969 / 0.938 | 0.735 / 0.733 |
| 64×8191 @ 8191×2051 | 0.696 / 0.696 | 0.866 / 0.868 | 0.691 / 0.693 |
| 65×4095 @ 4095×2051 | 0.432 / 0.437 | 0.535 / 0.531 | 0.425 / 0.426 |
| 1024×1024 @ 1024×1024 | 0.438 / 0.432 | 0.530 / 0.549 | 0.455 / 0.451 |

This measures synchronized launch into a preallocated output, not the full public
call. The small 257×129×513 case is mixed/close to noise; it does not justify
promotion either. Large losses do reproduce with very close controls. The current
Vulkan kernel is retained. This does not conclude that Vulkan lacks room for
improvement: only that reducing these tiles did not improve it. The compiler
and backend already behave differently with similar geometries; the difference
is not attributed to a specific instruction without additional ISA/profiling.
Both Vulkan confirmations preserve control bits. The second detects one
element outside tolerance in `65×8191 @ 8191×2051`, identical in both controls
and all five variants (−0.10885158 versus −0.10890857 in float64). This is also
an accuracy warning for the existing kernel, not a candidate-introduced regression.
The first round passes the threshold.

## Integration verification

- **1,652 tests pass and one is skipped**: runtime and benchmark tools, with
  OpenCL, Vulkan, and CUDA required and native CLBlast. The skipped test requires
  a second OpenCL device. All 277 PyOpenCL compiler warnings are retained.
- Wheel built from sdist: **525 pass and 909 are skipped** in the CPU environment
  without GPU bindings; optional imports blocked during initial smoke testing.
  Packaged runtime matches sources byte for byte.
- Global Ruff check and formatting pass; `pip check` finds no broken dependencies.
- The CPU workflow now includes `tests` and `benchmarks` with their optional
  extra; its remote execution and PoCL remain pending.

Passing functional tests does not cancel numerical benchmark warnings: they use
different input sets and objectives. Accumulation guarantees remain an explicit
task before the first stable release.

## Next boundary

Transfers still reduce many gains observed with resident data. All three
improvements apply within the compiler/backend, without changing example
algorithms to favor OpenPacho. The first release also needs contract, precision,
and distribution hardening; see [criteria and estimate](release-readiness.md).
