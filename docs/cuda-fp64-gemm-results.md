# Custom CUDA float64 GEMM: diagnosis and experiment

The [dense network](dense-network-results.md) revealed a repeated difference
between custom CUDA and OpenCL/cuBLAS in float64. This iteration investigates
that operation before expanding the API. Results are local to an RTX 5060 Ti,
Windows, driver 617.14; they do not support the same conclusion for other GPUs.

## Hypothesis and profiling

The current custom CUDA implementation computes each float64 output by traversing
K and reading both operands from global memory. The existing register-tiled GEMM
only supports float32. OpenCL already has a shared float64 tile.
Comparing application time alone is insufficient to attribute the difference to
GEMM: SGD also materializes transposes, reduces, and evaluates elementwise functions.

Launches from four compiled plans were instrumented with CUDA events on the
execution stream, retaining ten full-plan timing samples without instrumentation
and three instrumented repetitions per case. Outputs from every repetition
preserved their bits and passed the numerical check against NumPy float64.
Runtime files did not change during profiling.

| Float64 case | Uninstrumented plan median, ms | GEMM fraction of recorded GPU events |
| --- | ---: | ---: |
| Inference, 4096×128×256×32 | 2.130 | 95.28% |
| SGD, 4096×128×256×32 | 4.909 | 91.22% |
| Inference, 4095×127×255×31 | 2.128 | 95.21% |
| SGD, 4095×127×255×31 | 4.831 | 92.89% |

Events alter the measured program. Their fractions diagnose the recorded
launches, rather than exactly decomposing full uninstrumented time. Nor is the
difference between wall time and events interpreted as Python time. The profile
justifies a bounded GEMM experiment.

## Candidate

`benchmarks/cuda_fp64_tile.py` generates a 16×16 shared tile: 256 threads and
4096 bytes of static shared memory per block. It loads operands cooperatively
and preserves exactly K products in increasing order, without adding padding
products in the last iteration. It keeps the current CUDA options: no FMA
contraction, no subnormal flushing, and no reduced precision.

Generation supports contiguous matrices and batch broadcasting. Vectors, empty
dimensions, and K=0 are outside the candidate; the existing path retains those
contracts. Its launch requirements are checked separately from the float32 kernel's.

## Validation and comparison

Native checking includes 27 candidate cases with two destinations prefilled
with NaN and infinity: 54 launches. It covers boundaries 8/15/16/17/31/32/33,
rank-four broadcasting, cancellation, overflow, subnormals, and signed zeros.
All preserve the direct kernel's finite bits and NaN/+Inf/-Inf classes; preserving
NaN payloads is not required. Nine additional cases check that vectors, empty
arrays, and K=0 remain outside the candidate and preserve their previous result.

Selected samples are also checked against exact products and sums of fractions
of the original binary values, separating product rounding and accumulation.
Equivalence with the control does not imply mathematical exactness: cancellation,
rounding, and range errors of the original algorithm are preserved.
NumPy float64 is labeled as a same-precision reference.

The `benchmarks.cuda_fp64_gemm` runner captures the direct kernel's actual sources,
without executing them during capture, and retains two controls with that same
source. The comparison uses 24 GEMMs and six applications: small, regular, and
odd inference/SGD. Each case has three modes, 24 repetitions, and all six order
permutations with equal frequency. The output retains captured sources and allows
reuse through `--control-report` even if the runtime changes in the future.

GEMM with preallocated output and matmul with allocation are separated.
Applications use three independent plans, original state on every call, and two
scopes: resident execution and the full flow with uploads and download of all
outputs. First call, route discovery, preparation, and warm-up are excluded from
repeated samples. Numerical checks are outside the timer. Route-selection
instrumentation is common to all three modes and adds Python work: these times
are not equivalent to running the runtime without it.

The two rounds use separate processes and observed idle conditions before
starting. The same data and sources are preserved across rounds. Raw samples
and numerical failures are saved even when they invalidate a comparison.

## Result and decision

**The candidate is not promoted.** The two rounds complete 60 cases, 8,640 samples,
and 9,360 accepted checks. There is no consistent application improvement to
justify adding a runtime path. Narrow kernels with large K repeatedly worsen;
several large products remain nearly unchanged.

The table expresses candidate time divided by the average of the two controls'
medians. Below 1 favors the candidate. R1/R2 are independent processes; small
differences are not treated as firm gains.

| GEMM M×K×N | Preallocated R1 | Preallocated R2 | With allocation R1 | With allocation R2 |
| --- | ---: | ---: | ---: | ---: |
| 4096×128×256 | 0.998 | 1.000 | 1.004 | 1.001 |
| 4096×256×32 | 1.085 | 1.025 | 1.051 | 1.076 |
| 256×4096×32 | 1.088 | 1.117 | 1.099 | 1.077 |
| 4096×32×256 | 1.001 | 1.001 | 1.004 | 1.035 |
| 128×4096×256 | 1.013 | 1.028 | 1.016 | 1.010 |
| 4095×127×255 | 1.005 | 1.006 | 1.001 | 1.002 |
| 4095×255×31 | 1.060 | 1.071 | 1.124 | 1.049 |
| 255×4095×31 | 1.122 | 1.110 | 1.108 | 1.095 |
| 4095×31×255 | 1.006 | 1.013 | 1.004 | 1.006 |
| 127×4095×255 | 1.006 | 1.029 | 0.999 | 1.027 |

| Application | Resident R1 | Resident R2 | Full R1 | Full R2 |
| --- | ---: | ---: | ---: | ---: |
| inference, 31×7×11×3 | 1.012 | 1.011 | 1.093 | 0.927 |
| sgd, 31×7×11×3 | 1.070 | 0.954 | 0.961 | 1.000 |
| inference, 4096×128×256×32 | 1.014 | 1.004 | 1.024 | 1.005 |
| sgd, 4096×128×256×32 | 1.041 | 1.029 | 1.027 | 1.034 |
| inference, 4095×127×255×31 | 1.018 | 1.023 | 1.003 | 1.004 |
| sgd, 4095×127×255×31 | 1.028 | 1.028 | 1.008 | 1.037 |

Across the ten large GEMMs, the largest ratio between the two controls' medians
is 1.038. It reaches 1.753 for small cases and 1.158 for small applications.
Variation particularly limits the interpretation of small micromeasurements.
The two identical controls are retained to expose that variation, not to eliminate
it by averaging.

Changing memory organization while retaining arithmetic order and operations
does not explain the historical difference between APIs here. CUDA compiles
GEMM with `--fmad=false`; the OpenCL GEMM program uses `acc += left*right` and
does not include `FP_CONTRACT OFF`. This supports investigating contraction, but
source code alone does not establish which float64 instructions OpenCL executed.
OpenCL elementwise/map-reduce lowerers do disable contraction.

A later bounded investigation can save PTX/binaries for these shapes and compare
identical sources changing only contraction. This would be a different rounding
policy: it needs error validation, not bitwise equality, and does not authorize
changing defaults.

Targeted verification comprises 19 CPU generation/indexing tests and seven tests
of capture, restoration, validation, and plan protocols. The global suite was
not run. The runtime, its defaults, and public API remain unchanged.

Independent QA: **APPROVE**. All 26 CPU tests were rerun, tables and samples were
verified, all 23 control sources were reproduced without launching them, and
172 exact-oracle points were checked. All 33 runtime files are identical to the
previous state. The review did not rerun the GPU.

## Local evidence

Private reports and scripts are retained in
`.artifacts/iterations-2026-10-22/`: `cuda-fp64-profile.json`,
`profile_cuda_fp64.py`, `cuda-fp64-correctness.json`, `validate_cuda_fp64_tile.py`,
`cuda-fp64-tile-cpu.xml`, and `cuda-fp64-gemm/`. The latter contains the two reports,
the session, and the reproducible summary. The profile retains per-launch events,
shapes, source and result hashes, GPU state, and samples.
These local artifacts are not included in the distributed package.
