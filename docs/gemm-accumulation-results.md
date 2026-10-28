# Float32 GEMM accumulation — 2026-10-22

This investigation preserves the runtime and its defaults. Blocked summation
eliminates tolerance violations in the random cases studied, at low cost on
CUDA and higher cost on Vulkan. Neumaier also resolves severe cancellation
cases, but multiplies large-matrix time by 2.37 to 4.99. No variant is promoted
as a universal solution.

## Motivation and scope

The [padding](padding-reassessment.md) and
[rectangular GEMM](three-iterations-results.md) rounds retained float32 errors
with long K. Before expanding a public mathematical policy, three causes are
separated: product rounding, accumulation, and multiply-add contraction.
The analysis in [Higham, *The Accuracy of Floating Point Summation*](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf)
explains why cancellation and order matter and why no summation method uniformly
dominates the others. The decision here uses local measurements.

Only experimental sources in `benchmarks/` change. The generator starts from
`_gemm`/`_gemm_rect`, preserves each backend's actual tile selection, and fails
if the structure it transforms changes. It adds no auxiliary GPU buffers,
packing, reduced precision, or backend-configuration changes.

| Variant | Operation per output |
| --- | --- |
| `current` | Current source, unchanged; control for the public path |
| `sequential` | Float32 product and float32 sum, without contraction |
| `blocked` | Ordinary summation within each K tile and summation of partials |
| `kahan` | Kahan correction for each already-rounded product |
| `neumaier` | Neumaier correction for each product; final correction sum |
| `blocked_neumaier` | Ordinary partials per tile, partials merged with Neumaier |

K tiles are 16 or 32 according to the geometry already selected. Candidates use
`FP_CONTRACT OFF` in OpenCL, `precise` on Vulkan accumulators/intermediates, and
the existing NVRTC `--fmad=false` option in CUDA. Current OpenCL/Vulkan paths
allow contraction. This is not a general promise of bitwise equality between
APIs. Corrections skip the zero-padded K tail and handle nonfinite operands/sums
without introducing `inf-inf` into the correction. They do not repair product
overflow, sum overflow, or every possible correction overflow.

## Protocol and verification

- RTX 5060 Ti, driver 617.14, Windows, Python 3.12.14, NumPy 2.5.3.
- Six sequential processes: OpenCL, Vulkan, and custom CUDA, followed by reverse
  order. Observed idle time of 10.50–10.58 s before each process; maximum allowed
  temperature 52 °C and tolerance relative to baseline 2 °C. Observed starting
  temperatures were 38–41 °C. Thresholds were not relaxed.
- Five fixed random inputs: `(33,31)@(31,65)`, two historical seeds 972/973 for
  `(65,8191)@(8191,2051)`, `(65,16383)@(16383,257)`, and 1024³.
  The second round repeats the same data and changes measurement order;
  it does not count as five new inputs.
- Seven matrices with three orderings of `[1e8, 1, -1e8]`, K=15/16/17/31/32/33/8191,
  and one case covering NaN, infinities, `0*inf`, product overflow, and sum overflow.
  K=15 is a forced generator test below the public API's tiled GEMM threshold.
- **78 case/process groups, 468 variant results, and 4,320 samples**:
  24 repetitions per variant and random input; each variant occupies each
  position four times. First call, compilation, and warm-up excluded.
  Minimum warm-up of three calls and 0.10 s per variant.
- Synchronized wall times with preallocated resident inputs and output,
  including Python dispatch. They do not measure transfers, allocation, full
  application execution, or isolated GPU time. Adversarial cases are not timed.
- Control bitwise identical to public `matmul` in every case; timed results also
  preserve their bits after repetitions.
- Full float64 reference, inherited tolerance `atol=rtol=5e-5`, without relaxation.
  Selected points—including the control's worst normalized error—use exact
  dyadic sums for both the original product and the rounded float32 product.
  Absolute and relative errors and errors divided by the sum of absolute
  products are computed with Decimal80. The oracle does not depend on BLAS.
- **54 CPU tests pass** for generation, barriers/tails, algorithms,
  cancellation/nonfinites, the oracle, and runner controls. Ruff check/format
  pass. An independent oracle review also cross-checked 240 points with Fraction.
  Final independent QA: **APPROVE**; verified all 342 saved points, hashes,
  controls, counts, and report claims.

## Observed precision

Number of elements outside tolerance in the five random matrices, per round.
Both rounds reproduced the same counts:

| Variant | OpenCL | Vulkan | Custom CUDA |
| --- | ---: | ---: | ---: |
| Current | 3 | 3 | 4 |
| Sequential without contraction | 4 | 4 | 4 |
| Blocked | 0 | 0 | 0 |
| Kahan | 0 | 0 | 0 |
| Neumaier | 0 | 0 | 0 |
| Blocked + Neumaier | 0 | 0 | 0 |

Maximum absolute random error drops from approximately `1.83e-4` to `2.34e-6`
with Kahan/Neumaier. Blocked + Neumaier ranges from `3.56e-6` to `4.84e-6`
depending on backend; ordinary blocking ranges from `2.96e-5` to `4.97e-5`.
These are suite maxima, not mathematical bounds or guarantees for other inputs.

Under severe cancellation, only Neumaier passes all seven complete matrices on
all three backends. Kahan and blocked variants still lose terms: compensating
the sum of partials does not recover information already lost within the tile.
For example, eleven repetitions of `[1e8,1,-1e8]` with K=33 must sum to 11.
Retaining this case prevents promoting a random-case improvement as robust precision.

All variants without contraction reproduce sequential float32 nonfinite
classification. Current OpenCL/Vulkan paths differ in the case of overflowing
products with opposite signs; CUDA agrees. The difference is consistent with
contraction allowed by their sources, but assembly was not inspected here to
attribute each instruction. Disabling contraction does not automatically improve
random-case precision either: the strict control introduces one more violation
on OpenCL/Vulkan.

## Measured cost

Range of **candidate time / current time**, using the four large inputs in both
rounds. Below 1 means less time. The small case is excluded from this table
because dispatch cost dominates and CUDA changed from 33.1 to 20.7 microseconds
between processes; this is not presented as a stable gain.

| Variant | OpenCL | Vulkan | Custom CUDA |
| --- | ---: | ---: | ---: |
| Blocked | 0.98–1.03 | 1.12–1.18 | 0.93–0.98 |
| Kahan | 2.15–2.80 | 3.19–3.37 | 1.82–2.80 |
| Neumaier | 3.57–4.99 | 3.70–4.12 | 2.37–3.45 |
| Blocked + Neumaier | 1.05–1.17 | 1.43–1.70 | 1.00–1.09 |

For seed 972 of `(65,8191)@(8191,2051)`, the control takes 0.992–1.029 ms on
OpenCL, 0.754–0.758 ms on Vulkan, and 0.964–0.965 ms on CUDA. Neumaier takes
3.662–3.673 ms, 3.083–3.085 ms, and 2.295–2.321 ms, respectively.
These numbers compare algorithms within each backend and existing geometry;
they are not a general comparison between APIs or against BLAS.

## Decision and reproduction

Current policy is retained. The next precision investigation should compare an
explicit robust-accumulation option with float64 accumulation, including product,
subnormals, range, batches, and full application cost. Blocked + Neumaier merits
follow-up as a compromise for random errors; it must not be called robust against
severe cancellation. This evidence does not justify an autotuner, device-based
selection, or changing tolerances.

Runner: `python -m benchmarks.gemm_accumulation --backend cuda-custom --repeats 24
--seed 1011 --output <new-file.json>`; run each backend in an isolated process
and use the idle gate in `benchmarks.stability` between processes.
Reports retain samples, order, source/input/kernel hashes, first call, warm-up,
errors, classification, and telemetry. Experiment completion and numerical
acceptance are separate fields.

Local artifacts: `.artifacts/iterations-2026-10-22/gemm-accumulation/` contains the
six reports, logs, `session.json`, and `summary.json`. The supervisor and summary
script are in the parent directory. The wheel was not rebuilt and the full
suite was not run: this batch only adds research infrastructure.
