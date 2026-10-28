# Padding reassessment with the corrected OpenCL baseline

Date: 2026-09-19. **`padding=True` remains an explicit option, disabled
by default.** We found a reproducible advantage in products with few rows
and long K, even including padding and cropping. Most measured shapes
regress; automatic activation or prioritizing its expansion is not justified.

The [previous fix](kernel-inspection-results.md) removed accidental
float64 conversions in OpenCL loads. It invalidated the historical padding advantage
in the large irregular product tested then. This round uses that corrected
baseline on **both** sides of every comparison. It does not change the runtime, its
kernels, math flags or tolerances.

## Main evidence

RTX 5060 Ti, Windows, driver 616.64, float32, custom providers. For
`(65,8191) @ (8191,2051)`, these are medians and speedups from three rounds
with different seeds. A factor above one favors padding.
The control combines the two unpadded medians using their geometric mean;
speedup divides that value by the padded median from the same round.

| Backend | Resident without padding, ms | Resident with padding, ms | Resident speedup | Speedup with upload and download |
| --- | ---: | ---: | ---: | ---: |
| OpenCL | 1.643–1.666 | 1.173–1.220 | **1.366–1.400×** | **1.040–1.041×** |
| CUDA custom | 1.759–1.769 | 1.451–1.458 | **1.212–1.214×** | 0.843–0.865× |
| Vulkan | 0.782–0.820 | 1.111–1.212 | 0.677–0.704× | 0.902–0.913× |

Resident time includes the prepared public call, allocations, copies,
padding, GEMM, cropping and synchronization. It does not use prepacked inputs or remove
copies between invocations. The complete path adds uploading both inputs and
downloading the result of that same prepared plan; it does not measure eager `matmul`.
CUDA wins with resident data and loses when transfers are included on this shape.
OpenCL retains a small gain over the complete path.

The two independent unpadded controls differ by at most 0.47% in
this OpenCL case, 0.33% on CUDA and 1.94% on Vulkan, across scopes and rounds. The
favorable resident signal exceeds that observed variability. This is not a statistical
guarantee for other sizes, devices or applications.

The physical shape here becomes `(96,8192) @ (8192,2080)`. `explain` reports
**72,101,888 additional bytes** of private buffers and three copy dispatches.
That value excludes original inputs, logical output, pools and driver memory.
The memory cost is real even though the GEMM tile count and K iteration count
were already the same before padding.

## Distribution of results

Minimum–maximum resident speedup across the three rounds. All shapes are
retained in the table, including unfavorable ones. Ranges show variation between
rounds, not confidence intervals.

| M,K,N / batches | OpenCL | CUDA custom | Vulkan |
| --- | ---: | ---: | ---: |
| 257,255,259 | 0.373–0.563× | 0.448–0.455× | 0.544–0.608× |
| 1024,1024,1024, aligned | 0.994–1.000× | 1.002–1.011× | 1.008–1.015× |
| 1025,1023,1027 | 0.910–0.917× | 0.833–0.868× | 0.766–0.832× |
| 2049,2047,2051 | 0.954–0.955× | 0.999–1.010× | 0.896–0.902× |
| 4095,4095,4095 | 1.006–1.008× | 0.919–0.928× | 0.857–0.866× |
| 4097,4093,4101 | 1.002–1.008× | 0.911–0.963× | 0.777–0.795× |
| 2048,2047,2048 | 0.987–0.988× | 0.965–0.966× | 0.925–0.938× |
| 2047,2048,2048 | 0.986–0.988× | 0.964–0.965× | 0.927–0.936× |
| 2048,2048,2049 | 0.993–0.996× | 0.970–0.980× | 0.937–0.943× |
| 1024,8191,1024 | 0.949–0.951× | 0.945–0.947× | 0.898–0.899× |
| 65,8191,2051 | **1.366–1.400×** | **1.212–1.214×** | 0.677–0.704× |
| 4097,255,4099 | 0.825–0.830× | 0.425–0.433× | 0.385–0.391× |
| 8191,513,257 | 0.907–0.927× | 0.768–0.848× | 0.726–0.784× |
| 127,2047,8191 | 0.796–0.803× | 0.699–0.714× | 0.706–0.713× |
| batch 4; 257,255,259 | 0.524–0.587× | 0.531–0.546× | 0.581–0.663× |
| batches (2,1)/(1,3); 257,511,259 | 0.623–0.720× | 0.611–0.636× | 0.641–0.701× |

Only one of the fifteen irregular shapes exceeds 5% resident improvement in all three
OpenCL and CUDA rounds. None does on Vulkan. OpenCL improvements below 1%
on large matrices do not change the decision. The fully
aligned case does not pad or copy: small differences there come from
guard specialization or variation, and do not prove the value of physical padding.

## Nearby-size check

Before closing the decision, seven nearby shapes were tested on OpenCL and CUDA,
with two new rounds and the same methodology: **28 groups and 5,040 samples**.
The benefit survives on several shapes but does not extend uniformly:

| M,K,N | OpenCL resident | CUDA resident | OpenCL complete | CUDA complete |
| --- | ---: | ---: | ---: | ---: |
| 65,8191,2051 | 1.394× | 1.213–1.218× | 1.041× | 0.863× |
| 64,8191,2051 | 1.469–1.472× | 0.918–0.920× | 1.045–1.047× | 0.833–0.835× |
| 127,8191,2051 | 1.153–1.159× | 0.608–0.610× | 1.016–1.018× | 0.775–0.794× |
| 256,8191,2051 | 0.871× | 0.775–0.778× | 0.975–0.977× | 0.804–0.821× |
| 65,4095,2051 | 1.290–1.343× | 1.167–1.178× | 1.032–1.037× | 1.025–1.026× |
| 65,8191,2048 | 1.431–1.437× | 0.847–0.873× | 1.043–1.045× | 0.830–0.839× |
| 65,8192,2051 | 1.376–1.403× | 1.215–1.223× | 1.039–1.041× | 0.864–0.865× |

The main favorable case wins again with different data. Reducing M from 65
to 64 or aligning N is enough to reverse the CUDA advantage; OpenCL behaves
differently. A rule based only on “long K” or “irregular size” would be insufficient.
This follow-up selects neighbors of a shape discovered in the suite; it must not
be interpreted as an unbiased sample of all possible workloads.

## Accuracy

The **144 shape/backend/seed groups** produce bitwise identical results
between `off-a`, `on` and `off-b`; repetition in each scope also preserves
bits. That demonstrates no observed numerical changes from padding, not
universal float32 GEMM accuracy.

`rtol=5e-5, atol=5e-5` is retained against NumPy float64 built from inputs
already quantized to float32. OpenCL and Vulkan pass that comparison in all 48 groups
per backend. CUDA has one element outside tolerance in each of
these three groups, **with and without padding**:

- Seed 972, `(1024,8191) @ (8191,1024)`.
- Seeds 972 and 973, `(65,8191) @ (8191,2051)`.

For seed 972 and the second shape, element `[16,653]` is
`0.05214071273803711`, versus `0.05208303933545566` in the reference. Reproducing
the float32-rounded products and sequential sum on the CPU gives exactly
the CUDA value. This is evidence of accumulation/rounding with long K and cancellation,
not corruption introduced by padding. Improving this accumulation remains
independent work; FMA was not changed and the threshold was not relaxed.

In the main suite, maximum absolute error is 1.518e-4 on OpenCL/Vulkan and
1.613e-4 on CUDA; maximum relative L2 error is approximately 1.630e-6. A
small global metric can coexist with an element outside tolerance.
See also the [numerical follow-up](numerical-accuracy.md).

All 28 follow-up groups also preserve bits between modes and
pass the threshold with their seeds. Their maximum absolute errors are 1.691e-4
on OpenCL and 1.634e-4 on CUDA; maximum relative L2 error is approximately 1.629e-6.
In total there are **172 groups and 30,960 samples**, with 22 distinct shapes; the main
case appears in both suites with different inputs. The follow-up does not
remove the three tolerance failures recorded in the main suite.

The first attempt at the second CUDA round stopped in the unpadded
control because of that tolerance. The incomplete report is retained and excluded from
the tables. The worker was adjusted to record finite violations and continue
other modes, preserving data, flags and tolerances. Now `complete` indicates
completed execution and `all_reference_close` indicates accuracy; they are not equivalent.

## Method and reproduction

The main suite has 16 shapes, three backends and three seeds: nine complete
processes, 144 groups, 432 modes and **25,920 timed samples**. Each
group uses three independent plans (`off-a`, `on`, `off-b`), the same inputs,
30 samples per mode and scope, and all six possible orders with equal
frequency. Preparation, a minimum warmup of 0.25 s, references, validation
and telemetry are outside the clock. Every sample and the code hash are saved.

GPU processes are sequential. Before each process, at least
10 s of idle time is observed, three consecutive readings with GPU/memory utilization up to
5%, temperature up to 52 °C and within 2 °C of the idle reference;
the wait is up to 300 s without skipping the control. Backend and case order varies.
The thirteen selected preflight waits lasted between 10.4 and 62.2 seconds.
These controls do not lock frequencies or guarantee thermal equilibrium throughout
the run. Measurements are wall times, not pure device times.

```bash
python -m benchmarks.isolated --worker padding_decision --backends opencl,vulkan,cuda-custom --rounds 3 --repeats 30 --seed 971 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/padding-decision/reproduction
```

GPU runtimes and the Vulkan compiler must be configured as for other
suites. The worker also accepts `--case-indices` when run directly;
for example, index 10 selects the favorable shape. The new benchmark
records numerical violations without aborting other modes; its exit code
for completed execution **does not certify** `all_reference_close`.

Main local results are in `.artifacts/padding-decision/broad`
(round 1 only) and `continuation` (rounds 2 and 3). `broad/2-cuda-custom.json` is the
excluded incomplete attempt. `final-metrics.json` retains report selection and
hashes. Raw data and private tools are not included in the wheel.

The follow-up is in `.artifacts/padding-decision/neighbors`. Its adapter
`neighbor_worker.py` calls the same `benchmarks.padding_decision.run` with the
seven table shapes in that order, seeds 971 and 972 and 30 samples. Inputs
are generated with `default_rng(seed * 101 + shape_index)` and uniforms
in `[-0.5,0.5)` converted to float32. Thus the repeated case, index 0 here
and 10 in the main suite, has different values. Hashes identify both
the shared worker and shape adapter.

Literals `0.0` were also corrected to `0.0f` in the **experimental**
`benchmarks.kernel_variants` generator, where the same accidental promotion remained.
The three-language test protects that baseline for future experiments; the
benchmark for this decision uses the runtime generator, which was already corrected.

## Decision and next priority

Keep the current contract and `False` by default. Recommend `True` only
after measuring the complete workload and its memory; on this machine, there is no reason
to recommend it for Vulkan. Do not introduce an automatic heuristic from
a single GPU or extrapolate to all long-K products.

The next priority is improving unpadded GEMM for rectangular shapes
and its accumulation accuracy. Access alignment, edge handling and
tile selection can capture some of the benefit without materializing complete
inputs. This round changes stride, guards and submission together, so it does not
attribute the entire improvement to removing branches. Nor does it constitute a physical
limit for any backend. The possible extension to `mean` remains [noted](padding.md#future-extension-reductions),
with no implementation or additional priority in this round.
