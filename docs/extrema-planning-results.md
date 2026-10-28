# Preparing materialized extrema — 2026-10-22

The compiled reduction of an already materialized array can reuse the cooperative
lowerer with an empty elementwise group. The comparison justifies enabling
this path for **float32 min/max on OpenCL and CUDA**, retaining the previous path
on Vulkan. It does not change kernels, accuracy, BLAS providers or eager reductions.
Float64, integers, bool and sum retain their previous selection when the producer
is materialized; private-producer fusion continues working as before.

This iteration follows [resident extrema](extrema-results.md). The hypothesis
was specific: avoid repeated preparation and use the existing cooperative
reduction where the eager primitive traverses the axis serially. No different
normalization algorithm is introduced to make the comparison favorable.

## Experiment

`benchmarks/extrema_planning.py` prepares three independent plans on the same
float32 data: two eager controls and a candidate with
`prepare_map_reduce(node, (), node.inputs)`. Only the reduction step for
materialized extrema changes; other steps, dependencies, outputs and uses are preserved.
The runner explicitly forces both modes, even after promoting the path.
It does not change global policies or attribute fusion to a shared producer.

Eleven cases include row, column, noncontiguous-axis and total reductions,
a small case, three softmax cases and one logsumexp. Each process uses a different seed
per round and retains hashes for inputs, runtime, runner and recipe. Inputs
are uniform in [-4,4]; the performance test does not replace coverage of
NaN, signed zeros, empty arrays or dtype limits.

Two rounds, each with one process per backend, reverse the order between
OpenCL/Vulkan/CUDA and CUDA/Vulkan/OpenCL. Each process requires at least
ten seconds of idle time, three consecutive readings of GPU/memory utilization
≤5%, temperature ≤52 °C and ≤2 °C above baseline. All six gates passed:
10.48–16.75 seconds and 43–44 °C at startup. The policy was not relaxed.

Each mode warms up for at least 0.1 s and three calls. There are 30 measured calls
per mode and scope, using all six order permutations with equal frequency.
This gives **66 case/process groups and 11,880 samples**. Resident scope includes
allocations, computation and synchronization; complete scope adds upload and download.
Preparation, first call, warmup and validation are recorded separately.
These are flow wall times, not kernel times from GPU timestamps.

Before and after measurement, every output in both scopes matches the initial
control bit for bit. Extrema are exact against the same data in float64;
softmax/logsumexp meet rtol=2e-5 and atol=2e-6 against the stable float64 formula.
All six processes completed and all those checks passed. The report
distinguishes completed execution, numerical acceptance and bitwise equivalence.

## Results

Tables show the range across the two rounds for
`median(candidate) / mean(median(control A), median(control B))`.
Less than one means less time. All cases are retained.

### OpenCL custom

| Case | Resident | Complete |
| --- | ---: | ---: |
| max rows 3×8193 | 0.109–0.148 | 0.211–0.323 |
| max rows 1024×1024 | 0.256–0.266 | 0.884–0.887 |
| min rows 8193×33 | 0.595–0.599 | 0.892–0.910 |
| max columns 8193×33 | 0.074–0.110 | 0.233–0.248 |
| max total 1,048,577 | 0.585–0.636 | 0.886–0.917 |
| min axes (0,2), 17×129×33 | 0.265–0.267 | 0.668–0.691 |
| max rows 3×17 | 0.561–0.604 | 0.810–0.814 |
| softmax 32×8193 | 0.217–0.317 | 0.492–0.498 |
| softmax 1024×1024 | 0.787–0.790 | 0.969–0.993 |
| softmax 8193×33 | 0.820–0.822 | 0.947–0.961 |
| logsumexp 32×8193 | 0.142–0.252 | 0.292–0.315 |

### CUDA custom

| Case | Resident | Complete |
| --- | ---: | ---: |
| max rows 3×8193 | 0.759–0.830 | 0.883–0.917 |
| max rows 1024×1024 | 0.673–0.681 | 0.964–0.966 |
| min rows 8193×33 | 0.717–0.722 | 0.932–0.939 |
| max columns 8193×33 | 0.849–0.876 | 0.944–0.951 |
| max total 1,048,577 | 0.851–0.855 | 0.972–0.978 |
| min axes (0,2), 17×129×33 | 0.667–0.669 | 0.886–0.887 |
| max rows 3×17 | 0.621–0.683 | 0.845–0.846 |
| softmax 32×8193 | 0.937–0.943 | 0.990–1.002 |
| softmax 1024×1024 | 0.876–0.907 | 0.974–0.988 |
| softmax 8193×33 | 0.884–0.886 | 0.963–0.965 |
| logsumexp 32×8193 | 0.910–0.922 | 0.937–0.957 |

### Vulkan, candidate rejected as the default path

| Case | Resident | Complete |
| --- | ---: | ---: |
| max rows 3×8193 | 1.141–1.160 | 1.207–1.224 |
| max rows 1024×1024 | 0.880–0.885 | 1.003–1.004 |
| min rows 8193×33 | 0.916–0.931 | 1.019–1.058 |
| max columns 8193×33 | 1.152–1.160 | 1.007–1.007 |
| max total 1,048,577 | 0.756–0.777 | 0.995–0.999 |
| min axes (0,2), 17×129×33 | 0.883–0.893 | 0.911–0.932 |
| max rows 3×17 | 0.881–0.897 | 0.933–0.934 |
| softmax 32×8193 | 1.039–1.057 | 1.025–1.078 |
| softmax 1024×1024 | 0.950–0.957 | 0.989–1.021 |
| softmax 8193×33 | 0.964–0.966 | 0.965–1.040 |
| logsumexp 32×8193 | 1.088–1.103 | 1.069–1.072 |

The resident improvement on OpenCL and CUDA repeats in every case. In the complete
flow, transfers attenuate many gains: CUDA softmax 32×8193 does not demonstrate
a consistent complete-flow improvement. Vulkan shows repeated regressions,
including logsumexp; selection is not generalized across backends.

The two controls also show noise: the largest ratio between their medians was
1.055 on OpenCL, 1.045 on Vulkan and 1.157 on CUDA. The last corresponds to
complete logsumexp in the first round; the small resident CUDA max reached
1.143 in the second. JSON files retain samples and dispersion. Differences of
a few percentage points do not by themselves prove a reliable improvement.

## Integration and limits

Selection belongs to the lowerer through an optional capability, resolved when
preparing the plan. The compiler does not select by backend name. An external
lowerer without that capability retains ordinary execution. The new step consumes
the materialized input as a leaf: shared, returned or nonelementwise producers
are not duplicated or improperly absorbed. `explain` shows
`map_reduce` and its operation. Existing private fusion takes priority.

Measurement uses Windows, RTX 5060 Ti/driver 617.14, Python 3.12.14 and NumPy 2.5.3.
It does not demonstrate performance portability to other devices or a benefit
for unmeasured dtypes. Nor does it compare against external libraries. The
mathematical path retains NaN, infinities, signed zeros and dtype; it adds no tolerances.

Local reports and the supervisor are in
`.artifacts/iterations-2026-10-22/extrema-planning/` and its parent directory.
`measured-runner.py` preserves the runner from before promotion; its hashes
match the reports. The current runner forces controls so repeating the experiment
does not accidentally compare two optimized plans.

```powershell
.venv/Scripts/python.exe -m benchmarks.extrema_planning --backend opencl --repeats 30 --seed 1101 --output .artifacts/benchmarks/extrema-opencl.json
```

The idle gate belongs to the supervisor, not that individual command.
Use a new file; the runner refuses to overwrite results.

## Integration verification

**106 targeted tests pass**, with four skips for unsupported integers
on CUDA. They include extrema and normalization contracts, the lowerer's optional
capability, shared/returned/transposed producers, rebinding and array release,
and the benchmark protocol and control inversion.
OpenCL, Vulkan and CUDA were required. The full suite was not run.

Local XML files are `extrema-planning-tests.xml` (102 pass, four skips)
and `extrema-materialized-native.xml` (four pass), under the iteration directory.

After promotion, a short integration run repeats the eleven cases
on each backend with six calls per mode/scope: **1,188 samples**, all
reference and bit checks passed. It checks runner v2 with forced
controls and the integrated runtime; it does not replace the two performance rounds
in the tables. It is retained in `extrema-planning-integration/`, with idle gates
and separate processes.

Independent QA: **APPROVE**. It reran 45 CPU tests and checked all
ratios, samples, orders, idle gates, hashes and integration results.
The review did not run GPU work or modify the implementation.
