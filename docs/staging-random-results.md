# Vulkan staging and random generation — 2026-09-13

The main track continues with Vulkan execution/transfers. P2 adds
[resident random generation](random.md), without changing example algorithms,
solvers, tolerances, or checking frequency. There are no new dependencies.

## Staging control

Host mapping was separated from the GPU lease lifetime. A staging allocation
can retain its mapping when returned to the pool; no pending buffer is recycled
because it is mapped. The backend retains barriers and the readback wait.
Eviction and release unmap before freeing memory; the pool budget remains
shared, 128 MiB and 128 buffers. Coherent host memory avoids explicit
flush/invalidate, but does not replace synchronization; see the
[Vulkan specification](https://github.khronos.org/Vulkan-Site/spec/latest/chapters/synchronization.html).

The first mode control compared transient/persistent mapping in the new code.
The next control loads only `upload/download` from the previous local wheel and
binds them to the **same backend/context**. It preserves kernels, pool, driver,
and application, restores methods on completion, and records the previous code's
hash. Each block warms up for at least 0.25 s / three calls. There are two rounds
of 31 samples, eight cases per float32/64 dtype: **64 configurations**.

Retaining all mappings eliminates repeated calls, but timings are mixed.
Float64 CG drops from 19.103→18.313 ms and 18.091→16.946 ms in full execution
(4–6%); its diagnostics record 51 mappings→0. Float32 CG is mixed in resident
mode and barely changes in full mode. A float64 energy reduction worsens from
0.529→0.584 ms and 0.540→0.587 ms in full mode. The general variant is not
presented as a universal improvement: readback-only persistence is tested next.

The final **readback-only** control, another 64 configurations, retains uploads
with map/unmap per copy and eliminates repeated download mappings.
This is the default selection. Barriers and kernels remain unchanged.

| Full Vulkan case | Previous → persistent readback, round 1 (ms) | Round 2 (ms) |
| --- | ---: | ---: |
| Energy 2049×33, float32 | 0.493 → 0.494 | 0.489 → 0.480 |
| Energy 2049×33, float64 | 0.679 → 0.675 | 0.516 → 0.495 |
| CG 129×129, float32 | 12.400 → 11.448 | 12.408 → 11.879 |
| CG 129×129, float64 | 21.136 → 19.265 | 20.279 → 17.675 |

Full CG improves approximately 4–8% for float32 and 9–13% for float64 in these
two rounds. Large downloads barely change; one scalar float32 download even
worsens in one round. Not every fluctuation is attributed to map/unmap, and
persistence is not claimed to resolve Vulkan's cost. Evidence favors this
bounded selection over retaining all mappings.

Differences between resident timings of unchanged controls also reveal noise.
A Vulkan host time includes binding, driver, and waits; wall time minus GPU
timestamp is not interpreted as pure Python time.

## P2: random values on GPU

The facade reserves counters; backends write output directly.
Philox4x32-10 shares an integer algorithm across APIs and uses their uint32
high-multiply instructions. It introduces no per-thread state and requires no
atomics, shaderInt64, or cuRAND. The seed is a scalar split into two words,
not a CPU-generated random array. The NumPy version of the same algorithm is a
functional reference; the performance comparator uses NumPy directly.

Distribution selection is capability-based. Vulkan float64 normal/exponential
transforms are outside the contract because GLSL lacks double-precision
transcendentals; the rest support float32/64 according to fp64 availability.
Stateful generation is rejected inside compile/fuse. Seed/counter restrictions
and guarantees are in [random](random.md).

Three shapes were tested: 33, 257×129, and 1025×1023, four distributions and two
dtypes, on NumPy, OpenCL, Vulkan, custom CUDA, and CuPy. Two rounds with 21
samples yield **228 valid configurations and 12 explicit exclusions** of Vulkan
float64 normal/exponential. There were no validation failures.

Normal float32, 1025×1023; each cell is round 1 / round 2, milliseconds:

| Implementation | Resident result | Generation and host delivery | NumPy followed by upload |
| --- | ---: | ---: | ---: |
| NumPy 1 thread | 7.811 / 8.000 | 8.044 / 7.909 | — |
| OpenPacho OpenCL | 0.222 / 0.263 | 1.148 / 1.045 | 8.799 / 8.569 |
| OpenPacho Vulkan, final selection | 0.233 / 0.218 | 1.668 / 1.641 | 8.915 / 8.866 |
| OpenPacho custom CUDA | 0.111 / 0.112 | 1.195 / 1.018 | 8.931 / 8.627 |
| CuPy | 0.037 / 0.036 | 0.946 / 1.072 | 8.767 / 8.714 |

Vulkan uses the repeat with final selection: 36 additional valid configurations
and 12 capability exclusions. Other providers retain the complete controls
from before final scalar-validation hardening; their kernels, counters, and
transforms did not change.

There is value here even when delivering to the host: generating on GPU avoids
uploading a CPU-created matrix. For 33 elements, NumPy still wins: normal takes
1.3–2.7 µs, versus roughly 200–218 µs on OpenCL/Vulkan, 104–168 µs on CUDA, and
30–31 µs in CuPy. No silent fallback is introduced for that size.

CuPy retains a clear advantage for large normal generation; our implementation
still uses one Philox block per output and computes only one Box-Muller component.
There is room to improve occupancy/instructions and work per thread while
respecting the counter contract. This difference is not declared a physical limit.

Comparators use different algorithms: custom Philox4x32-10, PCG64 from
`numpy.random.default_rng`, and the default cuRAND in `cupy.random.RandomState`.
NumPy uses random/standard_normal/standard_exponential with native dtype; uniform
has no dtype and is converted inside the timer. Measurements include allocation,
generation, and synchronization, and download only where indicated; none is a
pure kernel timestamp. First call and warm-up are separated.

## Precision and statistics

Three published Random123 known vectors check the integer algorithm.
GPU tests cross-check against that CPU core and check continuity across split
calls, uint32 counter carry, uint64 exhaustion, empty sizes, adjacent ranges,
0-D output, call independence, and absence of matrix uploads.

In comparisons, `random` matches the CPU Philox reference exactly.
Maximum observed float32 absolute error in transformed distributions is
9.54e-7 on OpenCL/CUDA and 2.75e-6 on Vulkan. For float64 the maximum observed
is 8.89e-16 (Vulkan uniform only). This reference uses the **same dtype**:
it is not a higher-precision oracle or a reproduction of NumPy/CuPy sequences.

Mean, variance, extrema, nonfinites, and lag-1 correlation are recorded.
Tests include a uniform histogram and normal/exponential tails/proportions with
a fixed seed. These are statistical regression controls, not TestU01/PractRand.

An initial test detected loss of precision in the GLSL normal transform when
`log(1-u)` is evaluated near zero. It was fixed with a series for u < 1/8,
retaining the original tolerance. OpenCL/CUDA/CPU use `log1p(-u)`. Working
precision is not changed and fast math is not enabled.

## Reproduction and scope

The isolated runner waits for the previous process to exit, observes idle
conditions, then warms up the next provider. The ten RNG idle periods lasted
10.4–31.1 s: there were longer waits of 16.6, 31.1, and 29.0 s. All met the
configured criterion; clocks, power, and fans are not fixed, and OS noise is
not entirely eliminated. Only RTX 5060 Ti / Windows / NVIDIA driver 616.64 was checked.

```powershell
python -m benchmarks.isolated --worker staging --backends vulkan --control-wheel <previous-wheel> --rounds 2 --repeats 31 --output <new-directory>
python -m benchmarks.isolated --worker random_generation --backends numpy-1t,opencl,vulkan,cuda-custom,cupy --rounds 2 --repeats 21 --output <new-directory>
```

Artifacts in `.artifacts/staging-random`: `staging-control` is exploratory;
`staging-final` compares all mappings against the historical wheel;
`readback-final` checks the final selection. `random-control` contains the ten
RNG runs. JSON files retain every sample, errors, telemetry, and hashes.
No version was published.

`random-vulkan-final` checks the backend after selecting readback.
`corrected-random` contains derived reports correcting only the
`reference_is_higher_precision` label: the RNG reference uses the same dtype
and must be false. Originals and their hashes are retained; timings, moments,
and error magnitudes are unchanged. The helper and its regression test now
prevent that error.

The full suite passed **1,202 tests, one skipped** because it requires two
OpenCL devices. After the final adjustments, **72 RNG/staging tests**, **198
Vulkan tests**, and **30 metrics/runner tests** passed. Wheel from sdist:
**354 pass, 772 skipped** on CPU, with import verified from inside the wheel
and independent extras. See the [exact verification scope](verification.md).
Adversarial matrix accuracy was not rerun: its kernels remain unchanged and
its earlier warnings remain open.
