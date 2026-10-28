# Optional padding: local results — 2026-09-18

**Historical snapshot.** The [subsequent inspection](kernel-inspection-results.md)
detected and fixed unnecessary float64 conversions on the OpenCL path without
padding. With the corrected baseline, the padding advantage in the large irregular
case does not persist. The measurements below preserve the previous version.
The [current reassessment](padding-reassessment.md) expands the shapes on the corrected
baseline and retains the option for specific gains on rectangular products.

Padding provides an improvement reproduced in both large irregular GEMMs on
**OpenCL custom**. On the measured Vulkan and CUDA custom shapes, the additional
copies do not produce a general improvement. It remains disabled by default.
The [contract](padding.md) allows selecting it without changing logical dimensions,
accuracy or BLAS provider.

## Method and scope

Windows, Python 3.12, RTX 5060 Ti, driver 616.64; a single vendor/device.
`benchmarks.padding` ran six float32 shapes, including rectangular, odd,
aligned and broadcasting `(2,1,65,47) @ (1,3,47,67)`. Two rounds with seeds 941
and 942, separate processes, reversed backend order and 21 samples per variant
and scope. Within each case, `padding=False` and `True` alternate with the same
data; preparation and warmup are outside the repeated clocks.
The recorded first execution may include kernel compilation; the Python plan
has already been prepared. It does not measure installation or initial imports.

All six preflight controls observed GPU/memory inactivity ≤5%, three consecutive
readings, at least a 10 s pause, temperature ≤52 °C and within the configured
relative margin. Actual waits were **10.5–156.4 s**; the control was not relaxed
to complete the session. Mean temperatures in the final windows were
between 41 and 49 °C. This limits interference between processes;
it does not guarantee constant frequency, system isolation or thermal equilibrium.
Variation between rounds is visible, especially in some small calls.

**6,048 public samples** were saved: resident compiled execution,
prepared executor, direct `matmul` and the complete path with upload/download.
All include padding and cropping copies. Another **756 samples** correspond
to the `prepacked_wall` diagnostic: already padded inputs and an uncropped
physical output. This diagnostic excludes necessary API costs and is not pure
GPU time: it includes dispatch, allocation and synchronization.

## Resident compiled execution

Each range spans the medians of the two rounds, **not** a confidence
interval. Speedup is `time_without / time_with`, calculated per round;
values below one indicate a regression when padding is enabled.

| Backend | M,K,N / batch | Without padding (ms) | With padding (ms) | Speedup per round |
| --- | --- | ---: | ---: | ---: |
| OpenCL | 33,17,65 | 0.055–0.057 | 0.132–0.139 | 0.41–0.42× |
| OpenCL | 257,255,259 | 0.093–0.096 | 0.133–0.158 | 0.61–0.70× |
| OpenCL | 1025,1023,1027 | 0.845–0.874 | 0.597–0.647 | **1.35–1.42×** |
| OpenCL | 2049,2047,2051 | 6.002–6.503 | 3.760–4.237 | **1.53–1.60×** |
| OpenCL | 1024,1024,1024 | 0.505–0.541 | 0.496–0.542 | 1.00–1.02× |
| OpenCL | batch 2×3: 65,47,67 | 0.059–0.082 | 0.142–0.217 | 0.38–0.42× |
| Vulkan | 33,17,65 | 0.141–0.172 | 0.231–0.270 | 0.61–0.64× |
| Vulkan | 257,255,259 | 0.178–0.179 | 0.272–0.283 | 0.63–0.66× |
| Vulkan | 1025,1023,1027 | 0.535–0.541 | 0.671–0.683 | 0.79–0.80× |
| Vulkan | 2049,2047,2051 | 2.826–2.992 | 3.317–3.319 | 0.85–0.90× |
| Vulkan | 1024,1024,1024 | 0.494–0.503 | 0.485–0.503 | 1.00–1.02× |
| Vulkan | batch 2×3: 65,47,67 | 0.168–0.171 | 0.266–0.269 | 0.63–0.63× |
| CUDA custom | 33,17,65 | 0.075–0.076 | 0.216–0.221 | 0.34–0.35× |
| CUDA custom | 257,255,259 | 0.098–0.151 | 0.215–0.462 | 0.33–0.46× |
| CUDA custom | 1025,1023,1027 | 0.529–0.546 | 0.616–0.688 | 0.79–0.86× |
| CUDA custom | 2049,2047,2051 | 3.700–3.720 | 3.689–3.713 | 1.00–1.01× |
| CUDA custom | 1024,1024,1024 | 0.470–0.470 | 0.461–0.466 | 1.01–1.02× |
| CUDA custom | batch 2×3: 65,47,67 | 0.083–0.107 | 0.215–0.306 | 0.35–0.38× |

The aligned control does not allocate private buffers or make additional copies.
Its small differences are not considered a robust gain. The marginal CUDA
difference at 2049 is not interpreted as a demonstrated improvement either.

## Complete cost and memory

On OpenCL, the 2049×2047×2051 case still wins with transfers:
**14.51–14.69 ms → 11.95–12.79 ms**, a speedup of **1.15–1.21×** per round.
It requests **51,384,320 private bytes (49.00 MiB)** in addition to inputs and
the logical output. The 1025×1023×1027 case requests 12.50 MiB and goes from
3.14–3.31 ms to 2.93–3.03 ms complete, 1.07–1.09×. The resident benefit cannot
be extrapolated without including transfers.

The prepacked OpenCL diagnostic at 2049 takes 3.52–3.89 ms, versus
3.76–4.24 ms for the public resident path with all copies. On Vulkan it takes
2.80–2.84 ms and the public path 3.32 ms; the ordinary path already takes 2.83–2.99 ms.
These data point toward reducing copies and retaining internal layouts across
compatible chains. They do not prove which instructions the compiler emitted or
that all source-code guards become machine branches.

## Numerical error and limits

In all **36 paired comparisons**, the padded result matches the unpadded
result exactly element by element on the same backend.
Both pass comparison against a NumPy float64 reference; the largest observed
relative L2 error was **8.11×10⁻⁷** and the largest absolute error **4.02×10⁻⁵**.
No nonfinite values appeared in these random data. The elementwise check uses
`rtol=5e-5, atol=5e-5`; this is not a guarantee for arbitrary matrices.

Specific tests add NaN/Inf, signed zeros, subnormals and extrema:
copies preserve uint32 words bit for bit, added values are exact positive zero
and arithmetic is checked against the backend's existing path.
They also verify cropping before `sum`/`mean`, independent results,
input rebinding, batches, memory/index limits and BLAS preservation.
Math flags are unchanged and binary identity across vendors is not promised.

## Reproduction and next opportunities

Command and requirements are in [padding](padding.md). Samples, plan metadata,
source hashes, telemetry and errors from this run are in
`.artifacts/padding/measurements`; the summary is in
`.artifacts/padding/summary.json`. Private artifacts are not included in the
wheel. [Verification](verification.md) records the suites and final package.

The next useful opportunity would be to amortize copies by preserving padding through
compatible operations, with explicit logical dimensions and rules for
reductions/nonfinite values. An explicit contract for reusable prepacked
operands could also be evaluated. This version copies the current inputs again
on every call: it does not silently assume they are immutable. Shape/hardware
autotuning can later explore which cases justify the cost.
Current results do not justify enabling it globally.
