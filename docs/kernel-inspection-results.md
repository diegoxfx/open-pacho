# Kernel inspection and copy review — 2026-09-19

Inspection found a fixable issue in OpenCL float32 GEMM:
`0.0` literals in conditional loads promoted values to float64.
Changing only those literals to `0.0f` removes the conversions and reduces
time for the measured large irregular case by approximately **38 %**. Part of the
previous padding advantage avoided that inefficiency. It was insufficient evidence
to attribute the entire improvement to removing branches.

Review of padding copies found no data or index corruption.
Repeated preparation is reduced and execution-scope selection is corrected
for a product with multiple submissions. There is no claim that copy kernels have
reached a global optimum or that padding will improve every product.

## What was actually inspected

Local hardware: RTX 5060 Ti, Windows, Python 3.12, NVIDIA driver 616.64. The same
custom GEMM uses 64-thread groups, 32×32 tiles, K blocks of 16 and 16 float32
accumulators per thread. Sources share that arithmetic, but compilers do not
necessarily generate the same program.

| Backend | Available evidence | Evidence limit |
| --- | --- | --- |
| OpenCL | Program binary returned by the driver, which contains PTX here | PTX is IR; its registers are virtual, not physical registers |
| CUDA | NVRTC PTX, loaded-function attributes and SASS obtained by relinking that PTX | Relinking is not a dump of the already loaded module and may differ from its JIT |
| Vulkan | Exact GLSL and SPIR-V, pipeline statistics with capture enabled | The driver returns zero internal representations; we did not obtain Vulkan ISA |

OpenCL PTX was also relinked with the CUDA Driver API, outside the clocks.
Its SASS retains the previous control's conversions and removes them with the fix.
This reinforces the diagnosis, but **is not the original OpenCL executable**.
Extraction saves sources, flags, hashes, PTX/SPIR-V, metadata and disassembly
in `.artifacts/kernel-inspection`; it adds no dependencies to the installable runtime.

The Vulkan adapter queries the
[`VK_KHR_pipeline_executable_properties`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_executable_properties.html)
extension through a typed ABI, initializes structures and repeats incomplete queries.
The driver's `Local Memory Size=68719476736` per-thread value is suspicious and has no
corroboration. An independent CFFI/ctypes check found no layout or read error;
**we do not use that value to infer spills or occupancy**.
New exports retain original bytes and flag this limitation.

## Compiler controls

Two rounds, three separate processes per round, three sizes and 21 samples per
variant: **72 configurations and 1,512 wall times**, plus five separate GPU
diagnostics per configuration. Inputs and outputs preallocated; compilation,
transfers and preparation are outside the clock. Each measured call does include
dispatch and synchronization. These are direct GEMM diagnostics, not application
or complete API speedups. OpenCL uses a profiling queue and
Vulkan pipelines use capture flags; these times do not replace a normal runtime
measurement.

Observed idle time and acceptable temperature are required before each process.
Backend order is reversed between rounds and variants alternate within
each case. Telemetry and all samples are retained; this reduces bias
but does not guarantee constant clocks or thermal equilibrium during each call.

All cases meet `rtol=5e-5, atol=5e-5` against NumPy float64. In this
suite, including experimental math modes, maximum observed relative
L2 error is **8.113×10⁻⁷** and maximum absolute error **4.141×10⁻⁵**. These are not bounds
for arbitrary matrices or a promise of equal results across backends.

### OpenCL: accidental double-precision promotion

For `(2049,2047) @ (2047,2051)`:

| Variant | Wall-time medians of the two rounds | float64 conversions in PTX | Branches in PTX |
| --- | ---: | ---: | ---: |
| Historical literal `0.0` | 6.322–6.323 ms | 28 | 38 |
| float32 literal `0.0f` | 3.907–3.943 ms | 0 | 38 |
| `0.0f` and fixed group declared to the compiler | 3.907–3.918 ms | 0 | 38 |

Counts are static, not instructions executed per element. Guards,
tiles, inputs, outputs and flags are identical in the first pair. Previous PTX
included `cvt.f64.f32` and `cvt.rn.f32.f64` around loads; relinked SASS
shows 14 `F2F.F64.F32` and 14 `F2F.F32.F64`, versus zero after fixing the literal.
The number of branches does not decrease with this change.

Both variants' results are numerically equal in the measurements.
The public dtype, product order and math flags are unchanged.
On already aligned sizes, where OpenCL removed those conditional loads,
no consistent gain appears. Declaring group size also provides no general
additional improvement: that variant is not promoted.

The fix is in `src/openpacho/_gemm.py`, shared by all three backends. On CUDA
the compiler already removed the conversions; on Vulkan both literals produce
exactly the same SPIR-V. We attribute no gains there to the suffix.

### Contraction, guards and backend differences

CUDA retains `fmad=false`; relinked loop PTX/SASS contains separate multiplication
and addition. The experimental `fmad=true` variant replaces 32 static `FMUL` and 32
`FADD` with 32 `FFMA`. The same irregular case goes from 3.487–3.629 ms with
float32 literals to 3.324–3.326 ms with contraction. Results **change
rounding**. This option remains exclusively diagnostic: FMA is not equivalent
to enabling all fast-math, and these examples establish no universal error
guarantees. See [NVRTC options](https://docs.nvidia.com/cuda/nvrtc/).

OpenCL generates FMA in the normal control. Disabling contraction shows the corresponding
change in PTX; the performance difference is small in this GEMM.
On Vulkan, `precise` adds 38 `NoContraction` decorations; removing those
decorations recovers exactly the control's SPIR-V. The noncontracting variant
costs 3.047–3.240 ms versus 2.704–2.741 ms for the control with float32
literals, but without Vulkan ISA we do not claim to have counted its native FMAs.

There is an important negative control: **Vulkan baseline and `float_literals` have
identical SPIR-V in all six pairs**. Even so, their irregular medians differ
by approximately 6–7 %. We do not interpret that difference as a shader
optimization; it shows that timing, execution and pipeline placement can also
introduce differences this experiment does not separate.

Removing guards on aligned sizes reduces intermediate CUDA/Vulkan branches,
but does not guarantee an improvement. On Vulkan the driver's register value rises from
60 to 73. The small gain at 1024³ does not reproduce at the larger aligned size.
We do not extrapolate effective occupancy from that counter alone. Compilers may
use predication and perform later transformations; a source `if`
does not necessarily correspond to a divergent ISA branch.

Thus float64 promotion explains a large and demonstrable part of the OpenCL gap.
Contraction policy explains another arithmetic difference and part
of the time. Instruction scheduling, unrolling, memory access and final
Vulkan native code remain insufficiently isolated. There is no evidence
that the remaining difference is a physical wall for OpenCL, CUDA or Vulkan.

## Copy implementation review

Kernels still copy uint32 words over float32 storage. This
preserves subnormals, signed zeros and NaN payloads, and writes `0x00000000` into
padding. Dimensions, limits, storage and metadata are checked on
every call. Batch shapes and cropping remain within buffers.

Selected changes:

1. A 128-entry LRU cache for validated geometry and the copy body.
   It retains only integer tuples and strings; no arrays, buffers or objects
   retained indirectly by `dtype.metadata`. Current values and metadata
   are checked even when geometry is already cached.
2. The compiler counts known physical submissions for padded GEMM:
   up to two copies, GEMM and cropping. CUDA can thus amortize the context for that
   step without changing kernels, tolerances, data or API. An aligned product
   retains the single-submission criterion. The eager path does not change its scope.

The host-preparation microbenchmark, nine rounds of 100,000 calls, gives:
`_body` 1.246 → 0.077 µs; `_validate` 2.321 → 2.246 µs. The second saving is
marginal. These times are not presented as GPU cost or complete-flow gain.

The public benchmark uses the previous wheel **only to load `_padding_copy.py`**;
all modes execute the corrected GEMM and current backends. `old_copy` and
`cached_copy` isolate preparation; `cached_copy` and `current` isolate CUDA
scope. On OpenCL/Vulkan the latter are equivalent controls, useful for
observing noise. `unpadded` is a different reference, not a cache ablation.
There are **144 configurations and 4,464 samples**, with six shapes, two rounds and 31
samples per mode. They include allocations, padding, GEMM, cropping and synchronization,
with resident inputs. All four outputs of each of the 36 groups are
bitwise identical; all meet tolerances against a float64 reference.
The maximum relative L2 error in this second suite is **8.106×10⁻⁷**, and absolute error
**4.093×10⁻⁵**. Special-value tests run separately from the finite-random-data
benchmark.

Representative results, medians of each round:

| Case | Before → current, round 1 | Before → current, round 2 | Interpretation |
| --- | ---: | ---: | --- |
| Vulkan 33×17×65 | 222.1 → 214.1 µs | 222.7 → 211.7 µs | Small preparation saving |
| Vulkan 1025×1023×1027 | 641.1 → 623.8 µs | 631.9 → 619.6 µs | Small, repeated saving |
| CUDA 33×17×65 | 214.0 → 200.1 µs | 414.6 → 410.6 µs | Lower cost, large variation between rounds |
| CUDA 2049×2047×2051 | 3770.3 → 3751.7 µs | 3672.7 → 3695.3 µs | Mixed; no demonstrated improvement |

The CUDA diagnostic reduces 12 push/pop pairs to 5, retaining four launches
in that instrumented scope. It does not claim the kernel becomes faster. OpenCL
cache results are small or mixed. Further prioritizing that microadjustment
over GEMM, submissions, temporaries or layouts with greater removable cost is unwarranted.

After fixing literals, large irregular OpenCL **without padding** costs
3.925–3.940 ms in the prepared executor, versus **4.110–4.112 ms with padding**.
This comparison includes copies. The previous padding advantage does not persist
in that case with the corrected baseline path. The option remains disabled by
default; CUDA remains mixed and Vulkan varies between rounds. Historical
[padding results](padding-results.md) remain as the previous snapshot.

Indiscriminate `uint4` vectorization is not introduced: odd widths change each
row's alignment. No necessary copy guard is removed either. A next experiment
with more room would write the logical output from GEMM to avoid the intermediate
buffer and cropping, checking the effect of stores with irregular stride.
It needs its own control and is not implemented here.

## Reproduce and continue

```sh
python -m benchmarks.isolated --worker kernel_inspection --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 21 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/inspection-new
python -m benchmarks.isolated --worker padding_copy_review --control-wheel <previous-wheel.whl> --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 31 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/copy-review-new
```

They require backend dependencies and a configured Vulkan compiler.
`PACHO_NVDISASM` is optional for relinking/disassembling CUDA PTX without changing the
runtime. Vulkan capture requires the introspection extension; its absence
produces an explicit error. Historical controls and new captures retain
their own hashes. Initial reports call numerical equality `exact_baseline_equality`;
the current worker distinguishes numerical and bitwise equality.

Pending: measure explicit math policies, access/unrolling variants
with accuracy controls, obtain Vulkan ISA evidence through vendor tools
when available, and validate other devices. `mean` with
a padded sum and logical divisor is noted as a future extension in
[padding](padding.md#future-extension-reductions), without implementation in this round.
Verified tests and distributions are recorded in [verification](verification.md).
