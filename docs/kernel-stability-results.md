# Kernel selection and benchmark stability — 2026-09-13

This round turns optimization proposals into measurable controls. It maintains
the priority of Vulkan execution/transfers and pending CUDA boundaries; it does
not replace that work with a general autotuner or modify solvers.

## Completion, idle conditions, and warm-up

These are three separate stages. Synchronizing the queue guarantees completion
of our application's work. Downloading results is included only in the end-to-end
timer; validation downloads occur outside the resident timer. A timed pause alone
does not guarantee completion of a GPU queue.

The new `benchmarks.isolated` runs one provider per process and waits for its
exit before checking GPU state. The worker already synchronizes all samples and
finishes its validation downloads. The next provider receives a minimum
10-second pause and an idle check with three consecutive readings, two seconds
apart: GPU/memory utilization ≤5%, temperature stable within 2 °C and, after the
first provider, no higher than its initial reference plus 2 °C. A maximum
temperature can also be set; 52 °C was used here.

UUID, PCI bus, temperature, utilization, frequencies, power, pstate, and clock
throttling reasons are recorded. An active thermal signal prevents continuation.
If the criterion is not met within 120 seconds, **the next worker does not start**.
Timeout is not confused with approval. Without telemetry, only
`--allow-unverified` permits a fixed pause explicitly labeled unverified; it does
not allow ignoring a previously observed hot or busy GPU.

The current collector uses `nvidia-smi` only in benchmarks. The runtime and its
extras remain independent. `wait_ready` accepts a reading provider so AMD/Intel
telemetry can be added later. On multi-GPU machines, users must associate the
observed GPU with the selected devices in each API: the runner does not
automatically verify that correspondence by UUID. Here all paths executed on
the single observed RTX 5060 Ti.

After idling, each variant receives at least three calls and 0.25 seconds of
warm-up, outside the timer. First compilation, idle time, and warm performance
are separated. Initial order is randomized and reversed in the second round.
The pause avoids carrying excessive work/temperature forward; it does not
guarantee constant clocks or absence of OS noise. Power or pstate reasons are
recorded without forcing clocks, fans, or power limits.

In the final variant control, all six idle periods met the criterion: five
lasted approximately 10.4 s and one required **16.6 s**. Temperatures in accepted
windows were 40–42 °C. Slow samples are not discarded to make a backend look
better. [NVIDIA telemetry definitions](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
explain that utilization and clock reasons are observations over sampling windows,
not certification of stability throughout the run.

## What is being optimized

The current GEMM already uses 32×32 output tiles, blocks of 64 threads, and shared
memory for two 32×16 tiles. Each thread accumulates **4×4 results in registers**:
it already implements assigning several elements to one thread.

| Technique | Criterion and cost to measure |
| --- | --- |
| Tiling | Reuse versus shared memory, registers, occupancy, and partial-tile cost. A larger tile does not imply higher performance. |
| More elements per thread | Reuse and instructions versus register pressure, spills, and parallelism. Varying per-thread blocks and tiles is a joint decision. |
| Vector types | Verify alignment, stride, tail accesses, and emitted code. `float4` can help loads/stores without implying four simultaneous arithmetic operations per thread. |
| Memory banks | Model reads and writes: fixing conflicts in one phase can worsen another. Reads from the same address can be broadcast; they are not necessarily conflicts. |
| Transposition/packing | Distinguish flags/views, rearrangement within the tile, and global copying. Measure preparation and reuse before materializing an operand. |
| Padding | Account for bytes and limits before allocation or launch; padding data and accesses must be valid. Separate local padding from expanding global matrices. |
| Removing conditions | Remove only provably redundant checks; preserve edge guards and uniform barrier participation. Measure compiler output, which can predicate or remove conditions. |

For a global copy reused R times, break-even requires
`T_preparation + R*T_prepared_kernel < R*T_original_kernel`. Selection must
include extra memory, relevant uploads/readback, and validity of preparation
under mutation. No automatic global copy was implemented in this round.

OpenCL defines `float2/4/8/16` as vectors of float values; `float16` does not mean
FP16. Their existence in the language does not guarantee the same machine
instruction or an advantage in another API. See [OpenCL vector types](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#vector-data-types).
The [CUDA shared-memory and register guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)
details conflicts, broadcasts, and resource costs; physical banks and generated
code must be verified on the corresponding hardware.

## Controlled experiment

`benchmarks.kernel_variants` measures five float32 shapes: 32×16×32, 33×17×65,
257×129×513, 1025×513×769, and 1024³. Ten variants per shape, three backends, and
two rounds produce **300 configurations**, with 21 samples per variant. All pass
numerical checks and preserve the 16 sentinel values at the end of the output.
Output is initialized with NaN to detect omitted writes.

Comparisons cover the current kernel, a generated baseline without batch indices,
left/both padding, local transposition with/without padding, a 2×2 register block,
a K=32 tile, and edge specialization. Two edge controls separate the experimental
generator change from the actual runtime generator change.

Variants declare 2,048–8,192 bytes of shared memory and check device limits.
The logical data budget is checked before allocating matrices. This does not
measure actual peak VRAM or memory used by the driver, compiler, or allocator
caches. No operand is globally padded or transposed.

The timer measures launch and synchronization with preallocated output:
**it is not equivalent to a full public call or pure kernel time**. Downloads,
checks, compilation, warm-up, and telemetry are excluded. Errors are compared
against represented float32 inputs, recomputed in NumPy float64. This is not
an adversarial accuracy test or coverage of every memory pattern.

## Selecting the optimization

Removing only redundant conditions in the real generator for 1024³ gives:

| Backend | Guarded → full tiles, round 1 (ms) | Round 2 (ms) |
| --- | ---: | ---: |
| OpenCL | 0.790 → 0.477 | 0.778 → 0.465 |
| CUDA custom | 0.433 → 0.429 | 0.433 → 0.430 |
| Vulkan | 0.473 → 0.468 | 0.491 → 0.478 |

Only the OpenCL variant is promoted, where savings are large and repeated.
Automatic selection requires eligible float32 GEMM and **M/N multiples of 32,
K a multiple of 16**. It preserves tiles, memory, arithmetic order, and batch
support. Other shapes retain guards, and CUDA/Vulkan retain their shaders.
No user flag is required, and dtype or mathematical modes do not change.

Padding/transposition variants are not enabled indiscriminately: several worsen
large Vulkan/CUDA GEMMs. A 2×2 per-thread block also does not offset its lost
reuse in those cases. A small indexing-code change can also change what each
compiler optimizes; hence the additional generated baseline. Full results,
including losing variants, remain in the JSON files.

The subsequent `a @ b` control, with output allocation and the same public
paths, ran **88 configurations** (22 cases × two variants × two rounds), without
errors. It reuses the same backend and alternates only the previous/current
GEMM component extracted from the historical wheel. Each pair's numerical metrics
agree. Unchanged operations are noise controls.

| OpenCL float32 | Previous → current resident, round 1 / round 2 (ms) | Full, round 1 / round 2 (ms) |
| --- | ---: | ---: |
| 512×256×768 | 0.156 → 0.129 / 0.159 → 0.128 | 0.988 → 0.947 / 0.980 → 1.012 |
| 1024³ | 0.837 → 0.534 / 0.843 → 0.534 | 3.245 → 2.899 / 3.183 → 2.819 |
| 1025×513×769, guards retained | 0.400 → 0.402 / 0.402 → 0.401 | 1.845 → 1.848 / 1.939 → 1.929 |

1024³ improves approximately **36–37% in resident wall time and 11% in full
execution**. For 512×256×768 resident savings are 17–20%, but full execution is
mixed. Selection keeps the same GPU memory cost and does not eliminate transfers.
A kernel improvement is insufficient to promise an application improvement on
every small call or workload with little computation per byte.

## Reproduction

From the repository, with extras, drivers, and compilers for the selected backends:

```powershell
python -m benchmarks.isolated --worker kernel_variants --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 21 --max-temperature 52 --output .artifacts/new-variants
python -m benchmarks.isolated --backends opencl --control-wheel <previous-wheel> --control-component gemm --rounds 2 --repeats 21 --output .artifacts/new-public-control
```

Use new directories and sequential GPU workloads. Wheel GEMM controls temporarily
alternate the previous/current generator in the same process and context,
outside timers; they do not allow concurrent runner use in that process.
Substitutions are restored even if execution fails. They do not alter the API
used by the application.

Artifacts: `.artifacts/kernel-stability/variants-final`, `public-control`, and
logs. The first `variants` folder was exploratory and precedes the additional
generator control; it does not replace the final round. Local hardware: RTX
5060 Ti, Windows 11, driver 616.64. Conclusions do not certify another vendor,
driver, or OS.

Final verification: **1,122 tests pass and one is skipped** because it needs two
OpenCL devices. Includes all three required backends and native CLBlast,
aligned/partial edge controls, and batch broadcasting. The **42 adversarial
OpenCL cases** retain the same nine earlier warnings, without exceptions;
cancellation issues are not claimed to be resolved. That adversarial suite was
not repeated for unchanged default CUDA/Vulkan shaders. After adjusting observed
GPU propagation from parent to worker, all **25 protocol and control tests** pass.
This keeps requested telemetry consistent; it does not add automatic association
with each API's device.

Wheel from sdist: **327 CPU tests pass, 727 skipped**, with import from the wheel
and GPU/example modules blocked in the independence check. Separate extras,
NumPy as the base dependency, and exclusion of examples/solvers/benchmarks from
the wheel were verified. Ruff and formatting pass (88 files); Python 3.10 syntax
parsed, execution on Python 3.12. Only `_gemm.py` changed in the runtime relative
to the previous wheel. All 12 saved-source controls are identical to the previous
ones. No new version was published.
