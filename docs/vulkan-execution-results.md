# Vulkan execution and transfers: continuing the main priority

Historical snapshot. The subsequent stage is in [map-reduce, preparation and batched BLAS](remaining-suite-results.md).

2026-09-06. This round advances the **first block** of the
[work order](roadmap.md): Vulkan execution, staging and resource reuse.
Solvers/map-reduce, prepared execution and batched BLAS retain their order.
[Hardware/OS specialization](hardware-specialization.md) remains a lower-priority
design; no family selector or NVIDIA variant was added.

## General changes implemented

1. **Upload without intermediate serialization.** The array is copied directly to
   the staging mapping; `data.tobytes()`, its allocation and extra copy are removed.
   Owned storage, dtypes, bool and noncontiguous inputs are preserved.
2. **Readback with a preference for CPU caching.** Among memory types compatible
   with the buffer, HOST_VISIBLE + HOST_COHERENT + HOST_CACHED is preferred.
   If none exists, a compatible coherent option is retained.
   “Host visible” is not confused with “fast to read from the CPU.”
3. **Pool separated by use.** Upload staging and readback have separate bins;
   recycling a buffer does not erase the memory preference. The shared limit
   remains 128 MiB of actual allocation and 128 free buffers.
4. **Reusable descriptor sets.** They are recycled only after GPU completion.
   All their bindings are updated before reuse. Layouts live until backend
   closure and survive pipeline eviction; sets for in-flight commands remain distinct.
5. **Grouped submission per compiled call.** A plan's command buffers are
   submitted together. Eager execution retains immediate submission. Synchronization,
   pipeline eviction and the 64-command limit can split the group.
   The lock covers group recording; recorded resources not yet submitted
   also retain strong references. The group is flushed on exit, including
   on an exception. This does not introduce global lazy execution or change solver
   convergence checks.

Command buffers are still recorded and structures built per operation.
Persistent command graphs, a native executor and Vulkan map-reduce were not
implemented. Grouping reduces submission calls but does not remove the
backend's remaining overhead.

The memory preference uses standard properties, not GPU names.
HOST_COHERENT avoids requiring manual host flush/invalidate; barriers and the
wait before reading remain. [Vulkan memory properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryPropertyFlagBits.html).
Layouts and sets are retained for the lifetime required by
[the descriptor contract](https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html).

## Transfers: before and after

RTX 5060 Ti, Windows 11, driver 616.64. Eleven warm samples, synchronized wall-time
median. The public API is measured, including allocation/pooling, map/unmap,
CPU copies and transfer; **this is not an isolated PCIe measurement**. Each result
is validated exactly, outside the clock. The 256 MiB buffers exceed the pool
limit and are allocated again. Before/after processes are independent and do
not lock frequencies; samples and quartiles are in the JSON files.

| Size | Upload before → after | Download before → after |
| --- | ---: | ---: |
| 256 bytes | 0.199 → 0.200 ms | 0.231 → 0.258 ms |
| 4 MiB | 1.873 → 1.267 ms | 17.871 → 1.528 ms |
| 64 MiB | 26.746 → 15.874 ms | 276.144 → 26.747 ms |
| 256 MiB | 122.536 → 105.615 ms | 1154.668 → 110.950 ms |

Large downloads improve approximately tenfold. The few-byte case does not
improve: latency dominates there. This improvement comes from runtime memory
selection and copies; it does not demonstrate that the PCIe link became faster
or that every GPU or OS will achieve the same factor.

## Resident control

The `benchmarks.gaps` diagnostic profile on the same machine compares the
frozen wheel from the previous stage against the current runtime. Seven samples,
resident data and synchronization before/after; cProfile runs separately.

| float32 computation | Before | After |
| --- | ---: | ---: |
| GEMV2048 | 0.366 ms | 0.315 ms |
| DOT2048 | 0.324 ms | 0.292 ms |
| Compiled energy 2048×32 | 0.611 ms | 0.541 ms |
| Diagnostic CG, 2 iterations | 13.522 ms | 11.107 ms |

These wall-time improvements are more modest than the large-download gains.
Each microsecond is not attributed to an individual change: this comparison groups
staging, descriptors and submissions. Diagnostic CG does not use the same input
as the complete solvers in `benchmarks.execution`.

## Verification and reproduction

Another **319 configurations** of operations/steps, solvers and size/batch
controls were repeated: seven rounds per mode, with no validation errors.
The complete tables (`.artifacts/benchmarks/vulkan-execution/report/all-results.md`)
and CSV (`.artifacts/benchmarks/vulkan-execution/report/all-results.csv`),
local artifacts not distributed, preserve
all implementations, including NumPy, OpenCL, CuPy, cuBLAS and nvmath.
Vulkan comparison against the previous gaps snapshot, float32 milliseconds:

| Case | Resident before → after | Upload+computation+download before → after |
| --- | ---: | ---: |
| Matmul2048² | 11.905 → 12.069 | 92.240 → 24.017 |
| Complete Jacobi2048 | 82.502 → 66.667 | 88.606 → 69.264 |
| Steepest descent2048 | 100.486 → 84.218 | 106.417 → 88.690 |
| Complete CG2048 | 49.800 → 42.927 | 54.800 → 46.353 |
| 100 updates of 1,048,576 elements | 72.459 → 79.128 | 94.944 → 78.696 |

The last row remains mixed: one hundred independent compiled calls do not
form a single submission or GPU loop. Resident mode did not improve in that measurement.
In the current round Jacobi2048 gives 10.120 ms on NumPy and 12.740 ms on OpenCL versus
66.667 ms on Vulkan: the execution gap remains large. This transfer improvement
does not change the priority order.

- Full suite: **530 pass, 1 skipped** because it requires two OpenCL devices.
- **118 Vulkan tests** included in that suite: transfers, types, fusion,
  solvers, limits, in-flight resources, set survival during pipeline eviction,
  grouped submission, synchronization and exceptions. Memory selection also
  has a CPU test for requirements, preferences and fallback.
- Physical validation on AMD/Intel and other systems is still missing.
- Wheel installed without GPU dependencies: **194 pass, 320 skipped**. Ruff and
  formatting pass; wheel/sdist generated in `.artifacts/dist-vulkan-execution`.

Private artifacts in `.artifacts/benchmarks/vulkan-execution`: the
`before-profile.json`/`after-profile.json` profiles and
`before-transfers.json`/`after-transfers.json` transfers retain the samples.
Source and its hash, device/vendor IDs and the raw Vulkan driver version
are included in the transfer measurement.

```powershell
$env:PACHO_VULKAN_COMPILER='<path to glslang or glslc>'
python -m benchmarks.vulkan_transfers --output .artifacts/benchmarks/vulkan-execution/after-transfers.json
python -m benchmarks.gaps --backends vulkan --output .artifacts/benchmarks/vulkan-execution/after-profile.json
```

To repeat the broad suite, also configure CLBlast, PyOpenCL and the
CUDA references as in the gaps report:

```powershell
python -m benchmarks.execution --output .artifacts/benchmarks/vulkan-execution/operations.json
python -m benchmarks.execution --solvers-only --output .artifacts/benchmarks/vulkan-execution/solvers.json
python -m benchmarks.scale --output .artifacts/benchmarks/vulkan-execution/scale.json
python -m benchmarks.gaps_tables --root .artifacts/benchmarks/vulkan-execution --operations operations.json --solvers solvers.json --scale scale.json --output .artifacts/benchmarks/vulkan-execution/report
```

To measure the previous state, use the frozen wheel through `PYTHONPATH` in
another process. Do not mix other GPU workloads into the measurement. The benchmark
is called Vulkan transfers because it studies a general path; it does not enable
hardware-family profiles in the runtime.
