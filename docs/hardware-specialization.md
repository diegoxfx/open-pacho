# Capability-based specialization and the hardware track

This document sets rules for device and operating-system specialization.
The main track remains portable. Implemented variants are distinguished from
future family-specific profiles and autotuning.

## Implemented variants — 2026-09-13

- Vulkan detects instance/device extensions, enables properties2 and
  push_descriptor when available, and checks maxPushDescriptors.
  Within the limit it uses push descriptors; beyond it, descriptor sets.
  Selection does not use the NVIDIA name or change shaders, barriers, or
  precision. The portable alternative is retained and tested on the same machine.
- CUDA checks thread and shared-memory limits before using the tiled float32
  GEMM shared with OpenCL/Vulkan. NVRTC compiles for the device's compute
  capability. It does not enable TF32, fast-math, or new FMA contraction.
- OpenCL uses int32 indices for elementwise/gather when a conservative bound
  proves accesses fit, with an int64 path available. This is specialization by
  shape/layout, not by vendor or data precision.

Macros can express a variant within a kernel, but do not replace capability
queries/enabling or host-call selection. Program caches are local to the
backend/device; shared caches store only text/metadata. Future subgroup
instructions, FMA, or special matrix products must declare their requirements
and numerical semantics before entering the selector. A faster instruction does
not establish equivalent rounding, accumulation, or subnormal handling.

See [implementation and measurements](backend-sharing-results.md). Profiles by
family/OS and autotuning remain a secondary track.

## Work order

The main track keeps this order:

1. Vulkan execution: reduce execution and staging costs; reuse descriptors
   and batch submissions.
2. GEMV and map-reduce, with resident measurements and external applications
   (solvers in `examples/`, with explicit convergence criteria).
3. A prepared executor that reuses plans, memory, and metadata without hiding
   validation or synchronization.
4. Batched BLAS and custom kernels where measurements justify crossing a
   provider boundary.

Hardware and operating systems are a lower-priority track and must not hold
back the [current main track](roadmap.md).

## Selection rules

- Check actual capabilities first: workgroup limits, local memory, types, and
  extensions. Do not infer them from the device family name.
- If a capability is missing and a valid portable path exists, use it within
  the same backend and numerical contract. Otherwise, raise an explicit error.
  Never switch backends implicitly.
- Provider selection, such as BLAS versus a custom kernel, must be explicit,
  observable, and reversible.
- Separate optional tuning (tiles, subgroups, or OS-specific strategies) from
  workarounds required for correctness. Portable batching, pooling, and fusion
  belong to the main track. Restricting a workaround to hardware requires
  evidence of its scope; when the cause is unknown, retain general protection.
  Do not change dtype, tolerances, or checking frequency.

## Limits of the hardware evidence

RTX50 is a candidate family for repeating tests, not a portability conclusion.
Available local evidence is limited to RTX 5060 Ti, Windows 11, and driver 616.64.
PCIe Gen3 ×8 describes the host configuration and observed link; it is not a
property of the RTX50 family. A bug is not attributed to specific hardware
without a comparative reproduction that isolates it.

## Numerical cases that must remain separate

- Norm underflow with `A=eye(2)`, `b=[1e-30,-1e-30]` float32, and `rtol=atol=0`
  is a general numerical-range case. It was verified on three local backends:
  the float32 sum of squares returned zero, while the float64 check returned
  approximately `1.414e-30`. It is not presented as a GPU-specific failure.
- CLBlast 1.7.0 produced NaN in GEMV with `beta=0` when the recycled destination
  contained NaN/Inf; this was reproduced on NVIDIA under Windows. The causal
  scope remains unknown: the observation is not extended to every GPU, driver,
  BLAS, or version. The adapter already applies a general preventive fill, with
  tests: it will not be restricted to NVIDIA without evidence that this is safe.

## Advancement criterion

The [kernel and stability round](kernel-stability-results.md) incorporates
observed idle conditions, uniform warm-up, reversed order between processes, and
variant controls. It promotes only removal of OpenCL guards for full tiles;
padding, local layouts, and alternative blocks remain experimental when they
show no consistent gains. It includes the amortization criterion for global
packing/transposes and the memory cost before selection.

A specialization enters the main track only with detected capabilities, a
fallback within the same backend, an explicit provider, and a reproducible
comparison. Results must preserve shape, dtype, tolerances, residual, and
iterations; measuring an improvement in one case does not authorize a promise
for the entire family.

References: [gaps and optimization](gaps-and-optimization.md), [Vulkan backend](vulkan.md),
and [local diagnostics](../benchmarks/gaps.py).
