# Agreed architectural direction — 2026-09-12

OpenPacho is a GPU array and matrix operations library with an API
close to NumPy and explicit backend selection. Domain algorithms are
applications: solvers, neural networks, and simulations belong in examples and
benchmarks. This decision defines the product's boundaries; it does not require abandoning their tests.

## Status of this revision

- Implemented: solvers and prepared systems moved to `examples/solvers.py`,
  outside the wheel and package exports. Examples use the public API;
  `get_backend(array)` obtains their existing instance without transfers.
- Already present: NumPy as the only mandatory dependency; independent
  `opencl`, `vulkan`, and `cuda` extras, with deferred GPU runtime imports.
- The remainder of this document defines future work. No new capability,
  layout, math-mode, or autotuning contracts, or native Vulkan executor, have
  been implemented here. Historical results retain their original code.

The [later Vulkan dispatch iteration](vulkan-dispatch-results.md) implements
a typed CFFI table and separate diagnostics. There is still no native C/C++
group recorder; this does not change the pending status of layouts/modes/autotuning.
The [cross-backend round](backend-sharing-results.md) adds capability-selected
variants and shares tiled GEMM with CUDA. It still does not constitute the general
capability contract proposed here.

The [routing and completion round](routing-completion-results.md) implements two
bounded interfaces: backend-specific RNG adapters and an optional lowering
factory, resolved when preparing the plan. This is still neither a general
layout/capability protocol nor a public plugin mechanism.

## Separation and abstraction cost

The existing separation evolves without adding a chain of proxies per
operation or replacing it with a general Python compiler:

1. **Semantic facade:** operations, shapes, broadcasting, promotion, and errors.
2. **Optional planning:** IR for generic operations, layouts, uses, fusion,
   and materialization decisions. Eager remains an independent path.
3. **Internal contracts:** a typed primitive protocol and immutable capability
   description queried when creating a context/specializing a plan. Include
   dtype, limits, layouts, and numerical guarantees per operation; a Boolean
   “supports float64” does not describe all its functions.
4. **Backend execution:** buffers, lifetimes, queues, kernels, and providers.
   Resolve implementations when preparing the plan and bind calls directly;
   retain only the validation needed per call. Do not infer capabilities from
   vendor names or repeat dynamic negotiation per element/dispatch.

There is no “solver” primitive in the contract. A backend enables an example
when it has the required operations. CUDA already has stable float32/64 norms
for eager and compiled execution with custom or cuBLAS providers; the
algorithms remain in examples. The documentation matrix must be derived/checked
against those contracts and tests, without equating “implemented” with “validated on all
devices.”

Context/storage and BLAS provider are distinct responsibilities. Keeping
a clear boundary facilitates kernels and tests; sharing arrays between provider
instances or transferring directly between APIs is **low priority**. No automatic
backend migration is introduced, and OpenCL/Vulkan/CUDA interoperability need not
be resolved to optimize each backend.

## Custom kernels, providers, and layouts

Competitive custom kernels for GEMM, GEMV, DOT, reductions, and their compositions
are ongoing core work. CLBlast/cuBLAS are retained as providers and
references; winning in a fused solver does not demonstrate a win in isolated GEMM.
Evaluate precision, memory, and the size distribution, including rectangular,
odd, and non-power-of-two sizes. “Majority” requires a predefined suite and published
regressions/mixed cases, rather than counting only favorable shapes.

Begin layouts with logical 2D transposes and BLAS flags, with kernels able
to consume them. Choose packing/materialization according to the consumer, reuse,
and memory: for k uses, compare `T_pack + k*T_packed` with `k*T_transposed`.
Include packing in complete time and peak memory; do not hide it in preparation.
A copy may also be needed if the consumer does not accept that layout:
use a valid alternative or materialize explicitly, without promising that copying
never occurs. This does not yet extend to mutable views/full NumPy aliasing.

## Vulkan: diagnostics before replacing the adapter

The measured gap does not establish a theoretical Vulkan barrier. It mixes
Python/CFFI preparation, internal management, driver calls, GPU work, and waits. The
already verified pool improvement demonstrates implementation headroom; it does not
quantify the remaining percentage attributable to the Python binding.

Priority: measure host time, native calls, and GPU timestamps separately;
count allocations, descriptors, submissions, bytes, and memory. Do not call wall
time minus kernel duration “Python time”: queues, the driver, and intermediate waits are involved.
Khronos documents [command-buffer reuse and cost](https://docs.vulkan.org/samples/latest/samples/performance/command_buffer_usage/README.html)
and [host synchronization responsibilities](https://docs.vulkan.org/guide/latest/threading.html).

If the profile warrants it, build a small native executor for commands,
descriptors, pools, and submission that consumes the same plan. Compare it with the current
adapter using the same kernels, inputs, barriers, precision, memory policy, and
outputs. Also publish latency, errors, and packaging costs per OS.
Retain protection for resources in flight and locks where appropriate. Extend it
only with reproducible gains; a complete custom binding is not a prerequisite.
Neither this path nor Vulkan promises to overcome driver/hardware limitations.

## Integrity of examples and comparisons

Improvements must be activated through ordinary operations and documented public
APIs (optional `compile`), within the compiler/backend. Do not recognize
algorithm names or change check frequencies, tolerances, or data to
favor OpenPacho. Existing compile/prepare phases in solvers are
retained and declared; they are not presented as direct execution without adaptation.

An algorithm improvement requires justification and a separate comparison with
equivalent references. Compare eager/compiled, first execution, preparation,
resident, and end-to-end execution, with the same work or convergence and comparable error.
CuPy speed is a practical reference, not a demonstrated physical limit.
Matching it would be valuable for portability, backend selection, and integration;
CuPy also offers a high-level array interface.

The scope does not include autodiff, distributed training, or all
[JAX transformations](https://docs.jax.dev/en/latest/key-concepts.html).
There is overlap in array operations and compilation; comparing those workloads
can be useful without making JAX parity a product goal.

## Math modes and autotuning: future tracks

Preserve the current contract by default and design explicit policies for
fast arithmetic and more robust accumulation. Specify per operation: input/accumulator/output
dtype, FMA contraction, reassociation, transcendental approximations,
subnormals, NaN/Inf, signed zeros, and permission for reduced
precision. An “accurate” policy promises neither bitwise reproducibility nor representable
results when they do not fit the dtype. Reject unsupported guarantees.

FMA computes multiplication plus addition with a single rounding and can improve both
accuracy and speed. MAD is not a portable guarantee equivalent to correctly
rounded FMA; see [OpenCL C math functions](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#math-functions).
Do not treat “disable FMA” as a universal synonym for accuracy. Future policies
must participate in kernel/plan/tuning keys and benchmark metadata;
do not compare fast-math with conservative modes as though they had the same contract.

Autotuning is a secondary track, with explicit activation/budget, a bounded
search, and a portable path without calibration. Select among correct variants of
tiles, groups, vectorization, packing, and fusion according to measurements. Cache by
device, driver/compiler, kernel version, operation, shape/layout, dtype,
and math policy; invalidate on changes and do not extrapolate a local winner to
all AMD/NVIDIA/Intel hardware. Separate tuning cost from sustained execution. If the user
fixes a provider, tuning respects that selection.

## Validation and expansion

The first release must state its validated matrix precisely. Community
contributions for AMD/Intel or other OSes do not prevent testing the CPU wheel, extras
isolation, and reproducible CI now. Do not confuse PoCL CPU conformance with GPU validation.
A stable API release may still have experimental hardware support.

Add dense networks and physical simulations as example workloads according to the
[application plan](application-measurement-plan.md), with errors and memory as well
as timings. Do not add domain frameworks to the package or impose them as dependencies.

For major changes, use a bounded, read-only documentation audit
with an economical subagent (available Luna/Spark): identify files,
claims, and links needing changes. The main agent decides, edits, and
verifies. Historical reports are retained as dated snapshots; the audit
creates no permanent automation and does not replace tests.
