# Scope and priorities

OpenPacho is a library of GPU matrix operations with a limited NumPy API,
an explicit backend and optional compilation. Solvers, prepared systems and future
network/simulation applications live in `examples/`, outside the installable package.
The version remains **0.1.0a1**, with no PyPI publication.

The [agreed architectural direction](architecture-evolution.md) defines contracts,
accuracy and measurement rules. The [Vulkan dispatch round](vulkan-dispatch-results.md) adds a typed
CFFI boundary, an alternating control and timestamp diagnostics. It preserves kernels
and algorithms. The [cross-backend round](backend-sharing-results.md) adds capability-based Vulkan
push descriptors, OpenCL caching/indices and CUDA batches/tiles;
there is still room in preparation/submission and it does not imply parity with
OpenCL/CuPy. The [initial CUDA round](cuda-initial-results.md) retains its snapshot.

## Main line

The [stability and kernel round](kernel-stability-results.md) adds observed idle time
between processes and promotes OpenCL guard specialization on full
tiles. Other variants retain experimental status; the following execution gaps
retain their priority.

The [staging and random round](staging-random-results.md) retains Vulkan mappings
only for readback after rejecting general persistence as the default due to
mixed results. P2 adds a resident RNG. Submissions, waits
and command grouping remain the main priority.

The [routing and completion round](routing-completion-results.md) removes duplicate
RNG dispatch and resolves lowering/execution scope when preparing
plans. GEMM/GEMV/norms still contain name-based routes: their migration will be gradual,
when those operations are extended or measured, preserving shared arithmetic.

The [CUDA fusion round](cuda-fusion-results.md) shifts focus from Vulkan waits
to eliminating launches and full temporaries on CUDA. It implements elementwise
and map-reduce through the backend factory without changing domain algorithms.

The [CUDA argument preparation round](cuda-dispatch-results.md) reuses
host storage per kernel. Resident savings repeat and isolated packing
already costs less than a microsecond in the local control. The next
investigation moves beyond that substep: context/submission, binding and matrix operations
offer more remaining room than further tuning the same preparation. The
[three-optimization round](three-actions-results.md) implements shared positional
binding, nested CUDA context with selective scope and cooperative custom GEMV/DOT.
It retains measurements separated by change and acknowledges mixed results.

The following order is a set of areas, not a permanent preference for
an API. Iteration selection considers usage frequency, removable cost,
complete/resident gain, repeatability, accuracy and effort. If an adjustment gives only
small or mixed improvements across several rounds, another with greater potential
savings takes priority. The adjustment remains pending; it does not mean the backend
has exhausted its opportunities. Correctness regressions retain priority over performance.

[Kernel inspection](kernel-inspection-results.md) removes accidental
float64 conversions in OpenCL GEMM. Subsequent variants must start from
that corrected baseline: do not extrapolate the old padding advantage. Copy preparation
already offers marginal savings; prioritize eliminating temporaries/submissions and
improving arithmetic/layouts before further tuning those caches.

The [padding reassessment](padding-reassessment.md) retains the option for
reproduced gains on some shapes with few rows and long K. Most
regress; expanding padding has low priority. Prioritize rectangular GEMM without
extra copies and study CUDA accumulation with long K, where elements
fall outside tolerance both with and without padding.

The [next three iterations](three-iterations-results.md) add rectangular tiles
without copies on OpenCL/CUDA, selected cuBLAS vector primitives
and transpose flags for private temporaries compiled with CLBlast.
Earlier padding numbers are historical: a faster direct path
changes its break-even point. Those advantages are not extrapolated to the new runtime.

1. **Vulkan execution and observability:** decompose Python/CFFI preparation,
   driver, GPU, copies and waits. Improve commands/descriptors/staging and memory
   according to evidence. A small native executor is evaluated with an equivalent control
   before deciding on a complete adapter.
2. **Competitive custom kernels:** GEMM, GEMV, DOT, reductions and common compositions;
   a continuing goal of approaching or exceeding CLBlast on a representative
   suite. Retain CLBlast/cuBLAS by contract and mixed results.
   CUDA requires expanding measured cuBLAS primitive selection, tiling and reducing
   preparation/launch around the already available fused lowerer;
   immediate parity with more advanced backends is not required.
3. **Layouts and plans:** extend the BLAS flags already available for CLBlast; materialization according to
   consumer, reuse and memory. Reduce intermediates, binding and submissions
   without changing domain code or comparator algorithms. Also measure
   repeated work when fusing small operations with large broadcasting.
4. **Internal contracts:** gradually replace backend-name decisions
   with operation capabilities, layouts and numerical policies. Resolve routes
   when preparing plans and bind direct calls to contain abstraction costs.
5. **Applications and accuracy:** extend covariance/ridge with simple dense networks
   and simulations; keep measuring rectangular, odd and non-power-of-two shapes.
   Retain solver benchmarks to convergence as external clients.
   Publish error, preparation, first call, memory and resident/complete time.

These lines are prioritized using profiles; a verified general regression precedes
new optimizations. Improving the executor does not authorize changing tolerances,
check frequency or specializing by a solver's name.

## Secondary and future tracks

- **P2, random generation:** [first resident API](random.md) for uniform,
  normal and exponential via Philox4x32-10. Pending: unbiased integer distribution,
  spawn/checkpoint, extensive statistical batteries and using multiple
  outputs per thread without breaking the counter contract. It does not displace Vulkan
  execution or matrix kernels. Randomness in compile requires modeling effects.

- **Limited autotuning:** opt-in/explicit budget, candidate validation,
  cache invalidatable by device/driver/kernel/shape/layout/dtype/math mode.
  Respect the explicit provider and retain a portable path without calibration.
- **Reduction padding:** `mean` can sum with zeros and divide by the logical size
  of reduced axes. Measure separate division versus final fusion;
  preserve dtype, axes, empty arrays and nonfinite values. [Pending, not implemented](padding.md#future-extension-reductions).
- **Hardware/OS:** [family specialization](hardware-specialization.md) when
  locally verifiable. Separate optional optimizations from necessary workarounds
  for verified bugs; do not extrapolate to whole vendors.
- **Mathematical policies:** fast modes and more robust accumulation, with per-operation
  guarantees and explicit FMA/reassociation/subnormal/reduced-precision behavior.
  They participate in cache keys and every comparison; there are no public flags yet.
  The [GEMM accumulation investigation](gemm-accumulation-results.md) retains
  blocked and compensated variants as experiments: they improve accuracy in
  different cases, with different costs and limits. The default is unchanged.
- **Community portability:** verified hardware/driver/OS matrix. Prepare
  reproducible reports; physical AMD/Intel hardware depends on additional equipment or
  contributions. CPU/PoCL CI and packaging remain verifiable by the project.
- **Interoperability:** sharing storage between providers/contexts or moving
  directly between APIs has low priority. Backend choice is
  explicit; automatic migrations are not introduced.
- **Sparse operations, mutable views and full NumPy compatibility:** outside this stage.
  Autodiff, solvers as a library API and domain frameworks are not goals.

## Providers and installation

NumPy is the only mandatory dependency. OpenCL, Vulkan and CUDA are independent
extras; an OpenCL user does not install CUDA. Missing capabilities
produce a valid alternative within the selected backend/contract, or an explicit
error. Do not silently change dtype, accuracy or backend.

CLBlast already has batched/strided-batched GEMM and GEMV/DOT when symbols are
available; custom OpenCL/Vulkan already uses local/register tiles. CUDA has
memory, NVRTC, cuBLAS, batches, custom tiled float32 GEMM, elementwise/map-reduce fusion
and stable eager/compiled float32/64 norms for both BLAS providers. The common API
also includes resident `where(condition, x, y)` selection with three-argument
broadcasting and composition with elementwise/map-reduce fusion; graphs remain
pending. See the [selection results](selection-results.md). Historical
reports describe each stage and do not replace the [current contract](compatibility.md).

The [extrema iteration](extrema-results.md) adds resident min/max along axes,
dtype preservation and compiled composition. Initial softmax
and logsumexp recipes remain in `examples`, as clients of those primitives. They do not
add autodiff or a network API to the library.

[Materialized extrema preparation](extrema-planning-results.md) promotes
the existing cooperative path for compiled float32 min/max on OpenCL/CUDA.
Two rounds reproduce the resident improvement and show uneven complete-flow gains;
Vulkan retains its path because of repeated regressions.

The [dense-network client](dense-network-results.md) already exercises inference and
explicit gradients using existing primitives. It retains NumPy, CuPy and
native providers, small/regular/odd cases and explicit errors.
The CUDA custom float64 gap justifies studying its direct GEMM after
separating that cost from the remaining flow; it does not add a network API to the runtime.

The [float64 GEMM experiment](cuda-fp64-gemm-results.md) confirms GEMM's
weight but rejects promoting a shared 16×16 tile: two rounds do not improve
the flow consistently and repeat narrow-shape regressions.

The [float64 contraction experiment](fp64-contraction-results.md) isolates FMA
within each geometry and compiler, preserving the runtime's default policies.
The next priority is to reproduce its effect on complete CUDA custom inference and SGD,
with explicit experimental selection and an error budget.
Only a repeated applied gain would justify designing a public opt-in
policy;
the microbenchmark does not promote a tile or demonstrate greater accuracy in every case.

[Coupled oscillators](oscillators-results.md) add a fixed-step client
with repeated GEMV and a reusable compiled step. The modal trajectory
and energy distinguish integrator error from backend differences. The
model remains in examples without expanding the public API with physics or GPU loops.
Its times justify, as an independent area, decomposing dispatch,
allocations and waits in the prepared executor before optimizing them.

CuPy/cuBLAS/nvmath are performance and accuracy references; PyTorch/JAX are
added when a comparable workload justifies the benchmark dependency.
They are not mandatory dependencies and their time is not used as a universal physical ceiling.

## Publication and documentation

Before publication: review API, name/version, wheel/sdist, errors for missing
capabilities and the validated matrix. Announce API stability separately from experimental
support for unverified devices; do not promise universal validation.
The [release estimate](release-readiness.md) reserves 2–4 hardening iterations
for a limited candidate, followed by external validation before
declaring it stable; it does not require performance parity with CuPy.

For large changes, an economical read-only audit identifies affected documents;
the main agent integrates and verifies. Keep README,
compatibility, capabilities and installation consistent while retaining historical results.
See [local verification](verification.md) and [next applications](application-measurement-plan.md).
