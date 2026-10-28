# Architecture

The [accepted architecture](execution-architecture-proposal.md) distinguishes the
future goal from the subset implemented here. See also
[compilation](compilation.md), [Vulkan](vulkan.md), and the
[gap and optimization analysis](gaps-and-optimization.md). The [agreed direction](architecture-evolution.md)
separates future decisions from current code. Solvers and prepared systems live
in `examples/solvers.py`, outside the matrix-operations package.

## Contract

`src/openpacho/_array.py` validates shapes and dtypes, implements NumPy protocols, and
delegates operations to a backend. Each ndarray has a shape, dtype, and contiguous
storage owned by its backend. Its metadata requires no GPU transfers.

`backends.py` maintains selection through ContextVar, so nested contexts
are restored even on errors. It does not initialize a GPU on import. Backends
are reused by device ID; arrays from distinct contexts are not combined
implicitly. `to_backend` is an explicit copy through the host.

`random.py` reserves counters from a backend-bound generator; `_random.py`
shares Philox4x32-10 arithmetic without knowing backend names. The adapters
`_random_opencl.py`, `_random_vulkan.py`, and `_random_cuda.py` own the dialect,
cached templates, and launch; each backend binds its implementation. The
`random_distributions(dtype)` query declares support before reserving the stream.
The GPU generates directly into the output, with seed/counter as dynamic scalars.
Stateful RNG is rejected during capture; see the [contract](random.md).

`_cuda.py` incorporates the initial CUDA backend: it retains the primary context,
uses its own stream, restores the caller's context, and retains buffers in flight.
NVRTC executes custom primitives; GEMM/GEMV/dot support cuBLAS or a direct kernel.
It uses neither CuPy arrays nor CPU fallback. `compile` binds the dedicated
`CUDALowering` through `create_lowering()` to fuse elementwise/map-reduce groups. It supports
broadcast batches: cuBLAS uses strided-batched for regular batches
and GEMM per batch for general broadcasting; custom executes batches
in one kernel and shares the tiled float32 generator with OpenCL/Vulkan.
With cuBLAS, DOT uses the native primitive with a resident result, and a vector
against batched matrices uses GEMV with contiguous offsets. Unbatched vector
products retain GEMM according to the measured comparison. DOT pointer mode is
restored under a lock; restoration failure requires recovery before submitting
another product. It does not change the provider chosen by the user.
`_cuda_gemv.py` cooperates across 128 threads per row for products with a right-hand
vector and K >= 256, including contiguous batches. Long DOT reduces partials on the device.
It changes neither dtype nor mathematical flags; the summation tree differs from the serial kernel.
The execution scope holds context/lock for plans with at least two known physical steps and avoids
redundant nested push/pop, checking the thread's actual context to respect other
libraries. Operations retain their synchronization boundaries.
Each compiled CUDA module retains a `ctypes` argument vector with stable host
storage. The backend rebinds it on each dispatch under the existing lock and
copies only numerical values and device addresses; the kernel template retains
neither arrays nor caller scalars. Allocations remain in `_pending` until
`synchronize()`. The 128-variant LRU and `close()`, after synchronization,
release the module and vector. This optimization changes no kernels, math,
contexts, or stream; see [CUDA dispatch results](cuda-dispatch-results.md).
Matmul and BLAS remain separate steps. Stable float32/64 norms
are supported in eager and compiled execution, regardless
of whether the selected BLAS provider is custom or cuBLAS. Boolean inputs can
be reduced with an explicit float32/64 accumulator; Boolean accumulation and CUDA graphs
remain pending. Solver algorithms remain in examples, outside the runtime;
the [CUDA contract](cuda.md) details precision and optional dependencies.

`_opencl.py` creates a context and an in-order queue per device and BLAS
configuration. Buffers are reused through MemoryPool. Kernels compile under
OpenCL C 1.2 and enable fp64 only
when the device offers it. They use no vendor-specific extensions
or floating-point atomics. Cooperative kernels choose a work-group size
compatible with device limits.

Elementwise/gather reuse immutable metadata with a bounded LRU, without retaining
values or arrays. They use int32 index arithmetic if every intermediate access fits,
retaining int64 for larger shapes/offsets. This is independent of the stored
dtype. Eager transposes and those exposed as independent results
remain materialized; a compiled product can omit a private transpose
through the layout capability described below.

There are kernels for elementwise operations with broadcasting indices,
gather for slicing/transposition, reductions, and vector/matrix products, including
batches with broadcasting of leading dimensions. Large eager full reductions
use partials in blocks of 256 elements; other eager reductions
assign one output element per work-item. The `blas="custom"` path cooperates
across work-items for sufficiently large right-vector GEMV and DOT;
long DOT uses partials. Eligible float32 GEMM combines local memory and register
tiles: 64 threads compute 32×32 outputs, with 4×4 accumulators per thread. Other
types/sizes retain local tiling or the simple kernel according to capabilities.
In OpenCL, M/N multiples of 32 and K a multiple of 16 allow redundant guards
to be removed from these tiles; other shapes retain the previous guards.
Public layout, precision, and allocations do not change.

`_gemm_rect.py` adds 16×32/K=16 tiles in OpenCL and 16×16/K=32 in CUDA
for a measured region of 2D float32 matrices with few rows and long K.
Each backend declares its configuration; the shared selector retains the
previous path outside that region, for batches, and with explicit padding. There are no new
buffers or packing. Vulkan retains the original tile. See
[selection and results](three-iterations-results.md).

The [padding](padding.md) option, disabled by default, also allows alignment of
custom float32 GEMM in OpenCL/Vulkan/CUDA. `_padding.py` prepares the physical shapes
and plan; `_padding_copy.py` copies uint32 words to preserve subnormals and
other float32 bit patterns. Each backend declares `prepare_padded_matmul(...)`; the
compiler binds the resulting plan without backend-name conditions and
without retaining inputs. If the capability is absent or rejected, the normal
path is retained. Private buffers are padded and cropped on every call; subsequent
steps always see logical dimensions. BLAS retains its explicit provider.
`blas="clblast"` uses the optional `_clblast.py` adapter on the same buffers:
native GEMV/DOT for rank-1/2 vector cases when their symbols are
present, and GEMM otherwise. Batches use native batched/strided-batched GEMM
when its symbols are available, with offsets into the original inputs.
The same-provider GEMM loop remains an alternative when a capability is absent.

The optional `prepare_layout_matmul(...)` hook allows the compiler to query the
backend during preparation, without backend-name conditions. OpenCL
with CLBlast supports private full 2D transposes consumed solely by
a float32/64 matmul. The compiler verifies reversed shape, zero offset, and strides
of a full transpose; it does not interpret an arbitrary slice as a transpose.
`_matmul_layout.py` retains metadata and the backend, not input arrays, and executes
GEMM with transpose flags and leading dimensions from the original storage.
New inputs are rebound on every call. Materialization is retained
if the temporary has another consumer or is a public output, if dimensions
are empty, or if the backend rejects the capability. Eligible explicit padding takes
precedence. No public views or epilogues within CLBlast are introduced, and
avoiding a copy does not guarantee that BLAS's internal kernel choice is faster.

`_ops.py` centralizes ufuncs and type resolution. `_fusion.py` captures values
through the same ndarray facade and a tracing backend. `_compiler.py` produces
typed plans with an LRU keyed by function, metadata, static scalars, and backend.
It rebinds arrays and scalars per call without retaining capture inputs.
The fully positional path uses precomputed parameter names to avoid
rebuilding the binding; other cases retain the general signature. All
validation and specialization keys remain active.
`_lowering.py` separates backend-specific group generation and execution from
shared graph scheduling. Each backend optionally provides `create_lowering()`;
the plan resolves it once and binds `execution_scope()` when present. The
`execution_scope_min_steps` minimum may discard that scope when preparing a short plan:
CUDA declares two; a `padded_matmul` counts its GEMM and required physical copies,
even though it appears as one logical step. Others retain previous behavior. If
there is no factory, ordinary primitives are retained; preparation errors propagate.
No central registry is queried by name, nor is negotiation performed for each node.
CUDA provides its own `create_lowering()`: `CUDALowering` compiles supported
elementwise/map-reduce groups with NVRTC. OpenCL and Vulkan fuse elementwise
groups. Both also fuse
expressions with their reduction when the intermediate value has a single consumer
and is not a public output: map-reduce accumulates directly, with bounded partials
for large reductions. Repeated references from the same consumer, such as
`x * x`, do not force materialization of `x`; distinct consumers and public outputs
retain that boundary. In CUDA, matrix products remain separate
steps. OpenCL custom and Vulkan can also fuse the private pattern
`b - A @ x` with GEMV when the floating dtypes match. CLBlast retains its boundary;
NumPy evaluates the primitives of the same
plan. `@fuse` uses the same core but retraces every call and supports only one
elementwise output. Outputs are ordinary resident arrays.

`_vulkan.py` manages device-local buffers, staging, and native Vulkan queues.
It compiles GLSL to SPIR-V through an external compiler. It retains buffers used
by pending commands until completion. Its pool shares a limit of
128 MiB of native allocations and 128 buffers across device and staging memory,
preferentially reusing exact matches, or the smallest compatible capacity up to
`2*max(requested,256)`. Logical size, VkBuffer capacity, and native allocated
bytes are accounted for separately. Old classes are evicted when
new completed allocations enter the pool. A set of shapes exceeding those limits
can still cause repeated allocations. It also reuses commands and, on the
descriptor-set path, completed pools and sets, with a maximum of 128 resources
of each type. It groups
dispatches from a compiled call into one command buffer, retains immediate eager
submission, and synchronizes upon reaching 64 recorded/pending operations. A reusable
fence per context signals prior work when observing results;
successful waits precede resource recycling, and an empty queue causes no native
wait. The pool tolerates finalizer reentry when finding and releasing capacities.
Barriers
still separate dependencies within the command buffer. Readback prefers compatible coherent,
cached host memory, separate from the upload pool. Read-staging mappings are
retained with the native allocation, including while in the completed pool;
they are unmapped before that memory is freed/evicted. Barriers, waits
before reads, and references to in-flight leases are retained. Uploads retain map/unmap
per copy, because persisting all mappings did not consistently improve
complete operations. Kernels
include cooperative reductions/GEMV, long DOT with partials, and batched matmul
with register tiles for eligible float32 GEMM and 8×8 tiles as an alternative.
A 128-template LRU retains GLSL, ABI, and geometry; native descriptor-set
structures are reused only after their use completes. Submission and barrier structures
are also reused under the lock. If `VK_KHR_push_descriptor` and its properties2 dependency
are detected and enabled, groups within
maxPushDescriptors write descriptors directly into the command buffer.
Host records can be reused after recording; GPU buffers retain
their lifetimes until commands complete. Other groups use completed
descriptor sets. There is no vendor-specific tuning.
`_vulkan_commands.py` adds a private per-device table with entry points
and typed CFFI arguments for common commands. The backend manages lifetimes
and barriers; VkResult is checked, and calls outside that table still
use the optional binding. No binary extension was added. See
[measurements at this boundary](vulkan-dispatch-results.md).

The in-memory kernel cache is limited to 128 variants per context. PyOpenCL
may maintain its own disk cache. A lock protects argument mutation
and kernel enqueue; the queue preserves execution order. `asnumpy`, `item`,
and `synchronize` wait for the corresponding completion.

The three solvers in `examples/solvers.py` use the public API, including
`get_backend(array)` to recover the existing instance without copying data. Vector state remains on the device;
OpenCL and Vulkan fuse updates and keep step coefficients on the GPU.
Residual state uses a scaled norm that avoids forming extreme squares.
CG and steepest descent use normalized directions for their coefficients.
Stopping and validity decisions still download scalars. Computing each candidate's
true residual uses an advance plan that groups the update,
GEMV, and norm. Curvature groups GEMV/DOT and is validated by reading its
scalar directly, retaining the value on the GPU for the typed update. They check the true
residual at every iteration and preserve inputs. Convergence state is part
of the result, not an exception for reaching the iteration limit.

## Decisions

- **OpenCL first:** expresses the intent of portability across vendors and works
  with local hardware. Vulkan is an independent backend with its own shaders and
  initial portable optimizations; performance parity is not assumed.
- **NumPy 2 as the contract:** its type resolution and broadcasting are used, without executing
  GPU arithmetic in NumPy. Documented constructors can generate inputs on the host.
- **Bounded compatibility:** functions outside the contract fail before a
  hidden conversion. Storage is read-only through the public API to
  avoid promising views and aliasing semantics that are not yet implemented.
- **Minimal CPU installation:** NumPy is the only mandatory dependency; the
  `opencl`, `vulkan`, and `cuda` extras are independent, with actionable errors if
  the driver/runtime is missing. Examples and benchmarks are not included in the wheel.
- **Honest measurement:** compilation, transfer, and computation are measured separately;
  the baseline uses NumPy directly, and no claim is made that every GPU will be faster.

## Verification

The same conformance tests run on the NumPy and OpenCL backends, with
results computed by NumPy. They include single/double precision, integer types,
broadcasting, rectangular matrices, empty dimensions, NaN/Inf, and API errors.
External solvers in `examples/solvers.py` are checked against
`numpy.linalg.solve` and independent residuals.
Vulkan has specific tests for fusion, types, batches, reductions, solvers, and
pending-resource lifetimes. Local verification does not replace testing
with another vendor.

CI includes CPU tests on different systems and CPU OpenCL with PoCL on Linux. PoCL checks
another implementation of the standard; it does not replace validation on AMD/Intel/NVIDIA GPUs.
`PACHO_REQUIRE_OPENCL=1` prevents a misconfigured CI installation from skipping tests.
