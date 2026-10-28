# Execution, fusion and CLBlast

The [three-iteration round of 2026-09-24](three-iterations-results.md) adds
rectangular tiles without copies on OpenCL/CUDA, selection of cuBLAS vector
primitives and private-transpose flags when compiling with CLBlast. It includes controls
for the previous path, two rounds, transfers and numerical errors. Figures
from earlier stages are snapshots; in particular, the historical padding
advantage does not automatically transfer to the faster direct GEMM.

The user chooses the backend and device. OpenPacho does not automatically switch
between CPU and GPU or move an array to decide where to execute an operation.
Implemented backends are `numpy`, `opencl`, `vulkan` and [initial CUDA](cuda.md). Having a GPU does not imply support for every backend.

For new repeated code, [`@pacho.compile`](compilation.md) is recommended:
it avoids retracing the function on every call and supports reductions, algebra and multiple
outputs. `fuse` remains as limited per-call capture. OpenCL and Vulkan
fuse elementwise expressions and integrate them with their reduction when
the intermediate is neither returned nor has multiple consumers; CUDA applies the same
scheme to supported elementwise/map-reduce groups.

The [gap and optimization analysis](gaps-and-optimization.md) distinguishes measured
costs and remaining opportunities without assuming parity between backends.

## Explicit fusion

```python
import numpy as np
import openpacho as pacho


@pacho.fuse
def update(x, b, rate=0.01):
    return (1 - rate) * x + rate * b


with pacho.use_backend("opencl"):
    x = pacho.asarray(np.ones(10000, dtype=np.float32))
    b = pacho.asarray(np.zeros(10000, dtype=np.float32))
    for _ in range(100):
        x = update(x, b)
    result = pacho.asnumpy(x)
```

Each call to `update` runs one elementwise kernel and creates a single
output buffer. Intermediates are computed inside the kernel. Scalars are
passed by value, without buffers or transfers of scalar arrays. The Python loop
still executes 100 calls: this version does not fuse the whole loop.

Already supported elementwise ufuncs, broadcasting, constants
and scalar arguments, and a single result are supported. Reductions, `@`,
indexing, writes, host conversion or Python conditions depending on
traced data are unsupported. The function must be pure: its Python body executes on every
call. Array arguments must be OpenPacho; conversion from NumPy is
performed explicitly before calling. Arbitrary Python is not captured.

Type resolution and intermediate conversions follow NumPy 2; fast-math is not
enabled. Automatic floating-point contraction is disabled
in fused kernels. This does not guarantee bitwise equality for transcendental
functions. The NumPy backend evaluates the same graph without generating native code.
OpenCL programs, CUDA NVRTC modules and Vulkan pipelines are stored in caches
limited to 128 variants per backend; shapes, types and expression determine the
variant, scalar values do not. On Vulkan, device descriptor and push-constant limits
still bound fused expression size.

With `compile`, a computation such as `np.sum((a - b) * (a - b), axis=1)` can run
map-reduce on OpenCL, Vulkan and CUDA without materializing the whole transformed array. Large
reductions retain partials on device and need a later stage.
On CUDA, `CUDALowering` compiles supported groups with NVRTC; matmul and BLAS
retain their provider boundary. See the [CUDA fusion results](cuda-fusion-results.md).
An expression shared by different consumers or returned as an output retains its
materialization; there is no promise that any graph becomes a single kernel.

## Memory and iterative methods

OpenCL reuses buffers through a `MemoryPool`. Uploads of already contiguous host data
avoid an intermediate CPU copy; the GPU copy completes before returning,
preserving isolation if the user later modifies the input. Noncontiguous
arrays require host preparation.

The in-order queue maintains dependencies between operations and reuse.
The pool retains free memory for later calls. To release it:
`pacho.get_backend("opencl").release_cached_memory()`; this synchronizes the queue and
preserves live arrays. If another BLAS configuration is used, it must be called
on that backend object.

Vulkan reuses device and staging buffers by exact size or bounded compatible
capacity, with a combined limit
of 128 MiB of native allocations and 128 free buffers. Pending commands
retain their buffers to prevent premature reuse. Completed commands and,
on the sets path, descriptor pools are also reused, with at most
128 resources of each type. `pacho.get_backend("vulkan").release_cached_memory()`
waits for pending work and frees those caches while preserving live arrays.
Compiled calls group dispatches in one command buffer, split by
synchronization or 64 recorded/pending operations; eager retains submission per
operation. Push descriptors are used when capability and binding count
permit; other groups reuse completed descriptor sets. Coherent readback
with CPU caching is preferred. Dispatch templates and native descriptor,
barrier and submission structures are retained. Host push records are reused
after recording; GPU buffers are retained until use completes.
See [current measurements](backend-sharing-results.md).
This does not remove command recording or the cost of many Python calls.

The solvers in [examples](../examples/README.md) reuse compiled plans for vector updates;
OpenCL and Vulkan fuse them, including the private `b - A @ x` pattern with compatible
custom GEMV. CG and steepest descent use the scaled norm of the residual
state and normalized directions, keep coefficients on the GPU and download validity
checks. The true residual is still checked **at every iteration**, preserving
convergence and failure information. Synchronizations remain for
stopping decisions; these are not fully autonomous on-device solvers.
The order and accuracy of coefficient computations may slightly change
the iteration count compared with the previous version.

Advance plans group update, residual and norm; curvature plans
group GEMV and DOT (plus normalization for steepest descent). Host validation
reads curvature directly; it avoids an extra boolean kernel and keeps
the coefficient on device for updating with the same dtype.

For repeated solves with the same matrix, there is an explicit contract:

```python
from examples import solvers  # requires the repository; not included in the wheel

system = solvers.prepare(A, method="cg")  # also "jacobi" or "sd"
first = system.solve(b1, rtol=1e-5)
second = system.solve(b2, rtol=1e-5)
```

Preparation copies the matrix and validates its shape and finiteness once; Jacobi also
stores the validated diagonal. Later changes to the original matrix
do not alter the system. Each solve validates b/x0, tolerances and the true residual at
every iteration. It does not prove that the matrix is SPD or that Jacobi converges.
Accuracy is fixed by A (float64 for nonfloating matrices); a b requiring
promotion is rejected. An explicitly converted matrix must be prepared.
The copy adds memory and preparation time: it is not recommended as acceleration
for a single solve. Compiled solver steps prepare their bindings
once per solve without changing the public convergence contract.

## Optional BLAS, custom kernels retained

```python
with pacho.use_backend(
    "opencl", device="0:0", blas="clblast", clblast_library="/path/to/CLBlast/library"
):
    A = pacho.asarray(np.eye(64, dtype=np.float32))
    y = A @ A
```

`clblast_library` is the path to the appropriate native binary for the system (`.dll`,
`.so` or `.dylib`); it can also be configured with `OPENPACHO_CLBLAST_LIBRARY`.
Without an explicit path, discovery of the installed library is attempted. CLBlast is
distributed separately: `pip install open-pacho[opencl]` does not install that binary.

`blas="custom"` is the default option and allows comparison of custom kernels.
`blas="clblast"` delegates supported floating-point products: native DOT/GEMV for
rank 1/2 vector operations when the library exposes their symbols, and
GEMM for other cases. If optional symbols are missing, the GEMM path remains.
Batched `matmul` uses native strided-batched GEMM for regular offsets,
or batched GEMM with offset arrays for irregular broadcasting, when
available. Offsets point to original buffers; broadcast inputs
are not duplicated. If those symbols are missing, the same provider's GEMM loop remains.
A native execution error propagates and does not activate that alternative path.
Integer products and
empty dimensions are handled by OpenPacho. A missing library or execution
error produces an explicit error; it does not cause a hidden engine
switch. Different configurations have different contexts; choose the
configuration before creating arrays.

Custom OpenCL kernels cooperate across work-items for GEMV with a right vector
and large DOT. OpenCL and Vulkan use a new float32 GEMM path for M/N≥32,
K≥16 and sufficient capability: each group computes 32×32 outputs and each thread holds
4×4 accumulators in registers, with 4 KiB of shared memory. It includes batch
broadcasting and edges. OpenCL retains local 16×16/8×8 tiles or the simple kernel
for other cases. In the register path, OpenCL removes guards when M/N are
multiples of 32 and K of 16; see [control and results](kernel-stability-results.md).
Partial cases retain guards with the default configuration.
[Optional padding](padding.md) aligns custom float32 GEMM on OpenCL, Vulkan and
CUDA in exchange for buffers and copies; public times include that entire cost.
The [reassessment](padding-reassessment.md) finds gains in some rectangular
OpenCL/CUDA shapes and losses in most; it remains disabled by
default, with no automatic selection or general recommendation for Vulkan.
Vulkan retains its 8×8 path for ineligible cases.
GEMM allows multiply-add
contraction; elementwise fusion preserves its roundings. Vulkan also
uses cooperative reductions/GEMV and partials for long DOT. See the
[CLBlast investigation and register controls](clblast-techniques.md).
Other shapes, sizes and vendors still need measurement and tuning.

Integration is limited to an adapter of BLAS operations over OpenCL buffers.
Expressions, allocations and reductions remain under OpenPacho's
control. A BLAS call is a fusion boundary: we cannot insert an arbitrary
update into its kernel. For operations where that boundary
costs too much, custom kernels can be developed and compared against CLBlast
without changing the public array API. Which implementation wins by
size or device is not assumed; there is no automatic selection or hardware tuning yet.

Sources for the interfaces used:
[PyOpenCL MemoryPool](https://documen.tician.de/pyopencl/tools.html),
[CLBlast](https://github.com/CNugteren/CLBlast).

Sparse matrices and preconditioners remain future work.

## Vulkan dispatch diagnostics

The [CFFI boundary round](vulkan-dispatch-results.md) compares the generated binding
with typed commands from the same backend. Counts and results are preserved;
instrumented GPU times and wall times are published separately. Measurements
do not automatically attribute the remaining time to Python or to a physical limit.
