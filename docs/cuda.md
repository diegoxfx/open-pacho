# Initial CUDA

Initial implementation from 2026-09-12, extended with batches, tiling, fusion, and
cooperative GEMV/DOT; stable-norm update: 2026-10-22.
CUDA is selected explicitly and preserves OpenPacho's
array API. It is an initial functional foundation; it still has fewer
operations and optimizations than OpenCL/Vulkan.

```powershell
python -m pip install ".[cuda]"
pacho devices --backend cuda
pacho benchmark --backend cuda --size 129 --repeat 7
```

The installation above runs from this repository; the package is not yet
published on PyPI. It requires an NVIDIA GPU, a driver compatible with the installed CUDA 13 libraries, and
NVRTC. The extra includes CUDA bindings, NVRTC, nvmath bindings, and cuBLAS.
CPU import does not load these modules. The runtime depends on neither CuPy arrays nor
operations; CuPy and nvmath Matmul comparators remain external.
Physical validation at this stage is on Windows and RTX 5060 Ti; Linux
and other devices still require verification.

```python
import numpy as np
import openpacho as pacho

gpu = pacho.get_backend("cuda", device="0", blas="cublas")
a = pacho.asarray(np.ones((33, 17), dtype=np.float32), backend=gpu)
b = pacho.asarray(np.ones((17, 65), dtype=np.float32), backend=gpu)
c = (a @ b) * 0.5 + 2
host = pacho.asnumpy(c)
```

`blas="custom"` selects custom products through NVRTC. Eligible float32 GEMM
reuses OpenCL/Vulkan's tiled register generator: 64 threads per 32×32 output
tile, with 4×4 accumulators per thread and 4 KiB of shared memory. Device
and index limits are checked before selecting it. Other sizes
and float64 retain the direct GEMM kernel. Products with a right-hand vector
and inner dimension of at least 256 use a cooperative reduction of 128 threads
per row, also for float64 and contiguous batches. DOT with more than 4096 elements uses
partials of 1024 terms and a subsequent GPU reduction. This adapts
the algorithm tested in OpenCL; it does not depend on extensions from an NVIDIA family.
There is no autotuning. The default provider is `cublas`; `clblast`
and `clblast_library` are not valid CUDA configurations. It does not switch to CPU
if a driver, library, or operation is missing.

Custom GEMM adds two smaller tile dimensions for matrices with few rows:
in CUDA, 16×16 outputs and K=32 blocks. Selection is internal and conservative;
ineligible cases keep the previous kernel. It introduces no padding,
additional copies, reduced types, or mathematical flags. See the scope and
[three-iteration measurements](three-iterations-results.md).

With `blas="custom"`, `compile(..., padding=True)` or
`matmul(..., padding=True)` allow [float32 GEMM alignment](padding.md) using
private buffers and GPU cropping. `padding=False` remains the default;
cuBLAS and other cases retain their path. Copies preserve float32
bits, and the option does not modify `ftz`, `fmad`, or other mathematical flags.

## Verifiable scope

| Operation | Scope |
| --- | --- |
| float32/float64; bool for results and storage | Supported; integers and other dtypes produce an explicit error |
| Construction, upload, download, copy, reshape | Supported; owned resident arrays |
| Random generation | Custom [Philox4x32-10](random.md): random/uniform/normal/exponential float32/64, without cuRAND or CuPy as a dependency |
| Arithmetic, comparisons, and elementwise functions in the shared subset | Includes `where(condition, x, y)` with three-argument broadcasting, Boolean or numeric condition, without downloads; dtype promoted from branches |
| Basic slicing, transposition, and diagonal | GPU gather; outputs copy according to the shared contract |
| sum/mean over axes, keepdims | Block reduction with partials for long axes; floating-point accumulation |
| min/max (amin/amax) over axes, keepdims | Preserve float32/64 or bool; NaN propagated and deterministic zero ties; eager and compile/map-reduce |
| Dot, GEMV, and GEMM, including broadcast batched matmul | cuBLAS or custom product; rectangular, odd, and empty shapes |
| compile as function/decorator | Captures and reuses the plan; `CUDALowering` fuses supported elementwise/map-reduce groups, including `where` selection |
| Stable norms | Scaled Euclidean/Frobenius norm in float32/float64, eager and compiled; available with `blas="custom"` and `blas="cublas"` |
| Boolean accumulator in sum | Unsupported; summing Boolean inputs with an explicit float32/64 accumulator is supported. min/max do preserve bool |
| CUDA elementwise/map-reduce fusion | Supported through `backend.create_lowering()` and NVRTC |
| Graphs, pinned/overlapped transfers | Pending |

Default Boolean summation requests int64 through the shared contract, which
this stage does not support. Float32/64 accumulation can be selected explicitly.
`compile` does not enable operations outside this table; benchmarks label
its compiled CUDA execution as `backend-plan`. CUDA offers its own
`create_lowering()` factory providing `CUDALowering` for supported elementwise and
map-reduce groups. Matmul and BLAS remain outside that fusion. Reduction
variants use 128 lanes and chunks of up to 4096 terms, with NVRTC without implicit FMA or
FTZ (`fmad=false`, `ftz=false`). See the [CUDA fusion results](cuda-fusion-results.md).
Stable norms use generic lowering when compiled and are
available with both BLAS providers; `blas` continues to choose only the
product provider. See the [CUDA norm scope](cuda-norm-results.md).
Solvers remain in [examples](../examples/README.md), outside the package;
their algorithms are not part of the runtime.

Regular cuBLAS batches use `gemm_strided_batched` in one call, with
positive strides in elements. General broadcasting reuses input
pointers and executes GEMM per batch; it neither replicates matrices nor uses zero strides outside
the documented contract. There is room for a batched adapter with device-side pointers.
Custom computes all batches in the same launch. Vector promotion,
empty results, and K=0 preserve shared matmul semantics.
See [extension measurements](backend-sharing-results.md).

Since the 2026-09-24 round, two vectors use `sdot`/`ddot`, with a resident scalar
output. If one operand is a vector and the other has an explicit batch, GEMV
is used per matrix with contiguous offsets; singleton batches and multiple axes are supported.
Unbatched matrix-vector/vector-matrix products retain GEMM: selecting
GEMV universally was slower in the measured suite. The batched gain includes
a simpler host loop and does not prove that every GEMV kernel beats GEMM.

## Memory, execution, and accuracy

The backend retains the primary context and uses its own nonblocking stream.
Each call temporarily activates that context and restores the caller's; it does
not reset the primary context shared with other libraries. A lock protects
state, launch, and arguments. Pending operations retain their buffers
until synchronization; the initial limit is 64 launches. CUDA's pool reuses
completed exact sizes, evicts old entries, and retains at most
128 buffers/128 MiB requested from the driver, without measuring its internal physical rounding.
It does not yet include Vulkan's capacity-based reuse.

A compiled plan with at least two steps holds the context and lock during
execution. Shorter plans avoid that additional host cost. Nested
activations check the thread's current context and avoid another
push/pop when it already matches. If another library activates a different one, the backend's own
context is temporarily activated and the other restored. No GPU wait is added and
the pending-operation limit is unchanged. See the
[Driver API context contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).

Each compiled CUDA module retains a `ctypes` argument vector with
stable host storage for `kernelParams`. The vector is rebound on
every dispatch under the same lock and contains only copies of numerical values and
device addresses; the template retains neither caller arrays nor
scalars. Device allocations remain retained in `_pending` until
`synchronize()`. The 128-entry LRU and `close()`, after synchronization,
release the module and its argument vector. This optimization changes only
parameter packing; it changes neither kernels, math, contexts, nor stream.
See [CUDA dispatch measurements](cuda-dispatch-results.md).

Upload through Driver API completes its copy before returning, so
modifying the NumPy input afterward does not modify the GPU. Download waits for the stream.
Upload explicitly completes the default stream after HtoD:
pageable host memory may finish copying to staging before device DMA
completes. Our nonblocking stream does not inherit that dependency.
This is an initial conservative boundary, not an implementation of overlapped copies.
[Copy contract](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html),
[stream dependencies](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html).
`release_cached_memory()` synchronizes and releases only free memory; live outputs
remain valid. `close()` releases owned resources and invalidates arrays from
that backend with a closed-backend error. It does not reset third-party resources.

cuBLAS uses `PEDANTIC_MATH`, host scalar pointers, and the dedicated stream. DOT temporarily
changes pointer mode to DEVICE to write the result on the GPU; the
lock protects the change and a `finally` restores HOST. If restoration fails, the
backend records it and must recover HOST before any new BLAS
product. Submitted buffers are retained even if restoration fails. NVRTC
compiles without fast-math, with `fmad=false`, `ftz=false`, and precise division/square root.
Other libraries' mathematical settings are not changed globally.
The [float64 contraction experiment](fp64-contraction-results.md) compares
isolated executables with and without FMA, with distinct oracles for each policy.
Its flags are not applied to the backend: CUDA custom retains `fmad=false`.
Reduction trees may differ from NumPy/cuBLAS: errors are compared,
and bitwise identity between providers is not promised. Ordinary summation may
overflow or lose accuracy through cancellation; it still neither substitutes for a scaled
norm nor provides general stability guarantees.

Caches are local to the context/backend and do not write kernels to the user's
home directory.

Primary sources for the contracts used:
[Driver API and NVRTC in CUDA bindings](https://nvidia.github.io/cuda-python/cuda-bindings/13.3.0/overview.html),
[cuBLAS math mode](https://docs.nvidia.com/cuda/cublas/index.html).

## Reproducible comparison

The applied suite supports both CUDA providers as additional options:

```powershell
python -m benchmarks.applications --backends opencl,opencl-clblast,vulkan,cupy,cuda-cublas,cuda-custom --repeats 7 --output .artifacts/benchmarks/cuda-applications.json
python -m benchmarks.applications --irregular --backends opencl,opencl-clblast,vulkan,cupy,cuda-cublas,cuda-custom --repeats 7 --output .artifacts/benchmarks/cuda-irregular.json
```

First compilation, preparation, and transfers are separated from resident
execution. Ridge compares twenty common steps, not time to convergence.
Timings of this initial implementation are not considered the CUDA ceiling.
