# The path to CUDA and other backends

Architecture decision dated 2026-09-08, updated 2026-09-18. An
[initial CUDA backend](cuda.md) already exists: context, memory, NVRTC, primitives,
and cuBLAS. CuPy/cuBLAS/nvmath comparisons remain external references.
The following stages retain the expansion plan; they do not represent parity
already implemented with OpenCL/Vulkan.

## Implemented foundation

`_compiler.py` retains capture, specialization, materialization decisions,
dependencies, and temporary lifetimes. Lowering adapters generate and execute
fused groups by backend. `create_lowering` in `_lowering.py` requests the optional
`backend.create_lowering()` factory once; OpenCL, Vulkan, and CUDA provide their
adapters (`CUDALowering` for CUDA). NumPy, without a factory, executes its own
primitives without fusion.
The plan also binds `execution_scope()` if present. Preparation failures
propagate; they do not authorize switching backend or copying to CPU.

Each adapter prepares elementwise/map-reduce groups and executes them with the
current operands. Its payload can store metadata and code, never arrays from a
call or bound scalar values. Broadcasting, dtype, shared-intermediate, and public
output rules remain common. GEMV epilogue conditions respect the provider and
the corresponding reduction tree.

Tests use an independent adapter running on CPU to check scalar rebinding,
input release, and shared outputs without assuming a particular GPU name; the
GPU suite verifies the actual implementations. This is a private interface,
not a stable plugin API.

## Staged CUDA implementation

1. **Context and memory — foundation implemented.** Optional `_cuda.py` module,
   explicit discovery, retained primary context per device, dedicated stream,
   buffers, and bounded pool. Isolate arrays from other contexts and retain
   in-flight references. Importing OpenPacho on CPU does not load CUDA. `upload`
   isolates host input before returning; `asnumpy`, `item`, and `synchronize`
   observe the correct stream.
2. **Primitives and BLAS — partial.** Elementwise, gather, and floating-point
   reductions already run on CUDA, including matmul with batches/broadcasting
   and float32 tiling. [Custom cooperative GEMV/DOT](three-actions-results.md)
   is already implemented; cuBLAS already selects DOT for two vectors and GEMV
   for matrices with explicit batching, according to the
   [selection measurements](three-iterations-results.md).
   Stable float32/64 norms are implemented for eager and compiled execution,
   with custom kernels available with both `blas="custom"` and `blas="cublas"`;
   products retain cuBLAS and the observable custom option.
   Extend `blas` validation by backend: `clblast` is not a CUDA provider.
   Do not invent CPU emulation when a symbol or capability is missing.
3. **Compilation — elementwise/map-reduce implemented.** `CUDALowering`
   implements the common private interface and uses NVRTC for supported groups.
   The cache is bounded by code, types, geometry, device, and numerical options.
   It preserves rounding in fused elementwise operations, the scalar ABI, and
   the reduction tree of 128 lanes/chunks of up to 4096 terms, with
   `fmad=false` and `ftz=false`. Matmul and BLAS remain provider boundaries;
   bool accumulators and CUDA graphs remain pending. The stable norms needed
   by external examples are implemented.
   See the [CUDA fusion results](cuda-fusion-results.md). Reuse graph decisions;
   do not mechanically translate GLSL to CUDA.
4. **External applications and packaging.** By-value scalars, the initial CUDA
   extra, and stable norms are implemented. The external solvers in
   `examples/solvers.py` can use CUDA; they are not packaged in the library,
   and their algorithms are not part of the runtime.
   Run common and adversarial tests with CUDA required; verify installation
   without CUDA and error messages before publishing the extra.
5. **Performance.** Numerical parity first; then measure direct cuBLAS against
   the OpenPacho adapter on the same buffers and stream. Graphs, pinned memory,
   and specialized kernels are incorporated when a control demonstrates
   amortization, without changing convergence-check frequency.

The initial implementation uses `cuda.bindings` (Driver API/NVRTC) and cuBLAS
bindings distributed with nvmath behind a small adapter. Both modules are present
in the local benchmark environment; that does not yet prove the initial backend
is installable on every platform. CuPy remains an independent reference, avoiding
turning the OpenPacho backend into a complete CuPy wrapper. The initial extra
requires CUDA bindings 13.3+, NVRTC 13.3+, cuBLAS 13, and nvmath 1; extending
versions/platforms requires additional verification. Sources:
[CUDA Python](https://nvidia.github.io/cuda-python/cuda-bindings/latest/index.html),
[NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html).

The default will remain float32/float64 without implicit TF32 or fast-math. Any
future reduced-precision mode will have explicit selection, benchmark labels,
and error metrics. Bitwise identity across reduction trees or providers is not
promised. See [cuBLAS compute modes](https://docs.nvidia.com/cuda/cublas/index.html).

## Other backends and priority

| Candidate | Contribution | Priority / condition |
| --- | --- | --- |
| CUDA | NVIDIA-specific libraries and execution; local hardware available for verification | Initial foundation implemented; expand gradually and retain independent OpenCL/Vulkan backends. |
| HIP/ROCm | Access to native AMD libraries and execution | After supported AMD hardware/OS is available for measurement; reuse contracts, do not assume automatic portability of every kernel. |
| Metal | Native access to Apple GPUs and their libraries | When a test Mac and demand exist; do not design around assumed FP64 guarantees. |
| SYCL/oneAPI | C++ toolchain with several backends and a path to Intel ecosystems | Evaluate incremental value over OpenCL/Vulkan; higher build and distribution cost. |

HIP aligns with the CUDA model, but details and optimizations may require
adaptation. [HIP documentation](https://rocmdocs.amd.com/projects/HIP/en/latest/index.html).
Metal offers compute and optimized libraries on Apple platforms.
[Apple documentation](https://developer.apple.com/metal/).
SYCL can target different backends depending on the implementation; it is not a
single driver API. [Khronos SYCL](https://www.khronos.org/sycl/).

These backends are not added simultaneously. Each multiplies precision,
synchronization, lifetime, installation, and provider testing; the main priority
still includes Vulkan execution and transfers.

## What we mean by physical limit

CuPy/cuBLAS and OpenPacho use the same local GPU. An optimized library's advantage
is an achieved reference, not proof that OpenPacho has reached the hardware limit.
Bandwidth, compute by dtype, latency, and capacity limits differ; Python and
driver overhead is measured separately. NVIDIA describes precisely that
separation in its [CUDA performance guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

Quantifying the remaining margin requires GPU timestamp measurements, bytes
actually moved, sustained resident copy rates, and throughput by shape/dtype.
The roofline must use those measured ceilings and the workload's arithmetic
intensity. The advertised maximum or CuPy's speed is insufficient. Nor can we
attribute the current gap to Tensor Cores: the published comparisons disable TF32.
