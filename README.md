# OpenPacho

**Vendor-independent GPU arrays with a familiar API for NumPy users.**

OpenPacho revisits the idea of parallel numerical operations on GPUs explored in
PyPacho, with a new implementation. OpenCL and Vulkan are selectable GPU
backends that do not require CUDA. NumPy is the CPU backend and the behavioral
reference. Version **0.1.0a1**, experimental.
An [initial CUDA backend](docs/cuda.md) is also available for NVIDIA, with a
narrower scope than OpenCL/Vulkan and selectable cuBLAS/custom providers.

See the [addition of stable CUDA norms](docs/cuda-norm-results.md), the
[three kernel and layout iterations](docs/three-iterations-results.md), and the
[path to the first stable release](docs/release-readiness.md), including exit
criteria and the limits of current validation.

## Install

Requires Python 3.10 or later. From this repository:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install ".[opencl]"
pacho devices
```

The package is not yet published on PyPI. For CPU use, `python -m pip install .` is sufficient.
To contribute: `python -m pip install -e ".[opencl,dev,benchmark]"`.
The `opencl`, `vulkan`, and `cuda` extras are independent and can be combined:
`python -m pip install ".[opencl,vulkan]"` does not install CUDA. The distribution
is named `open-pacho` and the import is `openpacho`; once published, the equivalent
command will be `python -m pip install "open-pacho[opencl]"`.

OpenCL requires the vendor's driver/runtime in addition to PyOpenCL. Available
devices depend on the system: NVIDIA, AMD, Intel, and other vendors may expose
OpenCL; its availability is not assumed on every machine. On devices without
double precision, use `dtype=pacho.float32`: **precision is never silently reduced**.
See the [PyOpenCL installation guide](https://documen.tician.de/pyopencl/misc.html#installation).

For Vulkan: `python -m pip install ".[vulkan]"`, a Vulkan driver, and a GLSL/SPIR-V
compiler; see [installation and limitations](docs/vulkan.md).

For the initial CUDA backend: `python -m pip install ".[cuda]"` and an NVIDIA
driver compatible with CUDA 13. Run `pacho devices --backend cuda`; see the
[CUDA scope](docs/cuda.md).

## Migrate a NumPy function

You can keep NumPy inside existing functions. NumPy's dispatch protocols route
supported operations to OpenPacho:

```python
import numpy as np
import openpacho as pacho


def distances(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


with pacho.use_backend("opencl"):
    a = pacho.asarray([[1, 2, 3], [4, 5, 6]], dtype=pacho.float32)
    b = pacho.asarray([1, 1, 1], dtype=pacho.float32)
    result = distances(a, b)  # operations stay on the GPU
    print(result.device)
    print(pacho.asnumpy(result))  # explicit transfer at the end
```

You can also change the import in code that uses the supported subset:

```python
import openpacho as np

np.set_backend("opencl")
a = np.asarray([[4, 1], [1, 3]], dtype=np.float32)
b = np.asarray([1, 2], dtype=np.float32)
result = a @ b
print(np.asnumpy(result))
```

The initial backend is `numpy`; importing the package does not initialize a GPU.
`set_backend` changes where subsequent arrays are created; `use_backend` restores
the selection on exit. Existing arrays keep their device. To select a specific one:

```python
print(pacho.devices())
pacho.set_backend("opencl", device="0:0")  # platform:device ID
```

The default OpenCL selection requires a GPU. An OpenCL CPU device can be selected
explicitly by ID. There is no automatic fallback when GPU execution fails.

## Features

- Arrays of any dimensionality with broadcasting, scalar operations, and NumPy 2 type promotion.
- `+`, `-`, `*`, `/`, floating-point powers, comparisons, and `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `abs`, `isfinite`.
- `where(condition, x, y)` with three-argument broadcasting; selection runs on the backend without downloading. See the [selection limits](docs/compatibility.md#conditional-selection).
- `sum`, `mean`, `min`, `max`, reshape, transpose, integer indexing and slices, diagonals, vector/matrix `dot`, and batched `@`/`matmul` with broadcasting of leading dimensions.
- `float32`, `float64`, `int32`, `int64`, `uint32`, `uint64`, and `bool`, depending on the operation and device capabilities.
- Scaled Euclidean/Frobenius norms to avoid extreme intermediate squares.
- [Resident random generation](docs/random.md) with `pacho.random.default_rng(seed, backend=...)`: uniform, normal, and exponential distributions; reproducible seeds and scalar parameters.
- A device CLI and reproducible benchmarks with synchronization and validation against NumPy.
- Optional compilation with `pacho.compile(f)`, `@pacho.compile`, and `@pacho.compile()`: cached/prepared plans, multiple outputs, and elementwise/map-reduce fusion in OpenCL, Vulkan, and CUDA for supported groups; see the [contract and examples](docs/compilation.md) and [CUDA fusion results](docs/cuda-fusion-results.md).
- Reusable OpenCL/Vulkan memory and optional CLBlast with GEMM, batched GEMM, GEMV, and floating-point DOT; see [execution and performance](docs/performance.md).
- Lower compiled-call overhead and cooperative custom CUDA GEMV/DOT; see the [three measured optimizations](docs/three-actions-results.md), including limitations and numerical errors.
- [Optional padding](docs/padding.md) with `@pacho.compile(padding=True)` or `pacho.matmul(a, b, padding=True)`: aligned buffers for custom float32 GEMM in OpenCL, Vulkan, and CUDA, with explicit memory and copy costs; disabled by default.

The [Vulkan staging and random generation round](docs/staging-random-results.md)
documents readback mapping selection, the resident RNG, and timing/error controls.

The [gap and optimization analysis](docs/gaps-and-optimization.md) distinguishes
hardware limitations, runtime costs, and remaining opportunities.
The [Vulkan, GEMM, and stability iteration](docs/numerics-vulkan-results.md)
reports measured improvements and unresolved limits. Benchmarks publish
[numerical error and adversarial cases](docs/numerical-accuracy.md) alongside timings.
The [solver and application follow-up](docs/solver-applications-results.md)
extends the measurements and prepares the [CUDA path](docs/cuda-backend-plan.md).
The [initial CUDA round](docs/cuda-initial-results.md) implements CUDA and improves
Vulkan reuse when shapes change. The [dispatch round](docs/vulkan-dispatch-results.md)
reduces Python/CFFI conversions and reports GPU/host diagnostics with an interleaved control.
The [cross-backend extension](docs/backend-sharing-results.md) adds Vulkan push
descriptors when supported, OpenCL metadata/indexing, and CUDA batches/tiles.
The [kernel and stability round](docs/kernel-stability-results.md) adds observed
idle checks between providers, variant controls, and specialized OpenCL guards.
The [generated-code inspection](docs/kernel-inspection-results.md) fixes accidental
float64 promotion in OpenCL GEMM, reviews copies, and distinguishes PTX/SPIR-V/SASS
evidence from hardware hypotheses.
The [padding reassessment](docs/padding-reassessment.md) retains the option for
specific gains in rectangular products, keeps it disabled by default, and
documents shapes where it loses performance.
The [compiler, Vulkan, and CUDA reference comparison](docs/execution-results.md)
separates resident execution, transfers, and first execution.
The [first optimization comparison](docs/optimization-results.md) includes the
snapshot preceding that stage, NumPy, and the paths measured in the same session.

**Full NumPy compatibility is not a goal.** We prioritize frequent matrix
computations with measurable improvements and a reasonable implementation cost.
The [compatibility table](docs/compatibility.md) documents differences: no indexed
writes, `out=`, advanced indexing, complex arrays, or sparse algebra. Unsupported
functions fail explicitly. `np.asarray(array_pacho)` does not silently copy to the
CPU; use `pacho.asnumpy` when a dependency needs a NumPy ndarray.

## Example applications

The library provides matrix operations, not numerical solvers. CG, Jacobi,
steepest descent, and prepared systems live in [examples](examples/README.md),
outside the installable package. From the repository:

```python
import openpacho as pacho
from examples.solvers import conjugate_gradient

with pacho.use_backend("opencl"):
    A = pacho.asarray([[4, 1], [1, 3]], dtype=pacho.float32)
    b = pacho.asarray([1, 2], dtype=pacho.float32)
    result = conjugate_gradient(A, b, rtol=1e-5, maxiter=100)
    print(result.converged, result.iterations, result.residual_norm)
    print(pacho.asnumpy(result.x))
```

CG and steepest descent require a symmetric positive-definite matrix; Jacobi
requires a nonzero diagonal and a convergent iteration. The examples do not prove
these properties. The stopping criterion uses the true residual
`||b - A @ x|| <= max(atol, rtol * ||b||)`. Failure to converge returns
`converged=False` and a reason; the last iterate is not presented as a guaranteed solution.

## Verify and measure

```sh
python -m pytest
python examples/numpy_migration.py --backend opencl
pacho benchmark --backend opencl --size 256 --repeat 5 --output benchmark.json
pacho benchmark --backend numpy --size 256 --repeat 5
python -m build
```

OpenCL tests are skipped when the runtime/device is unavailable. To make a missing
GPU fail verification:

```powershell
$env:PACHO_REQUIRE_OPENCL = "1"
python -m pytest
```

On Linux: `PACHO_REQUIRE_OPENCL=1 python -m pytest`. `PACHO_OPENCL_DEVICE=0:0`
selects a specific test device.

Benchmarks separate the initial transfer, first execution (including compilation),
operations on resident data, and download. No universal speedup is promised:
problem size, precision, driver, and dispatch overhead matter.
See the [current architecture](docs/architecture.md), [evolution decisions](docs/architecture-evolution.md), and [priorities](docs/roadmap.md).
Results verified on this machine are recorded in [local verification](docs/verification.md).
The extensive CPU/OpenCL/CUDA comparison, including transfers and solvers with
up to 8192 variables, is in the [benchmark results](docs/benchmark-results.md).

## Origins and acknowledgments

OpenPacho is developed by **Diego Alejandro Cifuentes**, who co-authored
[PyPacho](https://github.com/azapat47/pypacho) with Andrés Felipe Zapata Palacio,
Esteban Echeverri Jaramillo, and Juan David Arcila Moreno. Their work is described in
[PyPacho: A Python library that implements parallel basic operations on GPUs](https://ieeexplore.ieee.org/document/9623197).
PyPacho explored parallel basic operations on GPUs and comparisons of iterative
methods across platforms, problem sizes, and precisions. That work laid the
foundation for the idea developed in OpenPacho.

This implementation uses new code under the MIT license. The
[provenance note](docs/provenance.md) distinguishes inspected references from
verified findings. It does not claim to reproduce the paper's experimental results.
