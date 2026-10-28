# Compiler, Vulkan, and CUDA references: local results

**Historical snapshot before GEMV, map-reduce, and Vulkan optimizations.**
The continuation of the priorities at the end of this document is in
[gaps and optimization](gaps-and-optimization.md); this page's timings
are retained as a reference for the previous state.
The most recent continuation is in [map-reduce, preparation, and batched BLAS](remaining-suite-results.md).

Platform: Windows-11-10.0.26200-SP0. Devices: {'opencl': 'NVIDIA GeForce RTX 5060 Ti', 'opencl-clblast': 'NVIDIA GeForce RTX 5060 Ti', 'vulkan': 'NVIDIA GeForce RTX 5060 Ti'}. The original session from 2026-09-06 used Windows 11, RTX 5060 Ti 16 GB (driver 616.64), Ryzen 7 5700X3D, and 32 GB RAM. AMD/Intel have not been validated.

**235 configurations, seven rounds per mode, zero validation errors.** 172 operation/step configurations and 63 complete-solver configurations. JSON files retain all samples; CSV adds quartiles.

## What changes and what still costs

100 updates of 1,048,576 float32 values: 203.974 ms with the best NumPy, 33.846 ms with direct OpenCL, 11.280 ms with compile, and 6.187 ms with CuPy. Including transfers: 209.622 ms for NumPy and 13.418 ms for compiled OpenCL.

2048×2048 float32 matmul: 31.670 ms with NumPy, 3.771 ms with OpenCL+CLBlast, 1.458 ms with cuBLAS, and 1.589 ms with nvmath. The main advantage here comes from the matrix provider: compiling an isolated matmul neither fuses it nor necessarily improves its kernel.

Complete Jacobi at 2048 float32: 10.402 ms for NumPy, 25.121 ms for OpenCL custom, and 14.658 ms for CuPy. Steepest descent: 13.354 ms for NumPy, 34.135 ms for OpenCL, and 23.711 ms for CuPy. These are resident times: transfer is not the only cause of differences. GEMV/DOT, dispatch, allocations, and synchronization for convergence are involved. CLBlast alone does not resolve the algorithm.

Vulkan: resident 2048 float32 matmul 20.447 ms, with transfers 108.242 ms; 100 large updates, 303.090 ms. It is functional; pooling, fusion, and lower submission costs are still needed. These numbers do not represent a limit of Vulkan as a technology.

100 updates of only 256 values: 0.301 ms for NumPy and 11.023 ms for compiled OpenCL. Correctness is preserved; making the GPU win this case is not a priority.

## Method and comparison limits

- Synchronized wall time, with Python dispatch and result allocation. Rounds randomize implementation order. Asynchronous launch timings are not used as complete computation time.
- Resident: inputs already uploaded, plans prepared, and kernels warmed. Each repetition resets the logical iteration state. Validation and result download are outside this time.
- With transfers: new input buffers, computation, and download. Plans/programs are reused; nvmath rebinds operands without replanning. NumPy uses existing host arrays: no artificial copy is added.
- JSON files separate upload, planning, first execution, and download. First execution is for that configuration within the process; it may reuse caches from earlier configurations. It does not measure an independent cold process or include device initialization. OpenPacho capture is included in its first execution; explicit nvmath planning is listed separately.
- NumPy is measured with BLAS limited to 1, 8, and 16 threads. Tables show the best median per case and mode. Operations without BLAS do not become parallel by changing that limit; small differences there are noise.
- Original float32/float64, TF32 disabled; nvmath uses COMPUTE_32F/64F. Shape, dtype, and values are checked in every round. Bitwise identity is not required.
- `updates100`: x=(1-rate)*x+rate*b, one hundred steps. `energy`: sum((2*(a*b+c))²,axis=-1), size×32 arrays. Jacobi20 and gradient20 use 20 steps on a dense diagonally dominant SPD matrix; gradient20 uses a fixed rate.
- `_solve` cases run public solvers to convergence, with true residual at every iteration, rtol=1e-4/1e-7 for float32/64, and maxiter=400. Steepest descent uses exact line search. NumPy/CuPy execute the same recurrence through native operations in `benchmarks.suite.native_solver`. Scalar coefficients may have different paths/intermediate precision. The host residual is also validated against tolerance (2% margin for reduction order). Inputs are densely stored tridiagonal systems; this is not a sparse test.
- CUDA is the platform. The specific cuBLAS reference uses direct bindings; CuPy matmul can also use cuBLAS, and nvmath uses cuBLASLt. These are not three independent devices. cuBLAS/nvmath are compared only on matmul; no high-level solver is invented for those interfaces.
- PyTorch/JAX, peak memory usage, energy, all rectangular shapes, and performance portability across vendors were not measured. This battery covers the current scope; it does not encompass every possible workload.

## Resident timings

Milliseconds; lower is better. OpenPacho uses compile for operations/steps; in `_solve`, the public solver already compiles its updates. Complete eager and compiled modes are in the JSON/CSV.

| Case / size / dtype | Best NumPy | OpenCL custom | OpenCL CLBlast | Vulkan | CuPy | cuBLAS | nvmath |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul / 64 / float32 | 0.029 | 0.234 | 0.241 | 1.058 | 0.333 | 0.316 | 0.520 |
| matmul / 512 / float32 | 0.943 | 0.494 | 0.342 | 1.302 | 0.349 | 0.343 | 0.450 |
| matmul / 1024 / float32 | 5.172 | 2.481 | 0.761 | 3.460 | 0.469 | 0.489 | 0.582 |
| matmul / 2048 / float32 | 31.670 | 16.402 | 3.771 | 20.447 | 1.464 | 1.458 | 1.589 |
| matmul / 64 / float64 | 0.038 | 0.260 | 0.259 | 1.011 | 0.335 | 0.377 | 0.397 |
| matmul / 1024 / float64 | 8.530 | 8.620 | 5.883 | 9.357 | 6.886 | 6.893 | 7.047 |
| energy / 8 / float32 | 0.026 | 0.270 | 0.328 | 4.515 | 0.293 | — | — |
| energy / 2048 / float32 | 0.151 | 0.308 | 0.337 | 5.825 | 0.345 | — | — |
| energy / 32768 / float32 | 5.283 | 0.311 | 0.371 | 5.024 | 0.302 | — | — |
| updates100 / 256 / float32 | 0.301 | 11.023 | 10.924 | 270.404 | 7.564 | — | — |
| updates100 / 65536 / float32 | 2.565 | 11.067 | 11.234 | 264.744 | 6.872 | — | — |
| updates100 / 1048576 / float32 | 203.974 | 11.280 | 11.193 | 303.090 | 6.187 | — | — |
| jacobi20 / 512 / float32 | 0.370 | 2.217 | 2.518 | 70.555 | 2.623 | — | — |
| jacobi20 / 2048 / float32 | 3.738 | 6.730 | 2.441 | 73.488 | 3.738 | — | — |
| gradient20 / 512 / float32 | 0.414 | 2.850 | 3.422 | 70.073 | 2.844 | — | — |
| gradient20 / 2048 / float32 | 3.877 | 6.969 | 3.320 | 72.751 | 4.093 | — | — |
| jacobi_solve / 512 / float32 | 1.185 | 26.416 | 26.737 | 349.146 | 11.939 | — | — |
| jacobi_solve / 2048 / float32 | 10.402 | 25.121 | 27.532 | 350.699 | 14.658 | — | — |
| jacobi_solve / 1024 / float64 | 9.134 | 47.483 | 46.741 | 606.478 | 24.415 | — | — |
| sd_solve / 512 / float32 | 1.584 | 32.532 | 35.952 | 558.161 | 24.600 | — | — |
| sd_solve / 2048 / float32 | 13.354 | 34.135 | 34.451 | 538.382 | 23.711 | — | — |
| sd_solve / 1024 / float64 | 14.814 | 62.470 | 67.341 | 1033.909 | 47.033 | — | — |
| cg_solve / 512 / float32 | 0.916 | 19.386 | 19.520 | 323.175 | 13.047 | — | — |
| cg_solve / 2048 / float32 | 8.727 | 17.428 | 19.036 | 311.270 | 10.672 | — | — |
| cg_solve / 1024 / float64 | 6.583 | 29.073 | 31.723 | 530.994 | 20.406 | — | — |

## Upload + computation + download

Milliseconds; warm plans and programs.

| Case / size / dtype | Best NumPy | OpenCL custom | OpenCL CLBlast | Vulkan | CuPy | cuBLAS | nvmath |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul / 64 / float32 | 0.014 | 0.292 | 0.261 | 3.534 | 0.594 | 0.401 | 0.486 |
| matmul / 512 / float32 | 0.833 | 1.404 | 1.496 | 8.125 | 0.987 | 1.104 | 1.054 |
| matmul / 1024 / float32 | 4.943 | 4.685 | 3.330 | 27.957 | 2.966 | 2.980 | 2.859 |
| matmul / 2048 / float32 | 31.088 | 28.896 | 13.070 | 108.242 | 11.370 | 10.981 | 10.691 |
| matmul / 64 / float64 | 0.020 | 0.247 | 0.253 | 3.430 | 0.421 | 0.390 | 0.463 |
| matmul / 1024 / float64 | 10.346 | 12.464 | 10.471 | 52.349 | 12.765 | 11.779 | 11.831 |
| energy / 8 / float32 | 0.011 | 0.356 | 0.411 | 7.477 | 0.492 | — | — |
| energy / 2048 / float32 | 0.082 | 0.582 | 0.599 | 9.464 | 0.608 | — | — |
| energy / 32768 / float32 | 5.299 | 2.427 | 2.544 | 17.043 | 2.559 | — | — |
| updates100 / 256 / float32 | 0.295 | 10.835 | 11.152 | 267.471 | 7.336 | — | — |
| updates100 / 65536 / float32 | 2.515 | 11.527 | 10.988 | 296.495 | 7.290 | — | — |
| updates100 / 1048576 / float32 | 209.622 | 13.418 | 14.335 | 318.779 | 10.037 | — | — |
| jacobi20 / 512 / float32 | 0.346 | 2.407 | 2.697 | 73.664 | 2.811 | — | — |
| jacobi20 / 2048 / float32 | 2.853 | 9.660 | 5.131 | 90.634 | 4.477 | — | — |
| gradient20 / 512 / float32 | 0.372 | 3.110 | 3.766 | 73.844 | 3.056 | — | — |
| gradient20 / 2048 / float32 | 3.088 | 9.490 | 5.905 | 81.944 | 6.747 | — | — |
| jacobi_solve / 512 / float32 | 1.103 | 26.918 | 26.795 | 358.815 | 11.947 | — | — |
| jacobi_solve / 2048 / float32 | 9.954 | 27.671 | 28.881 | 360.976 | 15.301 | — | — |
| jacobi_solve / 1024 / float64 | 9.054 | 46.408 | 47.406 | 597.689 | 24.489 | — | — |
| sd_solve / 512 / float32 | 1.515 | 31.792 | 34.362 | 554.568 | 24.392 | — | — |
| sd_solve / 2048 / float32 | 12.775 | 48.296 | 41.136 | 539.987 | 27.616 | — | — |
| sd_solve / 1024 / float64 | 14.510 | 60.198 | 64.552 | 1068.319 | 46.568 | — | — |
| cg_solve / 512 / float32 | 0.807 | 18.753 | 19.533 | 330.216 | 12.697 | — | — |
| cg_solve / 2048 / float32 | 8.143 | 21.696 | 21.801 | 328.721 | 16.057 | — | — |
| cg_solve / 1024 / float64 | 6.428 | 30.813 | 32.579 | 544.778 | 19.665 | — | — |

## Next priority

1. Reduce solver synchronization and execution costs without silently changing check frequency or tolerances; optimize GEMV/DOT.
2. Fuse map-reduce and improve decisions about shared temporaries. Measure an executor with lower overhead when Python dominates.
3. Vulkan pooling and fusion, descriptor/command reuse, and optimized matrix kernels. Verify a second vendor.
4. Extend frequently used operations, such as batched matmul, according to real cases. Keep CLBlast and custom kernels interchangeable by contract; a CUDA backend and sparse operations remain future work.

## Reproduce

```powershell
$env:PYOPENCL_NO_CACHE='1'
$env:PACHO_VULKAN_COMPILER='<path to glslang or glslc>'
$env:OPENPACHO_CLBLAST_LIBRARY='<path to CLBlast>'
python -m benchmarks.execution --repeats 7 --output .artifacts/benchmarks/execution/raw.json
python -m benchmarks.execution --solvers-only --repeats 7 --output .artifacts/benchmarks/execution-solvers/raw.json
python -m benchmarks.execution_report --inputs .artifacts/benchmarks/execution/raw.json .artifacts/benchmarks/execution-solvers/raw.json --output docs/execution-results.md --artifacts .artifacts/benchmarks/execution-report
```

Additional dependencies: benchmark extra, CuPy and nvmath-python with compatible CUDA components, native CLBlast, Vulkan bindings, and a shader compiler. They are not part of OpenPacho's minimal CPU dependencies.

Measured versions: numpy 2.5.3, pyopencl 2026.1.4, vulkan 1.3.275.1, cupy-cuda13x 14.2.0, nvmath-python 1.0.0.
The original local artifacts are in .artifacts/benchmarks/execution-2026-09-06, .artifacts/benchmarks/execution-solvers-2026-09-06.
