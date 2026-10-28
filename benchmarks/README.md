# Transfer hypothesis benchmark

## Rectangular GEMM, cuBLAS vector operations, and compiled layouts

Three workers compare each change with two controls using the previous path on
the same inputs/context. They use 30 samples by default; `--repeats` must be a
multiple of six to balance the order of the three modes. Results include
accuracy and numerical status separately from experiment completion.

```powershell
python -m benchmarks.isolated --worker rectangular_gemm --backends opencl,cuda-custom --rounds 2 --repeats 30 --output .artifacts/benchmarks/rectangular
python -m benchmarks.isolated --worker compiled_layouts --backends opencl-clblast --rounds 2 --repeats 30 --output .artifacts/benchmarks/layouts
python -m benchmarks.isolated --worker blas_vectors --backends cuda-cublas --control-wheel <trusted-previous-wheel> --rounds 2 --repeats 30 --output .artifacts/benchmarks/blas-vectors
```

`rectangular_gemm` disables only the new tile selection; it includes excluded
neighboring shapes and square/batched controls. `compiled_layouts` disables the
layout hook when preparing each control, including centered covariance.
`blas_vectors` replaces only `CUDABackend.matmul` with the method from the previous
wheel; it includes float32/64 and GEMM as a negative control. Overrides remain
outside the clock and are restored even on failure. Do not run these workers
concurrently. See [decisions and results](../docs/three-iterations-results.md).

`benchmarks.suite` measures the original hypothesis with CuPy/CUDA as an external
control. Later runners also compare OpenPacho's native CUDA backend when selected.
Run benchmarks from the repository root.

## Setup

```sh
python -m pip install -e ".[opencl,benchmark,dev]"
python -m pip install "cupy-cuda13x[ctk]"
python -m pytest benchmarks/test_suite.py
```

The CUDA extra requires a compatible driver. For another CUDA version, choose
the appropriate wheel following [CuPy's instructions](https://docs.cupy.dev/en/stable/install.html);
do not install two CuPy variants in the same environment. The default suite
command requires OpenCL and CUDA; if a backend is unavailable, select explicitly
with `--backends opencl`, `--backends cuda`, or `--backends ""` for CPU-only measurements.

## Run

In PowerShell, after activating the environment:

```powershell
$env:OPENBLAS_NUM_THREADS = "16"
$env:NVIDIA_TF32_OVERRIDE = "0"
$env:CUPY_CACHE_DIR = "$PWD/.artifacts/benchmarks/cupy-cache"
python -m benchmarks.suite --profile smoke --rounds 3 --output .artifacts/benchmarks/smoke
python -m benchmarks.suite --rounds 11 --output .artifacts/benchmarks/full
python -m benchmarks.suite --solver-sizes 2048,4096,8192 --rounds 7 --max-batch 1 --skip-transport --output .artifacts/benchmarks/solvers-large
python -m benchmarks.cold_start --rounds 3 --output .artifacts/benchmarks/cold.json
python -m benchmarks.report .artifacts/benchmarks/full/raw.json --extra .artifacts/benchmarks/solvers-large/raw.json --cold .artifacts/benchmarks/cold.json --output .artifacts/benchmarks/report
```

The suite refuses to overwrite an existing `raw.json`: use a new directory for
another run. `checkpoint.json` preserves progress. Every error is recorded, and
execution exits with a nonzero code if any case fails. The `smoke` profile uses
small sizes and every operation type before the full run.

`PYOPENCL_NO_CACHE=1` can be used when the environment prevents writing the user
cache. The context's in-memory cache remains active; this option affects first
compilation and must not be confused with transfers or resident execution time.
Do not run other CPU/GPU tests concurrently with benchmarks.

## Applications and scope

Current solvers are imported from `examples/solvers.py`, outside the package.
Their recurrences, tolerances, and documented preparation are preserved. Their
hashes are included in the source record. Run from the repository root.
`LegacyAdapter` deliberately retains the solvers from the historical wheel being
compared; it does not replace that algorithm with the current example.

New optimizations belong in the compiler/backend and must benefit equivalent
user code. See [rules and priorities](../docs/architecture-evolution.md).

## Predefined design

- 122 configurations: float32/64; vectors from 1024 to 16,777,216 elements;
  square matmul from 16² to 2048² plus four rectangular shapes;
  matvec and transpose up to 4096²; three sizes per solver.
- Extension: 18 solver configurations with 2048, 4096, and 8192 variables,
  7 rounds, and one call per sample. The largest matrix occupies 512 MiB in float64.
- Direct NumPy with 1-, 8-, and 16-thread BLAS; the analysis uses the fastest
  measured result for each configuration and preserves every original measurement.
- OpenPacho/OpenCL and CuPy/CUDA: large resident data, upload+compute+download,
  upload separately, and download separately.
- Upload and download alone with existing buffers, from 4 KiB to 256 MiB.
- Sequences of 1, 10, and 100 steps with one upload/download of large arrays.
  OpenPacho's Python scalars do cause small uploads at each step.
- 11 randomized rounds per configuration; adaptive blocks of up to 64 calls,
  synchronized after **every** call. Each sample records the block's mean time
  per call. It does not measure enqueue time alone.
- Warmup and validation are outside the clock. Operation time includes
  temporaries, dispatch, and result destruction. CuPy's native pools are retained;
  OpenPacho uses its normal `cl.Buffer` allocations.
- Equivalent solvers with the same dtype, tolerance, and true residual.
  `numpy.linalg.solve` is not used as a timed baseline. Convergence and iterations
  are recorded, and outputs are validated against a known solution.
- TF32 is disabled in these runs; speed is not obtained by silently changing
  float32 precision.

The seed, order, versions, BLAS, telemetry, and source hashes are recorded in
`raw.json`. Plot generation uses `comparisons.csv`, which also preserves the
selected CPU baseline, medians, and dispersion. Bootstrap intervals describe the
run and do not establish universal guarantees.

## Interpretation

1. A GPU that loses with resident data is not losing because of large-array
   transfers: investigate kernels, allocators, dispatch, and synchronization.
2. Winning with resident data but losing end to end indicates that input/output
   and preparation costs erase that advantage. DMA is not necessarily the only cost.
3. More computation per byte can amortize copies, even in a single call.
   A sequence can reuse the same data.
4. CuPy's result does not prove CUDA is intrinsically superior to OpenCL: the two
   libraries use different algorithms, internal libraries, and allocators.
5. Process startup and the first operation are reported separately: they are not
   equivalent to sustained performance after a program has initialized its GPU.

## Compare optimizations in the same session

`optimization.py` compares NumPy (1/8/16 threads), custom OpenCL kernels,
explicit fusion, and optional CLBlast. It can also load a previous wheel under
an isolated module name without reinstalling it or changing the current environment.

```powershell
$env:OPENPACHO_CLBLAST_LIBRARY = "C:\path\clblast.dll"
python -m benchmarks.optimization --output .artifacts/benchmarks/optimization-new --clblast --legacy-wheel C:\path\open_pacho-previous.whl --rounds 7
```

The output directory must be new. `raw.json` preserves samples, hashes,
validation, iterations, and resident/with-transfer timings; `REPORT.md`
summarizes medians including transfers. The previous reference is extracted
inside the output directory. Omit `--clblast` if the native library is not
installed, and `--legacy-wheel` if no previous package is available.

In this comparison, `opencl` already includes memory pooling, scalars passed by
value, and solvers with fused updates. `opencl-fused` adds the decorator to generic
expressions; `opencl-clblast` replaces their floating-point products. CLBlast
already uses dedicated GEMV and DOT when their symbols are available; earlier
reports describe the code at their own stage. Historical results are not overwritten.

To run the CLBlast conformance suite, configure the path above and run:

```powershell
$env:PACHO_BLAS = "clblast"
$env:PACHO_REQUIRE_OPENCL = "1"
python -m pytest
```

## Vulkan boundary control

Requires the [Vulkan extra, driver, and compiler](../docs/vulkan.md); the original
suite's OpenCL/CuPy installation does not include them.

`python -m benchmarks.vulkan_dispatch --repeats 21 --output <new-file.json>`
interleaves wrapped/direct paths in the same backend with identical kernels,
algorithms, and precision. It measures resident operations, complete applications,
and copies; float32/64, rectangular/odd shapes, and external solvers down to the
agreed residual. `--quick` reduces scope to check the runner. Repeating with
`--seed 4782` changes the order, not the inputs.

API and timestamp instrumentation is installed only in a separate invocation,
restores functions on completion, and waits before destroying queries. GPU
intervals are not pure shader times; API timings include driver work/waits.
New allocations and cache use are recorded, not total peak VRAM.
See [results and limitations](../docs/vulkan-dispatch-results.md).

`--control descriptors` compares `sets`/`push`, keeping the typed CFFI boundary
the same in both. It records maxPushDescriptors, counts, and timestamps; the
requested control requires the push extension. The `commands` control retains
sets on both paths to isolate wrappers. See the [next round](../docs/backend-sharing-results.md).

## Shared operations across backends

The default set requires OpenCL, native CLBlast, Vulkan, native CUDA, and CuPy.
Install only the required backends following the [OpenCL](../README.md),
[Vulkan](../docs/vulkan.md), or [CUDA](../docs/cuda.md) instructions; configure the
native CLBlast library as in the previous comparison. Limit `--backends` to the
available providers. Missing capabilities are recorded in JSON and produce a
nonzero exit code; they are not silently skipped.

```powershell
python -m benchmarks.backend_sharing --repeats 21 --output .artifacts/operations-new.json
```

Measures elementwise chains, transpose/reversal, GEMV, GEMM, and regular/broadcast
batches, float32/64, and irregular shapes. It interleaves providers and reports
resident wall time, host-call duration, end-to-end execution, first call, errors,
and samples/p10/p90. Host duration includes internal waits and API calls;
it is not pure Python time or CPU utilization.

To compare a historical wheel without installing it, run the same module in a
separate process with `PYTHONPATH` pointing to the wheel. JSON records the origin
and hashes of the package actually imported. Repeating with `--seed 942` changes
the order, not the data. Historically missing operations are labeled `unsupported`;
they do not count as validated executions or zero timings. Output files must be
new, and GPU workloads must run sequentially.

`--control-wheel <previous-wheel>` alternates previous/current methods on the
same context and resident data: elementwise/gather in OpenCL, matmul in CUDA.
Select a single OpenCL or CUDA backend with `--backends`. Other functions come
from the current runtime, and method swapping occurs outside the clock. This
control reduces context differences; it is not equivalent to loading the entire
historical package. The runner records the wheel, its hash, and the control's scope.

`register_gemm` and `push_descriptor_limit` describe backend capabilities,
not a trace of which kernel each row executed. The Vulkan control also records
the effective mode and commands. The CUDA strategy is checked through native-call
tests and the runtime's selection conditions.

To compare typical use of each API, run one process per provider sequentially,
for example `--backends vulkan` followed by `--backends opencl`, with separate
files. Interleaving several APIs in one process can strongly perturb timings
on this Windows driver. Record that control too when investigating coexistence;
do not confuse it with isolated performance or a physical limit.

## Observed idle checks and kernel experiments

`python -m benchmarks.isolated --backends opencl,vulkan,cuda-custom --rounds 2 --output <new-directory>`
automates sequential processes and observed idle checks before each one: a
minimum 10 s pause, temperature/utilization, three readings, and a timeout that
prevents the next worker from starting. It records clocks, power, and thermal
signals; it does not force hardware settings. The worker warms up again outside
the clock for at least three calls and 0.25 s.

`--worker kernel_variants` compares tile, padding, local-layout, and guard
candidates. `backend_sharing` now adds GEMM 512×256×768 and 1024³, for 22 cases.
`--control-component gemm --control-wheel <wheel>` controls only the GEMM generator
on an OpenCL/CUDA backend, preserving the other current paths. Overrides are
temporary and restored on completion; this control requires a process without
concurrent runtime calls.

The current collector is NVIDIA-specific, and `--gpu-id` selects the monitored
GPU, not the backend device. Verify that they match when several GPUs are present.
Without telemetry, `--allow-unverified` permits only a pause labeled unverified;
it never turns a hot/busy reading into approval on timeout.
See the [protocol, constraints, and results](../docs/kernel-stability-results.md).

## Coupled oscillators

`python -m benchmarks.oscillators --backend numpy --profile quick --repeats 12 --output <new-json>`
evaluates 64 velocity-Verlet steps and separates the discrete recurrence, modal
solution, and energy. `--profile all` adds N=257/2049 to N=31, float32/64. Supported
providers are NumPy, OpenCL custom/CLBlast, Vulkan, CUDA custom/cuBLAS, and CuPy;
OpenPacho compares eager execution with a reused prepared step. Samples include
the initial acceleration on every trajectory, 65 GEMV operations, and start from
the original state. GPU processes require external idle checks and exclusive
device use. See the [protocol and results](../docs/oscillators-results.md).

## CUDA float64 experiment

`python -m benchmarks.cuda_fp64_gemm --profile all --repeats 24 --output <new-json>`
compares the current direct GEMM with a 16×16 shared-memory candidate without
changing the runtime. It requires exclusive device use and an external observed
idle gate before the process. It retains two controls, all six order permutations,
samples, bitwise validation, and applications with independent plans.
`--control-report <previous-json>` reuses captured direct sources.
See the [scope, evidence, and decision](../docs/cuda-fp64-gemm-results.md).

## Float64 contraction

`python -m benchmarks.fp64_contraction --backend cuda-custom --profile all --repeats 24 --output <new-json>`
compares separate rounding and FMA in two CUDA geometries. `--backend opencl`
compares its current program with the same source and `FP_CONTRACT OFF`. The
benchmark owns the modules; they are not installed in runtime caches. It saves
sources, effective options, PTX/binaries, and oracles for both policies, alongside
exact error references and two performance controls. It requires an exclusive
GPU process and external idle checks. It changes no default policy.
See the [protocol and evidence](../docs/fp64-contraction-results.md).

## Staging and random generation

`python -m benchmarks.isolated --worker staging --backends vulkan --control-wheel <previous-wheel> --rounds 2 --repeats 31 --output <new-directory>`
compares transfer methods from a trusted local wheel with current methods on the
same backend/context. Without a wheel, it compares transient/readback modes in
the current code. It alternates blocks with identical warmup and separates diagnostics.

`python -m benchmarks.isolated --worker random_generation --backends numpy-1t,opencl,vulkan,cuda-custom,cupy --rounds 2 --repeats 21 --output <new-directory>`
measures resident results, delivery to the host, and NumPy followed by upload.
It uses the runner's observed pauses; each provider runs in a separate process.
The algorithms differ (OpenPacho Philox, NumPy default_rng, CuPy RandomState);
the report records moments, ranges, and, for OpenPacho, error against the same
Philox transformations on the CPU. NumPy uniform includes conversion to float32;
random and the standard normal/exponential variants use NumPy's native dtype.
It does not measure comprehensive statistical quality or promise sequence
reproduction across providers.
