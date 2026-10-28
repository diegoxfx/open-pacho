# Vulkan: typed CFFI boundary and diagnostics — 2026-09-12

## Result and scope

The per-device command table reduces generated-wrapper costs in frequent calls.
It remains a Python executor with CFFI calls; no C/C++ extension was added and
the full Vulkan adapter was not rewritten. It does not change the public array
API, kernels of measured workloads, barriers, pools, submission count, solver
checks, or precision.

Local control: RTX 5060 Ti, Windows 11, driver 616.64, and Vulkan binding 1.3.275.1.
Two independent processes, 30 cases per process, interleaved wrapped/direct paths
with 21 samples per mode. Includes float32/64, rectangular/odd shapes, operations,
applications, solvers to convergence, and scalar/large transfers.
The following ranges are the **two medians**, not confidence intervals.

| Case | dtype | Wrapped ms | Direct ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| gemv 2049x1023 | float32 | 0.309–0.316 | 0.277–0.314 | -1.8–12.3 % |
| gemv 2049x1023 | float64 | 0.353–0.366 | 0.335–0.338 | 5.2–7.7 % |
| gemm 257x129x513 | float32 | 0.266–0.270 | 0.245–0.247 | 7.7–8.4 % |
| gemm 257x129x513 | float64 | 0.393–0.409 | 0.358–0.372 | 8.9–9.2 % |
| covariance 4097x129 | float32 | 0.771–0.792 | 0.658–0.666 | 13.5–16.9 % |
| covariance 4097x129 | float64 | 1.093–1.129 | 0.997–1.002 | 8.3–11.6 % |
| ridge 4097x129 | float32 | 7.024–7.101 | 5.424–5.624 | 20.8–22.8 % |
| ridge 4097x129 | float64 | 7.151–7.369 | 5.466–5.602 | 23.6–24.0 % |
| cg 513x513 | float32 | 14.922–15.997 | 12.435–13.650 | 14.7–16.7 % |
| cg 513x513 | float64 | 24.392–24.735 | 20.141–20.338 | 16.6–18.6 % |
| jacobi 513x513 | float32 | 18.653–20.292 | 15.155–15.832 | 18.8–22.0 % |
| jacobi 513x513 | float64 | 31.548–32.179 | 25.454–25.595 | 18.9–20.9 % |
| sd 513x513 | float32 | 30.513–32.963 | 25.049–26.119 | 17.9–20.8 % |
| sd 513x513 | float64 | 57.564–59.371 | 46.700–48.915 | 17.6–18.9 % |

In these controls, solvers reduce resident time by 14.7–22.0%; ridge by
20.8–24.0%. Float32 GEMV is mixed across processes; large uploads can remain
unchanged or become slightly worse. No universal improvement is claimed.

Full mode also includes upload, preparation, and download. For ridge
4097×129 float32, full medians are:

- Wrapped: 7.514–8.054 ms.
- Direct: 5.809–6.312 ms.

Isolated transfers measure the upload/download API with staging and
synchronization, not pure PCIe bandwidth. The complete local table
(`.artifacts/vulkan-dispatch/comparison.csv`, not distributed) retains both runs,
all cases, L2 errors, and iteration counts.

## What changed

`_vulkan_commands.py` resolves a table per VkDevice through
[vkGetDeviceProcAddr](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceProcAddr.html).
The backend passes typed handles, native arrays, and NULLs: it avoids generic
per-argument inspection/conversion and uses device entry points. The existing
optional binding is retained for creation, allocation, and other calls.

- VkResult functions check errors and preserve binding exceptions.
- The table neither modifies the global Vulkan module nor shares pointers
  between VkDevices.
- `auto` uses wrappers if it cannot resolve the typed path; it does not switch
  backend. An execution error propagates without a concealing retry or fallback.
- Commands, leases, in-flight resources, locks, staging, barriers, and the
  64-operation limit retain their contract. Input validation is not relaxed.
- `_command_mode` is a private constructor control for diagnostics, not a setting
  the user needs to know to obtain the improvement.

`wrapped` is the generated binding with the current runtime's same typed records;
it is not a byte-for-byte old wheel. Initial `before-gaps` and `direct-gaps`
measurements were exploratory and do not support the causal table.

## What the diagnostics reveal

`benchmarks/vulkan_dispatch.py` records outside performance samples:

- Host time of each API (includes binding, driver, and possible waits).
- GPU timestamps around dispatch and copy; respects timestampValidBits, period,
  wraparound, query reset, and lifetime until commands complete.
- Counts of dispatches, barriers, descriptors, submissions, waits, mappings,
  and allocations.
- Bytes copied, new allocations, and retained cache. **It does not measure total peak VRAM.**

Timestamps use TOP/BOTTOM_OF_PIPE and add commands/dependencies. Their intervals
include instrumentation/stage effects; they are not pure shader times and are
not subtracted from uninstrumented wall time to invent “Python time.”
See [Vulkan timestamps](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html).

Diagnostic example from the first run, float32 ridge: the 80 dispatches,
80 barriers, 20 submissions, and two waits are identical. Total time measured
inside APIs changes from approximately 3.16 to 1.65 ms; GPU intervals sum to
0.352/0.355 ms. These values belong to an instrumented invocation and do not
replace its wall-time medians of 7.10/5.62 ms. They indicate room in
execution/preparation; they do not establish a physical limit for Vulkan,
NVIDIA, or PCIe.

## Independent numerical correction

The exploratory energy test using `** 2` discovered NaN for negative bases.
This was not an error in the new dispatch: GLSL defines
[pow as out of domain for x<0](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#exponential-functions).
The facade promised NumPy real-valued behavior for representable integer exponents.

A float32 helper was added that distinguishes integer/odd exponents, signed
zeros, NaN, and infinities and avoids calling pow outside its defined domain.
It is tested in eager, compile, and map-reduce with scalar rebinding. Float64
power remains out of scope. The energy control uses the existing benchmark's
`y*y`; the power correction is not presented as a dispatch speed improvement.

All wrapped/direct outputs and instrumented outputs agree in the final cases;
float64 error metrics, iterations, and counts were retained. For solvers, the
true residual of represented inputs is validated: stopping tolerance does not
guarantee a per-component relative solution tolerance. Error against the
float64 solution is published additionally.

The adversarial Vulkan suite was repeated with three samples: 42 configurations,
ten warnings, and zero exceptions. These are the same ten warnings in the Vulkan
subset of the initial CUDA round. Cancellation, subnormal, and ill-conditioned
system limitations remain visible; improving the executor does not fix them.

## Complementary application comparison

The existing covariance and ridge suite was run for 4095×127 and 4097×129
matrices, float32/64: 72 configurations and seven samples, without errors or
missing backends. The table shows resident medians in ms for 4097×129. OpenPacho
uses compile; CuPy executes its existing version of the same operations. NumPy
is measured with one and eight threads. Ridge retains twenty steps in all providers.

| Application | dtype | NumPy 1t | NumPy 8t | OpenCL custom | OpenCL CLBlast | Vulkan | CuPy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Covariance | float32 | 1.690 | 1.099 | 2.081 | 1.538 | 1.179 | 1.131 |
| Covariance | float64 | 2.740 | 1.784 | 3.044 | 3.170 | 1.702 | 1.858 |
| Ridge | float32 | 1.530 | 2.971 | 6.042 | 6.671 | 6.694 | 5.407 |
| Ridge | float64 | 2.634 | 3.525 | 7.542 | 8.628 | 6.586 | 6.088 |

In this sample, Vulkan is close to CuPy in covariance and compiled OpenCL in
ridge. Single-threaded NumPy still clearly wins for ridge at this size.
The small Vulkan/CuPy covariance difference does not establish a stable win:
it is a complementary run, not a paired repetition across processes. Its
absolute times must not be mixed with those of the wrapped/direct control either.

The JSON retains all modes, p10/p90, preparation, transfers, full time, and
numerical error. Modes use independent samples; their medians are not additive
components of one invocation. For float32 covariance, OpenPacho GPU's relative
L2 error is around 7.9e-7 versus approximately 9.3e-8 for CuPy: approaching its
time does not imply matching precision. For float64, this suite's reference
has the same precision and does not certify exactness.

## Reproduction

```powershell
$env:PACHO_VULKAN_COMPILER='<path to glslang or glslc>'
python -m benchmarks.vulkan_dispatch --repeats 21 --output .artifacts/vulkan-dispatch/run-a.json
python -m benchmarks.vulkan_dispatch --repeats 21 --seed 4782 --output .artifacts/vulkan-dispatch/run-b.json
python -m benchmarks.accuracy --backends vulkan --repeats 3 --output .artifacts/vulkan-dispatch/accuracy-new.json
python -m benchmarks.applications --irregular --backends opencl,opencl-clblast,vulkan,cupy --repeats 7 --output .artifacts/vulkan-dispatch/applications-new.json
```

Use new files. Do not run other CPU/GPU workloads during measurement.
Warm-up, references, validation downloads, and diagnostics are outside the timer.
Solvers are applications in `examples/`; ridge retains twenty steps, not a
convergence claim. JSON files retain first call and applied preparation;
this is not a process cold-start study.

Final results: `.artifacts/vulkan-dispatch/final-{1,2}.json`. The `quick*` and
`paired-*` runs retain runner explorations/failures. `paired-1` incorrectly
applied per-component validation to a solver; the corrected criterion uses its
contractual residual without changing the algorithm, tolerances, or timed data.
See [verification](verification.md) for tests, adversarial accuracy, and packaging.

## Next room for improvement

1. Reduce preparation/recording and Python→C crossings per operation group;
   the next experiment could be a small native recorder with the same plan.
2. Measure waits and scalar staging without changing example convergence control.
3. Maintain generic kernel/layout and compiler improvements as parallel tracks.
4. Do not attribute driver/queue times to hardware before isolating them. A
   complete new adapter, family-specific autotuning, and new APIs remain later decisions.
