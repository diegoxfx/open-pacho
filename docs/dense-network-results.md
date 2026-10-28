# A dense-network client with explicit gradients — 2026-10-22

`examples/dense_network.py` implements inference and one SGD step for a regression
network with two affine layers and an intermediate tanh activation. It is a client
of existing operations: it neither extends the runtime, adds autodiff, nor installs
a network framework. Its purpose is to check and jointly measure GEMM,
broadcasting, activations, reductions, transposes, and temporaries.

The family comes from the [application plan](application-measurement-plan.md).
Domain code uses public operations and works directly with NumPy,
CuPy, and OpenPacho. `compile` captures the outer functions; it contains no nested
compiled calls, downloads, or backend changes. The wheel still excludes
examples, and NumPy remains the only mandatory dependency.

## Mathematical contract

For X of shape (n,d), W1 (d,h), b1 (h,), W2 (h,o), b2 (o,), and Y (n,o):

```text
H  = tanh(X @ W1 + b1)
P  = H @ W2 + b2
L  = sum((P - Y)^2) / (2*n*o)
D2 = (P - Y) / (n*o)
dW2 = H.T @ D2             db2 = sum(D2, axis=0)
D1  = (D2 @ W2.T) * (1-H*H)
dW1 = X.T @ D1             db1 = sum(D1, axis=0)
```

`predict` returns P. `loss_and_gradients` returns L and the four gradients.
`sgd_step` returns **the loss before the step** and four new parameters:
W1−rate·dW1, b1−rate·db1, W2−rate·dW2, b2−rate·db2. All gradients use
the original parameters; neither inputs nor their parameter arrays are modified.
The loss also averages over the output dimension, not just batch rows.

Dimensions must be positive, and arrays must share a float32/64 dtype and
backend. Shapes and dtypes are checked through metadata. Finite inputs and
a finite positive rate are the caller's responsibility; GPU data is not observed
to validate them. The example does not demonstrate that an arbitrary rate makes
training converge. Vulkan retains its explicit error for float64 tanh.

## Measurement protocol

`benchmarks/dense_network.py` uses the same domain functions across all
comparators. Each process selects one backend: NumPy with 1/8 threads, OpenCL
custom/CLBlast, Vulkan, CUDA custom/cuBLAS, or CuPy. OpenPacho has eager and
compiled rows; compiled rows prepare the plan once and reuse it.

The full profile contains three shapes (n,d,h,o): (31,7,11,3),
(4096,128,256,32), and (4095,127,255,31), in float32/64, for inference and one SGD step.
These are synthetic data with seed 47, inputs and weights scaled by dimension,
nonzero bias, and fixed targets. The rate is 0.1. Each SGD call starts from the same
original parameters: it does not measure a trajectory whose state changes between variants.

Inference uploads only X and the four parameters; it does not upload unused targets.
The SGD step uploads all six arrays and downloads the loss and four new parameters.
Transposes of activations, weights, and X are part of the computation; they are
not hidden in external preparation. The following are recorded:

- Initial upload, plan preparation, and synchronized first execution, which may
  compile device kernels.
- Warm-up of at least three calls and 0.1 s.
- Twelve resident samples per implementation: allocations, computation, and synchronization.
- Twelve complete samples: upload, computation with a reusable plan/eager execution, and download
  of all declared outputs. Preparation is reported separately.
- Upload and first-download samples outside the preceding timers, input/output
  bytes, versions, sources, threads, and device. **Peak memory
  is not measured**; bytes are not presented as a substitute.

The two implementations in each process alternate in balanced orders.
Diagnostics remain outside the timer and include every output before and during
repetitions. SGD also evaluates gradients and prediction; gradient
checks use compiled execution when the measured row is compiled.

The float64 reference converts the same already-rounded data without generating a
different input. For float64 rows it is identified as `sameprecision`: it is not
an oracle with higher precision. Thresholds fixed before measuring are
rtol=2e-4, atol=2e-6 for float32 and rtol=2e-10, atol=2e-12 for float64. Absolute
and relative errors, dtypes, shapes, and nonfinite values are recorded. A numerical
rejection retains timings; an execution failure remains in its row and allows
the remaining rows to be attempted. `attempts_complete`, `execution_complete`, and
`all_numerically_accepted` express distinct conditions.

```powershell
.venv/Scripts/python.exe -m benchmarks.dense_network --backend numpy --profile quick --repeats 12 --output .artifacts/benchmarks/dense-numpy.json
.venv/Scripts/python.exe -m benchmarks.dense_network --backend opencl --profile all --repeats 12 --output .artifacts/benchmarks/dense-opencl.json
```

Individual commands require new files and do not themselves apply
the supervisor's GPU idle gate. A nonzero return preserves the JSON of
errors or numerical rejections; it does not mean that remaining rows were lost.

## Functional verification

Central finite differences for every parameter in a small float64 network
independently verify the gradients, the n·o factor, and the
update. Saturated activations, input preservation,
float32/64, differing dimensions, a single row, and array/rate rebinding
in the compiled plan are also checked. Loss decreases in one case with a small rate; that
observation is not extrapolated to every problem.

24 CPU example tests and eight native tests pass: OpenCL, Vulkan,
CUDA custom, and CUDA cuBLAS, including the expected Vulkan float64 tanh rejection.
The runner's 13 CPU tests check references on rounded data,
download of every output, original parameters between calls, balanced
modes, actual preparation in NumPy, and failure retention. The full suite was not
run and the runtime was not changed.
## Local results

Two rounds with reversed backend order were completed, plus two supplementary
CuPy processes. The first two CuPy attempts failed when creating the cache
outside the allowed directory; their 24 error rows are retained. Retries
use `CUPY_CACHE_DIR` within the workspace, without changing inputs, code, or tolerances.
Because they ran after the original queue, their positions are not interleaved
with other backends. Original reports are not replaced.

There are **288 executed and accepted rows**, with **6,912 resident/complete samples**.
Also retained are 24 Vulkan float64 rows rejected because tanh is unsupported and
the 24 initial CuPy cache-failure rows. All reports finished their
attempts; reports containing failures claim neither complete execution nor global acceptance.
Input hashes per case match across comparators and rounds, and loaded
runtime, example, and runner sources remained constant.

All 14 GPU gates passed without relaxing policy: at least ten seconds,
three consecutive GPU/memory utilization readings ≤5%, maximum 52 °C and two
degrees above baseline temperature. Observed waiting time was 10.48–73.08 s
and starting temperature 40–42 °C. This describes idle state before each process,
not temperature measurement for each kernel. Environment: Windows, RTX 5060 Ti,
driver 617.14, Python 3.12.14, and NumPy 2.5.3; other versions and library/thread
information are in the JSON files.

Maximum observed absolute errors, including replay and all
executed comparators, were:

| Dtype | Prediction | Loss | Gradients | New parameters |
| --- | ---: | ---: | ---: | ---: |
| float32 | 3.66e-7 | 4.69e-9 | 1.19e-8 | 2.92e-8 |
| float64 | 6.67e-16 | 1.39e-17 | 2.61e-17 | 1.39e-17 |

They are rounded upward for this summary; they are not bounds for other inputs.
The 23,328 diagnostics for accepted rows met the published thresholds.

Tables show **milliseconds**, the range of the two medians, for the regular
shape (4096,128,256,32). R = resident; C = complete. Preparation and first
call are separate. These are neither confidence intervals nor a universal ranking.

### float32

| Implementation | Inference R | Inference C | SGD R | SGD C |
| --- | ---: | ---: | ---: | ---: |
| numpy-1t | 7.378–7.418 | 7.526–7.593 | 15.169–15.325 | 15.111–15.302 |
| numpy-8t | 4.709–4.772 | 4.920–4.985 | 8.904–9.125 | 9.009–9.211 |
| opencl-eager | 0.364–0.380 | 1.067–1.112 | 2.194–2.359 | 3.149–3.597 |
| opencl-compiled | 0.266–0.267 | 0.884–0.893 | 1.237–1.447 | 2.153–2.659 |
| opencl-clblast-eager | 0.517–0.545 | 1.393–1.596 | 2.550–2.551 | 3.435–3.497 |
| opencl-clblast-compiled | 0.308–0.317 | 1.007–1.274 | 1.214–1.228 | 2.165–2.189 |
| vulkan-eager | 0.727–0.932 | 1.360–1.447 | 3.901–4.292 | 5.170–5.491 |
| vulkan-compiled | 0.769–0.788 | 1.302–1.414 | 1.674–1.802 | 2.694–2.834 |
| cuda-custom-eager | 0.621–0.621 | 1.436–1.455 | 3.323–3.353 | 4.422–4.487 |
| cuda-custom-compiled | 0.298–0.300 | 1.082–1.091 | 1.486–1.501 | 2.554–2.589 |
| cuda-cublas-eager | 0.692–0.795 | 2.009–2.247 | 3.463–3.557 | 4.553–4.613 |
| cuda-cublas-compiled | 0.500–0.509 | 1.600–1.717 | 1.290–1.300 | 2.342–2.368 |
| cupy-eager | 0.308–0.423 | 1.009–1.340 | 1.103–1.127 | 1.767–1.783 |

### float64

| Implementation | Inference R | Inference C | SGD R | SGD C |
| --- | ---: | ---: | ---: | ---: |
| numpy-1t | 22.302–22.811 | 22.618–22.952 | 36.916–37.801 | 36.868–37.602 |
| numpy-8t | 16.474–16.983 | 17.319–17.640 | 24.931–26.244 | 24.482–25.740 |
| opencl-eager | 1.098–1.101 | 2.487–2.598 | 2.771–2.867 | 4.212–4.671 |
| opencl-compiled | 1.070–1.077 | 2.422–2.445 | 2.679–2.684 | 3.995–4.227 |
| opencl-clblast-eager | 1.145–1.156 | 2.500–2.503 | 2.745–2.750 | 4.040–4.169 |
| opencl-clblast-compiled | 1.116–1.117 | 2.492–2.594 | 2.618–2.634 | 4.029–4.060 |
| vulkan-eager | unsupported | unsupported | unsupported | unsupported |
| vulkan-compiled | unsupported | unsupported | unsupported | unsupported |
| cuda-custom-eager | 2.226–2.230 | 3.767–3.884 | 5.066–5.093 | 6.714–6.729 |
| cuda-custom-compiled | 2.178–2.181 | 3.686–3.705 | 4.917–4.923 | 6.441–6.525 |
| cuda-cublas-eager | 1.374–1.394 | 2.873–3.047 | 3.349–3.487 | 4.867–4.944 |
| cuda-cublas-compiled | 1.322–1.351 | 3.011–3.093 | 2.830–2.835 | 4.345–4.360 |
| cupy-eager | 1.296–1.350 | 2.639–2.714 | 2.680–2.727 | 3.938–4.045 |

### Odd shapes and small case

For (4095,127,255,31), these ratios compare the compiled median with the eager median
in the same process. Both rounds are shown, including mixed results:

| Backend/dtype | Inference R | Inference C | SGD R | SGD C |
| --- | ---: | ---: | ---: | ---: |
| opencl/float32 | 0.676–0.690 | 0.824–0.836 | 0.626–0.673 | 0.717–0.794 |
| opencl/float64 | 0.981–0.984 | 0.986–0.991 | 0.964–0.970 | 0.945–0.990 |
| opencl-clblast/float32 | 0.610–0.652 | 0.846–0.873 | 0.467–0.495 | 0.635–0.667 |
| opencl-clblast/float64 | 0.977–0.980 | 0.979–1.008 | 0.925–0.938 | 0.938–0.983 |
| vulkan/float32 | 0.748–0.816 | 0.831–0.831 | 0.387–0.460 | 0.492–0.575 |
| cuda-custom/float32 | 0.484–0.509 | 0.757–0.757 | 0.349–0.361 | 0.432–0.498 |
| cuda-custom/float64 | 0.977–0.982 | 0.986–0.998 | 0.968–0.983 | 0.959–0.991 |
| cuda-cublas/float32 | 0.467–0.512 | 0.767–0.777 | 0.303–0.380 | 0.448–0.528 |
| cuda-cublas/float64 | 0.962–0.974 | 0.968–0.982 | 0.781–0.860 | 0.886–0.904 |

For small float32 inference, NumPy takes 0.026–0.028 ms complete;
GPUs take 0.416–1.042 ms. Dispatch/transfer costs matter:
large-case results do not justify automatically moving small workloads to the GPU.
The backend remains an explicit user choice.

## What changes the priority

Existing fusion and preparation clearly reduce float32 SGD-step cost
in these cases. Float64 gains tend to be smaller; some inference paths
have mixed complete results, including cuBLAS. CuPy timings and the
two rounds also show variation: small differences are not attributed
to a demonstrated improvement, nor is universal parity between libraries claimed.

The client reveals a specific area: CUDA custom float64 trails
cuBLAS/OpenCL. Inspection confirms that its float64 GEMMs use the direct kernel,
while OpenCL uses a shared-memory tile. This justifies breaking down
the flow and testing a bounded variant; **the application comparison does not yet
quantify exclusive kernel time or demonstrate a possible speedup**.
No kernels or provider selection are changed in this batch.

Reports, samples, diagnostics, and supervisors are retained in
`.artifacts/iterations-2026-10-22/dense-network/` and its parent directory. Tables
come from those reports; the small case, odd shapes,
explicit failures, and distributions of all samples are retained.

Independent QA: **APPROVE**. It reran the 37 CPU tests, inspected the eight
native tests, and reconciled the 140 table cells, samples, diagnostics,
idle gates, and hashes. It confirmed that the runtime remains identical to the previous
batch. It did not execute GPU work during review.
