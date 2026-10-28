# Vulkan, GEMM and numerical stability

Historical snapshot. The [continuation](solver-applications-results.md) groups
solver steps and command buffers, expands applications and prepares the CUDA interface.

Iteration of 2026-09-07. Continues the [previous suite](remaining-suite-results.md).
RTX 5060 Ti 16 GB, Ryzen 7 5700X3D, 32 GB RAM, Windows 11, driver 616.64.
Python 3.12.14, NumPy 2.5.3, PyOpenCL 2026.1.4, Vulkan Python 1.3.275.1,
glslang 16.5.0, CLBlast 1.7.0, CuPy 14.2.0 and nvmath-python 1.0.0.
Physical validation remains limited to this machine.

**There are significant improvements within the runtime and kernels. Vulkan now exceeds
CLBlast on the measured resident 2048² float32 GEMM, but still loses on
transfers and solvers. Accuracy is now a visible part of the result.**

## Implementations

1. **Less Vulkan object construction.** A 128-entry LRU of dispatch templates
   with source, ABI and geometry; CFFI descriptor structures retained per
   lease and rebound after their use completes. Submission,
   barrier and copy structures are reused under the lock. In-flight resources remain retained;
   this improvement alone does not reduce the number of submits.
2. **GEMM with register blocks.** OpenCL custom and Vulkan float32 compute
   32×32 outputs per 64-thread group, with 4×4 accumulators per thread and K=16 tiles.
   Broadcasting and edges are preserved. Selection checks shape and capabilities;
   other precisions/sizes retain previous paths. The
   [CLBlast investigation](clblast-techniques.md) explains the technique and what is deferred.
3. **Fused GEMV and residual.** The compiler can turn the private pattern
   `b - A @ x` into an operation with an epilogue for OpenCL custom/Vulkan and matching
   floating dtypes. The CLBlast provider is unchanged. Shared products,
   public outputs and ineligible cases retain the boundary. This is not general
   fusion of any expression around matmul.
4. **Stable norms and coefficients.** A scaled-norm primitive avoids
   forming extreme squares. CG and steepest descent use normalized
   directions; use of `r·r` for those coefficients is removed. Dtype,
   true residual at every iteration, tolerance, validation and convergence
   information are preserved. It does not guarantee every extreme problem is representable.
5. **Observable accuracy.** Times accompanied by absolute error, relative L2 error,
   nonfinite count and reference. Solvers add an independent residual and
   backward error. An adversarial suite uses Decimal80 over the inputs actually
   represented. See [method and reproduction](numerical-accuracy.md).

## Controls isolating improvements

GEMM: eleven samples interleaved within the same process. The previous variant
disables only the register path. Synchronized resident medians:

| Backend / float32 | Previous local tiling | Local + registers | Improvement |
| --- | ---: | ---: | ---: |
| OpenCL 512² | 0.324 ms | 0.247 ms | 1.31× |
| OpenCL 2048² | 12.092 ms | 6.126 ms | 1.97× |
| Vulkan 512² | 0.477 ms | 0.331 ms | 1.44× |
| Vulkan 2048² | 11.262 ms | 2.930 ms | 3.84× |

At 2048² both variants obtain relative L2 error 8.099e-7 and maximum absolute error
1.517e-4 against float64. Metrics are equal on these inputs, without a promise of
universal bitwise equality. TF32 and fast-math were not enabled.

The Vulkan execution control alternates three processes of each version, with 31
resident samples per operation. A 64K-element sum goes from 0.328–0.332 to
0.223–0.235 ms; SAXPY from 0.331–0.382 to 0.218–0.238 ms; total reduced energy from
0.439–0.445 to 0.286–0.324 ms. Profiles count 36–41 % fewer Python calls.
Their inclusive time contains waits: it is not interpreted as CPU utilization.

The GEMV residual control interleaves fusion/separate subtraction on the same plan.
At 2048², OpenCL goes from 0.163 to 0.077 ms and Vulkan from 0.348 to 0.239 ms. At 8192²,
gains are smaller: 0.790→0.739 and 1.008→0.907 ms, respectively. Maximum
difference between results was zero in all six 512/2048/8192 controls.
During implementation, propagation of GLSL `precise` that changed contraction
of the previous GEMV was corrected; cancellation and NaN/Inf regressions exist.

## Current complete comparison

319 performance configurations: 172 operations, 63 solvers and 84 scale cases.
Each mode has seven samples, with alternating order between implementations,
no validation errors or missing references. First execution,
resident data and upload+computation+download are separated. Diagnostics are calculated outside
the timer, but may affect caches/thermal state between samples; wall time
also includes Python, driver and waiting. These are not isolated kernel times.

2048² float32 GEMM, medians; compiled OpenPacho paths are shown:

| Implementation | Resident | Upload + computation + download | Relative L2 error vs float64 |
| --- | ---: | ---: | ---: |
| NumPy, 16 threads | 29.258 ms | 33.179 ms | 3.077e-7 |
| OpenCL custom | 6.902 ms | 14.848 ms | 8.099e-7 |
| OpenCL + CLBlast | 3.929 ms | 11.830 ms | 8.099e-7 |
| Vulkan | 3.047 ms | 15.759 ms | 8.099e-7 |
| CuPy | 1.419 ms | 10.180 ms | 4.698e-7 |
| Direct cuBLAS | 1.429 ms | 10.059 ms | 4.698e-7 |
| nvmath-python | 1.517 ms | 10.229 ms | 4.698e-7 |

Vulkan approaches on computation, but its download/staging remains costly. This also
appears in float32 GEMV 8192² (256 MiB matrix): OpenCL 0.867/39.867 ms,
Vulkan 1.054/98.134 ms and CuPy 0.852/53.706 ms, resident/complete. NumPy with 1 thread
records 8.996/8.462 ms. The GPU benefits from residency, but uploading the matrix on every
call reverses the comparison. Vulkan remains far from OpenCL's transfer cost
on the same PCIe link: attributing the entire difference to the bus is incorrect.

Complete solvers, 2048 float32 variables, `rtol=1e-4`, true residual at every iteration:

| Implementation | Jacobi, 37 iter. | Steepest descent, 30 iter. | CG, 13 iter. |
| --- | ---: | ---: | ---: |
| NumPy, 8 threads | 9.648 ms | 12.722 ms | 8.390 ms |
| OpenCL custom | 8.839 ms | 21.198 ms | 9.120 ms |
| OpenCL + CLBlast | 12.079 ms | 22.438 ms | 11.841 ms |
| Vulkan | 21.533 ms | 41.966 ms | 20.424 ms |
| CuPy | 10.228 ms | 17.757 ms | 10.501 ms |

Independent relative residuals remain near 8.907e-5 / 9.568e-5 /
7.086e-5. Relative solution error against the generating solution is
approximately 8.874e-5 / 3.490e-4 / 1.712e-4. The generating solution is not an
exact reference for the system after rounding the RHS.

The previous suite recorded higher values for Vulkan Jacobi/SD/CG
at 2048 float32; that is historical context, not a causal control: this stage
also changes the recurrence and norm. NumPy/CuPy comparators now use
the same normalized recurrence; CuPy obtains the solver norm through
cuBLAS NRM2. The entire historical change is not attributed to the executor.

Mixed results are retained. Compiled Vulkan energy on
2048×32 went from 0.712 ms in the historical suite to 1.663 ms; on 32768×32,
from 0.840 to 1.272 ms. An additional control of the same computation in three processes per
version and 31 samples gave 0.369–0.401→0.223–0.236 ms and
0.405–0.420→0.265–0.276 ms, with the same maximum error 1.526e-5 against NumPy.
The regression does not reproduce in isolation; the mixed-suite observation
is not erased either, nor is its sensitivity to the measurement environment considered resolved.

The 48 preparation cases also have seven samples per mode and
validation. Preparation does not ensure acceleration: CG OpenCL float32 2048 gives
10.274 ms ordinary and 13.477 ms prepared in this round; Vulkan gives 22.124 and
21.737 ms. The snapshot/validation contract is useful, but it is not yet a
reusable factorization or an autonomous GPU executor. This benchmark uses
`rtol=1e-5`, so it must not be mixed with times from the previous table.

## What errors show

264 adversarial configurations, three samples per case: **68 indicators outside the
diagnostic threshold, zero exceptions and zero missing backends**. These are not 68
proven bugs; the fixed threshold aids investigation and does not account for the conditioning
of each operation. The suite retains failures from references and OpenPacho.

| Case | Evidence | Interpretation |
| --- | --- | --- |
| float32 norms of order 1e-25/1e25 | OpenPacho: relative error 4.511e-8 / 2.664e-8; native NumPy/CuPy norm functions give zero/Inf | Algorithm defect fixed with scaling; does not describe cuBLAS NRM2. |
| float64 norms and solvers of order 1e±200 | OpenPacho cases remain below their diagnostic threshold | Extreme squares are not formed and no dtype change is required. |
| Sum/DOT/GEMM with adversarial cancellation | 40 indicators across implementations and dtypes | Accumulation order and accuracy matter; compensation or wider accumulation are future measurable decisions. |
| CG, Hilbert 8 float32, Vulkan | Decimal relative residual 5.455e-6; solution error 0.9663 | Meeting residual tolerance does not guarantee an accurate solution; estimated condition number 1.499e9. |
| CG, Hilbert 8 float64, Vulkan | Decimal relative residual 5.487e-12; solution error 1.498e-4 | Double precision helps, but conditioning and stopping criteria still matter. |
| Sum of the smallest float32 subnormal | Vulkan/CuPy: relative error 1; OpenCL/NumPy preserve it in this control | Requires distinguishing execution mode, driver and possible alternative algorithms. |

The total comprises 40 cancellation cases, 18 Hilbert cases, 8 native
NumPy/CuPy norm cases and 2 subnormal-sum cases. Hidden accuracy
changes are not enabled to make indicators disappear.

The local `VkPhysicalDeviceFloatControlsProperties` query reports
`shaderDenormPreserveFloat32=false` and `shaderDenormPreserveFloat64=false`.
Thus this driver does not offer the standard `DenormPreserve 32` mode to fix the
subnormal sum. The backend does not query those controls today either; on another
device advertising them there would be an improvement path. This does not prove
that the silicon cannot handle subnormals through another API or emulation.
See [Vulkan properties](https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceFloatControlsPropertiesKHR.html)
and [SPIR-V modes](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_float_controls.html).
The investigation and query remain in `.artifacts/float-controls`.

## Gaps and next priority

| Gap | What we control | What limits improvement | Next step |
| --- | --- | --- | --- |
| Resident solvers | Fusion, reduction and plan execution | Each host decision retains a synchronization; it is not matrix PCIe traffic | Measure scalar grouping and a small native executor preserving checks per iteration. |
| Vulkan transfers | Staging, host copies, allocation and submission | The local Gen3×8 link limits payload; it does not explain the entire difference from OpenCL | Profile large-transfer paths and reuse before family specialization. |
| Custom GEMM vs CLBlast/CUDA | Register reuse, vectorization, unroll and geometry | Compiler/driver, register pressure, hardware FP64 | Portable vectorization/unroll controls; packing only when amortized. |
| Cancellation and conditioning | Accumulation algorithm, norms, diagnostics and future solver strategy | Finite precision and information lost in inputs | Evaluate compensation and optional accumulation; do not promise that a certified residual implies small forward error. |
| Vulkan subnormals | Capability querying and explicit mode selection where available | This driver does not offer DenormPreserve32 | Retain regression/diagnostic case; investigate alternatives only with measured cost/value. |
| Environment-sensitive results | Protocol, isolation and interleaved controls | Caches, clocks and OS scheduling | Keep complete results and controls; do not turn isolated noise into a selection rule. |

Hardware/OS remains a [secondary track](hardware-specialization.md). There is no
new CUDA backend, sparse support or silent provider selection. AMD/Intel,
other driver versions and remote CI are not considered verified here.

## Verification and artifacts

- **809 tests pass, 1 skipped** because it requires two OpenCL devices. GPU
  backends required and native CLBlast. Includes extreme range, cancellation, limits,
  broadcasting, plans and in-flight resources.
- Wheel built from sdist and installed in an environment without PyOpenCL/Vulkan:
  **249 pass, 527 skipped**, importing from site-packages outside `src`.
- Ruff and formatting pass. Final distributions: `.artifacts/dist-numerics-vulkan`.
- Complete JSON and logs: `.artifacts/benchmarks/numerics-vulkan`;
  `operations.json`, `solvers.json`, `scale.json`, `prepared.json`, `accuracy.json`.
- CSV/Markdown tables with accuracy: `report` subdirectory; adversarial cases in
  `accuracy.md`. Figures: `report/gaps.png` and `report/precision.png`, also SVG.
- Register/energy controls: `.artifacts/numerics-vulkan`; executor/residual
  controls: `.artifacts/vulkan-execution-next`. JSON files record provenance.

Float64 results from the large suite are compared against float64: that part
does not certify accuracy. Decimal80 is reserved for small adversarial cases and
also has finite precision. Those references are identified in each row.
