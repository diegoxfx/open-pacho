# Techniques to adopt from CLBlast

Research dated 2026-09-07. The local native library is CLBlast 1.7.0;
the linked sources explain its design and do not establish exactly which variant
that binary selected for each call. Timing differences are not attributed entirely
to the Python layer: the custom kernel also had considerable room for optimization.

## The main difference

OpenPacho already used tiles in local/shared memory, but each thread accumulated
only one output. CLBlast combines that reuse with private accumulator blocks:
a thread computes several outputs and reuses each loaded value in several
multiplications. Its direct kernel declares private A, B, and C blocks, loads
from local memory into registers, and applies multiply-add to those blocks.
This is visible in the [direct kernel](https://github.com/CNugteren/CLBlast/blob/master/src/kernels/level3/xgemm_direct_part3.opencl).

The technique applied here is an independent implementation shared by OpenCL
and Vulkan in `_gemm.py`: each group of 64 threads computes 32×32 outputs; each
thread holds 4×4 accumulators and traverses K in blocks of 16. It requires 4 KiB
of shared memory. The path covers float32, M/N≥32, and K≥16, preserves broadcasting,
and checks boundaries. Other types/sizes keep the previous paths.
CLBlast remains an explicit provider and is not replaced when selected.

## What to implement and what to defer

| Technique | Decision | Reason |
| --- | --- | --- |
| Private blocks and local→register reuse | Implemented in OpenCL/Vulkan | Improves computation without changing the API or public storage. |
| Contiguous global loads and accumulators with constant indices | Implemented | Lets the compiler optimize registers; requires no vendor-specific instructions. |
| Explicit vectorization and tuned unrolling | Next bounded experiment | Alignment, register pressure, and shape determine whether it wins. |
| Prior packing, padding, and transposition | Opt-in padding added on 2026-09-18; the rest deferred | [Padding contract](padding.md): adds memory, kernels, and preparation; must be amortized by enough work. |
| Parameter database by family and exhaustive autotuning | Secondary track | Requires more hardware and persistent/versioned results. |
| FP16/TF32 or fast-math to improve a comparison | Outside the current contract | Would change precision; any future mode must be explicit and measured with error metrics. |

CLBlast maintains a direct path for general shapes and an indirect path with
storage preparation. For large matrices, moving O(n²) elements can be amortized
against O(n³) operations; for small ones, it can dominate. Its parameters include
geometry, register blocks, vector widths, and local memory usage.
See [GEMM details](https://github.com/CNugteren/CLBlast/blob/master/doc/details_gemm.md).

Specific values are not universal. CLBlast has tuning results by device/family
and a two-stage search for GEMM; it also tunes the choice between direct and
indirect paths. Reproducing its entire database is unnecessary to start improving
performance. See [official tuning documentation](https://github.com/CNugteren/CLBlast/blob/master/doc/tuning.md)
and the [CLBlast paper](https://arxiv.org/abs/1705.05249).

## Local evidence for the implemented technique

Interleaved resident control, eleven samples per variant, float32; same process,
inputs, and backend. The previous column disables only the new register path.
These are medians of full synchronized time, not isolated kernel times.

| Backend / GEMM | Previous local tiling | Local + registers | Relative L2 error against float64, both variants |
| --- | ---: | ---: | ---: |
| OpenCL 512² | 0.324 ms | 0.247 ms | 4.045e-7 |
| OpenCL 1024² | 1.681 ms | 0.934 ms | 5.741e-7 |
| OpenCL 2048² | 12.092 ms | 6.126 ms | 8.099e-7 |
| Vulkan 512² | 0.477 ms | 0.331 ms | 4.045e-7 |
| Vulkan 1024² | 1.679 ms | 0.694 ms | 5.741e-7 |
| Vulkan 2048² | 11.262 ms | 2.930 ms | 8.099e-7 |

At 2048² the maximum absolute error was 1.517e-4 in both paths. Identical metrics
in the tested cases do not promise bitwise equality for every matrix. Neither
fast-math nor reduced precision was enabled. Adversarial cancellation cases are
published separately and can have large relative error.

The control code and data are in `.artifacts/numerics-vulkan/register_gemm.py`
and `.artifacts/benchmarks/numerics-vulkan/register-gemm.json`. Full comparisons
against CLBlast, CuPy, and nvmath, including transfers, are in the
[report for this iteration](numerics-vulkan-results.md).
