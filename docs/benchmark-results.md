# Benchmark results — 2026-09-06

**The hypothesis holds in many cases, but not as a universal claim.**
OpenPacho and CuPy won on certain workloads even including transfers.
Jacobi and gradient descent lost at smaller sizes and won at 8192
variables in the test systems.

Hardware: Ryzen 7 5700X3D (8 cores, 16 threads), approximately 32 GB RAM,
16 GB RTX 5060 Ti, Windows 11. NumPy 2.5.3/OpenBLAS 0.3.34.106.0,
PyOpenCL 2026.1.4, CuPy 14.2.0. CuPy uses CUDA as an independent control;
OpenPacho still runs its OpenCL backend. TF32 disabled.

A subsequent check under repeated 64 MiB copies confirmed an active
**PCIe Gen3 ×8** link on this machine; measured large copies were around
6–7 GB/s. This configuration affects transfer costs and does not represent
all possible links for this GPU. Local evidence:
`.artifacts/benchmarks/report-2026-09-06/pcie-probe.json` (not distributed).

**140 configurations** were tested, with **1540 timing series**: 122 general
cases of 11 rounds and 18 additional solver cases of 7 rounds. All
configurations passed their numerical validation. Comparisons used the best
measured NumPy result among 1, 8 and 16 threads, without changing library code.

## Times including upload, computation and download

Medians in milliseconds. Runtime initialized; data generation and validation
outside the clock. The CPU already receives and returns host arrays, so no
artificial copy is added to its baseline.

| Case | NumPy | OpenPacho/OpenCL | CuPy/CUDA |
| --- | ---: | ---: | ---: |
| Sum of 16,777,216 elements, float32 | **14.11** | 94.28 | 43.52 |
| sin/exp/cos sequence of 16,777,216 elements, float32 | 139.79 | 98.16 | **39.74** |
| Matmul 2048×2048, float32 | 29.81 | 48.26 | **13.13** |
| Matmul 2048×2048, float64 | **66.15** | 116.76 | 102.50 |
| Jacobi, 1024 variables, float32 | **4.34** | 53.51 | 19.37 |
| Jacobi, 8192 variables, float32 | 385.58 | 276.98 | **119.18** |
| Gradient descent, 1024 variables, float32 | **4.32** | 68.95 | 35.92 |
| Gradient descent, 8192 variables, float32 | 554.52 | 335.75 | **155.56** |
| Jacobi, 8192 variables, float64 | 1440.83 | 751.98 | **299.05** |
| Gradient descent, 8192 variables, float64 | 2382.81 | 1130.29 | **438.74** |

The methods used identical recurrences and tolerances, and resident solvers
recorded the same iteration count as reference NumPy. The data
is an SPD matrix with diagonal 2.5 and sub/superdiagonal -1, **stored and processed
as dense**. The comparison is not against a specialized tridiagonal or sparse
solver; these results do not describe all systems or condition numbers.

## What explains the differences

1. **Transfers and preparation.** A large sum has little computation per byte.
   The GPU can compute quickly and lose the complete comparison once copies,
   allocations and array preparation are included.
2. **Dispatch and control per iteration.** Jacobi with 1024 variables in float32 takes
   48.41 ms on OpenPacho and 19.02 ms on CuPy even with resident data, versus
   4.34 ms on NumPy. Matrix upload does not explain that loss. Small kernels,
   temporaries and synchronizations to extract scalars remain.
3. **Amortization.** As the test systems grow, work on the same
   data offsets those costs. At 8192 variables the advantage persists with
   transfers included. This refutes “it can never win” without promising general gains.
4. **Implementation.** CuPy uses optimized libraries and kernels and a memory
   pool. OpenPacho uses simple kernels and new buffers. The difference between
   them does not demonstrate an intrinsic superiority of CUDA over OpenCL.

Across the full grid, OpenPacho showed a clear end-to-end advantage in
**15/140** configurations; CuPy in **33/140**. “Clear” requires at least a 10% median
advantage and a descriptive 95% bootstrap interval that does not cross 1. Counts depend
on the chosen cases; they are not probabilities of winning in real applications.

## Startup

In new processes, for a single 1024×1024 float32 multiplication, the sum of
imports/initialization and the first operation cost approximately **0.233 s on
NumPy**, **1.694 s on OpenPacho** and **1.216 s on CuPy**. NumPy clearly won here.
There were three processes per engine and dtype, retaining existing CuPy disk caches
and disabling the OpenCL disk cache due to environment restrictions.

## Reproduce and review

The design and commands are in [benchmarks/README.md](../benchmarks/README.md).
Complete local output is in
`.artifacts/benchmarks/report-2026-09-06/REPORT.md`, alongside `raw.json`,
`comparisons.csv`, `timings.csv`, startup data and PNG/SVG plots. These local
files are not part of the distributed code.

Runner snapshots were saved in each measurement folder. Their hashes
match those recorded: the runner was extended with large-solver options
after the first trial. The **nine OpenPacho runtime files
remained identical** between both trials and throughout each run.

Runner code passed 7 tests of semantics, validation and synchronization.
Raw files preserve every replicate and failure (none), along with versions,
BLAS configuration, seed, order and telemetry. Other GPUs,
specialized sparse solvers, fused kernels and all condition numbers were not measured.
