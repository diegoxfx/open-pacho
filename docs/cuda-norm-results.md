# Stable CUDA norms and external clients

Date: 2026-10-22. Local validation on Windows, Python 3.12.14,
NVIDIA RTX 5060 Ti, driver 617.14. The version remains `0.1.0a1`.

## Why this iteration

CUDA could multiply matrices and reduce arrays, but could not compute a stable
norm. This prevented using a common primitive for magnitudes, normalization, and
residuals, and blocked external solvers. Reusing the existing scaled algorithm
provides more utility than another microoptimization with marginal gains.
It adds neither domain algorithms to the package nor a new dependency.

## Implementation and contract

`_norm.py` shares combination of `(scale, ssq)` pairs across OpenCL, Vulkan,
and CUDA. CUDA adds only its source and launch: 128 threads, shared
memory, and blocks of up to 4096 terms. Partials remain on the GPU;
an extreme input is not squared before being scaled.

`pacho.linalg.norm` and `np.linalg.norm` support the same Euclidean and
Frobenius subset in float32/64, both eager and with `compile`. The norm preserves dtype
and backend, supports existing axes and `keepdims`, and works with both
CUDA providers. The `blas` choice still affects matrix products;
the norm uses the shared custom kernel. NaN dominates infinity and an empty
reduction returns zero. Dtype range and subnormals remain limits.

Solvers in `examples/solvers.py` can now use CUDA, including their prepared
forms. Recurrences, tolerances, and true-residual checks
at every iteration are retained. Validation counts nonfinite inputs with an explicit
floating accumulator and requires zero: it does not need to count exactly every
element of a large array or require int64 in CUDA. CPU retains eager
validation. Solvers remain outside the wheel and library API.

## Numerical evidence

The **52 existing norm and solver cases** in
`benchmarks.accuracy` corresponding to `cuda-custom` and `cuda-cublas` were run, with
Decimal80 references on represented inputs and unchanged thresholds.
There were no exceptions or exclusions for missing operations.

The eight extreme norms pass. The largest relative error observed was
**4.52e-8 in float32** and **6.24e-17 in float64**; both providers use the same
kernel and produced the same results. The 36 ordinary-system,
small-RHS, and large-RHS cases also pass diagnostics and converge.

**Six forward-error warnings** persist for Hilbert systems:
per provider, sizes 4 and 8 in float32, and size 8 in float64. All meet
their residual criterion, but relative error against the Decimal solution reaches
approximately 0.0109, 0.9663, and 0.000150 respectively. This demonstrates why
a small residual does not guarantee an accurate solution in an ill-conditioned
system. Stable norms resolve neither that problem nor the earlier GEMM cancellation
warnings. Tolerances are not relaxed to hide them.

These cases are used for numerical diagnostics. Their small timings, with one
repetition, support neither a speedup claim nor a provider
comparison.

## Reproduction and limits

Inputs and references already live in `benchmarks/accuracy.py`. The complete
run is obtained with:

```powershell
python -m benchmarks.accuracy --backends cuda-custom,cuda-cublas --repeats 1 --output accuracy.json
```

That command also includes sums/products and retains its nonzero exit when
there are diagnostic warnings. This iteration limited work to `norm` and `solve`;
the corresponding script and JSON are in
`.artifacts/iterations-2026-10-22/verify_norm_accuracy.py` and
`accuracy-cuda-{custom,cublas}.json`. Private artifacts are not distributed.

**257 targeted tests passed**, with all three GPU backends required and
51 PyOpenCL compiler warnings; the global suite was not run. The report
is in `.artifacts/iterations-2026-10-22/norm-final-sys-tests.xml`.
Final independent review approved the batch after correcting the CPU validation
instrumentation regression and reviewing those results.
Functional tests cover extreme ranges, empty inputs, scalars, irregular
axes, 4096/4097/8193 boundaries, NaN/Inf precedence across blocks,
compiled execution without downloads, and ordinary/prepared flows for all three
solvers. Shared CPU/OpenCL/Vulkan paths are also checked.

Local execution encountered insufficient permissions in the sandbox TEMP when
compiling Vulkan; TEMP/TMP were configured in the private artifact directory and
the explicit path to the installed GLSL compiler was set. A native Windows exception
`0xc0000008` was also observed within NVRTC with pytest descriptor capture.
The smoke test mixing all four backends and CUDA suites in separate processes
passed. The integrated suite also passed using `--capture=sys` instead of
capturing native descriptors. That option is retained for local verification;
the internal cause of the NVRTC exception has not been diagnosed and is not attributed
to an arithmetic error or declared fixed in the driver.
