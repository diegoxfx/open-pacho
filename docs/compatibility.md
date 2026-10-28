# NumPy compatibility

The [CUDA backend](cuda.md) implements a smaller subset: float32/64 and bool,
elementwise operations, gather, floating-point sum/mean and float32/64 or bool min/max, stable float32/64 norms
(eager and compiled) and matmul with vectors, matrices and batch broadcasting.
Integers and bool accumulators for sum on CUDA remain pending; bool inputs can
be reduced with an explicit float32/64 accumulator. CUDA elementwise/map-reduce
fusion is available for supported groups and matmul/BLAS
retain their boundary. Norms are available with both BLAS providers.
See the [CUDA fusion results](cuda-fusion-results.md) and [CUDA norm
scope](cuda-norm-results.md). The general tables below document
the shared contract, with backend differences indicated where applicable.

OpenPacho implements an explicit subset; it does not re-export NumPy functions that
could accidentally execute operations on the CPU. `__array_ufunc__` and
`__array_function__` retain the arrays' backend.

| Area | Supported | Limits of this version |
| --- | --- | --- |
| Creation | array, asarray, empty, zeros, ones, full, zeros_like, ones_like, arange, linspace, eye | C order; scalar full. arange/linspace/eye are generated on the host and uploaded once. No files. |
| Random | random.default_rng: random, uniform, normal, exponential | [Resident generation](random.md), scalar float32/64 parameters. Vulkan normal/exponential only float32. No NumPy sequence parity, integers or capture. |
| Data | float32/64, int32/64, uint32/64, bool | No float16, 8/16-bit integers, complex, object or structured dtypes. float64 requires fp64. |
| Arithmetic | +, -, *, /, floating-point power, reflected operators | No integer power, //, %, bitwise operators, `out=`, `ufunc where=` or `ufunc.reduce`. |
| Functions | sqrt, exp, log, sin, cos, tanh, abs, isfinite, comparisons, `where(condition, x, y)` | `where` broadcasts all three arguments and accepts a bool or numeric condition; output dtype is promoted from `x` and `y` according to NumPy 2, with weak Python scalars. The one-argument form is unsupported. |
| Broadcasting | Arbitrary dimensions, Python scalars and 0-D arrays | Combining devices/contexts requires an explicit transfer. |
| Shape | shape, ndim, size, nbytes, dtype, reshape, transpose, T | C order only; reshape may share storage. |
| Indexing | Integers, slices with positive/negative steps, ellipsis, None/newaxis | Eager slices and transpose are device copies, not views. Compilation may omit a private CLBlast matmul transpose under the conditions below. No boolean masks or advanced indexing. |
| Writing | Operations create new results | No setitem. `a += b` and similar are not supported through out=; use `a = a + b`. |
| Reductions | sum, mean, min/max (amin/amax); axis=None/int/tuple and keepdims | sum promotes small integers to 64 bits. mean accumulates in float32/64. min/max preserve dtype. No initial/where/out. |
| Algebra | @/matmul with vectors, matrices and batch broadcasting; dot 0D/1D/2D; diag of a matrix; scaled Euclidean/Frobenius norm | CUDA supports float32/64 norms in eager/compile with custom and cuBLAS; no boolean matmul, dot above rank 2, diag from a vector, spectral order, direct solve, SVD/FFT or sparse. |
| Transfers | asnumpy, to_host, to_backend, item | No implicit conversion with np.asarray. item synchronizes and copies a scalar. |

## Conditional selection

`where(condition, x, y)` evaluates the truth of each condition element and
selects between `x` and `y` with broadcasting of all three arguments. Arrays
must belong to the same backend/device; selection remains on that
backend and does not download data. The eager path is available alongside
`compile`/`fuse` capture, including composition with supported elementwise and
map-reduce groups. Both branches are evaluated before selection; `where` is not
a lazy flow-control operation. `where(condition)` (the form returning
indices) and the `where=` keyword argument of ufuncs or reductions are unsupported. See the
[selection results](selection-results.md).

## Extrema along axes

`min`/`max`, their `amin`/`amax` aliases and the `a.min()`/`a.max()` methods preserve
the input dtype and remain resident. They propagate NaN; infinities
participate in comparisons. 64-bit integers are compared without passing
through floating point. CUDA supports float32/64 and bool; integers remain out of scope.

The sign of zero has an explicit rule shared across backends: in a tie
between `-0` and `+0`, min returns `-0` and max returns `+0`. If only one sign appears,
it is preserved. NumPy may choose the sign according to its reduction order/implementation;
bitwise equality with NumPy in that tie is not promised.

A reduced axis of length zero produces `ValueError`, even if the output
would also be empty. An empty output with nonempty reduced axes is valid.
`axis=()` preserves shape and values. Accumulation dtype, `initial`,
`where` and `out` are unsupported. `compile` captures extrema and fuses eligible producers with
their reduction; `fuse` remains limited to elementwise. See [scope and tests](extrema-results.md).

Scalar results are 0-D arrays on the same device. For Python flow
control, call `.item()`. `if array:` always fails to avoid hidden
synchronizations. The repr shows metadata and does not download values.

Noncontiguous NumPy inputs are copied to C order on upload. `asarray` may
reuse a Pacho array on the same backend/dtype, but host data is always
copied into owned storage. `copy=False` when constructing from the host does not promise
shared memory with the original object.

Floating-point results need not match NumPy bit for bit:
sum order and device math functions may vary.
Multiply-add contraction also depends on the path. CUDA custom disables
FMA in NVRTC; OpenCL custom GEMM allows compiler contraction, whereas its
elementwise/map-reduce lowerers disable it. The [local float64 check](fp64-contraction-results.md)
identifies the generated instructions and numerical effects of both policies.
There is not yet a uniform public contraction selector across providers.
Converting floating-point values to integers outside the representable range has no
guaranteed parity between NumPy and OpenCL; it should also be avoided in fused
expressions.
The norm uses a scaled sum of squares; it avoids directly forming the square
of very small or large inputs.
The dtype is not widened: a result outside its range remains unrepresentable,
and the device's subnormal arithmetic modes remain relevant.
The solvers in [examples](../examples/README.md), outside the package, retain
the residual check at every iteration. A small residual
does not certify an accurate solution in ill-conditioned systems; see
[error measurement](numerical-accuracy.md).

## Migration path

1. Run the original code's tests with NumPy and retain reference results.
2. Convert inputs with `pacho.asarray(..., dtype=...)` and select OpenCL explicitly.
3. Keep compatible NumPy calls, or change the import to `openpacho` for supported creators and operations.
4. Transfer with `asnumpy` only when handing results to libraries requiring the CPU.
5. Compare accuracy and times after a first execution to compile kernels.
6. For repeated computations, evaluate `@pacho.compile` according to the
   [capture and compilation limits](compilation.md). The backend remains explicit.

A dependency function that internally calls `np.asarray` may require
adaptation; supporting NumPy protocols does not imply universal compatibility
with SciPy, pandas, scikit-learn or other packages.


## Compilation and capabilities by backend

Compatibility is deliberately partial: common matrix computation takes priority
over replicating all of NumPy. [`compile`](compilation.md) adds
local purity/capture restrictions and uses the same operation semantics.
Direct execution remains available.

`compile(..., padding=True)` preserves the interchangeable NumPy body (`a @ b`)
and enables [internal padding](padding.md) only for eligible custom float32 GEMM.
`pacho.matmul(..., padding=True)` also exists for direct execution; that
keyword is specific to OpenPacho. The default is `False`. Logical shapes
and the selected provider are preserved, including unsupported cases.

With OpenCL and CLBlast, `compile` may replace a private full 2D transpose
with float32/64 GEMM flags, retaining the original storage
as input. This applies only if that transpose has a single matmul consumer and
is not a public output. Shared temporaries, independent outputs,
empty dimensions and backends without that capability retain the copy; eager
execution is also unchanged. Eligible explicit padding takes precedence. The plan
does not retain example inputs or introduce public views. This optimization
does not guarantee an advantage on every shape or bitwise equality with the product
of materialized transposes; see [compilation](compilation.md).

OpenCL and Vulkan are selected explicitly. Vulkan does not support float64
transcendentals or float64 power; int64/uint64 require shaderInt64. Its limits and
optimization status are detailed in [Vulkan](vulkan.md). The [CUDA backend](cuda.md)
is initial: cuBLAS may be its provider; CuPy remains an external comparator.
Installation extras are independent: OpenCL/Vulkan do not require CUDA.
