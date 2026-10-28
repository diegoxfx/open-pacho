# Optional function compilation

Implemented in 0.1.0a1, 2026-09-06. Direct execution remains the default.
`compile` optimizes a bounded function and keeps its arrays' backend. It does
not create a global lazy graph or automatically switch from CPU to GPU.

```python
import numpy as np
import openpacho as pacho


@pacho.compile
def energy(a, b, c):
    x = a * b + c
    y = 2 * x
    return np.sum(y * y, axis=-1)


host = np.ones((1024, 32), dtype=np.float32)
original = energy(host, host, host)  # NumPy executes the original function
# Also "vulkan", "numpy", or "cuda"; CUDA fuses supported groups.
with pacho.use_backend("opencl"):
    a = pacho.asarray(host)
    result = energy(a, a, a)
    print(energy.explain(a, a, a))
    print(energy.cache_info())
    host_result = pacho.asnumpy(result)
```

The forms `pacho.compile(function)`, `@pacho.compile`, and `@pacho.compile()` are
equivalent. Assignments to `x` and `y` do not interrupt capture. The `return`
establishes plan outputs: an array or a tuple of arrays. The public result is
again an ordinary resident ndarray; downloading it requires `asnumpy` or `item`.

## Specialization and scalars

The first call with OpenPacho arrays captures the Python body, generates a plan,
and executes its steps. Subsequent compatible calls rebind buffers and scalars
without executing the Python body again. Scalars computed from scalar arguments
are recalculated on the host per call without downloading arrays. The first
call's rate is not frozen:

```python
@pacho.compile()
def update(x, b, rate=0.01):
    return (1 - rate) * x + rate * b


x = update(a, a, 0.01)
x = update(x, a, 0.02)  # reuses the plan; changes the rate value
```

Shapes, axes, and other parameters controlling structure are declared static.
Changing their value generates another variant:

```python
@pacho.compile(static_argnames=("axis",), cache_size=16)
def average(a, axis=-1):
    return np.mean(a, axis=axis)
```

Each function has an LRU of up to 32 plans by default. The key includes each
array's shape/dtype and backend object, each scalar's type, and static argument
values. All public arrays are contiguous; there is not yet an arbitrary-stride
key or a persistent disk plan cache.
`clear_cache()` releases plans and resets statistics; backends may retain their
programs and free buffers. `explain(*args)` captures if needed, without executing
kernels, and shows backend, BLAS, types, shapes, and steps. It is not a memory
profiler and does not measure driver synchronization.

Calls supplying every parameter positionally avoid rebuilding signature bindings
if the signature declares only positional or positional/keyword parameters.
Types and specialization are still validated on every call; scalars are rebound.
Keyword arguments, omitted defaults, and keyword-only parameters retain the
general path. This requires no user-code change or `prepare` call.
Capturing a new variant does bind the signature; the repeated gain applies to
calls reusing a cached specialization.

To repeat a specialization without looking it up in the LRU on every call:

```python
execute = compiled_energy.prepare(a, b, c)
result = execute(other_a, other_b, other_c)
```

`prepare` returns a callable that validates each array's shape, dtype, and context,
dynamic scalar types, and static values. It preserves defaults and accepts keyword
arguments; it neither retains example arrays nor freezes dynamic scalars.
An incompatibility raises an error rather than recapturing. Outputs of successive
calls are independent. This is a prepared Python executor, not native code or
a persistent GPU command graph.

## Current operations and optimization

`compile(..., padding=True)` and `@compile(padding=True)` enable
[optional GEMM padding](padding.md). `prepare` inherits that policy. Supported
steps use aligned physical buffers and crop on GPU before continuing; logical
shapes do not change. The default is `False`, and the usual path is retained
when backend, provider, dtype, or shape is unsupported. `explain` shows applied
steps and their requested private memory.

With OpenCL and `blas="clblast"`, compiled float32/64 matmul can directly consume
the original storage of a private complete 2D transpose, as in `a.T @ b`,
`a @ b.T`, or `a.T @ b.T`. It must have a single consumer and must not be a
function output. The backend prepares GEMM with the corresponding transpose
flags and leading dimensions; `explain` identifies it as `layout_matmul`.
The function's NumPy body need not change. Shared or returned transposes, empty
cases, and unsupported capabilities retain materialization. Eligible explicit
padding is resolved first. Eager execution retains its usual copies; this adds
no mutable views and retains no capture inputs. Flags can change the provider's
kernel choice and rounding, so neither a general gain nor bitwise equality with
the materialized-transpose path is promised.

Capture reuses ndarray rules: arithmetic, broadcasting and NumPy 2 promotion,
`where(condition, x, y)`, `sum`/`mean`/`min`/`max` reductions, matmul with batch
broadcasting, reshape, slicing, transposition, diagonals, and Euclidean/Frobenius
norms. `where` promotes only `x` and `y` and accepts bool or numeric conditions.
Both branches are evaluated before selection. Constructors such as `zeros_like`,
`ones_like`, and `pacho.full` with compatible constant parameters are supported.
Coverage and dtypes also depend on the backend. In `reduce` and `map_reduce`
steps, `explain` indicates `operation`: sum, min, or max. Extrema preserve dtype,
propagate NaN, and use eager's zero-tie rule; see
[compatibility](compatibility.md#extrema-along-axes).

OpenCL and CUDA also prepare float32 `min/max` on materialized inputs as
`map_reduce`, even without a producer to fuse. Shared or returned producer
boundaries are preserved. Vulkan, other dtypes, and sums keep their previous
selection. The [extrema plan comparison](extrema-planning-results.md) documents
the measured gain and regressions preventing extension to Vulkan.

The [random generator](random.md) is stateful: its calls are rejected during
`compile`/`fuse` before consuming counters, even from a closure. Generate random
inputs outside and pass them as arguments; a capture sample is not frozen for
reuse in every execution.

- **OpenCL:** fuses elementwise chains and elementwise producers with
  `sum`/`min`/`max`, respecting intermediate conversions and accumulator dtype.
  `x*x` counts as one consumer of x; it does not force storage of x. A producer
  shared between different consumers or returned by the function is materialized.
  Map-reduce uses compact partials for large reductions; one logical step does
  not promise a single kernel. Gather and matmul keep their boundaries except
  for the private CLBlast transposes above and the private `b - A @ x` pattern
  with compatible custom GEMV. There is no general elimination of equivalent
  expressions or epilogues inside the CLBlast provider.
- **Vulkan:** fuses elementwise chains with by-value scalars and producers with
  their reduction, with the same consumer and public-output boundaries.
  Map-reduce uses typed accumulation, cooperative groups, and compact partials.
  It reuses pipelines, buffers, and command resources. It groups each invocation's
  command submission synchronously on the host and executes asynchronously on
  GPU; it does not wait for the result or create a global lazy graph. A group
  splits if synchronization is required or the pending-resource limit is reached.
- **NumPy:** executes NumPy primitives. It does not generate native CPU code;
  wrapping OpenPacho CPU arrays does not promise to accelerate NumPy. With native
  NumPy inputs, the original function executes directly.

`where` selection can also combine into elementwise and map-reduce groups
supported by `compile`/`fuse`; it neither evaluates branches lazily nor supports
the one-argument form or ufunc `where=` keyword. Results and limits are in
[conditional selection](compatibility.md#conditional-selection) and
[selection results](selection-results.md).

CLBlast remains behind a matmul operation. It is selected through the OpenCL
backend before creating arrays; it does not limit IR extension. Choosing another
BLAS provider still creates another context, so it cannot be swapped on existing
buffers without explicit transfer.

Live temporaries belong to each invocation. The plan does not store arrays passed
during capture; it stops referencing intermediates when their uses are exhausted.
Backends retain in-flight resources as long as needed. The plan cache has a lock,
and concurrent calls with different inputs are tested.

## Contract limits

The function must be pure. This operator-overloading tracer cannot detect every
Python effect: modifying a global list or reading a mutable global is the author's
responsibility and outside the contract. Pass changing data as arguments.
There is no silent CPU reexecution when capture fails.

Array mutation, `out=`, host reads, Python conditions depending on arrays/dynamic
scalars, nested compiled calls, array arguments inside containers, `*args`/`**kwargs`,
and arbitrary Python within the plan are unsupported. Undecorated pure helper
functions can participate in capture. Array constants must be converted first
and passed explicitly; closure arrays are not captured. `np.*` constructors
without array arguments do not necessarily pass through NumPy dispatch and are
not automatically portable.

A Python loop with static bounds is unrolled during capture; it is not converted
into a native GPU loop. For many iterations, compile one step and keep the loop
outside. Solvers in `examples/solvers.py` already reuse update plans, preserving
validity and true-residual checks at every iteration. On GPU, the residual and
its scaled norm are computed through a common plan. The norm is an IR operation
in its own right, preserving stability under compilation. The `b - A @ x` pattern
can fuse GEMV and subtraction if the product is private, floating dtypes match,
and the provider is custom OpenCL or Vulkan. A shared/returned product or CLBlast
retains its boundary. CG and steepest descent use normalized directions and do
not form the squared norm for their coefficients. Tolerance and checking frequency
are not relaxed. Compiling steps does not remove those host observations or
guarantee beating NumPy.

`fuse` remains available over the same IR for a single elementwise result.
It retraces every call; use `compile` for repeated work.

The [accepted architecture](execution-architecture-proposal.md) describes future
extensions. The [gap analysis](gaps-and-optimization.md) quantifies current scope.

## Initial CUDA and external applications

Internally, a backend can provide `create_lowering()` and an `execution_scope()`
context to group execution. The plan binds these methods during preparation;
it does not decide by names or renegotiate them on every operation. Without a
lowerer it uses the chosen backend's primitives. Preparation or execution errors
propagate without switching backend or relaxing precision.

CUDA reuses capture and plans. `backend.create_lowering()` binds its
`CUDALowering`, which compiles supported elementwise/map-reduce groups with NVRTC.
Fused reductions use 128 lanes and chunks of up to 4096 terms; NVRTC retains
`fmad=false` and `ftz=false`. Matmul and BLAS remain separate steps.
Each compiled CUDA module retains a `ctypes` argument vector with stable host
storage. Each call copies scalars and device addresses into that vector under
the existing lock before enqueue; the template retains no caller arrays or
scalars. Allocations remain in `_pending` until `synchronize()`. The 128-variant
LRU and `close()`, after synchronization, release the module and vector.
This dispatch optimization changes no kernels, mathematics, contexts, or stream;
see [CUDA dispatch results](cuda-dispatch-results.md).
`compile` does not expand dtypes/operations in the [CUDA contract](cuda.md);
bool accumulators in reductions are unsupported, although bool inputs can be
reduced with float32/64 accumulators. Stable float32/64 norms are available in
eager and compiled CUDA, with both BLAS providers; graphs remain pending.
Solvers are [external examples](../examples/README.md), not a compiler extension,
and their algorithms did not change when enabling CUDA. See the
[CUDA fusion results](cuda-fusion-results.md). Optimized patterns must correspond
to generic operations and benefit equivalent user functions.
