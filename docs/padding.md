# Optional padding for matrix products

`padding=False` preserves the usual path. Enabling it allows private buffers
aligned to the custom GEMM kernel's tiles, padded with **exact positive
zeros**. The result is cropped on the GPU before passing it to the next step.
Public shape, broadcasting, and dtype are unchanged.

The [reassessment with the corrected OpenCL baseline](padding-reassessment.md) retains
the option for specific gains in rectangular products with few rows and
long K. Most shapes lose performance, and no useful Vulkan advantage was found
on the measured machine. Measure the complete flow before
enabling it; there is no automatic selection.

```python
import openpacho as pacho


@pacho.compile(padding=True)
def product(a, b):
    return a @ b


# Also pacho.compile(function, padding=True).
# a and b are OpenPacho arrays on the backend chosen by the user.
c = product(a, b)
execute = product.prepare(a, b)  # inherits padding=True
d = execute(other_a, other_b)

# Direct execution, without decorating a function:
e = pacho.matmul(a, b, padding=True)
print(product.explain(a, b))
```

The argument accepts `bool` and `numpy.bool_`. The `@` operator acquires the
compiler's policy; outside it, the default is retained. An explicit
`pacho.matmul(..., padding=True)` call is also captured inside `@pacho.compile`
without options. `fuse` remains elementwise only and does not accept this option.
The compiled function called with native NumPy arrays executes the original body.
`padding` is an OpenPacho extension, not a `numpy.matmul` keyword.

## Initial scope

| Path | Behavior with `padding=True` |
| --- | --- |
| OpenCL custom, Vulkan, CUDA custom | Eligible float32 GEMM, including batches and broadcasting |
| OpenCL with CLBlast / CUDA with cuBLAS | Retains the provider and its usual path |
| NumPy, float64, vectors, small/empty matrices | Retains the usual operation |
| Other operations | No padding; consume original logical dimensions |

Eligibility retains that of the existing register GEMM: operands with at
least two dimensions, `M,N >= 32`, `K >= 16`, and sufficient group and
shared-memory capacity. **M and N are rounded to multiples of 32, K to multiples of 16**;
not to the next power of two. Batch sizes are not expanded. The 32-bit index
limits and, when available, the device's maximum buffer size
are checked before preparing this path. If exceeded, the
usual operation is retained. An actual allocation failure propagates; it does not change
the selected backend or provider.

The contract is an execution option, not a promise of better performance or
of padding every operation. `explain` indicates the requested policy and the
`padded_matmul` steps actually selected. Each shows logical/physical
dimensions, alignment, `copy_dispatches`, and `scratch_bytes`.

## Memory and cost

For example, `(33,17) @ (17,65)` physically uses `(64,32) @ (32,96)` and crops
`(64,96)` to `(33,65)`. Its private buffers request **45,056 additional bytes**:
the two padded inputs and physical result. The logical result occupies its
own buffer. An operand is copied only if its size changes; if all
dimensions are already aligned, there are no additional copies or private buffers.

`scratch_bytes` sums those private buffers for **one operation**. It excludes original
inputs, logical output, pools, driver, and other pending calls; it does not
represent the process's physical peak memory. Each invocation copies the current
input values again. The plan caches metadata, not input buffers
or an assumption of immutability. There is no propagation of padded layouts
across multiple operations yet: cropping occurs at every GEMM boundary.

The aligned GEMM kernel removes per-element checks on tile
tails. This **does not mean the entire pipeline is branch-free**: there are
loops, copy guards, and executor guards. The balance depends on shape,
backend, and hardware: copying two inputs, writing/cropping the output, and
computing discarded elements may cost more than the guards removed.

## Accuracy and subnormals

Copies reinterpret every float32 word as uint32. They therefore perform no
arithmetic on subnormals, signed zeros, or NaN payloads. New cells
are written with the `0x00000000` pattern. Both operands are padded along
K; additional positions do not read data outside the original product.
Values in discarded rows/columns, including possible NaNs from `0*Inf`,
do not reach subsequent reductions such as `sum` or `mean`.

No fast-math, FTZ, or precision changes are enabled. Arithmetic retains the
existing backend policy: preserving bits during copies does not guarantee that all
devices preserve subnormals during multiplication. OpenCL allows an
implementation without denormal support to flush them to zero; CUDA retains
existing `ftz=false` and `fmad=false` options.
References: [OpenCL C numerical requirements](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html)
and [NVRTC options](https://docs.nvidia.com/cuda/nvrtc/).

## Reproducible measurement

```bash
python -m benchmarks.isolated --worker padding --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 21 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/padding/measurements
```

The worker pairs `False`/`True` with the same data, alternates order, warms up
outside the timer, and records all samples, errors against float64, shapes, and
requested memory. It measures the resident compiled function, prepared executor,
direct `matmul`, and complete flow with transfers. **All include
padding and cropping**. The `prepacked_wall` diagnostic is presented separately: it reuses
already-padded inputs and excludes cropping; it measures wall time with dispatch,
allocation, and synchronization, neither pure GPU time nor an API gain.

Processes run sequentially, with idle/temperature checks
before changing backends and reversed order in the second round. A busy
or hot device stops the session without inventing samples. Validation on other
vendors remains necessary.

See the [local results with errors and complete cost](padding-results.md).
That report is historical; the [current decision](padding-reassessment.md) uses
the corrected OpenCL kernel, two unpadded controls, and additional shapes:

```bash
python -m benchmarks.isolated --worker padding_decision --backends opencl,vulkan,cuda-custom --rounds 3 --repeats 30 --seed 971 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/padding-decision/reproduction
```

In this worker, `complete` records finished execution, and `all_reference_close`
records accuracy. A completed process may contain finite numerical
violations, which remain visible under the original tolerances.

## Future extension: reductions

`mean` can benefit from a padded implementation: sum including
zeros and divide by the number of **logical** elements in the reduced axes,
not their physical size. Division can be another kernel or integrated into the
final reduction stage; which is preferable must be measured. The current restriction
is one of scope, not a mathematical impossibility. Implementation remains pending.

This requires preserving axes, `keepdims`, accumulation dtype, and the contract
for empty reductions, NaN/Inf, and subnormals. Padding must be the additive identity
zero; it cannot automatically be extrapolated to other reductions. The complete
cost of padding, reduction, and normalization must be compared, without promising
that removing guards offsets copies or additional work.
