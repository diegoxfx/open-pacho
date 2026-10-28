# Resident selection with where

Date: 2026-10-22. Local implementation and verification of `0.1.0a1`.

## Purpose and scope

Comparisons already produced resident masks, but a primitive for choosing values
from them was missing. `where(condition, x, y)` completes that workflow for data
cleaning, thresholds, and piecewise functions. It integrates with the existing
elementwise operations without introducing advanced indexing, a domain framework,
or another dependency.

```python
import numpy as np
import openpacho as pacho


@pacho.compile
def valid_energy(x):
    cleaned = np.where(np.isfinite(x), x, 0.0)
    return np.sum(cleaned * cleaned)


x = pacho.asarray([1.0, float("nan"), -2.0], dtype=pacho.float32, backend="opencl")
print(valid_energy(x).item())  # 5.0; explicit scalar download
```

The public `pacho.where` API and `np.where` dispatch support broadcasting across
all three arguments. A numeric condition is true when it is nonzero; NaN and
infinity are true. Only `x` and `y` determine the output dtype, using NumPy 2
promotion for weak Python scalars and typed NumPy scalars.
The arrays' backend is preserved, and mixing contexts is rejected.

Both branch expressions are evaluated before selection: `where` does not provide
lazy control flow or hide an unsupported operation in the discarded branch.
Selecting a branch is also different from multiplying it by a mask: NaNs in the
other branch do not contaminate the selected value.
The one-argument form that returns indices and the `where=` keywords of
ufuncs/reductions remain out of scope.

## Integration

The semantic resolver adds promotion and broadcasting for three operands; each
backend emits native conditional selection. NumPy uses its own operation.
OpenCL, Vulkan, and CUDA integrate it into eager execution, `fuse`, `compile`, and
map-reduce. During fusion, converting a numeric condition to bool compares it
with zero: it does not truncate a fractional value to a boolean storage integer.

The tests verify that `sum(where(...))` executes with one launch per call for the
case of three rows of 3001 elements, without uploading or downloading data during
the operation. This is evidence of fusion and residency for that shape; it is
not a speed measurement or a guarantee of a single launch for arbitrarily large
reductions.

## Verification

- **44 new tests pass** across NumPy, OpenCL, Vulkan, and custom CUDA. One is
  skipped because CUDA does not support int64 storage; it is not emulated on CPU.
- **53 targeted regression tests pass**: promotion, booleans, rounding,
  broadcasting, rebinding, map-reduce, and shared outputs in the modified paths.
  They include both CUDA providers.
- Independent QA: **APPROVE**, with additional CPU checks of types and scalars
  against NumPy. There were no simultaneous GPU tests during this round.

Coverage includes fractional and nonfinite conditions, selection of NaN/Inf/signed
zero, integers greater than 2**53 where supported, empty arrays, scalars, bool
outputs, incompatible contexts, unsupported branches, and plan reuse with new
scalars. Neither the global suite nor a speed benchmark was run.

Environment: Windows, Python 3.12.14, RTX 5060 Ti, driver 617.14, with all three
GPU backends required. The reports are `where-tests.xml` and
`where-regression-tests.xml` in `.artifacts/iterations-2026-10-22`.
`--capture=sys`, accessible TEMP/TMP directories, and an explicit Vulkan compiler
were used, as in the [norm iteration](cuda-norm-results.md).
