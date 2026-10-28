# Resident extrema and normalization — 2026-10-22

OpenPacho adds `min` and `max`, the `amin`/`amax` aliases, array methods and
NumPy dispatch. They run on NumPy, OpenCL, Vulkan and CUDA, both eagerly and
through `compile`. This enables normalization along axes and recipes that
shift by the maximum without downloading arrays to obtain that maximum.

Their utility is tied to the [application plan](application-measurement-plan.md):
normalization, preprocessing transforms and future dense networks. The
`softmax` and `logsumexp` recipes live in `examples/normalization.py`; no
domain algorithms or mandatory dependencies are added to the runtime.

```python
import numpy as np
import openpacho as p
from examples.normalization import softmax, logsumexp


@p.compile
def normalize(x):
    return softmax(x), logsumexp(x)


x = p.asarray([[10000, 10001, 9999]], dtype=np.float32, backend="opencl")
weights, log_total = normalize(x)
host_weights = p.asnumpy(weights)  # explicit observation at the end
```

These recipes require finite floating-point inputs and a nonempty axis. They do
not check data through downloads; NaN and infinities follow ordinary
arithmetic. `logsumexp` retains the reduced axis for broadcasting. Each backend's
existing limits also remain: for example, Vulkan does not support
float64 `exp`/`log` in this GLSL implementation. The verification example uses
float32 on every backend.

## Contract

- `axis=None`, a positive/negative integer or a tuple of axes; `keepdims` and scalars.
- Input dtype preserved: float32/64, bool and int32/64/uint32/64 where the
  backend supports them. CUDA remains limited to float32/64 and bool.
- Integers never pass through a floating-point representation for comparison.
- NaN propagates; infinities participate normally. NaN payloads
  have no bitwise guarantee across backends.
- For zeros of different signs, min prefers `-0` and max prefers `+0`. If all
  zeros have the same sign, it is preserved. This associative rule also
  governs partials; it may differ from NumPy's sign choice.
- Reducing an empty axis fails explicitly, even when the output would also
  be empty. An empty output with nonempty reduced axes is valid.
  `axis=()` preserves shape, values and dtype.
- No `initial`, `where`, `out`, `nanmin`/`nanmax` or accumulation dtype.
  `fuse` remains elementwise only; reductions use `compile`.

## Implementation and limits

`_reductions.py` shares the identity and extrema combination between eager
kernels, per-lane accumulation, the cooperative tree and partial combination.
It does not reuse `fmin`/`fmax` with different NaN semantics. Integer
limits are constructed in their own type, including the int64 minimum without overflow.

The IR distinguishes `sum`, `min` and `max`; the Vulkan body cache also includes
the operation. `explain` shows it in `reduce`/`map_reduce` steps. Eligible private
elementwise producers fuse with extrema, without a full temporary.
Shared or returned producers retain the existing boundaries.

The geometry of existing reductions is preserved. In particular, a
compiled reduction without an elementwise producer may use the eager primitive.
This batch demonstrates functionality and residency; it does not promise acceleration or
compare times against NumPy, CuPy or neural-network libraries.

Sum paths retain their previous expressions and rules. CUDA enables
boolean min/max, but sum with a bool accumulator remains unsupported;
summing bool inputs with an explicit float32/64 accumulator remains valid.

## Verification

**267 targeted tests pass and four are skipped** for integers unsupported
on CUDA: 48 new cases and 219 targeted regressions/controls. OpenCL, Vulkan and
CUDA were required; the full suite was not run and the wheel was not rebuilt.

New coverage checks axes, keepdims, NumPy dispatch, capture, partials
larger than 4096 elements, NaN/infinity, zeros of both signs, integer extrema,
bool identity, empty arrays, explicit errors, operation caching and rebinding of
prepared arrays/scalars. A `where` producer followed by min/max over
`(3,8193)` uses two launches per call on each GPU: partials and combination,
with upload/download forbidden during execution.

Softmax and logsumexp are compared against a float64 reference for positive/negative
inputs of magnitude 10,000, both eager and compiled. Weights remain
finite and sum to one within tolerance. Regressions cover sum/mean,
promotion, rounding before accumulation, partials, shared outputs and the
lowerer's scalar ABI. Its harness now tests all three operations.
Independent QA: **APPROVE**, with 2,072 additional NumPy checks
covering dtypes, shapes, axes, empty arrays, NaN, zeros and eager/compiled execution.

Environment: Windows, RTX 5060 Ti/driver 617.14, Python 3.12.14, NumPy 2.5.3.
Uses `--capture=sys`, an accessible TEMP and an explicit GLSL compiler. QA also
accidentally ran a GPU selection; it did not overlap with other active GPU tests or
measurements. Reports are in
`.artifacts/iterations-2026-10-22/extrema-*.xml`: 202 regressions passed in the
first run and 65 final tests after fixing CUDA and updating the harness.
Evidence of the initial failures is also retained; they are not hidden through
skips. The runtime retains NumPy as its only base dependency.
