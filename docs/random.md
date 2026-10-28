# Resident random generation

`openpacho.random.default_rng` creates a backend-bound generator. GPU samples
are computed directly in device memory; no NumPy matrix is generated for
upload. GPU selection is explicit and its extras are independent.

```python
import openpacho as pacho

rng = pacho.random.default_rng(47, backend="opencl")
A = rng.uniform(-1, 1, size=(1025, 513), dtype=pacho.float32)
noise = rng.normal(0, 0.05, size=A.shape, dtype=pacho.float32)
times = rng.exponential(scale=2, size=(257, 129), dtype=pacho.float32)
result = A + noise  # remains on the device
```

`backend=` also accepts an existing instance. The generator retains it
even if the default backend later changes. The library's initial default
remains NumPy.

## API and capabilities

| Method | Distribution and parameters |
| --- | --- |
| `random(size=None, dtype=float64)` | Uniform on [0, 1). |
| `uniform(low=0, high=1, size=None, *, dtype=float64)` | Uniform on [low, high); equal bounds produce a constant. |
| `normal(loc=0, scale=1, size=None, *, dtype=float64)` | Normal with mean loc and standard deviation scale ≥ 0. It is not bounded to an interval. |
| `exponential(scale=1, size=None, *, dtype=float64)` | Exponential with mean scale ≥ 0; zero produces a constant zero. |

| Backend | random/uniform | normal/exponential |
| --- | --- | --- |
| NumPy, OpenCL, CUDA | float32 and float64 | float32 and float64 |
| Vulkan | float32 and float64 | float32 |

float64 requires fp64. Vulkan does not implement the necessary GLSL transcendental
functions in double precision: request float32 or explicitly select another backend. Precision
is not reduced and execution does not switch to CPU. `random_distributions(dtype)` declares
support through an internal contract; the facade does not decide by backend name.

Parameters are finite real scalars interpreted in the requested dtype.
Negative scales, reversed bounds, and a uniform width overflowing
that dtype are rejected. Conversion may round bounds; interval membership
refers to those represented bounds. Rounding toward the upper endpoint
is corrected to keep `high` excluded. Extreme parameters and normal/
exponential tails may still produce values beyond the dtype's range; they are not
silently clipped. Subnormals also depend on the device.

`size` accepts integers and tuples, empty dimensions, rectangular and odd shapes.
`None` produces a resident 0-D ndarray; `.item()` downloads the scalar. The initial
limit is 2³¹−1 elements per call, in addition to backend memory
limits. There is no `out=`, array parameters, random integers, choice/permutations,
truncated distributions, or `spawn`.

The API resembles NumPy but does not promise equal sequences. `dtype=` in
uniform/normal/exponential is an extension; NumPy does offer dtype in `random`.

## Parallelism and reproducibility

The facade delegates to `backend.random_data(...)` after validating capability and
reserving counters. Each backend owns its generation/launch adapter;
the shared algorithm does not query its name. Caches store immutable
source, never buffers, seeds, stream positions, or distribution parameters.
Separating adapters neither changes sequences nor adds mandatory GPU dependencies.

It uses **Philox4x32-10**, from the counter-based family described by
[Random123](https://github.com/DEShawResearch/random123). Each element evaluates
ten integer rounds on its counter. There is no per-thread mutable state, atomics,
or sequential chain of GPU states. High uint32 multiplication uses
OpenCL `mul_hi`, GLSL `umulExtended`, and CUDA `__umulhi`. The core requires no
64-bit GPU integers.

1. `seed` is an integer in [0, 2⁶⁴); its halves form the key. `None` obtains
   a new seed from system entropy without modifying a global RNG.
2. The counter occupies two uint32 words; the other two are zero. Each output
   consumes a block of four words, in C order.
3. Calls consume as many counters as elements; empty calls consume none.
   Changing shape or splitting calls with the same distribution, dtype, and parameters
   preserves values. An error occurs before wrapping at 2⁶⁴.
4. `random` converts 24 bits for float32 and 53 for float64 to [0, 1). The integer
   core and that conversion have exact checks across CPU and local GPUs.
5. Box-Muller is used for normal and inverse CDF for exponential. Math
   functions may differ across APIs/compilers: bitwise equality between
   devices or drivers is not guaranteed for transformed distributions.
   Finite resolution also limits tails.

A lock reserves disjoint ranges when multiple threads share a generator.
Scheduling determines which call receives each range; order is not
reproducible across threads. Failed validations do not consume counters. After
reservation, an allocation/compilation/dispatch failure does consume the range to
avoid reusing it after an uncertain asynchronous result. There is no public checkpoint/restore
in this version.

Normal/exponential use `log1p` where available. Vulkan uses an eight-term series
for `-log(1-u)` when u < 1/8, avoiding precision loss near zero; outside
that interval it retains `log`. This is neither a fast-math mode nor an
NVIDIA-specific optimization.

This is not a cryptographic generator. The [Random123 known-answer vectors](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors)
and local statistical tests are not equivalent to TestU01/PractRand and do not certify
statistical independence across all seeds.

## Compilation and measurement

Stateful generation is not captured inside `compile` or `fuse`, even if
the generator comes from a closure. It fails before advancing the counter. Generate
outside and pass arrays as arguments. Randomness within the graph will require
an explicit state/effect contract; a sample is not frozen at compile time.

`benchmarks.random_generation` separates generation with a resident result,
generation and download to the host, and NumPy generation followed by GPU upload.
It includes allocation, retains outliers, and measures first call/warms up separately.
Comparator algorithms may differ: distributions and destinations
are compared, not sequences. See [this round's results](staging-random-results.md).
