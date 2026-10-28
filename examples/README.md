# Example applications

OpenPacho provides GPU array/matrix operations. Domain algorithms live here;
they are neither installed in the wheel nor exported as `openpacho.solvers`.
The repository and sdist include the examples to reproduce tests and benchmarks.

From the repository root, after installing the package:

```sh
python examples/numpy_migration.py --backend numpy
python -m examples.solve_system --backend numpy
python -m examples.solve_system --backend opencl
```

`solvers.py` contains CG, Jacobi, steepest descent, and `PreparedSystem`.
It uses the public API, including `get_backend(array)` to preserve the inputs'
exact context/provider. CUDA supports these examples through its stable
float32/64 norms, in both eager and compiled execution and with the custom and
cuBLAS providers. There is no "solver" capability in the runtime; the algorithms
remain in this directory.

The normalized recurrences, validation, tolerances, and true residual at every
iteration are preserved from the previous implementation. CG/steepest descent
require SPD matrices; Jacobi requires a nonzero diagonal and a convergent iteration.
These properties are not proved, and a small residual does not guarantee an
accurate solution.

These examples already include explicit `compile`/`prepare` phases. Benchmarks
must report that preparation and retain eager and compiled comparisons where
appropriate; they do not prove that arbitrary NumPy code becomes faster without
adaptation. New optimizations must be reusable through ordinary compiler/backend
operations, without recognizing a solver's name. Algorithmic changes are evaluated
separately and applied to the relevant references as well.

`normalization.py` demonstrates softmax and logsumexp with a maximum shift, using
min/max and existing operations without intermediate transfers. The recipes
require finite floating-point inputs and a nonempty axis; they do not download
data for validation. `logsumexp` retains the reduced axis for broadcasting.
They can be captured with `compile` and respect each backend's function/dtype
limits; they are not a neural-network API in the runtime. See the
[extrema results](../docs/extrema-results.md).

`dense_network.py` contains `predict`, `loss_and_gradients`, and `sgd_step`: two
affine layers, tanh activation, and half the mean squared error as the loss.
Gradients are explicit, and the step returns the pre-update loss and new parameters
without modifying the originals. The functions accept direct NumPy and
OpenPacho/CuPy arrays; the caller supplies floating-point arrays with the same
dtype and backend. Metadata is validated without downloading data. Float64 tanh
remains unsupported in Vulkan.

```python
import numpy as np
import openpacho as p
from examples.dense_network import predict, sgd_step

rng = np.random.default_rng(47)
shapes = ((31, 7), (31, 3), (7, 11), (11,), (11, 3), (3,))
host = tuple((0.2 * rng.standard_normal(shape)).astype(np.float32) for shape in shapes)
x, y, w1, b1, w2, b2 = (p.asarray(a, backend="opencl") for a in host)
step = p.compile(sgd_step).prepare(x, y, w1, b1, w2, b2, 0.1)
loss, w1, b1, w2, b2 = step(x, y, w1, b1, w2, b2, 0.1)
prediction = predict(x, w1, b1, w2, b2)
print(loss.item())  # explicit observation of the pre-update loss
```

Finite differences check every parameter in the small case. The
[dense-network report](../docs/dense-network-results.md) separates inference,
one SGD step, preparation, residency, and transfers. It is not a training
framework and does not promise convergence for any learning rate.

`oscillators.py` implements velocity-Verlet for coupled oscillators with unit
masses, `q'' = -K @ q`. The stiffness matrix must be symmetric positive-definite;
the caller is responsible for that property and a stable time step. `verlet_step`
returns new position, velocity, and acceleration; `integrate` retains acceleration
between steps and executes exactly `steps + 1` matrix-vector products. `energy`
returns a resident scalar for diagnostics outside the loop.

```python
import numpy as np
import openpacho as p
from examples.oscillators import integrate, verlet_step

K, q, v = (
    p.asarray(a, backend="opencl")
    for a in (
        np.array([[2.0, -0.5], [-0.5, 1.0]], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.25], dtype=np.float64),
    )
)
dt = 1 / 32
step = p.compile(verlet_step).prepare(K, q, v, -(K @ q), dt)
q, v, a = integrate(K, q, v, dt, 64, step=step)
```

One step is prepared while the Python loop is retained; compiling the complete
integrator would unroll its steps. The example does not inspect scalars or download
arrays at every iteration. Its [report](../docs/oscillators-results.md) distinguishes
error against the discrete recurrence, time-discretization error against the modal
solution, and energy.
