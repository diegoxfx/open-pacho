"""A dense regression network using public NumPy-compatible array operations.

Two affine layers surround one hidden tanh activation. Arrays must have the
same float32/float64 dtype, reside on the same backend, and contain finite values.
The caller chooses that backend and uploads inputs before calling these helpers.
The learning rate is a finite, positive scalar supplied by the caller. These
conditions are not checked by downloading arrays or observing device scalars.

The loss averages over every batch/output element. Gradients are explicit;
there is no autodiff or mutable model state. Compile the outer plain function
with ``openpacho.compile`` when desired. Native NumPy arrays work directly, and
the functions themselves never transfer or construct arrays. Vulkan float64
tanh remains unsupported by that backend's current mathematical contract.
"""

import numpy as np


def _validate(x, w1, b1, w2, b2, y=None):
    """Check metadata only, so validation also works during plan capture."""
    arrays = (x, w1, b1, w2, b2) if y is None else (x, w1, b1, w2, b2, y)
    if any(a.dtype not in (np.dtype("float32"), np.dtype("float64")) for a in arrays):
        raise TypeError("dense network inputs require float32 or float64 arrays")
    if any(a.dtype != x.dtype for a in arrays):
        raise TypeError("dense network inputs must share one dtype")
    if x.ndim != 2 or w1.ndim != 2 or w2.ndim != 2 or b1.ndim != 1 or b2.ndim != 1:
        raise ValueError("expected x/w1/w2 matrices and b1/b2 vectors")
    n, d = x.shape
    if any(size <= 0 for a in arrays for size in a.shape):
        raise ValueError("dense network dimensions must be positive")
    if w1.shape[0] != d or b1.shape != (w1.shape[1],):
        raise ValueError("expected w1=(d, h) and b1=(h,) for x=(n, d)")
    if w2.shape[0] != w1.shape[1] or b2.shape != (w2.shape[1],):
        raise ValueError("expected w2=(h, o) and b2=(o,)")
    if y is not None and y.shape != (n, w2.shape[1]):
        raise ValueError("expected targets y=(n, o)")


def _forward(x, w1, b1, w2, b2):
    hidden = np.tanh(x @ w1 + b1)
    return hidden, hidden @ w2 + b2


def predict(x, w1, b1, w2, b2):
    """Return predictions of shape ``(n, o)`` without changing the inputs."""
    _validate(x, w1, b1, w2, b2)
    return _forward(x, w1, b1, w2, b2)[1]


def loss_and_gradients(x, y, w1, b1, w2, b2):
    """Return ``(mean((prediction-y)**2)/2, dw1, db1, dw2, db2)``.

    The mean divides by ``n*o``, rather than just the number of batch rows.
    Bias gradients sum the already normalized residual derivatives over rows.
    """
    _validate(x, w1, b1, w2, b2, y)
    hidden, prediction = _forward(x, w1, b1, w2, b2)
    residual = prediction - y
    loss = (residual * residual).mean() / 2
    d2 = residual / (x.shape[0] * w2.shape[1])
    dw2 = hidden.T @ d2
    db2 = d2.sum(axis=0)
    d1 = (d2 @ w2.T) * (1 - hidden * hidden)
    dw1 = x.T @ d1
    db1 = d1.sum(axis=0)
    return loss, dw1, db1, dw2, db2


def sgd_step(x, y, w1, b1, w2, b2, rate):
    """Return pre-update loss and four new parameters from one explicit step.

    Every gradient uses the original parameters. ``rate`` can be a Python float
    or a captured dynamic scalar; inputs and existing parameter arrays are not
    mutated. The caller is responsible for a finite positive learning rate.
    """
    loss, dw1, db1, dw2, db2 = loss_and_gradients(x, y, w1, b1, w2, b2)
    return loss, w1 - rate * dw1, b1 - rate * db1, w2 - rate * dw2, b2 - rate * db2
