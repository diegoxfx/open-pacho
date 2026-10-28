"""Resident normalization recipes for finite floating-point inputs.

These examples use only public array operations and remain outside the runtime.
They are directly capturable with openpacho.compile. Inputs must be finite and
the selected axis nonempty; NaNs/infinities follow the arithmetic's ordinary
propagation rules. No host validation or hidden synchronization is performed.
"""

import openpacho as p


def softmax(x, axis=-1):
    """Normalize exponentials after a max shift, preserving the input dtype."""
    x = p.asarray(x)
    if x.dtype.kind != "f":
        raise TypeError("softmax example requires floating-point inputs")
    shifted = x - p.max(x, axis=axis, keepdims=True)
    weights = p.exp(shifted)
    return weights / p.sum(weights, axis=axis, keepdims=True)


def logsumexp(x, axis=-1):
    """Log of a sum of exponentials, retaining the reduced axis for broadcasting."""
    x = p.asarray(x)
    if x.dtype.kind != "f":
        raise TypeError("logsumexp example requires floating-point inputs")
    maximum = p.max(x, axis=axis, keepdims=True)
    return maximum + p.log(p.sum(p.exp(x - maximum), axis=axis, keepdims=True))
