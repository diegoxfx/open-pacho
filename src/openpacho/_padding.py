"""Opt-in physical padding for eligible kernels, without changing logical arrays.

Backends own capability and copy dispatch. Plans retain only shape metadata and
backend callables: inputs are packed anew on every invocation, never frozen.
"""

from dataclasses import dataclass
from math import prod

import numpy as np


def validate_padding(value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("padding must be a boolean")
    return bool(value)


def prepare_padding(backend, ashape, bshape, shape, dtype):
    """An absent capability declines the hint; preparation failures propagate."""
    factory = getattr(backend, "prepare_padded_matmul", None)
    return None if factory is None else factory(ashape, bshape, shape, dtype)


@dataclass(frozen=True)
class PaddedGemm:
    backend: object
    input_shapes: tuple
    output_shape: tuple
    padded_input_shapes: tuple
    padded_output_shape: tuple
    copy: object

    @property
    def dispatch_count(self):
        return (
            1
            + sum(a != b for a, b in zip(self.input_shapes, self.padded_input_shapes))
            + (self.output_shape != self.padded_output_shape)
        )

    def run(self, a, b):
        from ._array import ndarray
        from ._gemm import register_gemm_data

        operands = []
        for array, shape in zip((a, b), self.padded_input_shapes):
            if array.shape != shape:
                data = self.copy(self.backend, array, shape)
                array = ndarray._wrap(data, shape, array.dtype, self.backend)
            operands.append(array)
        data = register_gemm_data(
            self.backend, *operands, self.padded_output_shape, specialize_bounds=True
        )
        if self.padded_output_shape != self.output_shape:
            padded = ndarray._wrap(data, self.padded_output_shape, a.dtype, self.backend)
            data = self.copy(self.backend, padded, self.output_shape)
        return data

    def explain(self):
        private = [
            prod(padded) * 4
            for original, padded in zip(self.input_shapes, self.padded_input_shapes)
            if original != padded
        ]
        if self.output_shape != self.padded_output_shape:
            private.append(prod(self.padded_output_shape) * 4)
        return {
            "input_shapes": self.input_shapes,
            "padded_input_shapes": self.padded_input_shapes,
            "output_shape": self.output_shape,
            "padded_output_shape": self.padded_output_shape,
            "alignment": (32, 16, 32),  # M, K, N; not next powers of two.
            "scratch_bytes": sum(private),
            "copy_dispatches": len(private),
            "bounds_checks_removed": True,  # Element guards inside the GEMM tile.
        }


def prepare_register_padding(backend, ashape, bshape, shape, dtype, copy, *, max_buffer_bytes=None):
    """Keep the existing register-GEMM eligibility and bound every padded buffer.

    Backend/provider checks belong to the caller. A limit decline keeps ordinary
    matmul; a later allocation failure remains an error, never a CPU/BLAS switch.
    """
    ashape, bshape, shape = tuple(ashape), tuple(bshape), tuple(shape)
    if (
        np.dtype(dtype) != np.dtype("float32")
        or len(ashape) < 2
        or len(bshape) < 2
        or not prod(shape)
        or min(ashape[-2], bshape[-1]) < 32
        or ashape[-1] < 16
    ):
        return None
    m, k, n = ashape[-2], ashape[-1], bshape[-1]
    expected = np.broadcast_shapes(ashape[:-2], bshape[:-2]) + (m, n)
    if k != bshape[-2] or shape != expected:
        raise ValueError("Invalid padded matmul shapes")
    mp, kp, npad = (
        (value + tile - 1) // tile * tile for value, tile in zip((m, k, n), (32, 16, 32))
    )
    ap, bp, cp = ashape[:-2] + (mp, kp), bshape[:-2] + (kp, npad), shape[:-2] + (mp, npad)
    sizes = [prod(s) for s in (ap, bp, cp)]
    if max(sizes) > 2**31 - 1 or (
        max_buffer_bytes is not None and max(sizes) * 4 > max_buffer_bytes
    ):
        return None
    return PaddedGemm(backend, (ashape, bshape), shape, (ap, bp), cp, copy)
