"""Prepared provider execution of private, full 2D transpose operands."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransposedGemm:
    """Store metadata and backend only; replacement inputs remain dynamic."""

    backend: object
    ashape: tuple
    bshape: tuple
    shape: tuple
    dtype: np.dtype
    transpose_a: bool
    transpose_b: bool

    def run(self, a, b):
        out = self.backend.empty(self.shape, self.dtype)
        self.backend._blas.matmul(
            self.backend.queue,
            a,
            b,
            out,
            self.shape,
            self.dtype,
            transpose_a=self.transpose_a,
            transpose_b=self.transpose_b,
        )
        return out

    def explain(self):
        return {
            "provider": "clblast",
            "transpose_a": self.transpose_a,
            "transpose_b": self.transpose_b,
            "stored_shapes": (self.ashape, self.bshape),
            "leading_dimensions": (self.ashape[1], self.bshape[1], self.shape[1]),
            # Count operands, not unique gathers: a private transpose may be
            # used twice by the same product without requiring two copies.
            "transpose_operand_bytes": self.dtype.itemsize
            * sum(
                shape[0] * shape[1]
                for shape, transpose in (
                    (self.ashape, self.transpose_a),
                    (self.bshape, self.transpose_b),
                )
                if transpose
            ),
        }
