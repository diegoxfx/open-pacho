# openpacho/la/operations.py

import numpy as np
import math
from openpacho.core.array import OpenPachoArray
from openpacho.opencl_backend.opencl_context import OpenCLContextManager
from openpacho.opencl_backend import opencl_kernels

def dot(x: OpenPachoArray, y: OpenPachoArray):
    """
    Compute the dot product of two vectors or matrix-vector.
    If x and y are both 1D, returns a scalar.
    If x is 2D and y is 1D, returns a 1D array (matrix-vector).
    """
    if x.backend.backend_name != y.backend.backend_name:
        raise ValueError(f"Both arrays must belong to the same backend. Got '{x.backend.backend_name}' and '{y.backend.backend_name}'.")

    if len(x.shape) == 1 and len(y.shape) == 1:
        result = x.backend.vector_dot(x, y)
        return result
    elif len(x.shape) == 2 and len(y.shape) == 1:
            result = x.backend.matrix_vec(x, y, return_vec=True)
    else:
        result = x.backend.matrix_matrix(x, y)
        if result.shape == (1, 1):
            return result._extract_scalar()
    return result


def norm(x: OpenPachoArray):
    vecdot = x.backend.vector_dot(x, x)
    if isinstance(vecdot, OpenPachoArray):
         vecdot = vecdot._extract_scalar()
    return math.sqrt(vecdot)