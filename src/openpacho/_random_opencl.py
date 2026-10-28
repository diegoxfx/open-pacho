"""OpenCL Philox4x32-10 random-data adapter."""

import re
from functools import lru_cache
from math import prod

import numpy as np

from ._random import _random_scalars, random_body

_CAST_RE = re.compile(r"\b(float|double|uint)\(([^()]*)\)")
_PRODUCT = (
    "uint h0=mul_hi(c0,0xD2511F53u), l0=c0*0xD2511F53u, "
    "h1=mul_hi(c2,0xCD9E8D57u), l1=c2*0xCD9E8D57u;"
)


@lru_cache(maxsize=128)
def _source(dtype, distribution):
    """Return the immutable OpenCL kernel source for a dtype/distribution."""

    dtype = np.dtype(dtype)
    dt = "float" if dtype == np.dtype("float32") else "double"
    body = random_body(dtype, distribution, _PRODUCT, f"{dt} nl=-log1p(-u);\n")
    # OpenCL C uses C-style casts (GLSL/CUDA accept function-style casts).
    body = _CAST_RE.sub(r"((\1)(\2))", body)
    params = f"uint seed0,uint seed1,uint base0,uint base1,{dt} first,{dt} second,{dt} upper"
    return f"#pragma OPENCL FP_CONTRACT OFF\n__kernel void compute({params},__global {dt}*out) {{ uint i=get_global_id(0); {body} }}"


def random_data(backend, shape, dtype, distribution, seed, start, first, second):
    """Generate resident data by launching the cached OpenCL kernel."""

    dtype = np.dtype(dtype)
    count = prod(shape)
    out = backend.empty(shape, dtype)
    if not count:
        return out
    scalars = _random_scalars(dtype, distribution, seed, start, first, second)
    backend._run(_source(dtype, distribution), count, *scalars, out)
    return out


__all__ = ["random_data"]
