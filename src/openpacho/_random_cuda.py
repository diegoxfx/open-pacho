"""CUDA Philox4x32-10 random-data adapter."""

from functools import lru_cache
from math import prod

import numpy as np

from ._random import _random_scalars, random_body

_PRODUCT = (
    "uint h0=__umulhi(c0,0xD2511F53u), l0=c0*0xD2511F53u, "
    "h1=__umulhi(c2,0xCD9E8D57u), l1=c2*0xCD9E8D57u;"
)


@lru_cache(maxsize=128)
def _source(dtype, distribution, count):
    """Return the immutable CUDA kernel source for one launch geometry."""

    dtype = np.dtype(dtype)
    dt = "float" if dtype == np.dtype("float32") else "double"
    body = random_body(dtype, distribution, _PRODUCT, f"{dt} nl=-log1p(-u);\n")
    params = f"uint seed0,uint seed1,uint base0,uint base1,{dt} first,{dt} second,{dt} upper"
    return f"""typedef unsigned int uint;
            extern "C" __global__ void compute({params},{dt}*out) {{
                uint i=blockIdx.x*blockDim.x+threadIdx.x; if(i>={count}u) return;
                {body}
            }}"""


def random_data(backend, shape, dtype, distribution, seed, start, first, second):
    """Generate resident data by launching the cached CUDA kernel."""

    dtype = np.dtype(dtype)
    count = prod(shape)
    out = backend.empty(shape, dtype)
    if not count:
        return out
    scalars = _random_scalars(dtype, distribution, seed, start, first, second)
    backend._run(_source(dtype, distribution, count), (count + 127) // 128, *scalars, out)
    return out


__all__ = ["random_data"]
