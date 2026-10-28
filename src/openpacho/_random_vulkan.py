"""Vulkan GLSL Philox4x32-10 random-data adapter."""

from functools import lru_cache
from math import prod

import numpy as np

from ._random import _random_scalars, random_body

_PRODUCT = (
    "uint h0,l0,h1,l1; umulExtended(c0,0xD2511F53u,h0,l0); umulExtended(c2,0xCD9E8D57u,h1,l1);"
)


def _log1m(dtype):
    """Return Vulkan's accurate small-u approximation for ``-log(1-u)``."""

    dtype = np.dtype(dtype)
    dt = "float" if dtype == np.dtype("float32") else "double"
    cast = lambda x: f"{dt}({x})"
    # GLSL lacks log1p. Avoid loss near zero with eight terms of -log(1-u);
    # at u<1/8 the omitted tail is <1e-9 absolute.
    polynomial = cast("0.125")
    for degree in range(7, 1, -1):
        polynomial = f"({cast(str(1 / degree))}+u*{polynomial})"
    return f"{dt} nl; if(u<{cast('0.125')}) nl=u+u*u*{polynomial}; else nl=-log({cast('1')}-u);\n"


@lru_cache(maxsize=128)
def _source(dtype, distribution):
    """Return the immutable Vulkan body/push-constant template.

    Count belongs to the final dispatch wrapper in ``backend._run``, not this
    body. Shapes therefore share this template without caching runtime values.
    """

    dtype = np.dtype(dtype)
    dt = "float" if dtype == np.dtype("float32") else "double"
    body = random_body(dtype, distribution, _PRODUCT, _log1m(dtype))
    bindings = ("seed0", "seed1", "base0", "base1", "first", "second", "upper")
    prefix = "\n".join(
        f"{'uint' if j < 4 else dt} {symbol}=pc.s{j};" for j, symbol in enumerate(bindings)
    )
    return prefix + body.replace("out[i]", "x0[i]")


def random_data(backend, shape, dtype, distribution, seed, start, first, second):
    """Generate resident data by launching the cached Vulkan shader body."""

    dtype = np.dtype(dtype)
    count = prod(shape)
    out = backend.empty(shape, dtype)
    if not count:
        return out
    from ._fusion import HostScalar

    scalars = _random_scalars(dtype, distribution, seed, start, first, second)
    backend._run(
        _source(dtype, distribution),
        count,
        [(out, dtype)],
        [HostScalar(v, np.asarray(v).dtype) for v in scalars],
    )
    return out


__all__ = ["random_data"]
