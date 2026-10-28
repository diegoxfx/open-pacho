"""Shared Philox4x32-10 arithmetic and backend-neutral lowering.

Algorithm: Salmon et al., SC11 / Random123. Implemented here using the published
round constants and permutation; known-answer vectors protect the bit stream.

The GPU adapters provide the small pieces of source syntax that differ between
OpenCL, CUDA and GLSL. Keeping those pieces outside this module lets the shared
algorithm stay independent of backend names and runtime objects.
"""

from functools import lru_cache
from math import prod

import numpy as np

_MASK = 0xFFFFFFFF


def _philox_numpy(words, key):
    c0, c1, c2, c3 = (np.asarray(w, dtype=np.uint64) for w in words)
    k0, k1 = map(int, key)
    for _ in range(10):
        p0, p1 = c0 * np.uint64(0xD2511F53), c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            (p1 >> 32) ^ c1 ^ np.uint64(k0),
            p1 & _MASK,
            (p0 >> 32) ^ c3 ^ np.uint64(k1),
            p0 & _MASK,
        )
        k0, k1 = (k0 + 0x9E3779B9) & _MASK, (k1 + 0xBB67AE85) & _MASK
    return c0, c1, c2, c3


def _numpy_data(shape, dtype, distribution, seed, start, first, second):
    # Chunk CPU reference workspace independently of the output size.
    output = np.empty(prod(shape), dtype)
    for offset in range(0, output.size, 65536):
        count = min(65536, output.size - offset)
        counters = np.arange(count, dtype=np.uint64) + np.uint64(start + offset)
        words = _philox_numpy((counters & _MASK, counters >> 32, 0, 0), (seed & _MASK, seed >> 32))

        def unit(a, b):
            if dtype == np.dtype("float32"):
                return (a >> 8).astype(dtype) * dtype.type(2**-24)
            return ((a >> 5).astype(dtype) * 67108864.0 + (b >> 6).astype(dtype)) * 2**-53

        u = unit(words[0], words[1])
        if distribution == "random":
            values = u
        elif distribution == "uniform":
            values = np.clip(first + (second - first) * u, first, np.nextafter(second, first))
        elif distribution == "normal":
            angle = dtype.type(2 * np.pi) * unit(words[2], words[3])
            values = first + second * (np.sqrt(dtype.type(-2) * np.log1p(-u)) * np.cos(angle))
        else:
            values = second * -np.log1p(-u)
        output[offset : offset + count] = values
    return output


@lru_cache(maxsize=128)
def random_body(dtype, distribution, product, negative_log1m):
    """Build the shared Philox body from backend-provided source fragments.

    ``product`` contains the backend's 32-bit multiply-high spelling and
    ``negative_log1m`` contains its ``-log(1-u)`` implementation. Both are
    immutable source strings supplied by an adapter; this function has no
    backend-name or runtime dispatch.
    """

    dt = "float" if dtype == np.dtype("float32") else "double"
    cast = lambda x: f"{dt}({x})"
    unit = lambda a, b: (
        f"{cast(a + ' >> 8')}*{cast('0.000000059604644775390625')}"
        if dt == "float"
        else f"({cast(a + ' >> 5')}*{cast('67108864.0')}+{cast(b + ' >> 6')})*{cast('1.1102230246251565404236316680908203125e-16')}"
    )
    text = f"""
    uint c0=base0+uint(i), c1=base1+uint(c0<base0), c2=0u, c3=0u;
    uint k0=seed0,k1=seed1;
    for(int round=0;round<10;++round) {{
        {product}
        uint n0=h1^c1^k0, n2=h0^c3^k1;
        c0=n0; c1=l1; c2=n2; c3=l0;
        k0+=0x9E3779B9u; k1+=0xBB67AE85u;
    }}
    {dt} u={unit("c0", "c1")};
    """
    if distribution in ("normal", "exponential"):
        text += negative_log1m
    if distribution == "random":
        expr = "u"
    elif distribution == "uniform":
        text += f"{dt} value=first+(second-first)*u;\n"
        expr = "value<first ? first : (value>upper ? upper : value)"
    elif distribution == "normal":
        text += f"{dt} v={unit('c2', 'c3')};\n"
        expr = (
            f"first+second*(sqrt({cast('2')}*nl)*cos({cast('6.283185307179586476925286766559')}*v))"
        )
    else:
        expr = "second*nl"
    return text + f"out[i]={expr};"


def _random_scalars(dtype, distribution, seed, start, first, second):
    """Pack invocation scalars while leaving all runtime values per-call."""

    words = tuple(np.uint32(x) for x in (seed & _MASK, seed >> 32, start & _MASK, start >> 32))
    upper = np.nextafter(second, first) if distribution == "uniform" else dtype.type(1)
    return (*words, first, second, upper)
