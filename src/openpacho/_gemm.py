"""Portable float32 direct GEMM with local tiles and a 4x4 register block.

No packing, reduced precision, device database or vendor-specific instructions.
Each 64-lane group computes a 32x32 output tile and reuses two 32x16 local tiles.
"""

from functools import lru_cache
from math import prod

import numpy as np


def _offset(batch_shape, source_shape):
    terms = []
    padding = len(batch_shape) - len(source_shape)
    for axis, dim in enumerate(source_shape):
        if dim != 1:
            terms.append(
                f"((batch/{prod(batch_shape[padding + axis + 1 :])})%{dim})*{prod(source_shape[axis + 1 :])}"
            )
    return "+".join(terms) or "0"


@lru_cache(maxsize=128)
def _source(name, ashape, bshape, *, specialize_bounds=False):
    m, k, n = ashape[-2], ashape[-1], bshape[-1]
    batch_shape = np.broadcast_shapes(ashape[:-2], bshape[:-2])
    nr, nc = (m + 31) // 32, (n + 31) // 32
    abase = f"({_offset(batch_shape, ashape[:-2])})*{m * k}"
    bbase = f"({_offset(batch_shape, bshape[:-2])})*{k * n}"
    init = " ".join(f"float c{r}{c}=0.0f;" for r in range(4) for c in range(4))
    # Explicit private scalars keep all accumulator indices static for both
    # OpenCL and GLSL compilers. Adjacent lanes cover adjacent output columns.
    inner = " ".join(f"float a{r}=left_tile[(ly+{r * 8})*16+kk];" for r in range(4))
    inner += " ".join(f"float b{c}=right_tile[kk*32+lx+{c * 8}];" for c in range(4))
    inner += " ".join(f"c{r}{c}+=a{r}*b{c};" for r in range(4) for c in range(4))

    # Every workgroup covers one complete tile on an aligned axis. Keep guards
    # on all other axes; no thread may leave before a cooperative barrier.
    def conditions(items):
        return " && ".join(test for test, aligned in items if not specialize_bounds or not aligned)

    stores = []
    for r in range(4):
        for c in range(4):
            condition = conditions(
                [(f"row+{r * 8}<{m}", m % 32 == 0), (f"col+{c * 8}<{n}", n % 32 == 0)]
            )
            guard = f"if({condition}) " if condition else ""
            stores.append(guard + f"x2[(batch*{m}+row+{r * 8})*{n}+col+{c * 8}]=c{r}{c};")
    stores = " ".join(stores)
    left_guard = conditions([(f"ar<{m}", m % 32 == 0), (f"ak<{k}", k % 16 == 0)])
    right_guard = conditions([(f"bk<{k}", k % 16 == 0), (f"bc<{n}", n % 32 == 0)])
    left_load = f"x0[abase+ar*{k}+ak]"
    right_load = f"x1[bbase+bk*{n}+bc]"
    # The conditional's two arms must stay float32. An unsuffixed 0.0
    # promotes the load to double in OpenCL C on fp64-capable devices;
    # this driver retained both conversions around each guarded tile load.
    if left_guard:
        left_load = f"({left_guard})?{left_load}:0.0f"
    if right_guard:
        right_load = f"({right_guard})?{right_load}:0.0f"
    body = f"""
        int lx=lane%8, ly=lane/8, batch=i/{nr * nc};
        int row=((i/{nc})%{nr})*32, col=(i%{nc})*32;
        int abase={abase}, bbase={bbase}; {init}
        for(int start=0;start<{k};start+=16) {{
            for(int t=lane;t<512;t+=64) {{
                int ar=row+t/16, ak=start+t%16;
                int bk=start+t/32, bc=col+t%32;
                left_tile[t]={left_load};
                right_tile[t]={right_load};
            }}
            BARRIER;
            for(int kk=0;kk<16;++kk) {{ {inner} }}
            BARRIER;
        }}
        row+=ly; col+=lx; {stores}
    """
    count = prod(batch_shape) * nr * nc
    if name == "opencl":
        body = body.replace("BARRIER", "barrier(CLK_LOCAL_MEM_FENCE)")
        return (
            f"""__kernel void compute(__global const float*x0,
            __global const float*x1,__global float*x2) {{
            int i=get_group_id(0),lane=get_local_id(0);
            __local float left_tile[512],right_tile[512]; {body} }}""",
            count,
        )
    if name == "cuda":
        return (
            'extern "C" __global__ void compute(const float*x0,const float*x1,float*x2) {'
            "int i=blockIdx.x,lane=threadIdx.x;"
            "__shared__ float left_tile[512],right_tile[512];"
            + body.replace("BARRIER", "__syncthreads()")
            + "}",
            count,
        )
    return "int lane=int(gl_LocalInvocationID.x);" + body.replace("BARRIER", "barrier()"), count


def register_gemm_data(backend, a, b, shape, *, specialize_bounds=None):
    if max(a.size, b.size, prod(shape)) > 2**31 - 1:
        raise ValueError("Register GEMM exceeds GPU indexing limits")
    # Measured benefit in OpenCL for complete tiles. Other backends and tails
    # retain their previous shaders; this is not a blanket fast-math mode.
    complete_opencl_tile = (
        backend.name == "opencl"
        and a.shape[-2] % 32 == 0
        and b.shape[-1] % 32 == 0
        and a.shape[-1] % 16 == 0
    )
    # Padding explicitly selects its aligned 32x32 kernel. A backend may enable
    # a smaller tile for unpadded, small-row problems without changing storage.
    from ._gemm_rect import select_tile, source

    tile = select_tile(backend, a.shape, b.shape) if specialize_bounds is None else None
    if tile is not None:
        body, count = source(backend.name, a.shape, b.shape, tile)
    else:
        if specialize_bounds is None:
            specialize_bounds = complete_opencl_tile
        body, count = _source(backend.name, a.shape, b.shape, specialize_bounds=specialize_bounds)
    out = backend.empty(shape, a.dtype)
    if backend.name == "opencl":
        backend._run(body, count * 64, a._data, b._data, out, local_size=64)
    elif backend.name == "cuda":
        backend._run(body, count, a._data, b._data, out, threads=64)
    else:
        backend._run(
            body,
            count,
            [(a._data, a.dtype), (b._data, b.dtype), (out, a.dtype)],
            cooperative=True,
            shared="shared float left_tile[512],right_tile[512];",
        )
    return out
