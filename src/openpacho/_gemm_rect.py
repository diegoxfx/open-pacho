"""Small-row GEMM tiles, with the same sequential K accumulation as direct GEMM.

The backend selects one of two fixed configurations. No packing, extra output
buffers, subgroup instructions, reduced precision or mathematical flags.
"""

from functools import lru_cache
from math import prod

import numpy as np

from ._gemm import _offset


@lru_cache(maxsize=128)
def source(name, ashape, bshape, tile):
    if tile not in ((16, 32, 16), (16, 16, 32)):
        raise ValueError("Unsupported rectangular GEMM tile")
    tm, tn, tk = tile
    m, k, n = ashape[-2], ashape[-1], bshape[-1]
    batch_shape = np.broadcast_shapes(ashape[:-2], bshape[:-2])
    nr, nc = (m + tm - 1) // tm, (n + tn - 1) // tn
    rows, columns = tm // 8, tn // 8
    init = " ".join(f"float c{r}{c}=0.0f;" for r in range(rows) for c in range(columns))
    inner = " ".join(f"float a{r}=left_tile[(ly+{r * 8})*{tk}+kk];" for r in range(rows))
    inner += " ".join(f"float b{c}=right_tile[kk*{tn}+lx+{c * 8}];" for c in range(columns))
    inner += " ".join(f"c{r}{c}+=a{r}*b{c};" for r in range(rows) for c in range(columns))
    stores = " ".join(
        f"if(row+{r * 8}<{m} && col+{c * 8}<{n}) "
        f"x2[(batch*{m}+row+{r * 8})*{n}+col+{c * 8}]=c{r}{c};"
        for r in range(rows)
        for c in range(columns)
    )
    body = f"""
        int lx=lane%8, ly=lane/8, batch=i/{nr * nc};
        int row=((i/{nc})%{nr})*{tm}, col=(i%{nc})*{tn};
        int abase=({_offset(batch_shape, ashape[:-2])})*{m * k};
        int bbase=({_offset(batch_shape, bshape[:-2])})*{k * n}; {init}
        for(int start=0;start<{k};start+={tk}) {{
            for(int t=lane;t<{tm * tk};t+=64) {{
                int ar=row+t/{tk}, ak=start+t%{tk};
                left_tile[t]=(ar<{m} && ak<{k})?x0[abase+ar*{k}+ak]:0.0f;
            }}
            for(int t=lane;t<{tk * tn};t+=64) {{
                int bk=start+t/{tn}, bc=col+t%{tn};
                right_tile[t]=(bk<{k} && bc<{n})?x1[bbase+bk*{n}+bc]:0.0f;
            }}
            BARRIER;
            for(int kk=0;kk<{tk};++kk) {{ {inner} }}
            BARRIER;
        }}
        row+=ly; col+=lx; {stores}
    """
    shared = f"float left_tile[{tm * tk}],right_tile[{tk * tn}];"
    if name == "opencl":
        body = (
            "__kernel void compute(__global const float*x0,__global const float*x1,"
            "__global float*x2){int i=get_group_id(0),lane=get_local_id(0);__local "
            + shared
            + body.replace("BARRIER", "barrier(CLK_LOCAL_MEM_FENCE)")
            + "}"
        )
    elif name == "cuda":
        body = (
            'extern "C" __global__ void compute(const float*x0,const float*x1,float*x2){'
            "int i=blockIdx.x,lane=threadIdx.x;__shared__ "
            + shared
            + body.replace("BARRIER", "__syncthreads()")
            + "}"
        )
    else:
        raise ValueError("Rectangular GEMM is enabled only for OpenCL and CUDA")
    return body, prod(batch_shape) * nr * nc


def select_tile(backend, ashape, bshape):
    """Conservative measured region; batching and wider grids retain direct GEMM."""
    config = getattr(backend, "_small_row_gemm_config", None)
    if config is None or len(ashape) != 2 or len(bshape) != 2:
        return None
    tile, minimum_k = config
    m, k, n = ashape[-2], ashape[-1], bshape[-1]
    if m % 32 == 0 and k % 16 == 0 and n % 32 == 0:
        return None
    if (
        32 <= m < 96
        and k >= minimum_k
        and 256 <= n <= 4096
        and ((m + 31) // 32) * ((n + 31) // 32) <= 224
    ):
        return tile
    return None
