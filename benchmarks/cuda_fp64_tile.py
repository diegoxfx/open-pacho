"""Benchmark-only FP64 shared-memory GEMM candidate; no backend dispatch changes.

The caller must supply C-contiguous float64 buffers and use the existing CUDA
compiler math options, including ``--fmad=false``. Vector, empty, and zero-K
products are ineligible and must retain the caller's original matmul fallback.
``blocks`` is a CUDA block count, and ``shared_bytes`` describes static shared
storage (do not also request this amount as dynamic shared memory).
"""

from dataclasses import dataclass
from math import prod
from operator import index

import numpy as np


@dataclass(frozen=True)
class Kernel:
    source: str
    blocks: int
    threads: int
    shared_bytes: int
    shape: tuple[int, ...]


def _shape(value):
    shape = tuple(index(dim) for dim in value)
    if len(shape) < 2 or any(dim <= 0 for dim in shape):
        raise ValueError("FP64 tile requires rank >= 2 and strictly positive dimensions")
    return shape


def _batch_offset(batch, operand):
    """Flatten a right-aligned broadcast batch using signed 64-bit arithmetic."""
    padding = len(batch) - len(operand)
    return (
        " + ".join(
            f"((batch/{prod(batch[axis + padding + 1 :])}LL)%{dim}LL)*{prod(operand[axis + 1 :])}LL"
            for axis, dim in enumerate(operand)
            if dim != 1
        )
        or "0LL"
    )


def source_for(ashape, bshape, tile=16) -> Kernel:
    """Generate one fixed 16x16 tile kernel for nonempty broadcast matrix products.

    This experiment accepts matrix ranks >= 2 only, rejects zero dimensions/K,
    and requires float64 C-contiguous operands at launch. Shape incompatibility
    raises ValueError; byte extents exceeding signed 64-bit addressing raise
    OverflowError. The caller remains responsible for device launch limits.
    Every output accumulates exactly K products in increasing K order, including
    an incomplete final tile, under the existing noncontracting compiler flags.
    """
    if index(tile) != 16:
        raise ValueError("The benchmark candidate supports only tile=16")
    ashape, bshape = _shape(ashape), _shape(bshape)
    m, k = ashape[-2:]
    if bshape[-2] != k:
        raise ValueError("Incompatible inner dimensions")
    n = bshape[-1]
    batch = np.broadcast_shapes(ashape[:-2], bshape[:-2])
    shape = batch + (m, n)
    if any(prod(s) * 8 > 2**63 - 1 for s in (ashape, bshape, shape)):
        raise OverflowError("FP64 tile byte extents exceed signed 64-bit addressing")
    tiles_r, tiles_c = (m + 15) // 16, (n + 15) // 16
    tiles_k = (k + 15) // 16
    abase = _batch_offset(batch, ashape[:-2])
    bbase = _batch_offset(batch, bshape[:-2])
    source = f"""extern "C" __global__ void compute(
    const double *a, const double *b, double *out) {{
    const long long g=(long long)blockIdx.x;
    const int lane=(int)threadIdx.x;
    const int ly=lane/16;
    const int lx=lane%16;
    const long long batch=g/{tiles_r * tiles_c}LL;
    const long long row=((g/{tiles_c}LL)%{tiles_r}LL)*16LL+ly;
    const long long col=(g%{tiles_c}LL)*16LL+lx;
    const long long abase=({abase})*{m * k}LL;
    const long long bbase=({bbase})*{k * n}LL;
    __shared__ double left[256];
    __shared__ double right[256];
    double acc=0.0;
    for(long long tile_k=0;tile_k<{tiles_k}LL;++tile_k) {{
        const long long start=tile_k*16LL;
        const int valid_k=(int)(({k}LL-start<16LL)?{k}LL-start:16LL);
        left[lane]=(row<{m}LL && start+lx<{k}LL)?a[abase+row*{k}LL+start+lx]:0.0;
        right[lane]=(start+ly<{k}LL && col<{n}LL)?b[bbase+(start+ly)*{n}LL+col]:0.0;
        __syncthreads();
        for(int j=0;j<valid_k;++j)
            acc+=left[ly*16+j]*right[j*16+lx];
        __syncthreads();
    }}
    if(row<{m}LL && col<{n}LL) out[(batch*{m}LL+row)*{n}LL+col]=acc;
}}"""
    return Kernel(source, prod(batch) * tiles_r * tiles_c, 256, 4096, shape)
