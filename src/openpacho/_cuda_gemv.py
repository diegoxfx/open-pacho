"""CUDA-owned matrix-vector and vector-dot kernels.

The public CUDA backend dispatches single-vector products with a sufficiently
large inner dimension here.  Inputs are C-contiguous, so flattening every
leading dimension into a row gives the same indexing contract as the existing
OpenCL cooperative GEMV path without materializing broadcasted copies.
"""

from math import prod

import numpy as np

from ._cuda import _type

_THREADS = 128
_DOT_CHUNK = 1024
_LONG_DOT_THRESHOLD = 4096


def _source(dtype, inner, ndim, chunks):
    """Return the fixed-width cooperative reduction kernel for one dispatch."""

    dt = _type(dtype)
    read_a = "a[j]" if ndim == 1 else f"a[row*{inner}LL+j]"
    start = "0LL" if chunks == 1 else f"chunk*{_DOT_CHUNK}LL"
    stop = f"{inner}LL" if chunks == 1 else f"min((chunk+1)*{_DOT_CHUNK}LL,{inner}LL)"
    return f"""extern "C" __global__ void compute(const {dt}*a, const {dt}*b, {dt}*out) {{
    long long block=(long long)blockIdx.x;
    long long row=block/{chunks}LL, chunk=block%{chunks}LL;
    int lane=(int)threadIdx.x;
    long long start={start};
    long long stop={stop};
    __shared__ {dt} sums[{_THREADS}];
    {dt} acc=({dt})0;
    for(long long j=start+(long long)lane;j<stop;j+={_THREADS}LL)
        acc+={read_a}*b[j];
    sums[lane]=acc;
    __syncthreads();
    for(int step={_THREADS // 2};step>0;step/=2) {{
        if(lane<step) sums[lane]+=sums[lane+step];
        __syncthreads();
    }}
    if(lane==0) out[block]=sums[0];
}}"""


def matvec_data(backend, a, b, shape, dtype):
    """Compute ``a @ b`` for a single vector ``b`` on a CUDA backend.

    ``a`` may be a vector, a matrix, or a C-contiguous batched array.  Its
    leading dimensions are flattened into independent rows.  A vector dot
    whose length exceeds 4096 is split into 1024-term blocks and reduced from
    device-resident partials through the backend's normal reduction contract.
    """

    dtype = np.dtype(dtype)
    inner = int(a.shape[-1])
    rows = int(prod(shape))

    # The regular CUDA matmul entry point handles these cases before routing,
    # but keeping the helper safe makes direct use deterministic as well.
    if not rows:
        return backend.empty(shape, dtype)
    if not inner:
        return backend.full(shape, 0, dtype)

    chunks = (
        (inner + _DOT_CHUNK - 1) // _DOT_CHUNK if a.ndim == 1 and inner > _LONG_DOT_THRESHOLD else 1
    )
    groups = rows * chunks
    if chunks > 1:
        partial = backend.empty((groups,), dtype)
        backend._run(
            _source(dtype, inner, a.ndim, chunks),
            groups,
            a._data,
            b._data,
            partial,
            threads=_THREADS,
        )
        from ._array import ndarray

        temporary = ndarray._wrap(partial, (groups,), dtype, backend)
        return backend.reduce(temporary, (0,), False, dtype)

    out = backend.empty(shape, dtype)
    backend._run(
        _source(dtype, inner, a.ndim, chunks),
        groups,
        a._data,
        b._data,
        out,
        threads=_THREADS,
    )
    return out
