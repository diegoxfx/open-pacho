"""Portable resident GEMV epilogue for compiled residuals, b - A @ x.

The reduction layout follows each custom backend. The matrix product is rounded
in the array dtype before subtraction, preserving the eager expression boundary.
"""

import numpy as np


def residual_matvec_data(backend, a, x, b):
    if a.ndim != 2 or x.ndim != 1 or b.ndim != 1:
        raise ValueError("Residual GEMV requires a matrix and two vectors")
    rows, columns = a.shape
    if x.shape != (columns,) or b.shape != (rows,):
        raise ValueError("Residual GEMV dimensions do not match")
    if a.dtype != x.dtype or a.dtype != b.dtype or a.dtype.kind != "f":
        raise ValueError("Residual GEMV requires matching float32/float64 dtypes")
    if a.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Residual GEMV requires matching float32/float64 dtypes")
    if any(value._backend is not backend for value in (a, x, b)):
        raise ValueError("Residual GEMV inputs must belong to the same backend")
    if backend.name not in ("opencl", "vulkan") or getattr(backend, "_blas", None) is not None:
        raise NotImplementedError("Residual GEMV requires a custom OpenCL or Vulkan provider")
    if not rows:
        return backend.empty((rows,), a.dtype)
    if not columns:
        return backend.elementwise("cast", [b], b.shape, b.dtype)
    if backend.name == "vulkan" and rows == 1 and columns > 4096:
        # This existing special case uses a partitioned DOT tree. Keep its
        # reduction order rather than silently substituting a different GEMV.
        from ._array import ndarray

        product = ndarray._wrap(backend.matmul(a, x, b.shape, a.dtype), b.shape, a.dtype, backend)
        return backend.elementwise("subtract", [b, product], b.shape, b.dtype)
    out = backend.empty(b.shape, a.dtype)
    dt = "float" if a.dtype == np.dtype("float32") else "double"
    if backend.name == "vulkan":
        backend._run(
            f"int lane=int(gl_LocalInvocationID.x); {dt} acc={dt}(0); "
            f"for(int j=lane;j<{columns};j+=64) acc+=x0[i*{columns}+j]*x1[j]; "
            "scratch[lane]=acc; barrier(); "
            "for(int step=32;step>0;step/=2) { "
            "if(lane<step) scratch[lane]+=scratch[lane+step]; barrier(); } "
            f"if(lane==0) {{ {dt} residual=x2[i]-scratch[0]; x3[i]=residual; }}",
            rows,
            [(a._data, a.dtype), (x._data, a.dtype), (b._data, a.dtype), (out, a.dtype)],
            cooperative=True,
            shared=f"shared {dt} scratch[64];",
        )
    elif columns >= 64:
        lanes = backend.work_group_size
        source = f"""#pragma OPENCL FP_CONTRACT OFF
        __kernel void compute(__global const {dt}*a, __global const {dt}*x,
                              __global const {dt}*b, __global {dt}*out) {{
            long row=get_group_id(0), lane=get_local_id(0);
            __local {dt} scratch[{lanes}]; {dt} acc=0;
            for(long j=lane;j<{columns}L;j+={lanes}) acc+=a[row*{columns}L+j]*x[j];
            scratch[lane]=acc; barrier(CLK_LOCAL_MEM_FENCE);
            for(int step={lanes // 2};step>0;step/=2) {{
                if(lane<step) scratch[lane]+=scratch[lane+step]; barrier(CLK_LOCAL_MEM_FENCE);
            }}
            if(lane==0) out[row]=b[row]-scratch[0];
        }}"""
        backend._run(source, rows * lanes, a._data, x._data, b._data, out, local_size=lanes)
    else:
        # The eager small-K implementation accumulates serially with contraction
        # permitted. Volatile preserves its final dtype rounding before epilogue.
        source = f"""__kernel void compute(__global const {dt}*a, __global const {dt}*x,
                              __global const {dt}*b, __global {dt}*out) {{
            long row=get_global_id(0); {dt} acc=0;
            for(long j=0;j<{columns}L;++j) acc+=a[row*{columns}L+j]*x[j];
            volatile {dt} product=acc; out[row]=b[row]-product;
        }}"""
        backend._run(source, rows, a._data, x._data, b._data, out)
    return out
