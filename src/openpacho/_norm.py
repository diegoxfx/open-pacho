"""Scaled sum-of-squares reduction shared by eager and compiled norms.

Pairs (scale, ssq) represent sum(x*x) as scale**2 * ssq without forming
scale**2. Merging keeps ratios at most one. Device subnormal support applies.
"""

from math import prod

import numpy as np


def _index(shape, axes):
    output = tuple(d for i, d in enumerate(shape) if i not in axes)
    reduced = tuple(shape[i] for i in axes)
    terms = []
    oi = ri = 0
    for axis, dim in enumerate(shape):
        stride = prod(shape[axis + 1 :])
        if axis in axes:
            terms.append(f"((r/{max(1, prod(reduced[ri + 1 :]))})%{max(1, dim)})*{stride}")
            ri += 1
        else:
            terms.append(f"((o/{max(1, prod(output[oi + 1 :]))})%{max(1, dim)})*{stride}")
            oi += 1
    return "+".join(terms) or "0", prod(output), prod(reduced)


def _merge(dt):
    return f"""
        if (isnan(s) || isnan(v)) {{ s=s+v; t=s; }}
        else if (isinf(v)) {{ s=v; t={dt}(1); }}
        else if (!isinf(s) && v>{dt}(0)) {{
            if(s<v) {{ {dt} z=s/v; t=w+t*z*z; s=v; }}
            else {{ {dt} z=v/s; t+=w*z*z; }}
        }}
    """


def norm_data(backend, a, axes, keepdims, dtype):
    dtype = np.dtype(dtype)
    index, count, reduced = _index(a.shape, axes)
    if not count or not reduced:
        return backend.full((count,), 0, dtype)
    if backend.name == "numpy":
        data = a._data.reshape(a.shape)
        scale = np.max(np.abs(data), axis=axes, keepdims=True)
        with np.errstate(all="ignore"):
            normalized = np.divide(data, scale, out=np.zeros_like(data), where=scale != 0)
            total = np.sum(normalized * normalized, axis=axes, keepdims=True, dtype=dtype)
            result = scale * np.sqrt(total)
        result = np.where(np.isinf(scale), scale, result)
        return np.asarray(result, dtype=dtype).reshape(-1)
    if a.size > 2**31 - 1:
        raise ValueError("Norm exceeds GPU indexing limits")
    lanes = (
        64
        if backend.name == "vulkan"
        else 128
        if backend.name == "cuda"
        else min(128, backend.work_group_size)
    )
    chunks = (reduced + 4095) // 4096
    current, first = a._data, True
    while True:
        final = chunks == 1
        out = backend.empty((count,) if final else (count, chunks, 2), dtype)
        dt = "float" if dtype == np.dtype("float32") else "double"
        merge = _merge(dt)
        read = (
            f"{dt} v=abs(x0[{index}]), w={dt}(1);"
            if first
            else f"{dt} v=x0[(o*{reduced}+r)*2], w=x0[(o*{reduced}+r)*2+1];"
        )
        write = "x1[o]=s*sqrt(t);" if final else "x1[i*2]=s; x1[i*2+1]=t;"
        body = f"""
            int o=i/{chunks}, chunk=i%{chunks};
            {dt} s={dt}(0), t={dt}(0);
            for(int j=lane;j<min(4096,{reduced}-chunk*4096);j+={lanes}) {{
                int r=chunk*4096+j; {read} {merge}
            }}
            scales[lane]=s; sums[lane]=t; BARRIER;
            for(int step={lanes // 2};step>0;step/=2) {{
                if(lane<step) {{ {dt} v=scales[lane+step], w=sums[lane+step];
                    {merge} scales[lane]=s; sums[lane]=t;
                }} BARRIER;
            }}
            if(lane==0) {{ {write} }}
        """
        if backend.name == "vulkan":
            body = "int lane=int(gl_LocalInvocationID.x);\n" + body
            backend._run(
                body.replace("BARRIER", "barrier()"),
                count * chunks,
                [(current, dtype), (out, dtype)],
                cooperative=True,
                shared=f"shared {dt} scales[{lanes}], sums[{lanes}];",
            )
        elif backend.name == "cuda":
            source = f"""extern "C" __global__ void compute(const {dt}*x0,{dt}*x1) {{
                int i=blockIdx.x, lane=threadIdx.x;
                __shared__ {dt} scales[{lanes}], sums[{lanes}];
                {body.replace("abs(", "fabs(").replace("BARRIER", "__syncthreads()")}
            }}"""
            backend._run(source, count * chunks, current, out, threads=lanes)
        else:
            for number in (0, 1):
                body = body.replace(f"{dt}({number})", f"(({dt}){number})")
            body = body.replace("abs(", "fabs(").replace("BARRIER", "barrier(CLK_LOCAL_MEM_FENCE)")
            source = f"""#pragma OPENCL FP_CONTRACT OFF
                __kernel void compute(__global const {dt}*x0,__global {dt}*x1) {{
                    int i=get_group_id(0), lane=get_local_id(0);
                    __local {dt} scales[{lanes}], sums[{lanes}]; {body}
                }}"""
            backend._run(source, count * chunks * lanes, current, out, local_size=lanes)
        if final:
            return out
        current, reduced, first = out, chunks, False
        chunks = (reduced + 4095) // 4096
