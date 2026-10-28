"""CUDA-owned lowering of typed elementwise groups and mapped reductions.

Prepared payloads contain source and launch metadata, never runtime arrays or
scalar values. The compiler owns grouping and shared/public output boundaries.
"""

from math import prod

from ._array import ndarray
from ._cuda import _EXPRESSIONS, _offset, _type
from ._lowering import _Lowering
from ._norm import _index
from ._reductions import combine, identity


def _expression(root, group, leaves):
    params, lines, names = [], [], {}
    for index, leaf in enumerate(leaves):
        symbol = f"v{index}"
        names[id(leaf)] = symbol
        dtype = _type(leaf.dtype)
        scalar = leaf.op == "scalar"
        params.append(f"{dtype} x{index}" if scalar else f"const {dtype}*x{index}")
        read = f"x{index}" if scalar else f"x{index}[{_offset(root.shape, leaf.shape)}]"
        lines.append(f"{dtype} {symbol}={read};")
    for index, node in enumerate(group, len(leaves)):
        names[id(node)] = f"v{index}"
        dtype = _type(node.dtype)
        lines.append(f"{dtype} v{index}; {{")
        # Retain every operation's promotion and storage rounding, including
        # explicit narrowing casts before subsequent higher-precision work.
        for label, child, resolved in zip(("a", "b", "c"), node.inputs, node.resolved):
            dt = _type(resolved)
            value = names[id(child)]
            if resolved.kind == "b":
                value = f"({value} != 0)"
            lines.append(f"{dt} {label}=({dt}){value};")
        expression = _EXPRESSIONS[node.op]
        if node.op == "isfinite" and node.resolved[0].kind == "b":
            expression = "1"
        if node.dtype.kind == "b":
            expression = f"(({expression})!=0)"
        lines.append(f"v{index}=({dtype})({expression}); }}")
    return params, "\n".join(lines), names[id(root)]


class CUDALowering(_Lowering):
    def can_reduce_materialized(self, node):
        return (
            node.dtype.kind == "f"
            and node.dtype.itemsize == 4
            and node.attrs.get("operation") in ("min", "max")
        )

    def can_residual_matvec(self, matrix, vector, rhs, node):
        # Matmul stays owned by the explicitly chosen provider. This lowerer
        # does not implement the OpenCL/Vulkan residual GEMV epilogue.
        return False

    def prepare_fused(self, node, group, leaves):
        params, body, value = _expression(node, group, leaves)
        params.append(f"{_type(node.dtype)}*out")
        count = prod(node.shape)
        source = f"""extern "C" __global__ void compute({",".join(params)}) {{
            long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
            if(i>={count}LL) return;
            {body}
            out[i]={value};
        }}"""
        return tuple(leaves), (source, (count + 127) // 128)

    def run_fused(self, node, payload, arguments, operands):
        source, blocks = payload
        out = self.backend.empty(node.shape, node.dtype)
        self.backend._run(source, blocks, *(a._data for a in operands), out)
        return out

    def prepare_map_reduce(self, node, group, leaves):
        mapped = node.inputs[0]
        operation = node.attrs.get("operation", "sum")
        if node.dtype.kind == "b" and operation == "sum":
            raise NotImplementedError("Boolean reduction is not implemented in initial CUDA")
        if prod(mapped.shape) > 2**31 - 1:
            raise ValueError("Array exceeds initial CUDA indexing limits")
        params, body, value = _expression(mapped, group, leaves)
        index, count, reduced = _index(mapped.shape, node.attrs["axes"])
        chunks = max(1, (reduced + 4095) // 4096)
        dtype = _type(node.dtype)
        neutral = "0" if operation == "sum" else identity(operation, node.dtype, dtype, "cuda")
        accumulate = f"acc+=({dtype}){value};"
        merge = "sums[lane]+=sums[lane+step];"
        if operation != "sum":
            accumulate = (
                "acc="
                + combine(operation, node.dtype, dtype, "cuda", "acc", f"({dtype}){value}")
                + ";"
            )
            merge = (
                "sums[lane]="
                + combine(operation, node.dtype, dtype, "cuda", "sums[lane]", "sums[lane+step]")
                + ";"
            )
        params.append(f"{dtype}*out")
        # Same accumulation/tree/chunk contract as eager CUDA reduce: 128
        # lanes and at most 4096 terms per block, bounded partial storage.
        # All lanes reach barriers, including tails and empty reductions.
        source = f"""extern "C" __global__ void compute({",".join(params)}) {{
            long long g=blockIdx.x, o=g/{chunks}LL, chunk=g%{chunks}LL;
            int lane=threadIdx.x;
            __shared__ {dtype} sums[128]; {dtype} acc={neutral};
            for(long long r=chunk*4096LL+lane;
                r<min((chunk+1)*4096LL,{reduced}LL);r+=128) {{
                long long i={index};
                {body}
                {accumulate}
            }}
            sums[lane]=acc; __syncthreads();
            for(int step=64;step>0;step/=2) {{
                if(lane<step) {merge} __syncthreads();
            }}
            if(lane==0) out[g]=sums[0];
        }}"""
        return tuple(leaves), (source, count, chunks)

    def run_map_reduce(self, node, payload, arguments, operands):
        source, count, chunks = payload
        out = self.backend.empty((count, chunks), node.dtype)
        self.backend._run(source, count * chunks, *(a._data for a in operands), out)
        if chunks > 1:
            partial = ndarray._wrap(out, (count, chunks), node.dtype, self.backend)
            options = {"operation": node.attrs["operation"]} if "operation" in node.attrs else {}
            return self.backend.reduce(partial, (1,), False, node.dtype, **options)
        return out
