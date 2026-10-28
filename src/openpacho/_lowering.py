"""Private backend lowering seam for compiled plans.

The compiler owns grouping, materialization and array lifetimes. Lowerers own
kernel preparation, scalar ABI and dispatch of those groups. Prepared payloads
may retain expression metadata, but never runtime arrays or bound scalar values.
Adding a backend here does not register a public backend or enable fallback.
"""

from math import prod

from ._array import ndarray
from ._reductions import combine as reduction_combine
from ._reductions import identity


class _Lowering:
    def __init__(self, backend):
        self.backend = backend

    def can_residual_matvec(self, matrix, vector, rhs, node):
        product = node.inputs[1]
        return (
            getattr(self.backend, "blas", "custom") == "custom"
            and matrix.ndim == 2
            and vector.ndim == rhs.ndim == 1
            and rhs.shape == product.shape == node.shape
            and node.dtype.kind == "f"
            and node.dtype == matrix.dtype == vector.dtype == rhs.dtype == product.dtype
        )


class OpenCLLowering(_Lowering):
    def can_reduce_materialized(self, node):
        return (
            node.dtype.kind == "f"
            and node.dtype.itemsize == 4
            and node.attrs.get("operation") in ("min", "max")
        )

    def prepare_fused(self, node, group, leaves):
        source, arguments = self._opencl_source(node, group, leaves)
        return arguments, source

    def prepare_map_reduce(self, node, group, leaves):
        return tuple(leaves), self._opencl_map_reduce_source(
            node, group, leaves, self.backend.work_group_size
        )

    def run_fused(self, node, payload, arguments, operands):
        data = self.backend.empty(node.shape, node.dtype)
        self.backend._run(payload, prod(node.shape), *(a._data for a in operands), data)
        return data

    def run_map_reduce(self, node, payload, arguments, operands):
        code, count, chunks, launch_count, local_size = payload
        data = self.backend.empty((count, chunks), node.dtype)
        self.backend._run(
            code, launch_count, *(a._data for a in operands), data, local_size=local_size
        )
        if chunks > 1:
            partial = ndarray._wrap(data, (count, chunks), node.dtype, self.backend)
            options = {"operation": node.attrs["operation"]} if "operation" in node.attrs else {}
            data = self.backend.reduce(partial, (1,), False, node.dtype, **options)
        return data

    @staticmethod
    def _opencl_expression(root, group, leaves):
        from ._opencl import _EXPRS, _ctype, _offset

        names, params, lines = {}, [], []
        for i, leaf in enumerate(leaves):
            names[id(leaf)] = f"v{i}"
            ctype = _ctype(leaf.dtype)
            if leaf.op == "scalar":
                params.append(f"{ctype} x{i}")
                read = f"x{i}"
            else:
                params.append(f"__global const {ctype} *x{i}")
                read = f"x{i}[{_offset(root.shape, leaf.shape)}]"
            lines.append(f"{ctype} v{i} = {read};")
        for i, node in enumerate(group, len(leaves)):
            names[id(node)] = f"v{i}"
            dtype = _ctype(node.dtype)
            lines.append(f"{dtype} v{i}; {{")
            for label, child, dt in zip(("a", "b", "c"), node.inputs, node.resolved):
                value = names[id(child)]
                if dt.kind == "b":
                    value = f"({value} != 0)"
                lines.append(f"{_ctype(dt)} {label} = ({_ctype(dt)}){value};")
            expr = _EXPRS[node.op]
            if node.op == "isfinite" and node.resolved[0].kind in "biu":
                expr = "1"
            if node.op == "absolute" and node.resolved[0].kind == "f":
                expr = "fabs(a)"
            if node.dtype.kind == "b":
                expr = f"(({expr}) != 0)"
            lines.append(f"v{i} = ({dtype})({expr}); }}")
        return params, lines, names[id(root)]

    @staticmethod
    def _opencl_source(root, group, leaves):
        from ._opencl import _ctype

        params, lines, result = OpenCLLowering._opencl_expression(root, group, leaves)
        params.append(f"__global {_ctype(root.dtype)} *out")
        source = (
            "#pragma OPENCL FP_CONTRACT OFF\n"
            + f"__kernel void compute({', '.join(params)}) {{ long i=get_global_id(0);\n"
            + "\n".join(lines)
            + f"\nout[i]={result}; }}"
        )
        return source, tuple(leaves)

    @staticmethod
    def _opencl_map_reduce_source(node, group, leaves, work_group_size):
        """Map directly into typed accumulators, with bounded partial storage.

        A workgroup reduces up to 4096 mapped values. Larger reductions store
        one partial per group, never an array-sized mapped intermediate.
        Contiguous rows of at least 32 terms use cooperative lanes too: mapping
        a whole row in one lane would make adjacent lanes read strided memory.
        Both paths support arbitrary axes and broadcasting in the producer.
        """
        from ._opencl import _ctype, strides

        mapped = node.inputs[0]
        axes = node.attrs["axes"]
        out_shape = tuple(d for i, d in enumerate(mapped.shape) if i not in axes)
        reduced_shape = tuple(mapped.shape[i] for i in axes)
        reduction_size = prod(reduced_shape)
        count = prod(out_shape)
        ctype = _ctype(node.dtype)
        operation = node.attrs.get("operation", "sum")
        neutral = "0" if operation == "sum" else identity(operation, node.dtype, ctype, "opencl")
        params, lines, result = OpenCLLowering._opencl_expression(mapped, group, leaves)
        params.append(f"__global {ctype} *out")
        base_terms, reduction_terms = [], []
        oi = ri = 0
        for axis, (dim, stride) in enumerate(zip(mapped.shape, strides(mapped.shape))):
            if axis in axes:
                reduction_terms.append(
                    f"((r / {max(1, prod(reduced_shape[ri + 1 :]))}L) % {max(1, dim)}L) * {stride}L"
                )
                ri += 1
            else:
                base_terms.append(
                    f"((o / {max(1, prod(out_shape[oi + 1 :]))}L) % {max(1, dim)}L) * {stride}L"
                )
                oi += 1
        index = f"({' + '.join(base_terms) or '0'}) + ({' + '.join(reduction_terms) or '0'})"
        accumulate = (
            f"acc = acc || ({result} != 0);"
            if node.dtype.kind == "b"
            else f"acc += ({ctype}){result};"
        )
        if operation != "sum":
            accumulate = (
                "acc = "
                + reduction_combine(
                    operation, node.dtype, ctype, "opencl", "acc", f"({ctype}){result}"
                )
                + ";"
            )
        mapping = f"long i = {index};\n" + "\n".join(lines) + "\n" + accumulate
        contiguous_rows = axes == tuple(range(len(mapped.shape) - len(axes), len(mapped.shape)))
        cooperative = reduction_size > 256 or (contiguous_rows and reduction_size >= 32)
        work_group_size = min(work_group_size, 1 << max(0, reduction_size - 1).bit_length())
        if not cooperative:
            body = (
                f"long o = get_global_id(0); {ctype} acc = {neutral};\n"
                f"for (long r = 0; r < {reduction_size}L; ++r) {{ {mapping} }}\n"
                "out[o] = acc;"
            )
            chunks, local_size, launch_count = 1, None, count
        else:
            chunks = (reduction_size + 4095) // 4096
            combine = (
                "scratch[lane] || scratch[lane + step]"
                if node.dtype.kind == "b"
                else "scratch[lane] + scratch[lane + step]"
            )
            if operation != "sum":
                combine = reduction_combine(
                    operation, node.dtype, ctype, "opencl", "scratch[lane]", "scratch[lane + step]"
                )
            body = (
                f"long g = get_group_id(0), o = g / {chunks}L;\n"
                f"long chunk = g % {chunks}L; int lane = get_local_id(0);\n"
                f"{ctype} acc = {neutral}; __local {ctype} scratch[{work_group_size}];\n"
                f"for (long r = chunk*4096 + lane; r < min((chunk+1)*4096, {reduction_size}L); r += {work_group_size}) {{ {mapping} }}\n"
                "scratch[lane] = acc; barrier(CLK_LOCAL_MEM_FENCE);\n"
                f"for (int step = {work_group_size // 2}; step > 0; step >>= 1) {{\n"
                f"if (lane < step) scratch[lane] = {combine};\n"
                "barrier(CLK_LOCAL_MEM_FENCE); }\n"
                "if (lane == 0) out[g] = scratch[0];"
            )
            local_size, launch_count = work_group_size, count * chunks * work_group_size
        source = (
            "#pragma OPENCL FP_CONTRACT OFF\n"
            + f"__kernel void compute({', '.join(params)}) {{\n{body}\n}}"
        )
        return source, count, chunks, launch_count, local_size


class VulkanLowering(_Lowering):
    def can_residual_matvec(self, matrix, vector, rhs, node):
        return super().can_residual_matvec(matrix, vector, rhs, node) and not (
            matrix.shape[0] == 1 and matrix.shape[1] > 4096
        )

    def prepare_fused(self, node, group, leaves):
        return tuple(leaves), tuple(group)

    def prepare_map_reduce(self, node, group, leaves):
        return tuple(leaves), tuple(group)

    def run_fused(self, node, payload, arguments, operands):
        return self.backend.fused_elementwise(node, payload, arguments, operands)

    def run_map_reduce(self, node, payload, arguments, operands):
        return self.backend.fused_reduce(node, payload, arguments, operands)


def create_lowering(backend):
    """Ask the implementation once when preparing a plan; no registry by name.

    No factory means ordinary primitives. Factory failures propagate: they do
    not request fallback to a different backend or a less precise execution.
    """
    factory = getattr(backend, "create_lowering", None)
    return None if factory is None else factory()
