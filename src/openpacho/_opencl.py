"""Vendor-neutral OpenCL 1.2 kernels. Buffers stay resident across operations."""

from collections import OrderedDict
from functools import lru_cache
from math import prod
from threading import RLock

import numpy as np

from .backends import _opencl_devices, check_dtype
from .exceptions import BackendUnavailableError, UnsupportedDtypeError

_TYPES = {
    "bool": "uchar",
    "int32": "int",
    "int64": "long",
    "uint32": "uint",
    "uint64": "ulong",
    "float32": "float",
    "float64": "double",
}
_EXPRS = {
    "add": "a + b",
    "subtract": "a - b",
    "multiply": "a * b",
    "true_divide": "a / b",
    "divide": "a / b",
    "power": "pow(a, b)",
    "negative": "-a",
    "positive": "a",
    "absolute": "(a < 0 ? -a : a)",
    "sqrt": "sqrt(a)",
    "exp": "exp(a)",
    "log": "log(a)",
    "sin": "sin(a)",
    "cos": "cos(a)",
    "tanh": "tanh(a)",
    "equal": "a == b",
    "not_equal": "a != b",
    "less": "a < b",
    "less_equal": "a <= b",
    "greater": "a > b",
    "greater_equal": "a >= b",
    "isfinite": "isfinite(a)",
    "cast": "a",
    "where": "a != 0 ? b : c",
}


def _ctype(dtype):
    return _TYPES[np.dtype(dtype).name]


@lru_cache(maxsize=128)
def strides(shape):
    return tuple(prod(shape[i + 1 :]) for i in range(len(shape)))


@lru_cache(maxsize=128)
def _offset(out_shape, in_shape, variable="i", index_bits=64):
    padding = len(out_shape) - len(in_shape)
    suffix = "L" if index_bits == 64 else ""
    terms = []
    for axis, (dim, stride) in enumerate(zip(in_shape, strides(in_shape))):
        if dim != 1:
            oa = axis + padding
            terms.append(
                f"(({variable} / {max(1, prod(out_shape[oa + 1 :]))}{suffix}) % {max(1, dim)}{suffix}) * {stride}{suffix}"
            )
    return " + ".join(terms) or "0"


@lru_cache(maxsize=128)
def _elementwise_source(name, specs, shape, dtype):
    # Metadata only: values and buffers are rebound on every dispatch.
    bits = 32 if prod(shape) <= 2**31 - 1 else 64
    params, reads = [], []
    for j, ((arg_shape, arg_dtype, scalar), symbol) in enumerate(zip(specs, ("a", "b", "c"))):
        dt = _ctype(arg_dtype)
        params.append(f"{dt} x{j}" if scalar else f"__global const {dt} *x{j}")
        reads.append(
            f"{dt} {symbol} = x{j}"
            + (";" if scalar else f"[{_offset(shape, arg_shape, index_bits=bits)}];")
        )
    expr = _EXPRS[name]
    if name == "isfinite" and specs[0][1].kind in "biu":
        expr = "1"
    if name == "absolute" and specs[0][1].kind == "f":
        expr = "fabs(a)"
    if dtype.kind == "b":
        expr = f"(({expr}) != 0)"
    index_type = "int" if bits == 32 else "long"
    return f"""__kernel void compute({", ".join(params)}, __global {_ctype(dtype)} *out) {{
        {index_type} i = get_global_id(0);
        {" ".join(reads)}
        out[i] = ({_ctype(dtype)})({expr});
    }}"""


@lru_cache(maxsize=128)
def _gather_source(shape, offset, source_strides, dtype):
    # Bound every intermediate, including negative-stride gathers, before
    # narrowing address arithmetic. Storage dtype is entirely independent.
    bound = abs(offset) + sum((max(1, d) - 1) * abs(s) for d, s in zip(shape, source_strides))
    narrow = max(prod(shape), bound) <= 2**31 - 1
    suffix, index_type = ("", "int") if narrow else ("L", "long")
    terms = [
        f"((i / {max(1, s)}{suffix}) % {max(1, d)}{suffix}) * ({t}{suffix})"
        for s, d, t in zip(strides(shape), shape, source_strides)
    ]
    index = " + ".join([f"{offset}{suffix}", *terms])
    return f"""__kernel void compute(__global const {_ctype(dtype)} *a,
                   __global {_ctype(dtype)} *out) {{
        {index_type} i = get_global_id(0); out[i] = a[{index}];
    }}"""


class OpenCLBackend:
    # Small-row, long-K region only; the portable direct tile remains the fallback.
    _small_row_gemm_config = ((16, 32, 16), 4081)

    name = "opencl"

    def __init__(self, device=None, *, blas="custom", clblast_library=None):
        if blas not in ("custom", "clblast"):
            raise ValueError("blas must be 'custom' or 'clblast'")
        if clblast_library is not None and blas != "clblast":
            raise ValueError("clblast_library requires blas='clblast'")
        self.cl, found = _opencl_devices()
        if device is None:
            choices = [d for _, _, d in found if d.type & self.cl.device_type.GPU]
            if not choices:
                raise BackendUnavailableError(
                    "No OpenCL GPU found. Select a CPU OpenCL device explicitly by ID."
                )
            selected = choices[0]
        else:
            choices = [d for pi, di, d in found if f"{pi}:{di}" == device]
            if not choices:
                raise BackendUnavailableError(
                    f"OpenCL device {device!r} not found; call devices() for IDs"
                )
            selected = choices[0]
        self.device = selected.name.strip()
        self.fp64 = bool(selected.double_fp_config)
        limit = min(
            128,
            selected.max_work_group_size,
            selected.max_work_item_sizes[0],
            selected.local_mem_size // 8,
        )
        self.work_group_size = 1 << (int(limit).bit_length() - 1)
        self._register_gemm = limit >= 64 and selected.local_mem_size >= 4096
        self._max_allocation_size = int(selected.max_mem_alloc_size)
        gemm_limit = min(selected.max_work_group_size, selected.max_work_item_sizes[0])
        self._gemm_tile = (
            16
            if gemm_limit >= 256 and selected.local_mem_size >= 4096
            else 8
            if gemm_limit >= 64 and selected.local_mem_size >= 1024
            else 0
        )
        try:
            self.context = self.cl.Context([selected])
            self.queue = self.cl.CommandQueue(self.context)
        except self.cl.Error as exc:
            raise BackendUnavailableError(
                f"Cannot initialize OpenCL device {self.device}: {exc}"
            ) from exc
        self._kernels = OrderedDict()
        self._lock = RLock()
        from pyopencl.tools import ImmediateAllocator, MemoryPool

        self._pool = MemoryPool(ImmediateAllocator(self.queue))
        self.blas = blas
        self._blas = None
        if blas == "clblast":
            from ._clblast import CLBlast

            self._blas = CLBlast(clblast_library)

    def validate_dtype(self, dtype):
        dtype = check_dtype(dtype)
        if dtype == np.dtype("float64") and not self.fp64:
            raise UnsupportedDtypeError(
                f"{self.device} does not support float64; choose dtype=float32 explicitly"
            )
        return dtype

    def empty(self, shape, dtype):
        self.validate_dtype(dtype)
        with self._lock:
            return self._pool.allocate(max(1, prod(shape) * np.dtype(dtype).itemsize))

    def release_cached_memory(self):
        """Release unused pooled buffers; live arrays and queued work remain valid."""
        with self._lock:
            self.queue.finish()
            self._pool.free_held()

    def upload(self, a):
        a = np.asarray(a)
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        self.validate_dtype(a.dtype)
        if not a.size:
            return self.empty(a.shape, a.dtype)
        out = self.empty(a.shape, a.dtype)
        self.cl.enqueue_copy(self.queue, out, a, is_blocking=True)
        return out

    def download(self, a):
        out = np.empty(a.shape, a.dtype)
        if out.size:
            self.cl.enqueue_copy(self.queue, out, a._data, is_blocking=True)
        return out

    def _run(self, source, count, *args, local_size=None):
        if not count:
            return
        # Kernel argument mutation is not thread safe; serialize enqueue, not execution.
        with self._lock:
            if source not in self._kernels:
                prefix = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" if self.fp64 else ""
                program = self.cl.Program(self.context, prefix + source).build(
                    options=["-cl-std=CL1.2"]
                )
                self._kernels[source] = self.cl.Kernel(program, "compute")
                if len(self._kernels) > 128:
                    self._kernels.popitem(last=False)
            self._kernels.move_to_end(source)
            return self._kernels[source](
                self.queue, (count,), (local_size,) if local_size else None, *args
            )

    def random_distributions(self, dtype):
        return ("random", "uniform", "normal", "exponential")

    def create_lowering(self):
        from ._lowering import OpenCLLowering

        return OpenCLLowering(self)

    def prepare_padded_matmul(self, ashape, bshape, shape, dtype):
        if self._blas is not None or not self._register_gemm:
            return None
        from ._padding import prepare_register_padding
        from ._padding_copy import opencl_copy

        return prepare_register_padding(
            self,
            ashape,
            bshape,
            shape,
            dtype,
            opencl_copy,
            max_buffer_bytes=self._max_allocation_size,
        )

    def prepare_layout_matmul(
        self, ashape, bshape, shape, dtype, *, transpose_a=False, transpose_b=False
    ):
        """Resolve optional BLAS layout support once, during specialization."""
        if self._blas is None or np.dtype(dtype) not in (np.dtype("f4"), np.dtype("f8")):
            return None
        if len(ashape) != 2 or len(bshape) != 2 or not (transpose_a or transpose_b):
            return None
        left = ashape[::-1] if transpose_a else ashape
        right = bshape[::-1] if transpose_b else bshape
        if not all((*left, *right)):
            return None
        if left[1] != right[0] or tuple(shape) != (left[0], right[1]):
            return None
        from ._matmul_layout import TransposedGemm

        return TransposedGemm(
            self,
            tuple(ashape),
            tuple(bshape),
            tuple(shape),
            np.dtype(dtype),
            transpose_a,
            transpose_b,
        )

    def random_data(self, *args):
        from ._random_opencl import random_data

        return random_data(self, *args)

    def full(self, shape, value, dtype):
        out = self.empty(shape, dtype)
        scalar = np.asarray(value, dtype=dtype).reshape(())
        # enqueue_fill_buffer handles non-finite floating point patterns correctly.
        if prod(shape):
            self.cl.enqueue_fill_buffer(self.queue, out, scalar, 0, prod(shape) * scalar.itemsize)
        return out

    def elementwise(self, name, args, shape, dtype):
        out = self.empty(shape, dtype)
        specs = tuple((a.shape, a.dtype, getattr(a, "_host_scalar", False)) for a in args)
        source = _elementwise_source(name, specs, shape, np.dtype(dtype))
        self._run(source, prod(shape), *(a._data for a in args), out)
        return out

    def gather(self, a, shape, offset, source_strides):
        out = self.empty(shape, a.dtype)
        source = _gather_source(shape, offset, source_strides, a.dtype)
        self._run(source, prod(shape), a._data, out)
        return out

    def reduce(self, a, axes, keepdims, dtype, *, operation="sum"):
        from ._reductions import combine, identity

        neutral = "0" if operation == "sum" else identity(operation, dtype, _ctype(dtype), "opencl")
        out_shape = tuple(d for i, d in enumerate(a.shape) if i not in axes)
        reduced_shape = tuple(a.shape[i] for i in axes)
        out = self.empty(out_shape, dtype)
        base_terms, reduction_terms = [], []
        oi = ri = 0
        for axis, (dim, stride) in enumerate(zip(a.shape, strides(a.shape))):
            if axis in axes:
                # axes is canonical ascending order.
                reduction_terms.append(
                    f"((r / {max(1, prod(reduced_shape[ri + 1 :]))}L) % {max(1, dim)}L) * {stride}L"
                )
                ri += 1
            else:
                base_terms.append(
                    f"((i / {max(1, prod(out_shape[oi + 1 :]))}L) % {max(1, dim)}L) * {stride}L"
                )
                oi += 1
        base = " + ".join(base_terms) or "0"
        index = " + ".join(reduction_terms) or "0"

        def accumulate(value):
            if operation != "sum":
                expression = combine(operation, dtype, _ctype(dtype), "opencl", "acc", value)
                return f"acc = {expression};"
            if np.dtype(dtype).kind == "b":
                return f"acc = acc || ({value} != 0);"
            return f"acc += ({_ctype(dtype)}){value};"

        # Large total reductions use independent chunks then recursively reduce partials.
        if len(axes) == a.ndim and a.size > 256:
            count = (a.size + 255) // 256
            partial = self.empty((count,), dtype)
            source = f"""__kernel void compute(__global const {_ctype(a.dtype)} *a,
                __global {_ctype(dtype)} *out) {{
                long i = get_global_id(0); {_ctype(dtype)} acc = {neutral};
                for (long j = i*256; j < min((i+1)*256, {a.size}L); ++j)
                    {accumulate("a[j]")}
                out[i] = acc;
            }}"""
            self._run(source, count, a._data, partial)
            from ._array import ndarray

            temporary = ndarray._wrap(partial, (count,), dtype, self)
            options = {} if operation == "sum" else {"operation": operation}
            return self.reduce(temporary, (0,), False, dtype, **options)
        source = f"""__kernel void compute(__global const {_ctype(a.dtype)} *a,
                      __global {_ctype(dtype)} *out) {{
            long i = get_global_id(0); {_ctype(dtype)} acc = {neutral};
            for (long r = 0; r < {prod(reduced_shape)}L; ++r)
                {accumulate(f"a[({base}) + ({index})]")}
            out[i] = acc;
        }}"""
        self._run(source, prod(out_shape), a._data, out)
        return out

    def matmul(self, a, b, shape, dtype):
        if not prod(shape) or not a.shape[-1]:
            return self.full(shape, 0, dtype)
        if (
            self._register_gemm
            and self._blas is None
            and np.dtype(dtype) == np.dtype("float32")
            and a.ndim > 1
            and b.ndim > 1
            and min(a.shape[-2], b.shape[-1]) >= 32
            and a.shape[-1] >= 16
        ):
            from ._gemm import register_gemm_data

            return register_gemm_data(self, a, b, shape)
        out = self.empty(shape, dtype)
        if self._blas is not None and np.dtype(dtype).kind == "f":
            self._blas.matmul(self.queue, a, b, out, shape, dtype)
            return out
        k = a.shape[-1]
        n = b.shape[-1] if b.ndim > 1 else 1
        m = a.shape[-2] if a.ndim > 1 else 1
        batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        aoffset = f"({_offset(batch, a.shape[:-2], 'batch')})*{m * k}L"
        boffset = f"({_offset(batch, b.shape[:-2], 'batch')})*{k * n}L"
        # Coalesced row reductions, one workgroup per vector result.
        if b.ndim == 1 and k >= 64:
            lanes = self.work_group_size
            # Short DOT needs one launch; a second partial-reduction launch costs
            # more than the tiny vector's work. Large DOT still exposes many groups.
            chunks = (k + 1023) // 1024 if a.ndim == 1 and k > 4096 else 1
            groups = prod(shape) * chunks
            partial = self.empty((groups,), dtype) if chunks > 1 else out
            start = "(row * 1024 + lane)" if chunks > 1 else "lane"
            stop = f"min((row+1)*1024L, {k}L)" if chunks > 1 else f"{k}L"
            read_a = "a[j]" if a.ndim == 1 else f"a[({aoffset}) + (row % {m}L)*{k}L+j]"
            source = f"""#pragma OPENCL FP_CONTRACT OFF
            __kernel void compute(__global const {_ctype(dtype)} *a,
                __global const {_ctype(dtype)} *b, __global {_ctype(dtype)} *out) {{
                long row=get_group_id(0), lane=get_local_id(0), batch=row/{m}L;
                __local {_ctype(dtype)} scratch[{lanes}];
                {_ctype(dtype)} acc=0;
                for(long j={start}; j<{stop}; j+={lanes}) acc += {read_a}*b[j];
                scratch[lane]=acc; barrier(CLK_LOCAL_MEM_FENCE);
                for(int s={lanes // 2}; s>0; s/=2) {{
                    if(lane<s) scratch[lane]+=scratch[lane+s]; barrier(CLK_LOCAL_MEM_FENCE);
                }}
                if(lane==0) out[row]=scratch[0];
            }}"""
            self._run(source, groups * lanes, a._data, b._data, partial, local_size=lanes)
            if chunks > 1:
                from ._array import ndarray

                return self.reduce(
                    ndarray._wrap(partial, (groups,), dtype, self), (0,), False, dtype
                )
            return out
        if self._gemm_tile and a.ndim > 1 and b.ndim > 1 and m >= 8 and n >= 8 and k >= 8:
            tile = self._gemm_tile
            tiles_r, tiles_c = (m + tile - 1) // tile, (n + tile - 1) // tile
            ctype = _ctype(dtype)
            source = f"""__kernel void compute(__global const {ctype} *a,
                __global const {ctype} *b, __global {ctype} *out) {{
                long g=get_group_id(0), batch=g/{tiles_r * tiles_c}L;
                int lane=get_local_id(0), ly=lane/{tile}, lx=lane%{tile};
                long row=((g/{tiles_c}L)%{tiles_r}L)*{tile}+ly, col=(g%{tiles_c}L)*{tile}+lx;
                __local {ctype} left[{tile * tile}], right[{tile * tile}]; {ctype} acc=0;
                for(long start=0;start<{k}L;start+={tile}) {{
                    left[lane]=(row<{m}L && start+lx<{k}L)?a[({aoffset})+row*{k}L+start+lx]:0;
                    right[lane]=(start+ly<{k}L && col<{n}L)?b[({boffset})+(start+ly)*{n}L+col]:0;
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for(int j=0;j<{tile};++j) acc+=left[ly*{tile}+j]*right[j*{tile}+lx];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }}
                if(row<{m}L && col<{n}L) out[(batch*{m}L+row)*{n}L+col]=acc;
            }}"""
            self._run(
                source,
                prod(batch) * tiles_r * tiles_c * tile * tile,
                a._data,
                b._data,
                out,
                local_size=tile * tile,
            )
            return out
        source = f"""__kernel void compute(__global const {_ctype(dtype)} *a,
                     __global const {_ctype(dtype)} *b, __global {_ctype(dtype)} *out) {{
            long i = get_global_id(0), row = (i / {n}L) % {m}L, col = i % {n}L, batch=i/{m * n}L;
            {_ctype(dtype)} acc = 0;
            for (long j = 0; j < {k}L; ++j) acc += a[({aoffset})+row*{k}L+j]*b[({boffset})+j*{n}L+col];
            out[i] = acc;
        }}"""
        self._run(source, prod(shape), a._data, b._data, out)
        return out

    def synchronize(self):
        self.queue.finish()

    def norm(self, a, axes, keepdims, dtype):
        from ._norm import norm_data

        return norm_data(self, a, axes, keepdims, dtype)

    def residual_matvec(self, a, x, b):
        from ._gemv import residual_matvec_data

        return residual_matvec_data(self, a, x, b)
