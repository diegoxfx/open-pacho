"""Initial native CUDA backend: Driver API, NVRTC and optional cuBLAS.

No CuPy objects, implicit CPU execution or CUDA graphs. The
primary context is retained, never reset, and every call restores the caller's
current context. Only completed allocations enter the bounded buffer pool.
"""

from collections import OrderedDict
from contextlib import contextmanager
from math import prod
from threading import RLock
from weakref import WeakSet

import numpy as np

from ._cuda_arguments import KernelArguments
from .backends import DeviceInfo, check_dtype
from .exceptions import BackendUnavailableError, UnsupportedDtypeError

_TYPES = {"bool": "unsigned char", "float32": "float", "float64": "double"}
_EXPRESSIONS = {
    "add": "a+b",
    "subtract": "a-b",
    "multiply": "a*b",
    "true_divide": "a/b",
    "divide": "a/b",
    "power": "pow(a,b)",
    "negative": "-a",
    "positive": "a",
    "absolute": "fabs(a)",
    "sqrt": "sqrt(a)",
    "exp": "exp(a)",
    "log": "log(a)",
    "sin": "sin(a)",
    "cos": "cos(a)",
    "tanh": "tanh(a)",
    "equal": "a==b",
    "not_equal": "a!=b",
    "less": "a<b",
    "less_equal": "a<=b",
    "greater": "a>b",
    "greater_equal": "a>=b",
    "isfinite": "isfinite(a)",
    "cast": "a",
    "where": "a != 0 ? b : c",
}


def _check(result):
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA call failed: {getattr(status, 'name', status)} ({int(status)})")
    return values[0] if len(values) == 1 else tuple(values) if values else None


def _driver():
    try:
        from cuda.bindings import driver

        _check(driver.cuInit(0))
        return driver
    except Exception as exc:
        raise BackendUnavailableError(
            f'CUDA unavailable; install an NVIDIA driver and "open-pacho[cuda]": {exc}'
        ) from exc


def cuda_devices():
    driver = _driver()
    try:
        version = str(_check(driver.cuDriverGetVersion()))
        result = []
        for i in range(_check(driver.cuDeviceGetCount())):
            device = _check(driver.cuDeviceGet(i))
            name = _check(driver.cuDeviceGetName(256, device)).split(b"\0", 1)[0].decode()
            major = _check(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
                )
            )
            result.append(
                DeviceInfo(
                    str(i),
                    name,
                    "NVIDIA",
                    "GPU",
                    major >= 2,
                    int(_check(driver.cuDeviceTotalMem(device))),
                    version,
                )
            )
        return result
    except Exception as exc:
        raise BackendUnavailableError(f"CUDA device discovery failed: {exc}") from exc


def _type(dtype):
    return _TYPES[np.dtype(dtype).name]


def _offset(shape, source_shape, variable="i"):
    padding = len(shape) - len(source_shape)
    return (
        " + ".join(
            f"(({variable}/{max(1, prod(shape[axis + padding + 1 :]))}LL)%{max(1, dim)}LL)"
            f"*{prod(source_shape[axis + 1 :])}LL"
            for axis, dim in enumerate(source_shape)
            if dim != 1
        )
        or "0"
    )


class _Allocation:
    def __init__(self, backend, ptr, capacity):
        self.backend, self.ptr, self.capacity = backend, int(ptr), capacity
        backend._leases.add(self)

    def __del__(self):
        try:
            with self.backend._lock:
                if self.ptr:
                    ptr, self.ptr = self.ptr, 0
                    self.backend._recycle(ptr, self.capacity)
        except Exception:  # noqa: BLE001, S110 - interpreter shutdown only
            pass


class CUDABackend:
    _small_row_gemm_config = ((16, 16, 32), 2048)

    name = "cuda"
    _activation_depth = 0
    _cublas_pointer_mode_dirty = False
    # One fused step usually cannot amortize entering another host context scope.
    execution_scope_min_steps = 2

    def __init__(self, device=None, *, blas="cublas"):
        if blas not in ("custom", "cublas"):
            raise ValueError("CUDA blas must be 'custom' or 'cublas'")
        found = cuda_devices()
        identifier = "0" if device is None else str(device)
        choices = [d for d in found if d.id == identifier]
        if not choices:
            raise BackendUnavailableError(f"CUDA device {device!r} not found; call devices('cuda')")
        self.device, self.fp64, self.blas = choices[0].name, choices[0].fp64, blas
        self.driver = _driver()
        self._lock, self._leases = RLock(), WeakSet()
        self._context = self._stream = self._handle = None
        self._closed = False
        self._pending, self._kernels, self._pool = [], OrderedDict(), OrderedDict()
        self._submitted = self._pool_items = self._cached_bytes = 0
        self._pool_limit = 128 * 1024**2
        self._device = _check(self.driver.cuDeviceGet(int(identifier)))
        try:
            from cuda.bindings import nvrtc

            self.nvrtc = nvrtc
            _check(nvrtc.nvrtcVersion())
            self._context = _check(self.driver.cuDevicePrimaryCtxRetain(self._device))
            with self._activate():
                self._stream = _check(
                    self.driver.cuStreamCreate(self.driver.CUstream_flags.CU_STREAM_NON_BLOCKING)
                )
                attrs = self.driver.CUdevice_attribute
                self._register_gemm = (
                    _check(
                        self.driver.cuDeviceGetAttribute(
                            attrs.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self._device
                        )
                    )
                    >= 64
                    and _check(
                        self.driver.cuDeviceGetAttribute(
                            attrs.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, self._device
                        )
                    )
                    >= 4096
                )
                self._architecture = tuple(
                    _check(self.driver.cuDeviceGetAttribute(a, self._device))
                    for a in (
                        attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    )
                )
                if blas == "cublas":
                    from nvmath.bindings import cublas

                    self._cublas = cublas
                    self._handle = cublas.create()
                    cublas.set_stream(self._handle, int(self._stream))
                    cublas.set_pointer_mode(self._handle, cublas.PointerMode.HOST)
                    cublas.set_math_mode(self._handle, cublas.Math.PEDANTIC_MATH)
        except Exception as exc:
            self.close()
            raise BackendUnavailableError(
                f'CUDA initialization failed; check "open-pacho[cuda]" and CUDA libraries: {exc}'
            ) from exc

    @contextmanager
    def _activate(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("CUDA backend is closed")
            # The outer frame always owns a push/pop. Nested operations may
            # share it, but query the driver: another library/backend can have
            # temporarily changed this thread's current context.
            pushed = not self._activation_depth or (
                int(_check(self.driver.cuCtxGetCurrent())) != int(self._context)
            )
            if pushed:
                _check(self.driver.cuCtxPushCurrent(self._context))
            self._activation_depth += 1
            try:
                yield
            finally:
                self._activation_depth -= 1
                if pushed:
                    _check(self.driver.cuCtxPopCurrent())

    def execution_scope(self):
        """Hold the context/lock for one prepared plan, without an extra wait."""
        return self._activate()

    def validate_dtype(self, dtype):
        dtype = check_dtype(dtype)
        if dtype.name not in _TYPES or dtype == np.dtype("float64") and not self.fp64:
            raise UnsupportedDtypeError("Initial CUDA backend supports bool, float32 and float64")
        return dtype

    def _synchronize(self):
        _check(self.driver.cuStreamSynchronize(self._stream))
        self._submitted = 0
        self._pending.clear()

    def synchronize(self):
        with self._activate():
            self._synchronize()

    def empty(self, shape, dtype):
        dtype = self.validate_dtype(dtype)
        if prod(shape) > 2**31 - 1:
            raise ValueError("Array exceeds initial CUDA indexing limits")
        capacity = max(1, prod(shape) * dtype.itemsize)
        with self._activate():
            entries = self._pool.get(capacity)
            if entries:
                ptr = entries.pop()
                if not entries:
                    del self._pool[capacity]
                self._cached_bytes -= capacity
                self._pool_items -= 1
            else:
                ptr = _check(self.driver.cuMemAlloc(capacity))
            return _Allocation(self, ptr, capacity)

    def _recycle(self, ptr, capacity):
        if self._closed:
            return
        with self._activate():
            if capacity > self._pool_limit:
                _check(self.driver.cuMemFree(ptr))
                return
            while self._pool and (
                self._pool_items >= 128 or self._cached_bytes + capacity > self._pool_limit
            ):
                old = next(iter(self._pool))
                _check(self.driver.cuMemFree(self._pool[old][-1]))
                self._pool[old].pop()
                if not self._pool[old]:
                    del self._pool[old]
                self._pool_items -= 1
                self._cached_bytes -= old
            self._pool.setdefault(capacity, []).append(ptr)
            self._pool.move_to_end(capacity)
            self._pool_items += 1
            self._cached_bytes += capacity

    def release_cached_memory(self):
        with self._activate():
            self._synchronize()
            while self._pool:
                capacity = next(iter(self._pool))
                entries = self._pool[capacity]
                _check(self.driver.cuMemFree(entries[-1]))
                entries.pop()
                if not entries:
                    del self._pool[capacity]
                self._cached_bytes -= capacity
                self._pool_items -= 1

    def upload(self, a):
        a = np.ascontiguousarray(a) if np.ndim(a) else np.asarray(a)
        with self._activate():
            out = self.empty(a.shape, a.dtype)
            if a.size:
                _check(self.driver.cuMemcpyHtoD(out.ptr, a.ctypes.data, a.nbytes))
                # Pageable HtoD may return after host staging, before device DMA.
                # Our nonblocking compute stream does not inherit that default
                # stream dependency. Complete it before publishing/recycling out.
                _check(self.driver.cuStreamSynchronize(0))
            return out

    def download(self, a):
        with self._activate():
            self._synchronize()
            out = np.empty(a.shape, a.dtype)
            if out.size:
                _check(self.driver.cuMemcpyDtoH(out.ctypes.data, a._data.ptr, out.nbytes))
            return out

    def _compile(self, source):
        if source in self._kernels:
            self._kernels.move_to_end(source)
            return self._kernels[source][1]
        nvrtc = self.nvrtc
        program = _check(nvrtc.nvrtcCreateProgram(source.encode(), b"openpacho.cu", 0, [], []))
        try:
            major, minor = self._architecture
            options = [
                f"--gpu-architecture=compute_{major}{minor}".encode(),
                b"--std=c++11",
                b"--fmad=false",
                b"--ftz=false",
                b"--prec-div=true",
                b"--prec-sqrt=true",
            ]
            (status,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
            if int(status):
                log = bytearray(_check(nvrtc.nvrtcGetProgramLogSize(program)))
                _check(nvrtc.nvrtcGetProgramLog(program, log))
                raise RuntimeError(f"CUDA compilation failed: {log.decode(errors='replace')}")
            ptx = bytearray(_check(nvrtc.nvrtcGetPTXSize(program)))
            _check(nvrtc.nvrtcGetPTX(program, ptx))
        finally:
            _check(nvrtc.nvrtcDestroyProgram(program))
        if len(self._kernels) >= 128:
            # Retire first: a failed wait cannot orphan a newly loaded module.
            self._synchronize()
            old_source = next(iter(self._kernels))
            _check(self.driver.cuModuleUnload(self._kernels[old_source][0]))
            del self._kernels[old_source]
        module = _check(self.driver.cuModuleLoadData(bytes(ptx)))
        try:
            function = _check(self.driver.cuModuleGetFunction(module, b"compute"))
        except Exception:
            _check(self.driver.cuModuleUnload(module))
            raise
        self._kernels[source] = (module, function, None)
        return function

    def _retain(self, args):
        self._pending.extend(a for a in args if isinstance(a, _Allocation))
        self._submitted += 1
        if self._submitted >= 64:
            self._synchronize()

    def _run(self, source, blocks, *args, threads=128):
        if not blocks:
            return
        with self._activate():
            function = self._compile(source)
            module, _, parameters = self._kernels[source]
            if parameters is None:
                parameters = KernelArguments(args, _Allocation)
                self._kernels[source] = (module, function, parameters)
            pointers = parameters.bind(args)
            _check(
                self.driver.cuLaunchKernel(
                    function, blocks, 1, 1, threads, 1, 1, 0, self._stream, pointers, 0
                )
            )
            self._retain(args)

    def _element_kernel(self, params, body, count, *args):
        source = f"""extern "C" __global__ void compute({params}) {{
            long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
            if(i>={count}LL) return; {body}
        }}"""
        self._run(source, (count + 127) // 128, *args)

    def random_distributions(self, dtype):
        return ("random", "uniform", "normal", "exponential")

    def create_lowering(self):
        from ._cuda_lowering import CUDALowering

        return CUDALowering(self)

    def prepare_padded_matmul(self, ashape, bshape, shape, dtype):
        if self.blas != "custom" or not self._register_gemm:
            return None
        from ._padding import prepare_register_padding
        from ._padding_copy import cuda_copy

        return prepare_register_padding(self, ashape, bshape, shape, dtype, cuda_copy)

    def random_data(self, *args):
        from ._random_cuda import random_data

        return random_data(self, *args)

    def full(self, shape, value, dtype):
        out = self.empty(shape, dtype)
        dt = _type(dtype)
        self._element_kernel(
            f"{dt} value, {dt}*out",
            "out[i]=value;",
            prod(shape),
            np.asarray(value, dtype=dtype)[()],
            out,
        )
        return out

    def elementwise(self, name, args, shape, dtype):
        out = self.empty(shape, dtype)
        params, reads = [], []
        for j, (a, symbol) in enumerate(zip(args, ("a", "b", "c"))):
            dt = _type(a.dtype)
            scalar = getattr(a, "_host_scalar", False)
            params.append(f"{dt} x{j}" if scalar else f"const {dt}*x{j}")
            reads.append(
                f"{dt} {symbol}=x{j}" + (";" if scalar else f"[{_offset(shape, a.shape)}];")
            )
        expr = _EXPRESSIONS[name]
        if name == "isfinite" and args[0].dtype.kind == "b":
            expr = "1"
        if np.dtype(dtype).kind == "b":
            expr = f"(({expr})!=0)"
        params.append(f"{_type(dtype)}*out")
        body = "\n".join(reads) + f"out[i]=({_type(dtype)})({expr});"
        self._element_kernel(",".join(params), body, prod(shape), *(a._data for a in args), out)
        return out

    def gather(self, a, shape, offset, strides):
        out = self.empty(shape, a.dtype)
        terms = [
            f"((i/{max(1, prod(shape[j + 1 :]))}LL)%{max(1, d)}LL)*({s}LL)"
            for j, (d, s) in enumerate(zip(shape, strides))
        ]
        index = "+".join([f"{offset}LL", *terms])
        dt = _type(a.dtype)
        self._element_kernel(
            f"const {dt}*a,{dt}*out", f"out[i]=a[{index}];", prod(shape), a._data, out
        )
        return out

    def reduce(self, a, axes, keepdims, dtype, *, operation="sum"):
        from ._array import ndarray
        from ._norm import _index
        from ._reductions import combine, identity

        dtype = self.validate_dtype(dtype)
        if dtype.kind == "b" and operation == "sum":
            raise NotImplementedError("Boolean reduction is not implemented in initial CUDA")
        index, count, reduced = _index(a.shape, axes)
        if not reduced:
            return self.full((count,), 0, dtype)
        chunks = (reduced + 4095) // 4096
        out = self.empty((count, chunks), dtype)
        dt = _type(dtype)
        neutral = "0" if operation == "sum" else identity(operation, dtype, dt, "cuda")
        accumulate = (
            f"acc+=({dt})x0[{index}];"
            if operation == "sum"
            else f"acc={combine(operation, dtype, dt, 'cuda', 'acc', f'x0[{index}]')};"
        )
        merge = (
            "sums[lane]+=sums[lane+step];"
            if operation == "sum"
            else f"sums[lane]={combine(operation, dtype, dt, 'cuda', 'sums[lane]', 'sums[lane+step]')};"
        )
        # One block per output/chunk, a common tree for both floating precisions.
        source = f"""extern "C" __global__ void compute(const {_type(a.dtype)}*x0,{dt}*out) {{
            int i=blockIdx.x, lane=threadIdx.x, o=i/{chunks}, chunk=i%{chunks};
            __shared__ {dt} sums[128]; {dt} acc={neutral};
            for(int r=chunk*4096+lane;r<min((chunk+1)*4096,{reduced});r+=128)
                {accumulate}
            sums[lane]=acc; __syncthreads();
            for(int step=64;step>0;step/=2) {{
                if(lane<step) {merge} __syncthreads();
            }}
            if(lane==0) out[i]=sums[0];
        }}"""
        self._run(source, count * chunks, a._data, out)
        if chunks == 1:
            return out
        partial = ndarray._wrap(out, (count, chunks), dtype, self)
        options = {} if operation == "sum" else {"operation": operation}
        return self.reduce(partial, (1,), False, dtype, **options)

    def matmul(self, a, b, shape, dtype):
        if np.dtype(dtype).kind != "f":
            raise NotImplementedError("Initial CUDA matmul requires float32 or float64")
        m, k = (1, a.shape[0]) if a.ndim == 1 else a.shape[-2:]
        n = b.shape[-1] if b.ndim > 1 else 1
        a_batch, b_batch = a.shape[:-2], b.shape[:-2]
        batch = np.broadcast_shapes(a_batch, b_batch) if a_batch or b_batch else ()
        if not k:
            return self.full(shape, 0, dtype)
        if self.blas == "custom" and b.ndim == 1 and k >= 256:
            from ._cuda_gemv import matvec_data

            return matvec_data(self, a, b, shape, dtype)
        if (
            self.blas == "custom"
            and self._register_gemm
            and np.dtype(dtype) == np.dtype("float32")
            and a.ndim > 1
            and b.ndim > 1
            and min(m, n) >= 32
            and k >= 16
            and max(a.size, b.size, prod(shape)) <= 2**31 - 1
        ):
            from ._gemm import register_gemm_data

            return register_gemm_data(self, a, b, shape)
        out = self.empty(shape, dtype)
        if not prod(shape):
            return out
        if self.blas == "cublas":
            with self._activate():
                cublas = self._cublas
                if self._cublas_pointer_mode_dirty:
                    # A previous failed DOT mode restoration must not expose
                    # host scalar addresses to a handle still in DEVICE mode.
                    cublas.set_pointer_mode(self._handle, cublas.PointerMode.HOST)
                    self._cublas_pointer_mode_dirty = False
                if max(m, n, k, prod(batch)) > 2**31 - 1:
                    raise ValueError("CUDA cuBLAS matmul dimensions exceed the 32-bit API")
                # Native DOT and the simple contiguous GEMV batch loop help
                # measured vector workloads. Unbatched GEMV keeps GEMM: the
                # native alternative regressed the tested shapes/precisions.
                if a.ndim == b.ndim == 1 or (batch and (a.ndim == 1 or b.ndim == 1)):
                    from ._cuda_blas import vector_matmul

                    vector_matmul(self, a, b, out, dtype)
                    return out
                alpha, beta = np.array(1, dtype=dtype), np.array(0, dtype=dtype)
                function = cublas.sgemm if np.dtype(dtype) == np.dtype("float32") else cublas.dgemm
                # Row-major C=A@B is column-major C.T=B.T@A.T, with no copies.
                if not batch:
                    function(
                        self._handle,
                        cublas.Operation.N,
                        cublas.Operation.N,
                        n,
                        m,
                        k,
                        alpha.ctypes.data,
                        b._data.ptr,
                        n,
                        a._data.ptr,
                        k,
                        beta.ctypes.data,
                        out.ptr,
                        n,
                    )
                    self._retain((a._data, b._data, out))
                elif prod(batch) > 1 and a_batch == b_batch == batch:
                    function = (
                        cublas.sgemm_strided_batched
                        if np.dtype(dtype) == np.dtype("float32")
                        else cublas.dgemm_strided_batched
                    )
                    function(
                        self._handle,
                        cublas.Operation.N,
                        cublas.Operation.N,
                        n,
                        m,
                        k,
                        alpha.ctypes.data,
                        b._data.ptr,
                        n,
                        k * n,
                        a._data.ptr,
                        k,
                        m * k,
                        beta.ctypes.data,
                        out.ptr,
                        n,
                        m * n,
                        prod(batch),
                    )
                    self._retain((a._data, b._data, out))
                else:
                    # General broadcasting reuses input pointers; no repeated
                    # uploads or overlapping output matrices. Avoid undocumented
                    # zero strides in the strided-batched contract.
                    itemsize = np.dtype(dtype).itemsize
                    for output_index, index in enumerate(np.ndindex(batch)):
                        offsets = []
                        for source in (a_batch, b_batch):
                            offset = 0
                            for coordinate, dimension in zip(
                                index[len(batch) - len(source) :], source
                            ):
                                offset = offset * dimension + (coordinate if dimension != 1 else 0)
                            offsets.append(offset)
                        function(
                            self._handle,
                            cublas.Operation.N,
                            cublas.Operation.N,
                            n,
                            m,
                            k,
                            alpha.ctypes.data,
                            b._data.ptr + offsets[1] * k * n * itemsize,
                            n,
                            a._data.ptr + offsets[0] * m * k * itemsize,
                            k,
                            beta.ctypes.data,
                            out.ptr + output_index * m * n * itemsize,
                            n,
                        )
                        self._retain((a._data, b._data, out))
        else:
            dt = _type(dtype)
            if not batch:
                body = (
                    f"{dt} acc=0; for(int j=0;j<{k};++j) "
                    f"acc+=a[(i/{n})*{k}+j]*b[j*{n}+i%{n}]; out[i]=acc;"
                )
            else:
                abase = f"({_offset(batch, a_batch, 'batch')})*{m * k}LL"
                bbase = f"({_offset(batch, b_batch, 'batch')})*{k * n}LL"
                body = (
                    f"long long batch=i/{m * n}LL, row=(i/{n}LL)%{m}LL, col=i%{n}LL;"
                    f"{dt} acc=0; for(long long j=0;j<{k}LL;++j) "
                    f"acc+=a[({abase})+row*{k}LL+j]*b[({bbase})+j*{n}LL+col]; out[i]=acc;"
                )
            self._element_kernel(
                f"const {dt}*a,const {dt}*b,{dt}*out", body, prod(shape), a._data, b._data, out
            )
        return out

    def norm(self, a, axes, keepdims, dtype):
        from ._norm import norm_data

        return norm_data(self, a, axes, keepdims, self.validate_dtype(dtype))

    def close(self):
        if not hasattr(self, "_lock"):
            return
        with self._lock:
            self._close_locked()

    def _close_locked(self):
        if self._closed:
            return
        if self._context is not None:
            with self._activate():
                if self._stream is not None:
                    self.release_cached_memory()
                for lease in list(self._leases):
                    if lease.ptr:
                        _check(self.driver.cuMemFree(lease.ptr))
                        lease.ptr = 0
                if self._handle is not None:
                    self._cublas.destroy(self._handle)
                    self._handle = None
                while self._kernels:
                    source = next(iter(self._kernels))
                    _check(self.driver.cuModuleUnload(self._kernels[source][0]))
                    del self._kernels[source]
                if self._stream is not None:
                    _check(self.driver.cuStreamDestroy(self._stream))
                    self._stream = None
            _check(self.driver.cuDevicePrimaryCtxRelease(self._device))
            self._context = None
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - interpreter shutdown only
            pass
