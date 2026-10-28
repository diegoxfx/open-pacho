"""Explicit device selection; no GPU imports or initialization at package import."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .exceptions import BackendUnavailableError, UnsupportedDtypeError

SUPPORTED_DTYPES = tuple(
    map(np.dtype, ("bool", "int32", "int64", "uint32", "uint64", "float32", "float64"))
)


def check_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype not in SUPPORTED_DTYPES or not dtype.isnative:
        raise UnsupportedDtypeError(f"Unsupported dtype {dtype}; supported: {SUPPORTED_DTYPES}")
    return dtype


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    name: str
    vendor: str
    kind: str
    fp64: bool
    memory_bytes: int
    driver: str


def _opencl_devices():
    try:
        import pyopencl as cl
    except ImportError as exc:
        raise BackendUnavailableError(
            'Install OpenCL support: pip install "open-pacho[opencl]"'
        ) from exc
    try:
        devices = [
            (pi, di, d)
            for pi, p in enumerate(cl.get_platforms())
            for di, d in enumerate(p.get_devices())
        ]
    except cl.Error as exc:
        raise BackendUnavailableError(
            f"OpenCL discovery failed; install a vendor OpenCL driver: {exc}"
        ) from exc
    return cl, devices


def devices(backend="opencl"):
    """List devices for an explicitly selected GPU API."""
    if backend == "cuda":
        from ._cuda import cuda_devices

        return cuda_devices()
    if backend == "vulkan":
        from ._vulkan import vulkan_devices

        return vulkan_devices()
    if backend != "opencl":
        raise ValueError("Device discovery requires 'opencl', 'vulkan' or 'cuda'")
    cl, found = _opencl_devices()
    return [
        DeviceInfo(
            f"{pi}:{di}",
            d.name.strip(),
            d.vendor.strip(),
            "GPU" if d.type & cl.device_type.GPU else "CPU/other",
            bool(d.double_fp_config),
            int(d.global_mem_size),
            d.driver_version,
        )
        for pi, di, d in found
    ]


class NumpyBackend:
    name = "numpy"
    device = "CPU (NumPy)"

    def random_distributions(self, dtype):
        return ("random", "uniform", "normal", "exponential")

    def random_data(self, *args):
        from ._random import _numpy_data

        return _numpy_data(*args)

    def validate_dtype(self, dtype):
        return check_dtype(dtype)

    def upload(self, a):
        return np.array(a, copy=True, order="C").reshape(-1)

    def download(self, a):
        return a._data.copy().reshape(a.shape)

    def empty(self, shape, dtype):
        return np.empty(shape, dtype=dtype).reshape(-1)

    def full(self, shape, value, dtype):
        return np.full(shape, value, dtype=dtype).reshape(-1)

    def elementwise(self, name, args, shape, dtype):
        values = [a._data.reshape(a.shape) for a in args]
        if name == "cast":
            value = values[0].astype(dtype)
        else:
            with np.errstate(all="ignore"):
                value = getattr(np, name)(*values)
        return np.asarray(value, dtype=dtype).reshape(-1).copy()

    def gather(self, a, shape, offset, strides):
        # A view is used only internally; the public indexing operation returns a copy.
        if not np.prod(shape, dtype=np.int64):
            return np.empty(0, dtype=a.dtype)
        view = np.lib.stride_tricks.as_strided(
            a._data[offset:],
            shape=shape,
            strides=tuple(s * a.dtype.itemsize for s in strides),
            writeable=False,
        )
        return np.array(view, copy=True).reshape(-1)

    def reduce(self, a, axes, keepdims, dtype, *, operation="sum"):
        if operation != "sum":
            from ._reductions import numpy_extreme

            return numpy_extreme(a._data.reshape(a.shape), axes, keepdims, operation)
        return np.asarray(
            np.sum(a._data.reshape(a.shape), axis=axes, keepdims=keepdims, dtype=dtype)
        ).reshape(-1)

    def matmul(self, a, b, shape, dtype):
        return np.asarray(a._data.reshape(a.shape) @ b._data.reshape(b.shape), dtype=dtype).reshape(
            -1
        )

    def synchronize(self):
        pass

    def norm(self, a, axes, keepdims, dtype):
        from ._norm import norm_data

        return norm_data(self, a, axes, keepdims, dtype)


_NUMPY = NumpyBackend()
_DEFAULT = ContextVar("openpacho_backend", default=_NUMPY)


@lru_cache(maxsize=8)
def _get_opencl(device, blas="custom", clblast_library=None):
    from ._opencl import OpenCLBackend

    return OpenCLBackend(device, blas=blas, clblast_library=clblast_library)


@lru_cache(maxsize=8)
def _get_vulkan(device):
    from ._vulkan import VulkanBackend

    return VulkanBackend(device)


@lru_cache(maxsize=8)
def _get_cuda(device, blas="cublas"):
    from ._cuda import CUDABackend

    return CUDABackend(device, blas=blas)


def get_backend(backend=None, *, device=None, blas=None, clblast_library=None):
    """Resolve a selection, or retrieve an array's existing backend without moving data."""
    from ._array import ndarray

    if isinstance(backend, ndarray):
        if device is not None or blas is not None or clblast_library is not None:
            raise ValueError("An array's backend cannot be reconfigured")
        return backend._backend
    if getattr(backend, "_is_tracing", False):
        if device is not None or blas is not None or clblast_library is not None:
            raise ValueError("Captured backend does not accept reconfiguration")
        return backend
    if blas is not None and backend not in ("opencl", "cuda"):
        raise ValueError("BLAS configuration requires backend='opencl' or 'cuda'")
    if clblast_library is not None and backend != "opencl":
        raise ValueError("clblast_library requires backend='opencl'")
    if backend is None:
        if device is not None:
            raise ValueError("device requires backend='opencl', 'vulkan' or 'cuda'")
        return _DEFAULT.get()
    if isinstance(backend, NumpyBackend) or getattr(backend, "name", None) in (
        "opencl",
        "vulkan",
        "cuda",
    ):
        if device is not None:
            raise ValueError("device cannot accompany an existing backend")
        return backend
    if backend == "numpy" and device is None:
        return _NUMPY
    if backend == "opencl":
        return _get_opencl(device, blas or "custom", clblast_library)
    if backend == "vulkan":
        return _get_vulkan(device)
    if backend == "cuda":
        return _get_cuda(device, blas or "cublas")
    raise ValueError("backend must be 'numpy', 'opencl', 'vulkan', 'cuda', or a backend object")


def set_backend(backend, *, device=None, blas=None, clblast_library=None):
    """Set the backend for new arrays in this execution context; existing arrays stay put."""
    selected = get_backend(backend, device=device, blas=blas, clblast_library=clblast_library)
    _DEFAULT.set(selected)
    return selected


@contextmanager
def use_backend(backend, *, device=None, blas=None, clblast_library=None):
    selected = get_backend(backend, device=device, blas=blas, clblast_library=clblast_library)
    token = _DEFAULT.set(selected)
    try:
        yield selected
    finally:
        _DEFAULT.reset(token)


def synchronize(backend=None):
    get_backend(backend).synchronize()
