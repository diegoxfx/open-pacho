"""Optional CUDA matmul references used by benchmark experiments.

This module is deliberately outside :mod:`openpacho`'s runtime backend
registry.  It gives benchmark code three independently selectable references
on a CUDA device:

* ``CuPyReference`` uses the ordinary CuPy ``matmul`` path;
* ``CuBLASReference`` calls the low-level cuBLAS bindings shipped with CuPy;
* ``NVMATHReference`` uses nvmath-python's stateful ``advanced.Matmul`` API.

The adapters share a small lifecycle:

``resident = adapter.upload((a, b))``
``plan = adapter.setup(resident)``
``result = adapter.execute(plan)``

``compute(resident, plan=plan)`` and ``compute(plan)`` are convenience forms
for benchmark loops.  Keeping ``setup``/``plan`` separate from ``execute`` is
important for nvmath-python, where planning is a one-time cost that can be
amortized across repeated executions.

The implementation only accepts two-dimensional, same-dtype float32 or
float64 matrices.  TensorFloat-32 is disabled for the process before CUDA is
used, and nvmath is given an exact float32/float64 compute type.  No adapter is
registered as an OpenPacho backend.

The nvmath API and platform claims are documented by NVIDIA at:
https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html
https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/linalg/generated/nvmath.linalg.advanced.Matmul-class.html
"""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar

import numpy as np

# Keep all CUDA references on the same arithmetic contract as the existing
# benchmark suite.  NVIDIA documents this environment variable as the switch
# that prevents TF32 use in CUDA libraries.  Set it before importing CuPy.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_LAST_AVAILABILITY_ERRORS: dict[str, str] = {}


class ReferenceUnavailable(RuntimeError):
    """Raised when an optional CUDA reference cannot be initialized."""


@dataclass(frozen=True)
class MatmulSpec:
    """Validated matrix multiplication dimensions and arithmetic type."""

    m: int
    k: int
    n: int
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return (self.m, self.n)


class _Plan:
    """Internal common marker for plans owned by a reference adapter."""

    def __init__(self, owner: CUDAReference, a: Any, b: Any, spec: MatmulSpec):
        self.owner = owner
        self.a = a
        self.b = b
        self.spec = spec
        self.closed = False


class _CuPyPlan(_Plan):
    pass


class _CuBLASPlan(_Plan):
    pass


class _NVMATHPlan(_Plan):
    def __init__(self, owner: CUDAReference, a: Any, b: Any, spec: MatmulSpec, mm: Any):
        super().__init__(owner, a, b, spec)
        self.mm = mm


def _unpack_operands(operands: Any = None, b: Any = None) -> tuple[Any, Any]:
    """Accept either ``(a, b)`` or the convenient ``a, b`` form."""

    if b is not None:
        if operands is None:
            raise TypeError("the left operand is required")
        return operands, b
    if operands is None:
        raise TypeError("matmul operands are required")
    if isinstance(operands, (tuple, list)) and len(operands) == 2:
        return operands[0], operands[1]
    raise TypeError("matmul operands must be a pair (a, b)")


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _import_cupy() -> Any:
    """Import CuPy lazily so CPU-only users can import this module."""

    # CuPy's import initializes its on-disk kernel cache.  The benchmark
    # workspace is writable in the supported project setup, while a fallback
    # keeps this helper usable from a read-only checkout or CI sandbox.
    if not os.environ.get("CUPY_CACHE_DIR"):
        cache = Path.cwd() / ".artifacts" / "cuda-references" / "cupy-cache"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["CUPY_CACHE_DIR"] = str(cache)
        except OSError:
            pass
    try:
        import cupy as cp
    except Exception as exc:  # ImportError and CUDA DLL failures are both optional.
        raise ReferenceUnavailable(f"CuPy is unavailable: {_format_exception(exc)}") from exc
    return cp


def _require_device(cp: Any, device: int) -> Any:
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise ReferenceUnavailable(f"CUDA is unavailable: {_format_exception(exc)}") from exc
    if count <= 0:
        raise ReferenceUnavailable("CUDA is unavailable: no CUDA devices were reported")
    if device < 0 or device >= count:
        raise ReferenceUnavailable(f"CUDA device {device} is unavailable (device count: {count})")
    try:
        properties = cp.cuda.runtime.getDeviceProperties(device)
        cp.cuda.Device(device).use()
    except Exception as exc:
        raise ReferenceUnavailable(
            f"CUDA device {device} could not be selected: {_format_exception(exc)}"
        ) from exc
    return properties


def _device_id(value: Any) -> int | None:
    try:
        return int(value.device.id)
    except (AttributeError, TypeError, ValueError):
        return None


def _validate_spec(a: Any, b: Any, device: int) -> MatmulSpec:
    try:
        ndim_a, ndim_b = int(a.ndim), int(b.ndim)
        dtype_a, dtype_b = np.dtype(a.dtype), np.dtype(b.dtype)
        shape_a, shape_b = tuple(a.shape), tuple(b.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("matmul operands must be array-like objects") from exc
    if ndim_a != 2 or ndim_b != 2:
        raise ValueError("CUDA references currently support two-dimensional matrices only")
    if dtype_a not in _SUPPORTED_DTYPES or dtype_b not in _SUPPORTED_DTYPES:
        raise ValueError("CUDA references support only float32 and float64 operands")
    if dtype_a != dtype_b:
        raise ValueError("CUDA references require both operands to have the same dtype")
    if shape_a[1] != shape_b[0]:
        raise ValueError("matmul inner dimensions do not match")
    if any(int(dimension) <= 0 for dimension in (*shape_a, *shape_b)):
        raise ValueError("CUDA references require positive matrix dimensions")
    for operand_name, operand in (("a", a), ("b", b)):
        operand_device = _device_id(operand)
        if operand_device is not None and operand_device != device:
            raise ValueError(
                f"operand {operand_name} is on CUDA device {operand_device}, expected {device}"
            )
    return MatmulSpec(int(shape_a[0]), int(shape_a[1]), int(shape_b[1]), dtype_a)


class CUDAReference:
    """Common lifecycle for the benchmark-only CUDA matmul references."""

    name: ClassVar[str] = "cuda-reference"
    label: ClassVar[str] = "CUDA reference"
    platform: ClassVar[str] = "CUDA"
    implementation: ClassVar[str] = ""

    def __init__(self, device: int = 0):
        if isinstance(device, bool) or not isinstance(device, (int, np.integer)):
            raise TypeError("device must be an integer CUDA device index")
        self.device = int(device)
        self._closed = False
        self._last_plan: _Plan | None = None
        self._plans: list[_Plan] = []
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        self.cp = _import_cupy()
        self._properties = _require_device(self.cp, self.device)

    @classmethod
    def available(cls, device: int = 0) -> bool:
        """Return whether this reference can be initialized on ``device``."""

        instance = None
        try:
            instance = cls(device=device)
            return True
        except Exception:  # noqa: BLE001 - availability is optional by design.
            return False
        finally:
            if instance is not None:
                instance.close()

    def _activate(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.name} reference is closed")
        self.cp.cuda.Device(self.device).use()
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    def _device_operands(self, operands: Any = None, b: Any = None) -> tuple[Any, Any, MatmulSpec]:
        self._activate()
        left, right = _unpack_operands(operands, b)
        try:
            left = self.cp.asarray(left)
            right = self.cp.asarray(right)
        except Exception as exc:
            raise ValueError(
                f"could not move matmul operands to CUDA: {_format_exception(exc)}"
            ) from exc
        # A contiguous snapshot makes the row-major cuBLAS mapping explicit
        # and gives all references identical input layout and ownership.
        left = self.cp.ascontiguousarray(left)
        right = self.cp.ascontiguousarray(right)
        spec = _validate_spec(left, right, self.device)
        return left, right, spec

    def upload(self, operands: Any = None, b: Any = None) -> tuple[Any, Any]:
        """Copy two host matrices to the selected CUDA device."""

        self._activate()
        left, right = _unpack_operands(operands, b)
        try:
            # ``cp.array(..., copy=True, order="C")`` performs one H2D copy
            # for host inputs.  Calling ``cp.asarray`` first would upload and
            # then copy again on the device, distorting transfer benchmarks.
            left = self.cp.array(left, copy=True, order="C")
            right = self.cp.array(right, copy=True, order="C")
        except Exception as exc:
            raise ValueError(
                f"could not move matmul operands to CUDA: {_format_exception(exc)}"
            ) from exc
        _validate_spec(left, right, self.device)
        return left, right

    def host(self, value: Any) -> np.ndarray:
        """Copy a device result to a NumPy array."""

        self._activate()
        try:
            return self.cp.asnumpy(value)
        except Exception as exc:
            raise ValueError(
                f"could not copy CUDA result to host: {_format_exception(exc)}"
            ) from exc

    def sync(self) -> None:
        """Synchronize the current CUDA stream."""

        self._activate()
        self.cp.cuda.get_current_stream().synchronize()

    def setup(self, operands: Any = None, b: Any = None) -> _Plan:
        """Prepare a reusable matmul plan without executing it."""

        left, right, spec = self._device_operands(operands, b)
        plan = self._setup_plan(left, right, spec)
        self._remember_plan(plan)
        return plan

    def plan(self, operands: Any = None, b: Any = None) -> _Plan:
        """Alias for :meth:`setup`, matching common benchmark terminology."""

        return self.setup(operands, b)

    def execute(self, plan: _Plan) -> Any:
        """Execute a prepared plan once."""

        self._check_plan(plan)
        return self._execute_plan(plan)

    def compute(self, operands: Any = None, b: Any = None, *, plan: _Plan | None = None) -> Any:
        """Execute a plan, preparing one on first use when needed.

        The usual benchmark form is ``compute((a, b), plan=plan)``.  Passing
        ``compute(plan)`` executes an already prepared plan directly.
        """

        if b is None and isinstance(operands, _Plan) and plan is None:
            plan = operands
            operands = None
        if plan is None:
            if operands is None:
                if self._last_plan is None:
                    raise TypeError("operands or a prepared plan are required")
                plan = self._last_plan
            else:
                left, right, spec = self._device_operands(operands, b)
                previous = self._last_plan
                if previous is not None and not previous.closed and previous.spec == spec:
                    plan = previous
                    self._refresh_plan(plan, left, right)
                else:
                    plan = self._setup_plan(left, right, spec)
                    self._remember_plan(plan)
        elif operands is not None:
            left, right, spec = self._device_operands(operands, b)
            if plan.spec != spec:
                raise ValueError("operands do not match the prepared matmul plan")
            self._refresh_plan(plan, left, right)
        return self.execute(plan)

    def _check_plan(self, plan: _Plan) -> None:
        if not isinstance(plan, _Plan) or plan.owner is not self:
            raise TypeError("plan was not created by this CUDA reference")
        if plan.closed:
            raise RuntimeError("matmul plan is closed")
        self._activate()

    def _setup_plan(self, a: Any, b: Any, spec: MatmulSpec) -> _Plan:
        raise NotImplementedError

    def _remember_plan(self, plan: _Plan) -> None:
        self._plans.append(plan)
        self._last_plan = plan

    def _refresh_plan(self, plan: _Plan, a: Any, b: Any) -> None:
        plan.a, plan.b = a, b

    def _execute_plan(self, plan: _Plan) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        """Release adapter resources; prepared plans should not be reused."""

        self._closed = True
        for plan in self._plans:
            plan.closed = True

    def cleanup(self) -> None:
        """Alias for :meth:`close` that preserves subclass cleanup behavior."""

        self.close()

    def __enter__(self) -> CUDAReference:  # noqa: PYI034 - keep Python 3.10 support.
        self._activate()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class CuPyReference(CUDAReference):
    """Standard CuPy ``@``/``matmul`` CUDA reference."""

    name = "cupy"
    label = "CUDA / CuPy"
    implementation = "CuPy matmul"

    def _setup_plan(self, a: Any, b: Any, spec: MatmulSpec) -> _CuPyPlan:
        return _CuPyPlan(self, a, b, spec)

    def _execute_plan(self, plan: _CuPyPlan) -> Any:
        self._activate()
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        return self.cp.matmul(plan.a, plan.b)


class CuBLASReference(CUDAReference):
    """Raw cuBLAS GEMM reference using CuPy's existing low-level bindings."""

    name = "cublas"
    label = "CUDA / cuBLAS"
    implementation = "cupy_backends.cuda.libs.cublas"

    def __init__(self, device: int = 0):
        super().__init__(device=device)
        handle = None
        try:
            from cupy_backends.cuda.libs import cublas

            self._cublas = cublas
            handle = cublas.create()
            cublas.setMathMode(handle, cublas.CUBLAS_DEFAULT_MATH)
            self._handle = handle
        except Exception as exc:
            if handle is not None:
                with suppress(Exception):
                    cublas.destroy(handle)
            self._handle = None
            raise ReferenceUnavailable(
                f"cuBLAS bindings are unavailable: {_format_exception(exc)}"
            ) from exc

    def _setup_plan(self, a: Any, b: Any, spec: MatmulSpec) -> _CuBLASPlan:
        return _CuBLASPlan(self, a, b, spec)

    def _execute_plan(self, plan: _CuBLASPlan) -> Any:
        self._activate()
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        self._cublas.setMathMode(self._handle, self._cublas.CUBLAS_DEFAULT_MATH)
        stream = self.cp.cuda.get_current_stream()
        self._cublas.setStream(self._handle, int(stream.ptr))
        result = self.cp.empty(plan.spec.shape, dtype=plan.spec.dtype, order="C")
        alpha = np.asarray(1, dtype=plan.spec.dtype)
        beta = np.asarray(0, dtype=plan.spec.dtype)

        # cuBLAS consumes column-major matrices.  A C-contiguous row-major
        # result C = A @ B is represented as C.T = B.T @ A.T, so exchange A/B
        # and use (n, m, k) with leading dimensions (n, k, n).
        function = (
            self._cublas.sgemm if plan.spec.dtype == np.dtype(np.float32) else self._cublas.dgemm
        )
        function(
            self._handle,
            self._cublas.CUBLAS_OP_N,
            self._cublas.CUBLAS_OP_N,
            plan.spec.n,
            plan.spec.m,
            plan.spec.k,
            int(alpha.ctypes.data),
            int(plan.b.data.ptr),
            plan.spec.n,
            int(plan.a.data.ptr),
            plan.spec.k,
            int(beta.ctypes.data),
            int(result.data.ptr),
            plan.spec.n,
        )
        return result

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            with suppress(Exception):
                self.sync()
            with suppress(Exception):
                self._cublas.destroy(self._handle)
            self._handle = None
        super().close()


class NVMATHReference(CUDAReference):
    """Stateful nvmath-python advanced matrix multiplication reference."""

    name = "nvmath"
    label = "CUDA / nvmath-python"
    implementation = "nvmath.linalg.advanced.Matmul"

    def __init__(self, device: int = 0):
        super().__init__(device=device)
        try:
            import nvmath
            from nvmath.linalg import advanced

            self._nvmath = nvmath
            self._advanced = advanced
            self._matmul_type = advanced.Matmul
            self._compute_types = advanced.MatmulComputeType
        except Exception as exc:
            raise ReferenceUnavailable(
                f"nvmath-python is unavailable: {_format_exception(exc)}"
            ) from exc
        major = int(self._properties.get("major", 0))
        if major < 7:
            raise ReferenceUnavailable(
                "nvmath-python host GPU Matmul requires compute capability 7.0 or newer"
            )

    def _setup_plan(self, a: Any, b: Any, spec: MatmulSpec) -> _NVMATHPlan:
        compute_type = (
            self._compute_types.COMPUTE_32F
            if spec.dtype == np.dtype(np.float32)
            else self._compute_types.COMPUTE_64F
        )
        mm = None
        try:
            mm = self._matmul_type(a, b, options={"compute_type": compute_type})
            mm.plan()
        except Exception:
            if mm is not None:
                with suppress(Exception):
                    mm.free()
            raise
        return _NVMATHPlan(self, a, b, spec, mm)

    def _refresh_plan(self, plan: _NVMATHPlan, a: Any, b: Any) -> None:
        if plan.a is a and plan.b is b:
            return
        plan.mm.reset_operands(a=a, b=b)
        plan.a, plan.b = a, b

    def _execute_plan(self, plan: _NVMATHPlan) -> Any:
        self._activate()
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        return plan.mm.execute()

    def close(self) -> None:
        with suppress(Exception):
            self.sync()
        for plan in getattr(self, "_plans", []):
            if isinstance(plan, _NVMATHPlan) and not plan.closed:
                with suppress(Exception):
                    plan.mm.free()
                plan.closed = True
        super().close()


# Compatibility aliases make the adapter naming easy to discover alongside
# benchmarks.suite.Adapter while keeping the longer class names descriptive.
CuPyAdapter = CuPyReference
CuBLASAdapter = CuBLASReference
NvmathReference = NVMATHReference
NvmathAdapter = NVMATHReference
NVMATHAdapter = NVMATHReference


_REFERENCE_CLASSES = (CuPyReference, CuBLASReference, NVMATHReference)


def available_references(device: int = 0) -> list[CUDAReference]:
    """Return initialized CUDA references that are usable on ``device``.

    Optional dependencies and CUDA hardware are probed independently.  A
    missing package or unsupported device is skipped and recorded in
    :func:`availability_errors`; no synthetic result is returned.
    """

    _LAST_AVAILABILITY_ERRORS.clear()
    references: list[CUDAReference] = []
    for reference_class in _REFERENCE_CLASSES:
        try:
            references.append(reference_class(device=device))
        except Exception as exc:  # noqa: BLE001 - each optional reference may be absent.
            _LAST_AVAILABILITY_ERRORS[reference_class.name] = _format_exception(exc)
    return references


def availability_errors() -> dict[str, str]:
    """Return reasons recorded by the most recent :func:`available_references` call."""

    return dict(_LAST_AVAILABILITY_ERRORS)


__all__ = [
    "CUDAReference",
    "CuBLASAdapter",
    "CuBLASReference",
    "CuPyAdapter",
    "CuPyReference",
    "MatmulSpec",
    "NVMATHAdapter",
    "NVMATHReference",
    "NvmathAdapter",
    "NvmathReference",
    "ReferenceUnavailable",
    "availability_errors",
    "available_references",
]
