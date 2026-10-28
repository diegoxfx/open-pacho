"""Small optional ctypes adapter for the CLBlast C BLAS API.

The adapter deliberately owns no OpenCL objects.  OpenPacho's OpenCL backend
owns the queue and buffers and passes their ``int_ptr`` handles here.  This
keeps CLBlast optional and leaves backend selection and fallback policy to the
caller.

CLBlast's C API represents ``cl_command_queue`` as a pointer to an OpenCL
opaque handle (``cl_command_queue*``), so the queue handle is passed by
reference.  The same applies to the optional event output argument.
"""

from __future__ import annotations

import ctypes
import os
from ctypes.util import find_library
from itertools import pairwise

import numpy as np

from .exceptions import BackendUnavailableError, UnsupportedDtypeError

_ROW_MAJOR = 101
_NO_TRANSPOSE = 111
_TRANSPOSE = 112
_SUCCESS = 0

_HANDLE = ctypes.c_void_p
_HANDLE_PTR = ctypes.POINTER(_HANDLE)
_STATUS = ctypes.c_int
_SIZE = ctypes.c_size_t


def _load_library(path):
    """Load a CLBlast shared library with the platform's native ABI."""

    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    return loader(path)


def _library_candidates():
    """Return names that may resolve to a system CLBlast library."""

    found = find_library("clblast")
    candidates = [found] if found else []
    if os.name == "nt":
        candidates.extend(("clblast.dll", "libclblast.dll"))
    elif os.sys.platform == "darwin":
        candidates.extend(("libclblast.dylib", "clblast"))
    else:
        candidates.extend(("libclblast.so", "libclblast.so.1", "clblast"))
    # Preserve order while avoiding repeated names returned by find_library.
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _configure_gemm(function, scalar):
    """Attach the exact ABI from ``include/clblast_c.h`` to one GEMM symbol."""

    function.restype = _STATUS
    function.argtypes = [
        ctypes.c_int,  # CLBlastLayout
        ctypes.c_int,  # CLBlastTranspose a
        ctypes.c_int,  # CLBlastTranspose b
        _SIZE,
        _SIZE,
        _SIZE,
        scalar,
        _HANDLE,
        _SIZE,
        _SIZE,
        _HANDLE,
        _SIZE,
        _SIZE,
        scalar,
        _HANDLE,
        _SIZE,
        _SIZE,
        _HANDLE_PTR,  # cl_command_queue* queue
        _HANDLE_PTR,  # cl_event* event
    ]
    return function


def _configure_gemm_batched(function, scalar):
    """Attach the CLBlast ``*gemmBatched`` ABI from ``clblast_c.h``."""

    function.restype = _STATUS
    function.argtypes = [
        ctypes.c_int,  # CLBlastLayout
        ctypes.c_int,  # CLBlastTranspose a
        ctypes.c_int,  # CLBlastTranspose b
        _SIZE,
        _SIZE,
        _SIZE,
        ctypes.POINTER(scalar),  # alpha array
        _HANDLE,
        ctypes.POINTER(_SIZE),  # a offsets array
        _SIZE,
        _HANDLE,
        ctypes.POINTER(_SIZE),  # b offsets array
        _SIZE,
        ctypes.POINTER(scalar),  # beta array
        _HANDLE,
        ctypes.POINTER(_SIZE),  # c offsets array
        _SIZE,
        _SIZE,  # batch count
        _HANDLE_PTR,
        _HANDLE_PTR,
    ]
    return function


def _configure_gemm_strided_batched(function, scalar):
    """Attach the CLBlast ``*gemmStridedBatched`` ABI from ``clblast_c.h``."""

    function.restype = _STATUS
    function.argtypes = [
        ctypes.c_int,  # CLBlastLayout
        ctypes.c_int,  # CLBlastTranspose a
        ctypes.c_int,  # CLBlastTranspose b
        _SIZE,
        _SIZE,
        _SIZE,
        scalar,
        _HANDLE,
        _SIZE,
        _SIZE,
        _SIZE,  # a stride
        _HANDLE,
        _SIZE,
        _SIZE,
        _SIZE,  # b stride
        scalar,
        _HANDLE,
        _SIZE,
        _SIZE,
        _SIZE,  # c stride
        _SIZE,  # batch count
        _HANDLE_PTR,
        _HANDLE_PTR,
    ]
    return function


def _handle(value, label):
    """Read an OpenCL ``int_ptr`` without accepting host-pointer fallbacks."""

    try:
        pointer = value.int_ptr
    except AttributeError as exc:
        raise TypeError(f"{label} must expose an OpenCL int_ptr handle") from exc
    try:
        pointer = int(pointer)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label}.int_ptr must be an integer handle") from exc
    if pointer <= 0:
        raise ValueError(f"{label}.int_ptr must be a non-zero OpenCL handle")
    return _HANDLE(pointer)


def _zero_gemv_output(queue, out, count, dtype):
    """Make beta=0 independent of recycled destination contents.

    CLBlast 1.7.0 GEMV kernels can evaluate beta*y even when beta is zero.
    A recycled y containing NaN/Inf then contaminates an otherwise finite
    product. Initialize on the same in-order queue without a host wait.
    """
    import pyopencl as cl

    storage = out._data if hasattr(out, "_data") else out
    cl.enqueue_fill_buffer(queue, storage, np.zeros((), dtype=dtype), 0, count * dtype.itemsize)


def _batch_offsets(batch, dimensions, matrix_size):
    """Return element offsets for broadcast batch coordinates.

    The returned offsets address matrices in one existing OpenCL buffer.  A
    singleton dimension repeats its matrix offset, so broadcasting never
    expands or copies an input array.
    """

    batch = tuple(batch)
    dimensions = tuple(dimensions)
    if any(d == 0 for d in batch):
        return []
    offsets = []
    for coordinates in np.ndindex(batch):
        aligned = coordinates[len(batch) - len(dimensions) :] if dimensions else ()
        value = 0
        for coordinate, dimension in zip(aligned, dimensions):
            value = value * dimension + (0 if dimension == 1 else coordinate)
        offsets.append(value * matrix_size)
    return offsets or [0]


def _regular_batch_stride(offsets):
    """Return a constant positive stride, or ``None`` for irregular offsets."""

    if len(offsets) < 2:
        return 0
    stride = offsets[1] - offsets[0]
    if stride <= 0 or any(right - left != stride for left, right in pairwise(offsets)):
        return None
    return stride


class CLBlast:
    """Optional CLBlast adapter loaded from a shared library.

    ``path`` takes precedence over ``OPENPACHO_CLBLAST_LIBRARY``.  With no
    explicit path the platform loader is queried through ``find_library``.
    Construction fails immediately if CLBlast or either supported GEMM symbol
    is unavailable, so a selected ``blas='clblast'`` backend never silently
    executes a different implementation.
    """

    name = "clblast"

    def __init__(self, path=None):
        requested = path
        if requested is None:
            requested = os.environ.get("OPENPACHO_CLBLAST_LIBRARY") or None

        if requested is not None:
            requested = os.fspath(requested)
            candidates = (requested,)
        else:
            candidates = _library_candidates()

        if not candidates:
            raise BackendUnavailableError(
                "CLBlast was not found. Install the optional CLBlast shared library "
                "and set OPENPACHO_CLBLAST_LIBRARY to its absolute path "
                "(for example, clblast.dll, libclblast.so, or libclblast.dylib)."
            )

        library = None
        loaded_path = None
        errors = []
        for candidate in candidates:
            try:
                library = _load_library(candidate)
                loaded_path = candidate
            except (OSError, FileNotFoundError) as exc:
                errors.append(f"{candidate!r}: {exc}")
                if requested is not None:
                    break
        if library is None:
            detail = "; ".join(errors) or "the loader returned no library"
            if requested is not None:
                message = (
                    f"Cannot load CLBlast library {requested!r}: {detail}. "
                    "Check that the path names a library for this Python architecture "
                    "and that its OpenCL runtime dependencies are available."
                )
            else:
                message = (
                    "CLBlast was found by name but could not be loaded: "
                    f"{detail}. Set OPENPACHO_CLBLAST_LIBRARY to an explicit path."
                )
            if errors:
                raise BackendUnavailableError(message) from OSError(errors[-1])
            raise BackendUnavailableError(message)

        self._library = library
        self.path = loaded_path
        functions = {}
        for symbol, scalar in (("CLBlastSgemm", ctypes.c_float), ("CLBlastDgemm", ctypes.c_double)):
            try:
                function = getattr(library, symbol)
            except AttributeError as exc:
                raise BackendUnavailableError(
                    f"CLBlast library {self.path!r} does not export {symbol}. "
                    "Use a build exposing the CLBlast C API (include/clblast_c.h)."
                ) from exc
            functions[symbol] = _configure_gemm(function, scalar)
        self._gemm = functions
        # Batched entry points are optional across CLBlast builds.  A missing
        # symbol is recorded as unavailable so matmul can use the existing
        # regular-GEMM loop without changing the selected BLAS backend.
        self._gemm_batched = {}
        self._gemm_strided_batched = {}
        for prefix, scalar in (("S", ctypes.c_float), ("D", ctypes.c_double)):
            batched_symbol = f"CLBlast{prefix}gemmBatched"
            batched = getattr(library, batched_symbol, None)
            if batched is not None:
                self._gemm_batched[batched_symbol] = _configure_gemm_batched(batched, scalar)
            strided_symbol = f"CLBlast{prefix}gemmStridedBatched"
            strided = getattr(library, strided_symbol, None)
            if strided is not None:
                self._gemm_strided_batched[strided_symbol] = _configure_gemm_strided_batched(
                    strided, scalar
                )
        # Older/minimal CLBlast builds can retain the GEMM mapping. Full builds
        # use the actual level-1/2 routines instead of skinny GEMM calls.
        self._vectors = {}
        for prefix, scalar in (("S", ctypes.c_float), ("D", ctypes.c_double)):
            for operation in ("gemv", "dot"):
                symbol = f"CLBlast{prefix}{operation}"
                function = getattr(library, symbol, None)
                if function is None:
                    continue
                function.restype = ctypes.c_int
                function.argtypes = (
                    [
                        ctypes.c_int,
                        ctypes.c_int,
                        _SIZE,
                        _SIZE,
                        scalar,
                        _HANDLE,
                        _SIZE,
                        _SIZE,
                        _HANDLE,
                        _SIZE,
                        _SIZE,
                        scalar,
                        _HANDLE,
                        _SIZE,
                        _SIZE,
                        _HANDLE_PTR,
                        _HANDLE_PTR,
                    ]
                    if operation == "gemv"
                    else [
                        _SIZE,
                        _HANDLE,
                        _SIZE,
                        _HANDLE,
                        _SIZE,
                        _SIZE,
                        _HANDLE,
                        _SIZE,
                        _SIZE,
                        _HANDLE_PTR,
                        _HANDLE_PTR,
                    ]
                )
                self._vectors[symbol] = function

    def matmul(self, queue, a, b, out, shape, dtype, *, transpose_a=False, transpose_b=False):
        """Enqueue a row-major product into ``out`` and return immediately.

        OpenPacho arrays are C-contiguous.  A one-dimensional left operand is
        interpreted as a ``1 x K`` matrix and a one-dimensional right operand
        as ``K x 1``; this preserves NumPy's vector/matrix result shapes while
        allowing every supported product to use GEMM. Non-batched vector
        products use native GEMV/DOT when those symbols are available.
        """

        dtype = np.dtype(dtype)
        if dtype == np.dtype("float32"):
            symbol = "CLBlastSgemm"
            scalar = ctypes.c_float
        elif dtype == np.dtype("float64"):
            symbol = "CLBlastDgemm"
            scalar = ctypes.c_double
        else:
            raise UnsupportedDtypeError(
                f"CLBlast GEMM supports only float32 and float64, got {dtype}"
            )

        ashape = tuple(a.shape)
        bshape = tuple(b.shape)
        if transpose_a or transpose_b:
            if len(ashape) != 2 or len(bshape) != 2:
                raise ValueError("CLBlast transpose flags require two-dimensional operands")
            ashape = ashape[::-1] if transpose_a else ashape
            bshape = bshape[::-1] if transpose_b else bshape
        if not ashape or not bshape:
            raise ValueError("CLBlast GEMM requires at least one-dimensional arrays")
        if ashape[-1] != bshape[-2 if len(bshape) > 1 else 0]:
            raise ValueError("CLBlast GEMM inner dimensions do not match")

        k = ashape[-1]
        m = 1 if len(ashape) == 1 else ashape[-2]
        n = 1 if len(bshape) == 1 else bshape[-1]
        batch = np.broadcast_shapes(ashape[:-2], bshape[:-2])
        expected = batch + (() if len(ashape) == 1 else (m,)) + (() if len(bshape) == 1 else (n,))
        if tuple(shape) != expected:
            raise ValueError(f"CLBlast GEMM output shape {shape!r} does not match {expected!r}")

        # CLBlast rejects zero M/N/K.  The OpenCL backend handles these cases
        # before calling the adapter; keeping this guard makes direct use safe.
        if not m or not n or not k:
            return

        if np.dtype(a.dtype) != dtype:
            raise TypeError(f"left operand dtype {a.dtype} does not match {dtype}")
        if np.dtype(b.dtype) != dtype:
            raise TypeError(f"right operand dtype {b.dtype} does not match {dtype}")

        queue_handle = _handle(queue, "queue")
        a_handle = _handle(a._data, "a._data")
        b_handle = _handle(b._data, "b._data")
        out_handle = _handle(out._data if hasattr(out, "_data") else out, "out")

        prefix = "S" if dtype == np.dtype("float32") else "D"
        routine = "dot" if len(ashape) == len(bshape) == 1 else "gemv"
        vector_function = self._vectors.get(f"CLBlast{prefix}{routine}")
        if (
            vector_function is not None
            and len(ashape) <= 2
            and len(bshape) <= 2
            and (len(ashape) == 1 or len(bshape) == 1)
        ):
            if routine == "dot":
                arguments = (
                    _SIZE(k),
                    out_handle,
                    _SIZE(0),
                    a_handle,
                    _SIZE(0),
                    _SIZE(1),
                    b_handle,
                    _SIZE(0),
                    _SIZE(1),
                    ctypes.byref(queue_handle),
                    None,
                )
            else:
                _zero_gemv_output(queue, out, m if len(ashape) == 2 else n, dtype)
                matrix, vector = (a_handle, b_handle) if len(ashape) == 2 else (b_handle, a_handle)
                rows, columns = (m, k) if len(ashape) == 2 else (k, n)
                transpose = _NO_TRANSPOSE if len(ashape) == 2 else 112
                arguments = (
                    _ROW_MAJOR,
                    transpose,
                    _SIZE(rows),
                    _SIZE(columns),
                    scalar(1),
                    matrix,
                    _SIZE(0),
                    _SIZE(columns),
                    vector,
                    _SIZE(0),
                    _SIZE(1),
                    scalar(0),
                    out_handle,
                    _SIZE(0),
                    _SIZE(1),
                    ctypes.byref(queue_handle),
                    None,
                )
            status = int(vector_function(*arguments))
            if status != _SUCCESS:
                raise RuntimeError(
                    f"CLBlast{prefix}{routine} failed with CLBlast status code {status}"
                )
            return

        # Row-major leading dimensions describe stored rows, before op(A/B).
        a_ld = m if transpose_a else k
        b_ld = k if transpose_b else n
        c_ld = n
        function = self._gemm[symbol]
        a_offsets = _batch_offsets(batch, ashape[:-2], m * k)
        b_offsets = _batch_offsets(batch, bshape[:-2], k * n)
        if not a_offsets:
            return
        c_offsets = [index * m * n for index in range(len(a_offsets))]
        batch_count = len(a_offsets)

        strided_symbol = f"CLBlast{prefix}gemmStridedBatched"
        strided_function = self._gemm_strided_batched.get(strided_symbol)
        a_stride = _regular_batch_stride(a_offsets)
        b_stride = _regular_batch_stride(b_offsets)
        c_stride = _regular_batch_stride(c_offsets)

        # Pointer-array batching is reserved for irregular/multidimensional
        # broadcast offsets.  A contiguous batch with a strided symbol is
        # handled by StridedBatched below, even when both symbols are present.
        batched_symbol = f"CLBlast{prefix}gemmBatched"
        batched_function = self._gemm_batched.get(batched_symbol)
        if (
            batch_count > 1
            and batched_function is not None
            and not (
                strided_function is not None
                and a_stride is not None
                and b_stride is not None
                and c_stride is not None
            )
        ):
            offset_type = _SIZE * batch_count
            alpha_type = scalar * batch_count
            a_offset_array = offset_type(*a_offsets)
            b_offset_array = offset_type(*b_offsets)
            c_offset_array = offset_type(*c_offsets)
            alphas = alpha_type(*([1] * batch_count))
            betas = alpha_type(*([0] * batch_count))
            try:
                status = batched_function(
                    _ROW_MAJOR,
                    _NO_TRANSPOSE,
                    _NO_TRANSPOSE,
                    _SIZE(m),
                    _SIZE(n),
                    _SIZE(k),
                    alphas,
                    a_handle,
                    a_offset_array,
                    _SIZE(a_ld),
                    b_handle,
                    b_offset_array,
                    _SIZE(b_ld),
                    betas,
                    out_handle,
                    c_offset_array,
                    _SIZE(c_ld),
                    _SIZE(batch_count),
                    ctypes.byref(queue_handle),
                    None,
                )
            except (ctypes.ArgumentError, OSError) as exc:
                raise RuntimeError(f"{batched_symbol} call failed: {exc}") from exc
            status = int(status)
            if status != _SUCCESS:
                raise RuntimeError(f"{batched_symbol} failed with CLBlast status code {status}")
            return

        # Strided batching is compact for a fully contiguous batch.  A zero or
        # irregular stride would be incorrect for broadcast inputs, so those
        # cases use pointer-array batching when available or the regular GEMM
        # fallback below.
        # Strided batching is the compact path for a fully contiguous batch.
        # A zero or irregular stride is unsuitable for broadcast inputs.
        if (
            batch_count > 1
            and strided_function is not None
            and a_stride is not None
            and b_stride is not None
            and c_stride is not None
        ):
            try:
                status = strided_function(
                    _ROW_MAJOR,
                    _NO_TRANSPOSE,
                    _NO_TRANSPOSE,
                    _SIZE(m),
                    _SIZE(n),
                    _SIZE(k),
                    scalar(1),
                    a_handle,
                    _SIZE(a_offsets[0]),
                    _SIZE(a_ld),
                    _SIZE(a_stride),
                    b_handle,
                    _SIZE(b_offsets[0]),
                    _SIZE(b_ld),
                    _SIZE(b_stride),
                    scalar(0),
                    out_handle,
                    _SIZE(c_offsets[0]),
                    _SIZE(c_ld),
                    _SIZE(c_stride),
                    _SIZE(batch_count),
                    ctypes.byref(queue_handle),
                    None,
                )
            except (ctypes.ArgumentError, OSError) as exc:
                raise RuntimeError(f"{strided_symbol} call failed: {exc}") from exc
            status = int(status)
            if status != _SUCCESS:
                raise RuntimeError(f"{strided_symbol} failed with CLBlast status code {status}")
            return

        for index, (a_offset, b_offset) in enumerate(zip(a_offsets, b_offsets)):
            try:
                status = function(
                    _ROW_MAJOR,
                    _TRANSPOSE if transpose_a else _NO_TRANSPOSE,
                    _TRANSPOSE if transpose_b else _NO_TRANSPOSE,
                    _SIZE(m),
                    _SIZE(n),
                    _SIZE(k),
                    scalar(1),
                    a_handle,
                    _SIZE(a_offset),
                    _SIZE(a_ld),
                    b_handle,
                    _SIZE(b_offset),
                    _SIZE(b_ld),
                    scalar(0),
                    out_handle,
                    _SIZE(c_offsets[index]),
                    _SIZE(c_ld),
                    ctypes.byref(queue_handle),
                    None,  # cl_event* is optional; no event reference is retained.
                )
            except (ctypes.ArgumentError, OSError) as exc:
                raise RuntimeError(f"{symbol} call failed: {exc}") from exc

            status = int(status)
            if status != _SUCCESS:
                raise RuntimeError(f"{symbol} failed with CLBlast status code {status}")
        return
