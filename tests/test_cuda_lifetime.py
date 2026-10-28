"""CPU controls for CUDA ordering contracts that a synchronous fake can conceal."""

import ctypes
import sys
from collections import OrderedDict
from threading import RLock
from types import ModuleType, SimpleNamespace
from weakref import WeakSet

import numpy as np
import pytest

from openpacho._cuda import CUDABackend
from openpacho.exceptions import BackendUnavailableError


@pytest.mark.parametrize(
    "host",
    [np.arange(257, dtype=np.float32)[::2], np.asarray(4.5, dtype=np.float64)],
    ids=("noncontiguous", "scalar"),
)
def test_upload_finishes_default_stream_dma_before_publishing_allocation(host):
    backend = object.__new__(CUDABackend)
    backend._lock = RLock()
    backend._closed = False
    backend._context = "primary-context"
    backend._stream = "nonblocking-compute-stream"
    events, staged, completed = [], {}, {}

    def push(context):
        events.append(("push", context))
        return (0,)

    def pop():
        events.append(("pop",))
        return (0, "primary-context")

    def allocate(shape, dtype):
        events.append(("allocate", shape, dtype))
        return SimpleNamespace(ptr=123)

    def copy(destination, source, size):
        # Pageable HtoD may return once host bytes reach driver staging memory.
        # Device data becomes ready only after the implicit stream completes.
        events.append(("copy", destination, size))
        staged[destination] = ctypes.string_at(source, size)
        return (0,)

    def synchronize(stream):
        events.append(("synchronize", stream))
        if stream == 0:
            completed.update(staged)
            staged.clear()
        return (0,)

    backend.empty = allocate
    backend.driver = SimpleNamespace(
        cuCtxPushCurrent=push,
        cuCtxPopCurrent=pop,
        cuMemcpyHtoD=copy,
        cuStreamSynchronize=synchronize,
    )
    try:
        allocation = backend.upload(host)
        assert completed.get(allocation.ptr) == host.tobytes()
        assert not staged
        assert [event[0] for event in events] == [
            "push",
            "allocate",
            "copy",
            "synchronize",
            "pop",
        ]
        assert events[-2] == ("synchronize", 0)
    finally:
        backend._closed = True  # No real native resources belong to this fake.


def test_partial_cache_release_never_reuses_successfully_freed_pointers():
    backend = object.__new__(CUDABackend)
    backend._lock = RLock()
    backend._closed = False
    backend._context, backend._stream = "primary", "stream"
    backend._pending, backend._submitted, backend._leases = [], 0, WeakSet()
    backend._pool = OrderedDict([(4, [101, 102]), (8, [201])])
    backend._pool_items, backend._cached_bytes = 3, 16
    backend.fp64 = True
    attempts, freed, allocated = [], set(), []

    def free(pointer):
        attempts.append(pointer)
        if len(attempts) == 2:
            return (1,)
        assert pointer not in freed, "A completed free was attempted twice"
        freed.add(pointer)
        return (0,)

    def allocate(size):
        allocated.append(300 + len(allocated))
        return (0, allocated[-1])

    backend.driver = SimpleNamespace(
        cuCtxPushCurrent=lambda context: (0,),
        cuCtxPopCurrent=lambda: (0, "primary"),
        cuStreamSynchronize=lambda stream: (0,),
        cuMemFree=free,
        cuMemAlloc=allocate,
    )
    try:
        with pytest.raises(RuntimeError, match="CUDA call failed"):
            backend.release_cached_memory()
        cached = [pointer for entries in backend._pool.values() for pointer in entries]
        assert set(cached).isdisjoint(freed)
        assert backend._pool_items == len(cached) == 2
        assert backend._cached_bytes == sum(
            capacity * len(entries) for capacity, entries in backend._pool.items()
        )
        first = backend.empty((1,), np.dtype("float32"))
        second = backend.empty((1,), np.dtype("float32"))
        assert {first.ptr, second.ptr}.isdisjoint(freed)
        backend.release_cached_memory()
        assert not backend._pool
        assert backend._pool_items == backend._cached_bytes == 0
    finally:
        backend._closed = True


@pytest.mark.parametrize("failure", ["synchronize", "unload"])
def test_failed_module_eviction_does_not_load_untracked_module(failure):
    backend = object.__new__(CUDABackend)
    backend._closed = True  # This fake calls only _compile, under no real CUDA context.
    backend._architecture = (12, 0)
    backend._kernels = OrderedDict((str(i), (f"old-{i}", "fn")) for i in range(128))
    original = list(backend._kernels.items())
    events = []
    backend.nvrtc = SimpleNamespace(
        nvrtcCreateProgram=lambda *args: (0, "program"),
        nvrtcCompileProgram=lambda *args: (0,),
        nvrtcGetPTXSize=lambda *args: (0, 4),
        nvrtcGetPTX=lambda *args: (0,),
        nvrtcDestroyProgram=lambda *args: (0,),
    )

    def synchronize():
        events.append(("synchronize",))
        if failure == "synchronize":
            raise RuntimeError("Injected synchronization failure")

    def unload(module):
        events.append(("unload", module))
        return (1,) if failure == "unload" else (0,)

    backend._synchronize = synchronize
    backend.driver = SimpleNamespace(
        cuModuleLoadData=lambda *args: events.append(("load", "new")) or (0, "new"),
        cuModuleGetFunction=lambda *args: (0, "new-function"),
        cuModuleUnload=unload,
    )
    with pytest.raises(RuntimeError):
        backend._compile("new-source")
    assert list(backend._kernels.items()) == original
    assert not any(event[0] == "load" for event in events)
    assert events[0] == ("synchronize",)
    if failure == "unload":
        assert events[1] == ("unload", "old-0")


@pytest.mark.parametrize("failure", ["stream", "attributes"])
def test_constructor_failure_releases_owned_context_and_restores_caller(monkeypatch, failure):
    from openpacho import _cuda

    stack, events = [99], []

    def retain(device):
        events.append(("retain", device))
        return (0, 47)

    def release(device):
        events.append(("release", device))
        return (0,)

    def push(context):
        stack.append(context)
        return (0,)

    def pop():
        return (0, stack.pop())

    driver = SimpleNamespace(
        cuDeviceGet=lambda ordinal: (0, ordinal),
        cuDevicePrimaryCtxRetain=retain,
        cuDevicePrimaryCtxRelease=release,
        cuCtxPushCurrent=push,
        cuCtxPopCurrent=pop,
        cuCtxGetCurrent=lambda: (0, stack[-1]),
        cuStreamCreate=lambda flags: (1,) if failure == "stream" else (0, 789),
        cuDeviceGetAttribute=lambda *args: (1,),
        cuStreamSynchronize=lambda stream: (0,),
        cuStreamDestroy=lambda stream: events.append(("destroy-stream", stream)) or (0,),
        CUstream_flags=SimpleNamespace(CU_STREAM_NON_BLOCKING=1),
        CUdevice_attribute=SimpleNamespace(
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=0,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR=1,
        ),
    )
    cuda, bindings, nvrtc = (ModuleType(name) for name in ("cuda", "cuda.bindings", "nvrtc"))
    cuda.bindings, bindings.nvrtc = bindings, nvrtc
    nvrtc.nvrtcVersion = lambda: (0, 13, 0)
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.nvrtc", nvrtc)
    monkeypatch.setattr(_cuda, "_driver", lambda: driver)
    monkeypatch.setattr(
        _cuda, "cuda_devices", lambda: [SimpleNamespace(id="0", name="fake", fp64=True)]
    )
    with pytest.raises(BackendUnavailableError, match="CUDA initialization failed"):
        CUDABackend(blas="custom")
    assert stack == [99]
    assert events.count(("retain", 0)) == events.count(("release", 0)) == 1
    assert events.count(("destroy-stream", 789)) == (failure == "attributes")
