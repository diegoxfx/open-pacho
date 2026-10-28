"""CPU-only tests for CUDA kernel argument storage and dispatch lifetimes."""

import ctypes
import gc
import sys
import weakref
from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace
from weakref import WeakSet

import numpy as np
import pytest

from openpacho._cuda import CUDABackend, _Allocation
from openpacho._cuda_arguments import KernelArguments


class FakeAllocation:
    def __init__(self, pointer):
        self.ptr = pointer


def _bytes(value):
    return ctypes.string_at(ctypes.addressof(value), ctypes.sizeof(value))


def test_kernel_arguments_packs_native_cuda_abi_widths_and_signed_zero():
    allocation = FakeAllocation(0xFEDCBA9876543210)
    args = (
        np.bool_(True),
        np.float32(-0.0),
        np.float64(-0.0),
        np.uint32(0xFFFFFFFF),
        np.uint64(0xFEDCBA9876543210),
        allocation,
    )
    parameters = KernelArguments(args, FakeAllocation)

    assert parameters.bind(args) == parameters.address
    assert [ctypes.sizeof(value) for value in parameters.values] == [1, 4, 8, 4, 8, 8]
    assert all(
        pointer == ctypes.addressof(value)
        for pointer, value in zip(parameters.pointers, parameters.values)
    )
    assert parameters.address == ctypes.addressof(parameters.pointers)

    expected = [
        np.asarray(args[0]).tobytes(),
        np.asarray(args[1]).tobytes(),
        np.asarray(args[2]).tobytes(),
        np.asarray(args[3]).tobytes(),
        np.asarray(args[4]).tobytes(),
        int(allocation.ptr).to_bytes(8, byteorder=sys.byteorder),
    ]
    assert [_bytes(value) for value in parameters.values] == expected


def test_kernel_arguments_rebinds_values_without_retaining_inputs():
    first_allocation = FakeAllocation(0x1111222233334444)
    first = (
        np.bool_(True),
        np.float32(1.25),
        np.float64(-2.5),
        np.uint32(7),
        np.uint64(2**40),
        first_allocation,
    )
    parameters = KernelArguments(first, FakeAllocation)
    first_addresses = tuple(ctypes.addressof(value) for value in parameters.values)
    first_ref = weakref.ref(first_allocation)
    parameters.bind(first)

    del first, first_allocation
    gc.collect()
    assert first_ref() is None

    second_allocation = FakeAllocation(0xAAAABBBBCCCCDDDD)
    second = (
        np.bool_(False),
        np.float32(-3.5),
        np.float64(4.75),
        np.uint32(0xAABBCCDD),
        np.uint64(2**63 + 9),
        second_allocation,
    )
    assert parameters.bind(second) == parameters.address
    assert tuple(ctypes.addressof(value) for value in parameters.values) == first_addresses
    assert [value.value for value in parameters.values] == [
        0,
        np.float32(-3.5).item(),
        np.float64(4.75).item(),
        0xAABBCCDD,
        2**63 + 9,
        0xAAAABBBBCCCCDDDD,
    ]


def test_kernel_arguments_rejects_changed_count_or_exact_types():
    allocation = FakeAllocation(0x12345678)
    parameters = KernelArguments((np.float32(1), allocation), FakeAllocation)
    parameters.bind((np.float32(1), allocation))
    before = tuple(_bytes(value) for value in parameters.values)

    with pytest.raises(TypeError, match="parameter count"):
        parameters.bind((np.float32(2),))
    assert tuple(_bytes(value) for value in parameters.values) == before

    with pytest.raises(TypeError, match="parameter type"):
        parameters.bind((np.float64(2), allocation))
    assert tuple(_bytes(value) for value in parameters.values) == before

    with pytest.raises(TypeError, match="parameter type"):
        parameters.bind((np.float32(2), np.uint64(3)))

    assert parameters.bind((np.float32(2), allocation)) == parameters.address


def _native_allocation(backend, pointer):
    return _Allocation(backend, pointer, 8)


def test_cuda_run_retries_failed_launch_and_reuses_parameters_under_lock():
    backend = object.__new__(CUDABackend)
    backend._closed = False
    backend._context = "context"
    backend._stream = "stream"
    backend._lock = RLock()
    backend._leases = WeakSet()
    backend._kernels = OrderedDict()
    backend._pending = []
    backend._submitted = 0
    compile_calls, launch_addresses, snapshots, lock_state = [], [], [], []

    def compile_source(source):
        compile_calls.append(source)
        if source not in backend._kernels:
            backend._kernels[source] = ("module", "function", None)
        return backend._kernels[source][1]

    def launch(*arguments):
        launch_addresses.append(arguments[9])
        lock_state.append(backend._lock._is_owned())
        parameters = backend._kernels["source"][2]
        snapshots.append(tuple(_bytes(value) for value in parameters.values))
        return (1,) if len(snapshots) == 1 else (0,)

    backend._compile = compile_source
    backend.driver = SimpleNamespace(
        cuCtxPushCurrent=lambda context: (0,),
        cuCtxPopCurrent=lambda: (0, "context"),
        cuLaunchKernel=launch,
    )
    first = _native_allocation(backend, 0x1111222233334444)
    second = _native_allocation(backend, 0xAAAABBBBCCCCDDDD)
    try:
        with pytest.raises(RuntimeError, match="CUDA call failed"):
            backend._run("source", 1, np.float32(1.25), first)
        assert not backend._pending

        parameters = backend._kernels["source"][2]
        assert parameters is not None
        with pytest.raises(TypeError, match="parameter count"):
            backend._run("source", 1, np.float32(2.0))
        with pytest.raises(TypeError, match="parameter type"):
            backend._run("source", 1, np.float64(2.0), second)
        assert len(snapshots) == 1

        backend._run("source", 1, np.float32(-3.5), second)
        assert len(snapshots) == 2
        assert compile_calls == ["source"] * 4
        assert launch_addresses == [parameters.address, parameters.address]
        assert lock_state == [True, True]
        assert snapshots[0][0] == np.asarray(np.float32(1.25)).tobytes()
        assert snapshots[1][0] == np.asarray(np.float32(-3.5)).tobytes()
        assert snapshots[0][1] == int(first.ptr).to_bytes(8, byteorder=sys.byteorder)
        assert snapshots[1][1] == int(second.ptr).to_bytes(8, byteorder=sys.byteorder)
        assert backend._pending == [second]
    finally:
        backend._pending.clear()
        first.ptr = second.ptr = 0
        backend._closed = True


def test_module_eviction_releases_parameter_storage_after_completion():
    backend = object.__new__(CUDABackend)
    backend._closed = True  # This fake owns no driver resources.
    backend._architecture = (12, 0)
    parameters = KernelArguments((np.float32(1),), FakeAllocation)
    reference = weakref.ref(parameters)
    backend._kernels = OrderedDict(
        (str(i), (f"module-{i}", "function", parameters if i == 0 else None)) for i in range(128)
    )
    del parameters
    events = []
    backend.nvrtc = SimpleNamespace(
        nvrtcCreateProgram=lambda *args: (0, "program"),
        nvrtcCompileProgram=lambda *args: (0,),
        nvrtcGetPTXSize=lambda *args: (0, 4),
        nvrtcGetPTX=lambda *args: (0,),
        nvrtcDestroyProgram=lambda *args: (0,),
    )

    def complete():
        assert reference() is not None
        events.append("complete")

    backend._synchronize = complete
    backend.driver = SimpleNamespace(
        cuModuleUnload=lambda *args: events.append("unload") or (0,),
        cuModuleLoadData=lambda *args: events.append("load") or (0, "new-module"),
        cuModuleGetFunction=lambda *args: (0, "new-function"),
    )
    assert backend._compile("new-source") == "new-function"
    gc.collect()
    assert reference() is None
    assert len(backend._kernels) == 128
    assert events == ["complete", "unload", "load"]
