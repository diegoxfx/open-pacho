"""CPU-only protocol tests; no real backend initialization, compilation, or launch."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import fp64_contraction_kernels as kernels
from openpacho._cuda import CUDABackend
from openpacho._opencl import OpenCLBackend

PTX = b""".version 8.0
.target sm_75
.address_size 64
.visible .entry compute(.param .u64 a) {
 // fma.rn.f64 %d1,%d2,%d3,%d4;
 mul.rn.f64 %d1,%d2,%d3;
 add.rn.f64 %d1,%d1,%d4;
 ret;
}
.func helper() {
 fma.rn.f64 %d1,%d2,%d3,%d4;
}
"""


def array(shape, backend=None, data=None):
    return SimpleNamespace(
        shape=tuple(shape),
        ndim=len(shape),
        dtype=np.dtype("float64"),
        _data=data if data is not None else object(),
        _backend=backend,
    )


class CaptureCUDA:
    name, blas, _register_gemm = "cuda", "custom", False
    matmul, _element_kernel = CUDABackend.matmul, CUDABackend._element_kernel

    def __init__(self):
        self._kernels = OrderedDict([("existing", object())])
        self._lock = RLock()

    def empty(self, shape, dtype):
        return object()

    def _run(self, *args, **kwargs):
        raise AssertionError("Capture must never launch")


class CaptureOpenCL(CaptureCUDA):
    name, _blas, _gemm_tile = "opencl", None, 16
    matmul = OpenCLBackend.matmul


@pytest.mark.parametrize("k", [15, 16, 17, 31, 32, 33])
def test_capture_authoritative_sources_records_padding_and_restores_exact_instance(k):
    a, b = array((8, k)), array((k, 8))
    backend = CaptureOpenCL()
    before = vars(backend).copy()
    native = kernels.capture(backend, a, b, (8, 8), "native")
    assert vars(backend) == before and "_run" not in vars(backend)
    assert native.tile == 16 and native.local_size == 256 and native.shared_bytes == 4096
    assert native.accumulation_steps == ((k + 15) // 16) * 16
    assert native.extra_zero_terms == (-k) % 16
    assert f"start<{k}L" in native.source
    assert "for(int j=0;j<16;++j) acc+=" in native.source
    assert "FP_CONTRACT" not in native.source
    cuda = CaptureCUDA()
    direct = kernels.capture(cuda, a, b, (8, 8), "direct")
    tiled = kernels.capture(cuda, a, b, (8, 8), "tile16")
    assert "__shared__" not in direct.source
    assert "valid_k" in tiled.source and tiled.threads == 256
    assert direct.extra_zero_terms == tiled.extra_zero_terms == 0
    assert direct.accumulation_steps == tiled.accumulation_steps == k


def test_native_capture_restores_override_on_failure_and_checks_three_buffer_abi():
    backend = CaptureOpenCL()
    previous = object()
    backend._run = previous
    a, b = array((8, 17)), array((17, 8))

    def bad(a, b, shape, dtype):
        out = object()
        backend._run("source", 1, a._data, b._data, object())
        return out

    backend.matmul = bad
    with pytest.raises(AssertionError, match="ABI"):
        kernels.capture(backend, a, b, (8, 8), "native")
    assert backend._run is previous

    def broken(*args):
        raise RuntimeError("capture failure")

    backend.matmul = broken
    with pytest.raises(RuntimeError, match="capture failure"):
        kernels.capture(backend, a, b, (8, 8), "native")
    assert backend._run is previous


def test_ptx_counts_only_compute_entry_without_comments_or_helpers():
    text = PTX.decode().replace(" ret;", " {\n fma.rn.f64 %d1,%d2,%d3,%d4;\n }\n ret;")
    text = "/* .entry compute() { fma.rn.f64 x,y,z,w; } */\n" + text
    summary = kernels.entry_ptx_summary(text)
    assert (summary["fma_f64"], summary["mul_f64"], summary["add_f64"]) == (1, 1, 1)
    assert kernels.binary_evidence(b"opaque compiler blob")["f64_instruction_evidence"] is None
    assert "unknown" in kernels.binary_evidence(b"blob")["contraction_observation"]
    with pytest.raises(ValueError, match="exactly one"):
        kernels.entry_ptx_summary(".entry other() { ret; }")
    with pytest.raises(ValueError, match="unbalanced"):
        kernels.entry_ptx_summary(".entry compute() { ret;")


class FakeNVRTC:
    def __init__(self):
        self.options, self.destroyed = [], 0

    def nvrtcCreateProgram(self, *args):
        return 0, object()

    def nvrtcCompileProgram(self, program, count, options):
        self.options.append(options)
        return (0,)

    def nvrtcGetProgramLogSize(self, program):
        return 0, 1

    def nvrtcGetProgramLog(self, program, data):
        data[:] = b"\0"
        return (0,)

    def nvrtcGetPTXSize(self, program):
        return 0, len(PTX)

    def nvrtcGetPTX(self, program, data):
        data[:] = PTX
        return (0,)

    def nvrtcDestroyProgram(self, program):
        self.destroyed += 1
        return (0,)

    def nvrtcVersion(self):
        return 0, 13, 3


class FakeDriver:
    CUfunction_attribute = SimpleNamespace(
        **{
            "CU_FUNC_ATTRIBUTE_" + name: name
            for name in (
                "MAX_THREADS_PER_BLOCK",
                "NUM_REGS",
                "SHARED_SIZE_BYTES",
                "LOCAL_SIZE_BYTES",
            )
        }
    )
    CUdevice_attribute = SimpleNamespace(
        **{
            "CU_DEVICE_ATTRIBUTE_" + name: name
            for name in ("MAX_THREADS_PER_BLOCK", "MAX_SHARED_MEMORY_PER_BLOCK", "MAX_GRID_DIM_X")
        }
    )

    def __init__(self, backend):
        self.backend = backend
        self.events, self.modules = [], []
        self.max_function_threads = 1024

    def cuDeviceGetAttribute(self, name, device):
        assert self.backend.active
        return 0, {
            "MAX_THREADS_PER_BLOCK": 1024,
            "MAX_SHARED_MEMORY_PER_BLOCK": 49152,
            "MAX_GRID_DIM_X": 2**31 - 1,
        }[name]

    def cuModuleLoadData(self, data):
        assert self.backend.active
        module = object()
        self.modules.append(module)
        self.events.append("load")
        return 0, module

    def cuModuleGetFunction(self, module, name):
        return 0, object()

    def cuFuncGetAttribute(self, name, function):
        return 0, self.max_function_threads if name == "MAX_THREADS_PER_BLOCK" else 0

    def cuModuleUnload(self, module):
        assert self.backend.active
        self.events.append("unload")
        self.modules.remove(module)
        return (0,)

    def cuLaunchKernel(self, *args):
        assert self.backend.active
        self.events.append(("launch", args))
        return (0,)

    def cuDriverGetVersion(self):
        return 0, 13030


class OwnedCUDA(CaptureCUDA):
    device, _architecture, _device, _stream = "fake GPU", (7, 5), 0, "owned stream"

    def __init__(self):
        super().__init__()
        self.active = False
        self.driver, self.nvrtc = FakeDriver(self), FakeNVRTC()
        self.retained = []

    @contextmanager
    def _activate(self):
        previous, self.active = self.active, True
        try:
            yield
        finally:
            self.active = previous

    def synchronize(self):
        self.driver.events.append("sync")

    def _retain(self, args):
        self.retained.append(args)


def test_cuda_owns_independent_modules_changes_only_fmad_and_binds_stream(tmp_path, monkeypatch):
    backend = OwnedCUDA()
    a, b = array((8, 17)), array((17, 8))
    descriptor = kernels.capture(backend, a, b, (8, 8), "tile16")
    before = list(backend._kernels.items())
    separate = kernels.prepare(backend, descriptor, "separate", tmp_path / "separate")
    contracted = kernels.prepare(backend, descriptor, "contracted", tmp_path / "contracted")
    assert separate.module is not contracted.module
    assert backend.nvrtc.options[0][2] == b"--fmad=false"
    assert backend.nvrtc.options[1][2] == b"--fmad=true"
    assert backend.nvrtc.options[0][:2] + backend.nvrtc.options[0][3:] == (
        backend.nvrtc.options[1][:2] + backend.nvrtc.options[1][3:]
    )
    assert backend.nvrtc.destroyed == 2 and not backend.active
    assert separate.metadata["f64_instruction_evidence"]["fma_f64"] == 0
    assert (tmp_path / "separate" / "source.txt").read_text() == descriptor.source

    class Allocation:
        def __init__(self, ptr):
            self.ptr = ptr

    monkeypatch.setattr(kernels, "_Allocation", Allocation)
    a = array((8, 17), backend, Allocation(1))
    b = array((17, 8), backend, Allocation(2))
    out = array((8, 8), backend, Allocation(3))
    assert separate.call(a, b, out) is out
    launch = backend.driver.events[-1][1]
    assert launch[0] is separate.function and launch[8] == backend._stream
    assert launch[1:7] == (1, 1, 1, 256, 1, 1)
    assert backend.retained == [(a._data, b._data, out._data)]
    separate.close()
    separate.close()
    contracted.close()
    assert backend.driver.events[-4:] == ["sync", "unload", "sync", "unload"]
    assert not backend.driver.modules and list(backend._kernels.items()) == before
    with pytest.raises(RuntimeError, match="closed"):
        separate.call(a, b, out)
    with pytest.raises(FileExistsError):
        kernels.prepare(backend, descriptor, "separate", tmp_path / "separate")


def test_cuda_prepare_failure_releases_module_and_sync_failure_never_unloads(tmp_path):
    backend = OwnedCUDA()
    descriptor = kernels.capture(backend, array((8, 17)), array((17, 8)), (8, 8), "tile16")
    backend.driver.max_function_threads = 128
    with pytest.raises(ValueError, match="actual CUDA function"):
        kernels.prepare(backend, descriptor, "separate", tmp_path / "failed")
    assert not backend.driver.modules and backend.driver.events[-2:] == ["sync", "unload"]
    backend.driver.max_function_threads = 1024
    owner = kernels.prepare(backend, descriptor, "separate", tmp_path / "valid")

    def failure():
        raise RuntimeError("wait failed")

    backend.synchronize = failure
    with pytest.raises(RuntimeError, match="wait failed"):
        owner.close()
    assert owner.module is not None and not owner.closed
    del backend.synchronize
    owner.close()


class FakeProgram:
    def __init__(self, backend, source):
        self.backend, self.source, self.binaries = backend, source, [PTX]
        backend.programs.append(self)

    def build(self, *, options, devices):
        self.options, self.devices = options, devices
        if self.backend.fail_build:
            raise RuntimeError("build failed")
        return self

    def get_build_info(self, device, name):
        return "-cl-std=CL1.2 -DENV=1" if name == "OPTIONS" else "build log"

    def get_info(self, name):
        return self.devices

    def release(self):
        self.backend.events.append("program release")


class FakeKernel:
    def __init__(self, backend, program):
        self.backend, self.program = backend, program

    def get_work_group_info(self, name, device):
        return {
            "WORK_GROUP_SIZE": self.backend.function_threads,
            "LOCAL_MEM_SIZE": 4096,
            "PRIVATE_MEM_SIZE": 0,
        }[name]

    def __call__(self, *args):
        self.backend.events.append(("enqueue", args))

    def release(self):
        self.backend.events.append("kernel release")


class OwnedOpenCL(CaptureOpenCL):
    fp64, device = True, "fake GPU"

    def __init__(self):
        super().__init__()
        self.programs, self.events, self.fail_build, self.function_threads = [], [], False, 256
        device = SimpleNamespace(
            max_work_group_size=256,
            max_work_item_sizes=(256, 256, 64),
            local_mem_size=4096,
            address_bits=64,
            version="OpenCL 3.0",
            driver_version="1",
            opencl_c_version="OpenCL C 1.2",
            platform=SimpleNamespace(version="OpenCL 3.0"),
        )
        self.context, self.queue = object(), SimpleNamespace(device=device)
        self.cl = SimpleNamespace(
            Program=lambda context, source: FakeProgram(self, source),
            Kernel=lambda program, name: FakeKernel(self, program),
            kernel_work_group_info=SimpleNamespace(
                **{x: x for x in ("WORK_GROUP_SIZE", "LOCAL_MEM_SIZE", "PRIVATE_MEM_SIZE")}
            ),
            program_build_info=SimpleNamespace(OPTIONS="OPTIONS", LOG="LOG"),
            program_info=SimpleNamespace(DEVICES="DEVICES"),
        )

    def synchronize(self):
        self.events.append("sync")


def test_opencl_pair_only_pragma_differs_uses_owned_objects_and_effective_options(
    tmp_path, monkeypatch
):
    backend = OwnedOpenCL()
    descriptor = kernels.capture(backend, array((8, 17)), array((17, 8)), (8, 8), "native")
    before = list(backend._kernels.items())
    monkeypatch.setenv("PYOPENCL_BUILD_OPTIONS", "-DENV=1")
    native = kernels.prepare(backend, descriptor, "native", tmp_path / "native")
    separate = kernels.prepare(backend, descriptor, "no_contraction", tmp_path / "separate")
    assert native.program is not separate.program and native.kernel is not separate.kernel
    assert (
        separate.program.source.replace("#pragma OPENCL FP_CONTRACT OFF\n", "")
        == native.program.source
    )
    assert native.program.source.startswith("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
    assert native.program.options == separate.program.options == ["-cl-std=CL1.2"]
    assert native.metadata["effective_build_options"] == "-cl-std=CL1.2 -DENV=1"
    assert native.metadata["build_environment"]["PYOPENCL_BUILD_OPTIONS"] == "-DENV=1"
    a, b, out = (array(s, backend) for s in ((8, 17), (17, 8), (8, 8)))
    native.call(a, b, out)
    assert backend.events[-1] == (
        "enqueue",
        (backend.queue, (256,), (256,), a._data, b._data, out._data),
    )
    native.close()
    separate.close()
    assert backend.events[-6:] == ["sync", "kernel release", "program release"] * 2
    assert list(backend._kernels.items()) == before


def test_opencl_build_and_capability_failures_cleanup(tmp_path):
    backend = OwnedOpenCL()
    descriptor = kernels.capture(backend, array((8, 17)), array((17, 8)), (8, 8), "native")
    backend.fail_build = True
    with pytest.raises(RuntimeError, match="build failed"):
        kernels.prepare(backend, descriptor, "native", tmp_path / "failed")
    assert backend.events == ["sync", "program release"]
    backend.fail_build, backend.function_threads = False, 128
    with pytest.raises(ValueError, match="actual OpenCL kernel"):
        kernels.prepare(backend, descriptor, "native", tmp_path / "small-function")
    assert backend.events[-3:] == ["sync", "kernel release", "program release"]
    backend.queue.device.local_mem_size = 1024
    with pytest.raises(ValueError, match="4096"):
        kernels.prepare(backend, descriptor, "native", tmp_path / "small-device")
    with pytest.raises(ValueError, match="global work size"):
        backend.queue.device.local_mem_size = 4096
        kernels.prepare(backend, replace(descriptor, count=2**64), "native", tmp_path / "huge")
