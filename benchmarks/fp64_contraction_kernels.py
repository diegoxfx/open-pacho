"""Owned benchmark kernels for an isolated FP64 contraction experiment.

Import and capture do not compile or launch. Preparation compiles independent
objects and never installs them in a runtime cache. PTX evidence describes
compiler IR, not the driver's loaded machine code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.cuda_fp64_gemm import capture_current, device_limits, override
from benchmarks.cuda_fp64_tile import source_for
from benchmarks.kernel_inspection import compile_cuda, ptx_summary
from openpacho._cuda import KernelArguments, _Allocation, _check


def _sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _cache_snapshot(backend):
    cache = backend._kernels
    return id(cache), tuple((key, id(value)) for key, value in cache.items())


def _assert_cache(backend, snapshot):
    if _cache_snapshot(backend) != snapshot:
        raise AssertionError("The runtime kernel cache changed during owned-kernel work")


def _release_opencl(value):
    # Current PyOpenCL Program/Kernel wrappers use reference-counted destruction
    # and expose no release method. Support wrappers which do expose one, too.
    release = getattr(value, "release", None)
    if release is not None:
        release()


@dataclass(frozen=True)
class Descriptor:
    backend: str
    geometry: str
    source: str
    count: int
    threads: int | None
    local_size: int | None
    shared_bytes: int
    shape: tuple
    ashape: tuple
    bshape: tuple
    accumulation_steps: int
    extra_zero_terms: int
    tile: int | None = None

    def metadata(self):
        result = asdict(self)
        result.pop("source")
        result["source_sha256"] = _sha(self.source)
        result["count_unit"] = "blocks" if self.backend == "cuda" else "global work items"
        return result


def _validate_inputs(a, b, shape):
    if any(np.dtype(x.dtype) != np.dtype("float64") for x in (a, b)):
        raise ValueError("The experiment requires float64 operands")
    if min(a.ndim, b.ndim) < 2 or not all((*a.shape, *b.shape)):
        raise ValueError("The experiment requires nonempty matrices, rank >= 2")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("Incompatible inner dimensions")
    expected = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
    if tuple(shape) != expected:
        raise ValueError("Output shape differs from broadcast matrix product")


def capture(backend, a, b, shape, geometry):
    """Capture authoritative source without launching; native capture allocates output."""
    _validate_inputs(a, b, shape)
    if getattr(backend, "blas", "custom") != "custom" or getattr(backend, "_blas", None):
        raise ValueError("An explicit custom backend is required")
    snapshot = _cache_snapshot(backend)
    k, tile = a.shape[-1], None
    with getattr(backend, "_lock", nullcontext()):
        try:
            if backend.name == "cuda" and geometry in ("direct", "tile16"):
                kernel = (
                    capture_current(backend, backend.matmul, a, b, shape, np.dtype("float64"))
                    if geometry == "direct"
                    else source_for(a.shape, b.shape)
                )
                if geometry == "tile16":
                    tile = 16
                result = Descriptor(
                    "cuda",
                    geometry,
                    kernel.source,
                    kernel.blocks,
                    kernel.threads,
                    None,
                    kernel.shared_bytes,
                    tuple(shape),
                    a.shape,
                    b.shape,
                    k,
                    0,
                    tile,
                )
            elif backend.name == "opencl" and geometry == "native":
                launches = []

                def record(source, count, *args, local_size=None):
                    launches.append((source, count, args, local_size))

                with override(backend, "_run", record):
                    out = backend.matmul(a, b, shape, np.dtype("float64"))
                if len(launches) != 1:
                    raise AssertionError(f"Expected one native OpenCL launch, got {len(launches)}")
                source, count, args, local = launches[0]
                if len(args) != 3 or not all(x is y for x, y in zip(args, (a._data, b._data, out))):
                    raise AssertionError("Captured matmul changed its three-buffer ABI")
                if "__local double left[" in source:
                    tile = int(backend._gemm_tile)
                    if tile not in (8, 16) or local != tile * tile:
                        raise AssertionError("Unrecognized native OpenCL tile geometry")
                    steps = ((k + tile - 1) // tile) * tile
                    shared = tile * tile * 8 * 2
                else:
                    if "__local" in source or "acc +=" not in source:
                        raise AssertionError("Unrecognized native OpenCL accumulation")
                    steps, shared = k, 0
                result = Descriptor(
                    "opencl",
                    geometry,
                    source,
                    int(count),
                    None,
                    local,
                    shared,
                    tuple(shape),
                    a.shape,
                    b.shape,
                    steps,
                    steps - k,
                    tile,
                )
            else:
                raise ValueError("Supported geometries: CUDA direct/tile16; OpenCL native")
            return result
        finally:
            _assert_cache(backend, snapshot)


def entry_ptx_summary(text, entry="compute"):
    """Count f64 operations only inside the named PTX entry, ignoring comments."""
    clean = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    clean = re.sub(r"//[^\r\n]*", "", clean)
    matches = list(re.finditer(r"\.entry\s+" + re.escape(entry) + r"\s*\(", clean))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one PTX entry named {entry}")
    start = clean.find("{", matches[0].end())
    if start < 0:
        raise ValueError("PTX entry has no body")
    depth, end = 1, start + 1
    while depth and end < len(clean):
        depth += (clean[end] == "{") - (clean[end] == "}")
        end += 1
    if depth:
        raise ValueError("PTX entry body is unbalanced")
    summary = ptx_summary(clean[start + 1 : end - 1])
    ops = summary["static_ir_opcodes"]
    summary.update(
        {
            name + "_f64": sum(
                n for op, n in ops.items() if op.startswith(name + ".") and op.endswith(".f64")
            )
            for name in ("fma", "mul", "add")
        }
    )
    summary.update(entry=entry, scope="static PTX compute-entry IR; not loaded machine code")
    return summary


def binary_evidence(data):
    text = data.decode("utf-8", errors="replace")
    is_ptx = bool(
        re.search(r"(?m)^\s*\.version\s+", text) and re.search(r"\.entry\s+compute\s*\(", text)
    )
    return {
        "is_ptx": is_ptx,
        "f64_instruction_evidence": entry_ptx_summary(text) if is_ptx else None,
        "contraction_observation": "inspect f64 instruction evidence"
        if is_ptx
        else "unknown: opaque compiler binary",
    }


def _cuda_limits(backend, descriptor):
    # The paired probe deliberately requires the tile's capability even for direct.
    with backend._activate():
        limits = device_limits(backend)
    if not 0 < descriptor.count <= limits["grid_x"]:
        raise ValueError("Kernel block count exceeds CUDA grid-x capability")
    if not 0 < descriptor.threads <= limits["threads"]:
        raise ValueError("Kernel threads exceed CUDA capability")
    if descriptor.shared_bytes > limits["shared_bytes"]:
        raise ValueError("Kernel static shared storage exceeds CUDA capability")
    return limits


def _opencl_limits(backend, descriptor):
    device = backend.queue.device
    limits = {
        "threads": int(device.max_work_group_size),
        "work_item_x": int(device.max_work_item_sizes[0]),
        "shared_bytes": int(device.local_mem_size),
        "address_bits": int(device.address_bits),
    }
    if min(limits["threads"], limits["work_item_x"]) < 256 or limits["shared_bytes"] < 4096:
        raise ValueError("Probe requires 256 work items/group and 4096 local bytes")
    if descriptor.local_size and (
        descriptor.local_size > min(limits["threads"], limits["work_item_x"])
        or descriptor.count % descriptor.local_size
    ):
        raise ValueError("Invalid OpenCL local/global geometry")
    if descriptor.count <= 0 or descriptor.count > 2 ** limits["address_bits"] - 1:
        raise ValueError("OpenCL global work size exceeds address capability")
    if descriptor.shared_bytes > limits["shared_bytes"]:
        raise ValueError("OpenCL local storage exceeds device capability")
    return limits


class Prepared:
    """One privately owned executable. Caller must close before closing its backend."""

    def __init__(self, backend, descriptor, variant, folder):
        self.backend, self.descriptor = backend, descriptor
        self.variant, self.folder = variant, Path(folder)
        self.module = self.function = self.program = self.kernel = self.parameters = None
        self.closed = False
        self.snapshot = _cache_snapshot(backend)
        self.metadata = {
            "descriptor": descriptor.metadata(),
            "variant": variant,
            "evidence_scope": "compiler IR/program binary; not loaded machine code",
            "runtime_cache_unchanged": True,
        }

    def _save(self):
        (self.folder / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def call(self, a, b, out):
        if self.closed:
            raise RuntimeError("Prepared kernel is closed")
        d = self.descriptor
        _validate_inputs(a, b, d.shape)
        if tuple(a.shape) != d.ashape or tuple(b.shape) != d.bshape:
            raise ValueError("Prepared operand shape changed")
        if tuple(out.shape) != d.shape or np.dtype(out.dtype) != np.dtype("float64"):
            raise ValueError("Prepared output shape/dtype changed")
        if any(getattr(x, "_backend", self.backend) is not self.backend for x in (a, b, out)):
            raise ValueError("Prepared buffers must belong to the owning backend")
        args = (a._data, b._data, out._data)
        with getattr(self.backend, "_lock", nullcontext()):
            if d.backend == "cuda":
                with self.backend._activate():
                    if self.parameters is None:
                        self.parameters = KernelArguments(args, _Allocation)
                    pointers = self.parameters.bind(args)
                    _check(
                        self.backend.driver.cuLaunchKernel(
                            self.function,
                            d.count,
                            1,
                            1,
                            d.threads,
                            1,
                            1,
                            0,
                            self.backend._stream,
                            pointers,
                            0,
                        )
                    )
                    self.backend._retain(args)
            else:
                self.kernel(
                    self.backend.queue, (d.count,), (d.local_size,) if d.local_size else None, *args
                )
        return out

    def close(self):
        if self.closed:
            return
        with getattr(self.backend, "_lock", nullcontext()):
            # Do not unload/release after an unsuccessful wait: queued work may live.
            self.backend.synchronize()
            if self.module is not None:
                with self.backend._activate():
                    _check(self.backend.driver.cuModuleUnload(self.module))
                self.module = self.function = None
            if self.kernel is not None:
                _release_opencl(self.kernel)
                self.kernel = None
            if self.program is not None:
                _release_opencl(self.program)
                self.program = None
            self.parameters = None
            self.closed = True
            _assert_cache(self.backend, self.snapshot)
            self.metadata["closed"] = True
            self._save()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def prepare(backend, descriptor, variant, folder):
    """Compile one isolated variant, save source/binary/options, return its owner."""
    choices = {"cuda": ("separate", "contracted"), "opencl": ("native", "no_contraction")}
    if backend.name != descriptor.backend or variant not in choices.get(backend.name, ()):
        raise ValueError("Variant/backend mismatch")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    owner = Prepared(backend, descriptor, variant, folder)
    source = descriptor.source
    try:
        if backend.name == "cuda":
            limits = _cuda_limits(backend, descriptor)
            proxy = SimpleNamespace(
                nvrtc=backend.nvrtc,
                driver=backend.driver,
                _architecture=backend._architecture,
                _activate=backend._activate,
                _kernels={},
            )
            try:
                data, details = compile_cuda(proxy, source, contraction=variant == "contracted")
            finally:
                # compile_cuda writes only this cache, including when wrapped by a
                # diagnostic that raises after compilation. Transfer ownership now.
                if source in proxy._kernels:
                    owner.module, owner.function, _ = proxy._kernels.pop(source)
            attrs = details["actual_function_attributes"]
            if descriptor.threads > attrs["MAX_THREADS_PER_BLOCK"]:
                raise ValueError("Kernel threads exceed actual CUDA function capability")
            if attrs["SHARED_SIZE_BYTES"] > limits["shared_bytes"]:
                raise ValueError("Actual CUDA static shared size exceeds capability")
            with backend._activate():
                versions = {
                    "nvrtc": _check(backend.nvrtc.nvrtcVersion()),
                    "driver": _check(backend.driver.cuDriverGetVersion()),
                }
            owner.metadata.update(
                details,
                versions=versions,
                architecture=backend._architecture,
                effective_build_options=details["options"],
            )
            binary_name = "program.ptx"
        else:
            limits = _opencl_limits(backend, descriptor)
            prefix = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" if backend.fp64 else ""
            source = (
                prefix
                + ("#pragma OPENCL FP_CONTRACT OFF\n" if variant == "no_contraction" else "")
                + source
            )
            options = ["-cl-std=CL1.2"]
            # Retain even an unbuilt program so a build failure can release it.
            owner.program = backend.cl.Program(backend.context, source)
            owner.program.build(options=options, devices=[backend.queue.device])
            owner.kernel = backend.cl.Kernel(owner.program, "compute")
            device, info = backend.queue.device, backend.cl.kernel_work_group_info
            attrs = {
                name: int(owner.kernel.get_work_group_info(getattr(info, name), device))
                for name in ("WORK_GROUP_SIZE", "LOCAL_MEM_SIZE", "PRIVATE_MEM_SIZE")
            }
            if descriptor.local_size and descriptor.local_size > attrs["WORK_GROUP_SIZE"]:
                raise ValueError("Local size exceeds actual OpenCL kernel capability")
            if attrs["LOCAL_MEM_SIZE"] > limits["shared_bytes"]:
                raise ValueError("Actual OpenCL local size exceeds device capability")
            build = backend.cl.program_build_info
            actual_options = owner.program.get_build_info(device, build.OPTIONS)
            log = owner.program.get_build_info(device, build.LOG)
            devices = list(owner.program.get_info(backend.cl.program_info.DEVICES))
            data = bytes(owner.program.binaries[devices.index(device)])
            owner.metadata.update(
                options=options,
                effective_build_options=actual_options,
                log=log,
                build_environment={
                    key: os.environ.get(key)
                    for key in (
                        "PYOPENCL_BUILD_OPTIONS",
                        "PYOPENCL_NO_CACHE",
                        "POCL_EXTRA_BUILD_FLAGS",
                    )
                },
                actual_function_attributes=attrs,
                versions={
                    "device": device.version,
                    "driver": device.driver_version,
                    "opencl_c": device.opencl_c_version,
                    "platform": device.platform.version,
                },
            )
            binary_name = "program.bin"
        evidence = binary_evidence(data)
        if backend.name == "cuda" and not evidence["is_ptx"]:
            raise ValueError("NVRTC output did not contain the expected PTX entry")
        (folder / "source.txt").write_text(source, encoding="utf-8")
        (folder / binary_name).write_bytes(data)
        if backend.name == "opencl" and evidence["is_ptx"]:
            (folder / "program.ptx").write_bytes(data)
        _assert_cache(backend, owner.snapshot)
        owner.metadata.update(
            evidence,
            source_sha256=_sha(source),
            captured_source_sha256=_sha(descriptor.source),
            options_sha256=_sha(json.dumps(owner.metadata["options"], separators=(",", ":"))),
            effective_options_sha256=_sha(
                json.dumps(owner.metadata["effective_build_options"], separators=(",", ":"))
            ),
            binary_sha256=_sha(data),
            binary_file=binary_name,
            device=str(backend.device),
            device_limits=limits,
        )
        owner._save()
        return owner
    except BaseException:
        owner.close()
        raise
