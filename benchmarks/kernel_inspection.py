"""GEMM compiler ablations and saved IR; diagnostic only, no runtime policy changes."""

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import measure, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.suite import source_hashes
from openpacho._cuda import _check
from openpacho._gemm import _source


def ptx_summary(text):
    """Static IR counts, never dynamic executed instructions or physical registers."""
    # NVRTC/OpenCL PTX can contain either comment style. Strip both before
    # inspecting declarations as well as instructions, so commented examples
    # cannot become evidence of executed code or allocated virtual registers.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\r\n]*", "", text)
    instructions = Counter()
    predicated = branches = 0
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"(?:[\w$]+:\s*)?(@!?%[\w$.]+\s+)?([a-z][a-z0-9_.]*)(?:\s+[^;]*)?;", line)
        if match:
            predicate, opcode = match.groups()
            instructions[opcode] += 1
            predicated += bool(predicate)
            branches += opcode.split(".")[0] == "bra"
    return {
        "static_ir_opcodes": dict(instructions),
        "predicated_ir_instructions": predicated,
        "ir_branches": branches,
        "float64_conversions": sum(
            n for op, n in instructions.items() if op.startswith("cvt.") and ".f64" in op
        ),
        "fma_f32": sum(
            n for op, n in instructions.items() if op.startswith("fma.") and op.endswith(".f32")
        ),
        "virtual_register_declarations": re.findall(r"\.reg\s+(\.[\w]+)\s+%(\w+)<(\d+)>", text),
    }


def compile_cuda(backend, source, *, contraction=False):
    nvrtc = backend.nvrtc
    major, minor = backend._architecture
    options = [
        f"--gpu-architecture=compute_{major}{minor}".encode(),
        b"--std=c++11",
        b"--fmad=true" if contraction else b"--fmad=false",
        b"--ftz=false",
        b"--prec-div=true",
        b"--prec-sqrt=true",
    ]
    program = _check(nvrtc.nvrtcCreateProgram(source.encode(), b"inspect.cu", 0, [], []))
    try:
        (status,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
        log = bytearray(_check(nvrtc.nvrtcGetProgramLogSize(program)))
        _check(nvrtc.nvrtcGetProgramLog(program, log))
        if int(status):
            raise RuntimeError(log.decode(errors="replace"))
        data = bytearray(_check(nvrtc.nvrtcGetPTXSize(program)))
        _check(nvrtc.nvrtcGetPTX(program, data))
    finally:
        _check(nvrtc.nvrtcDestroyProgram(program))
    with backend._activate():
        module = _check(backend.driver.cuModuleLoadData(bytes(data)))
        try:
            function = _check(backend.driver.cuModuleGetFunction(module, b"compute"))
            enum = backend.driver.CUfunction_attribute
            attrs = {
                key: _check(
                    backend.driver.cuFuncGetAttribute(
                        getattr(enum, "CU_FUNC_ATTRIBUTE_" + key), function
                    )
                )
                for key in (
                    "NUM_REGS",
                    "LOCAL_SIZE_BYTES",
                    "SHARED_SIZE_BYTES",
                    "MAX_THREADS_PER_BLOCK",
                )
            }
        except Exception:
            _check(backend.driver.cuModuleUnload(module))
            raise
        backend._kernels[source] = (module, function, None)
    return bytes(data), {
        "options": [x.decode() for x in options],
        "actual_function_attributes": attrs,
        "log": log.decode(errors="replace"),
    }


def relink_sass(ptx, folder, cuda_backend):
    """Re-link saved PTX with the CUDA driver. NOT a dump of the OpenCL executable."""
    tool = os.environ.get("PACHO_NVDISASM")
    if not tool:
        return {"available": False, "reason": "PACHO_NVDISASM not configured"}
    driver = cuda_backend.driver
    with cuda_backend._activate():
        state = _check(driver.cuLinkCreate(0, [], []))
        try:
            _check(
                driver.cuLinkAddData(
                    state,
                    driver.CUjitInputType.CU_JIT_INPUT_PTX,
                    ptx,
                    len(ptx),
                    b"inspect.ptx",
                    0,
                    [],
                    [],
                )
            )
            pointer, size = _check(driver.cuLinkComplete(state))
            binary = ctypes.string_at(int(pointer), size)
        finally:
            _check(driver.cuLinkDestroy(state))
    target = folder / "relinked.cubin"
    target.write_bytes(binary)
    command = [tool, "-c", str(target)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    (folder / "relinked.sass").write_text(result.stdout, encoding="utf-8")
    return {
        "available": True,
        "scope": "PTX re-linked by CUDA Driver API; not original OpenCL executable, may differ from module JIT",
        "command": command,
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
    }


def variant_source(name, ashape, bshape, variant):
    aligned_opencl = (
        name == "opencl" and ashape[-2] % 32 == 0 and ashape[-1] % 16 == 0 and bshape[-1] % 32 == 0
    )
    source, count = _source(
        name,
        ashape,
        bshape,
        specialize_bounds=variant == "aligned_guards_removed" or aligned_opencl,
    )
    # Retain the historical control after the production literal fix. No other
    # source transform, tile/guard change or compiler flag differs in this pair.
    source = re.sub(r"\b0\.0f?\b", "0.0", source)
    if variant in ("float_literals", "float_literals_fixed_group", "no_contraction", "contraction"):
        source = source.replace("0.0", "0.0f")
    if variant == "float_literals_fixed_group":
        source = source.replace(
            "__kernel void", "__attribute__((reqd_work_group_size(64,1,1))) __kernel void"
        )
    if variant == "no_contraction" and name == "opencl":
        source = "#pragma OPENCL FP_CONTRACT OFF\n" + source
    if variant == "no_contraction" and name == "vulkan":
        source = re.sub(r"float (c\d\d)=", r"precise float \1=", source)
    return "// diagnostic variant " + variant + "\n" + source, count


def prepare(backend, shape, variant, arrays, output, folder):
    a, b = arrays
    source, count = variant_source(backend.name, a.shape, b.shape, variant)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "source.txt").write_text(source, encoding="utf-8")
    metadata = {}
    if backend.name == "opencl":
        backend._run(source, count * 64, a._data, b._data, output._data, local_size=64)
        kernel = backend._kernels[source]
        data = kernel.program.binaries[0]
        (folder / "program.bin").write_bytes(data)
        text = data.decode("utf-8", errors="replace")
        metadata = {
            "kind": "OpenCL program binary",
            "is_ptx": ".version " in text and ".entry " in text,
            "options": "-cl-std=CL1.2",
            "fp64_extension_enabled": backend.fp64,
            "kernel_private_mem": kernel.get_work_group_info(
                backend.cl.kernel_work_group_info.PRIVATE_MEM_SIZE, backend.context.devices[0]
            ),
            "kernel_local_mem": kernel.get_work_group_info(
                backend.cl.kernel_work_group_info.LOCAL_MEM_SIZE, backend.context.devices[0]
            ),
        }
        if metadata["is_ptx"]:
            (folder / "program.ptx").write_bytes(data)
            metadata["ptx"] = ptx_summary(text)
        call = lambda: backend._run(
            source, count * 64, a._data, b._data, output._data, local_size=64
        )
    elif backend.name == "cuda":
        data, metadata = compile_cuda(backend, source, contraction=variant == "contraction")
        (folder / "program.ptx").write_bytes(data)
        metadata["ptx"] = ptx_summary(data.decode())
        metadata["sass"] = relink_sass(data, folder, backend)
        call = lambda: backend._run(source, count, a._data, b._data, output._data, threads=64)
    else:
        from benchmarks.vulkan_artifacts import export_pipeline

        shader, *_ = backend._dispatch_template(
            source,
            count,
            (np.dtype("float32"),) * 3,
            (),
            True,
            "shared float left_tile[512],right_tile[512];",
        )
        metadata = export_pipeline(backend, shader, folder)
        call = lambda: backend._run(
            source,
            count,
            [(a._data, a.dtype), (b._data, b.dtype), (output._data, output.dtype)],
            cooperative=True,
            shared="shared float left_tile[512],right_tile[512];",
        )

    def invoke():
        call()
        return output

    return invoke, metadata


def device_diagnostic(backend, function):
    if backend.name == "cuda":
        from benchmarks.cuda_dispatch import diagnose

        return diagnose(backend, function)
    if backend.name == "vulkan":
        from benchmarks.vulkan_dispatch import Diagnostic

        _, report = Diagnostic(backend).run(function)
        return report
    original = backend._run
    events = []

    def recorded(*args, **kwargs):
        event = original(*args, **kwargs)
        events.append(event)
        return event

    backend._run = recorded
    try:
        function()
        backend.synchronize()
        return {
            "kernel_events_seconds": [(e.profile.end - e.profile.start) * 1e-9 for e in events],
            "scope": "profiling queue kernel events, separate diagnostic",
        }
    finally:
        backend._run = original


def run(output, backends, repeats=21, seed=941):
    if output.exists() or len(backends) != 1 or repeats < 3:
        raise ValueError("Use a new output, one backend and >=3 repeats")
    name = backends[0]
    if name == "vulkan":
        from benchmarks.vulkan_artifacts import create_backend

        backend = create_backend()
    elif name in ("cuda-custom", "opencl"):
        backend = (
            p.get_backend("cuda", blas="custom")
            if name == "cuda-custom"
            else p.get_backend("opencl", blas="custom")
        )
    else:
        raise ValueError(name)
    if name == "opencl":
        # Kept for the whole run; timings labelled queue-profiling-enabled.
        from pyopencl.tools import ImmediateAllocator, MemoryPool

        backend.queue = backend.cl.CommandQueue(
            backend.context, properties=backend.cl.command_queue_properties.PROFILING_ENABLE
        )
        backend._pool = MemoryPool(ImmediateAllocator(backend.queue))
    folder = output.with_suffix("")
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "openpacho.kernel-inspection.v1",
        "backend": name,
        "device": backend.device,
        "repeats": repeats,
        "seed": seed,
        "source_sha256": source_hashes(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline": "historical unsuffixed 0.0 literals; float_literals uses current float32 zeros",
        "rows": [],
        "complete": False,
        "scope": "raw GEMM with preallocated input/output; no padding copies or allocations in timed call; OpenCL queue profiling enabled; Vulkan pipeline capture flags enabled; not public API speedup",
    }
    try:
        with threadpool_limits(limits=1):
            for shape in ((1024, 1024, 1024), (2049, 2047, 2051), (2080, 2048, 2080)):
                m, k, n = shape
                rng = np.random.default_rng(seed)
                host = [rng.uniform(-0.5, 0.5, s).astype("float32") for s in ((m, k), (k, n))]
                reference = host[0].astype("float64") @ host[1].astype("float64")
                arrays = tuple(p.asarray(x, backend=backend) for x in host)
                variants = ["baseline", "float_literals"]
                if name == "opencl":
                    variants += ["float_literals_fixed_group", "no_contraction"]
                elif name == "cuda-custom":
                    variants += ["contraction"]
                else:
                    variants += ["no_contraction"]
                if m % 32 == 0 and k % 16 == 0 and n % 32 == 0:
                    variants += ["aligned_guards_removed"]
                functions = {}
                outputs = {}
                rows = {}
                for variant in variants:
                    out = p.empty((m, n), dtype="float32", backend=backend)
                    function, metadata = prepare(
                        backend,
                        shape,
                        variant,
                        arrays,
                        out,
                        folder / ("-".join(map(str, shape)) + "-" + variant),
                    )
                    warmup(function, p, backend)
                    actual = p.asnumpy(function())
                    np.testing.assert_allclose(actual, reference, rtol=5e-5, atol=5e-5)
                    functions[variant] = function
                    outputs[variant] = actual
                    rows[variant] = {
                        "shape": shape,
                        "variant": variant,
                        "artifact": metadata,
                        "error": error_metrics(actual, reference),
                        "samples": [],
                    }
                for pair in range(repeats):
                    order = variants[pair % len(variants) :] + variants[: pair % len(variants)]
                    for variant in order if pair % 2 else order[::-1]:
                        seconds, _, _ = measure(functions[variant], p, backend)
                        rows[variant]["samples"].append(seconds)
                for variant in variants:
                    row = rows[variant]
                    row["wall"] = _sample_stats(row.pop("samples"))
                    row["numeric_baseline_equality"] = bool(
                        np.array_equal(outputs[variant], outputs["baseline"])
                    )
                    row["bitwise_baseline_equality"] = bool(
                        np.array_equal(
                            outputs[variant].view(np.uint32), outputs["baseline"].view(np.uint32)
                        )
                    )
                    row["diagnostics"] = [
                        device_diagnostic(backend, functions[variant]) for _ in range(5)
                    ]
                    report["rows"].append(row)
                print(
                    shape,
                    {v: round(rows[v]["wall"]["median_seconds"] * 1000, 3) for v in variants},
                    flush=True,
                )
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["complete"] = True
    finally:
        getattr(backend, "close", backend.synchronize)()
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", default="opencl")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    args = parser.parse_args()
    run(args.output, args.backends.split(","), args.repeats, args.seed)
