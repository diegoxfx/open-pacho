"""Paired wrapped/direct Vulkan control with separate host/GPU diagnostics.

Wall samples never run with instrumentation. GPU timestamp intervals include
timestamp/stage effects and are not pure shader timings. Host API durations
include the selected binding, driver, and blocking waits; wall minus GPU time
is never labelled Python time. No workload or kernel is changed between modes.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.applications import (
    ApplicationCase,
    Implementation,
    PachoAdapter,
    ridge_regression,
    sample_covariance,
)
from benchmarks.applications import (
    make_inputs as application_inputs,
)
from benchmarks.execution import energy
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.suite import Case, make_inputs, source_hashes, telemetry
from examples import solvers
from openpacho._vulkan import VulkanBackend
from openpacho._vulkan_commands import RESULT_COMMANDS, VOID_COMMANDS


def timestamp_seconds(start, end, bits, period_ns):
    if not 1 <= bits <= 64 or not period_ns > 0:
        raise ValueError("Timestamp capability must provide 1..64 valid bits and positive period")
    return ((int(end) - int(start)) & ((1 << bits) - 1)) * period_ns * 1e-9


def solver_validation(a, b, actual, rtol):
    a, b, actual = (np.asarray(x, dtype=np.float64) for x in (a, b, actual))
    residual = float(np.linalg.norm(b - a @ actual))
    limit = rtol * float(np.linalg.norm(b))
    # The solver's contract bounds the residual, not componentwise forward error.
    # Independently evaluate it on the represented inputs, outside all clocks.
    if not np.isfinite(residual) or residual > limit * 1.05:
        raise AssertionError(f"Independent solver residual {residual} exceeds {limit}")
    return {"residual_float64": residual, "residual_limit": limit, "rtol": rtol}


class Diagnostic:
    """One warmed invocation; query storage survives every intervening queue drain."""

    def __init__(self, backend, max_commands=2048):
        self.backend, self.vk = backend, backend.vk
        self.maximum = max_commands
        self.originals = {}
        self.counts = Counter()
        self.seconds = defaultdict(float)
        self.events = []
        self.copy_bytes = 0
        self.new_allocation_bytes = 0
        self.pool = None

    def _wrap(self, name, function):
        def call(*args):
            event = name in ("vkCmdDispatch", "vkCmdCopyBuffer") and self.pool is not None
            query = 2 * len(self.events)
            if event:
                if query >= 2 * self.maximum:
                    raise RuntimeError("Diagnostic timestamp capacity exceeded")
                self.vk.vkCmdResetQueryPool(args[0], self.pool, query, 2)
                self.vk.vkCmdWriteTimestamp(
                    args[0], self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, self.pool, query
                )
            start = time.perf_counter_ns()
            try:
                result = function(*args)
            finally:
                self.seconds[name] += (time.perf_counter_ns() - start) * 1e-9
                self.counts[name] += 1
            if name == "vkCmdCopyBuffer":
                self.copy_bytes += sum(int(args[4][i].size) for i in range(args[3]))
            elif name == "vkAllocateMemory":
                self.new_allocation_bytes += int(args[1].allocationSize)
            if event:
                self.vk.vkCmdWriteTimestamp(
                    args[0], self.vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, self.pool, query + 1
                )
                self.events.append(name)
            return result

        return call

    def run(self, function):
        backend, vk = self.backend, self.vk
        backend.synchronize()
        before = backend._cached_bytes
        if backend._timestamp_valid_bits:
            self.pool = vk.vkCreateQueryPool(
                backend._device,
                vk.VkQueryPoolCreateInfo(
                    queryType=vk.VK_QUERY_TYPE_TIMESTAMP, queryCount=self.maximum * 2
                ),
                None,
            )
        names = (
            VOID_COMMANDS
            + RESULT_COMMANDS
            + (
                "vkMapMemory",
                "vkAllocateMemory",
                "vkFreeMemory",
                "vkCreateBuffer",
                "vkDestroyBuffer",
                "vkBindBufferMemory",
                "vkGetBufferMemoryRequirements",
                "vkAllocateDescriptorSets",
                "vkCreateDescriptorPool",
                "vkDestroyDescriptorPool",
            )
        )
        if getattr(backend, "_push_descriptor_limit", 0):
            names += ("vkCmdPushDescriptorSetKHR",)
        try:
            for name in names:
                original = getattr(vk, name)
                self.originals[name] = original
                setattr(vk, name, self._wrap(name, original))
            start = time.perf_counter_ns()
            value = function()
            backend.synchronize()
            wall = (time.perf_counter_ns() - start) * 1e-9
            intervals = []
            if self.events:
                count = 2 * len(self.events)
                data = vk.ffi.new("uint64_t[]", count)
                vk.vkGetQueryPoolResults(
                    backend._device,
                    self.pool,
                    0,
                    count,
                    count * 8,
                    data,
                    8,
                    vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
                )
                intervals = [
                    {
                        "command": name,
                        "seconds": timestamp_seconds(
                            data[2 * i],
                            data[2 * i + 1],
                            backend._timestamp_valid_bits,
                            backend._timestamp_period,
                        ),
                    }
                    for i, name in enumerate(self.events)
                ]
            result = {
                "instrumented_wall_seconds": wall,
                "host_api_seconds": dict(self.seconds),
                "command_counts": dict(self.counts),
                "gpu_intervals": intervals,
                "gpu_timestamps_supported": bool(backend._timestamp_valid_bits),
                "timestamp_valid_bits": backend._timestamp_valid_bits,
                "timestamp_period_ns": backend._timestamp_period,
                "copy_bytes": self.copy_bytes,
                "new_native_allocation_bytes": self.new_allocation_bytes,
                "cached_native_bytes_before": before,
                "cached_native_bytes_after": backend._cached_bytes,
                "memory_scope": "new VkDeviceMemory allocations and retained cache, not total/peak VRAM",
                "interpretation": "diagnostic only: API durations include binding/driver/waits; GPU intervals include timestamp/stage effects; clocks are not directly subtractable",
            }
            return value, result
        finally:
            for name, original in self.originals.items():
                setattr(vk, name, original)
            # Do not destroy query storage until pending commands have completed.
            # If the driver cannot establish completion, propagate its error.
            backend.synchronize()
            if self.pool is not None:
                vk.vkDestroyQueryPool(backend._device, self.pool, None)
                self.pool = None


def cases(quick=False):
    if quick:
        return [
            (op, shape, "float32")
            for op, shape in (
                ("add", (33,)),
                ("energy", (129, 17)),
                ("ridge", (129, 17)),
                ("cg", (33, 33)),
                ("upload", (1025,)),
                ("download", (1025,)),
            )
        ]
    return [
        (op, shape, dtype)
        for dtype in ("float32", "float64")
        for op, shape in (
            ("add", (33,)),
            ("add", (1048577,)),
            ("dot", (2049,)),
            ("gemv", (2049, 1023)),
            ("gemm", (257, 129, 513)),
            ("energy", (2049, 33)),
            ("covariance", (4097, 129)),
            ("ridge", (4097, 129)),
            ("cg", (513, 513)),
            ("jacobi", (513, 513)),
            ("sd", (513, 513)),
            ("upload", (1,)),
            ("download", (1,)),
            ("upload", (1048576,)),
            ("download", (1048576,)),
        )
    ]


def workload(backend, spec):
    op, shape, dtype = spec
    rng = np.random.default_rng(4781)
    host = lambda shape: rng.uniform(-0.5, 0.5, shape).astype(dtype)
    upload = lambda arrays: tuple(p.asarray(a, backend=backend) for a in arrays)
    if op in ("covariance", "ridge"):
        case = ApplicationCase(op, *shape, dtype)
        inputs = application_inputs(case)[: 1 if op == "covariance" else 2]
        impl = Implementation(PachoAdapter("vulkan", backend), "vulkan", compiled=True)
        start = time.perf_counter_ns()
        resident = impl.upload(inputs)
        prepared = impl.prepare(case, resident)
        backend.synchronize()
        preparation = (time.perf_counter_ns() - start) * 1e-9
        reference = (
            sample_covariance(inputs[0].astype(np.float64))
            if op == "covariance"
            else ridge_regression(*(a.astype(np.float64) for a in inputs))
        )
        function = lambda: impl.calculate(case, prepared)
        function.end_to_end = lambda: impl.host(
            impl.calculate(case, impl.prepare(case, impl.upload(inputs)))
        )
        return (
            function,
            reference,
            {"preparation_seconds": preparation, "input_bytes": sum(a.nbytes for a in inputs)},
        )
    if op in ("cg", "jacobi", "sd"):
        inputs, _ = make_inputs(Case(op, shape, dtype), seed=47)
        a, b = upload(inputs)
        method = {
            "cg": solvers.conjugate_gradient,
            "jacobi": solvers.jacobi,
            "sd": solvers.steepest_descent,
        }[op]

        def solve():
            result = method(a, b, rtol=1e-4 if dtype == "float32" else 1e-7, maxiter=400)
            if not result.converged:
                raise AssertionError(result.reason)
            solve.iterations = result.iterations
            return result.x

        solve.validate = lambda actual: solver_validation(
            inputs[0], inputs[1], actual, 1e-4 if dtype == "float32" else 1e-7
        )

        def full_solve():
            result = method(*upload(inputs), rtol=1e-4 if dtype == "float32" else 1e-7, maxiter=400)
            if not result.converged or result.iterations != solve.iterations:
                raise AssertionError("End-to-end solver changed convergence/iterations")
            return p.asnumpy(result.x)

        solve.end_to_end = full_solve

        return (
            solve,
            np.linalg.solve(inputs[0].astype(np.float64), inputs[1].astype(np.float64)),
            {"input_bytes": sum(x.nbytes for x in inputs)},
        )
    inputs = (host(shape),)
    if op in ("add", "dot"):
        inputs += (host(shape),)
    elif op == "gemv":
        inputs += (host((shape[1],)),)
    elif op == "gemm":
        inputs = (host(shape[:2]), host(shape[1:]))
    elif op == "energy":
        inputs += (host(shape), host(shape))
    if op == "upload":
        return (
            lambda: p.asarray(inputs[0], backend=backend),
            inputs[0],
            {"input_bytes": inputs[0].nbytes},
        )
    resident = upload(inputs)
    if op == "download":
        return lambda: p.asnumpy(resident[0]), inputs[0], {"input_bytes": inputs[0].nbytes}
    functions = {
        "add": lambda a, b: a + b,
        "dot": lambda a, b: np.dot(a, b),
        "gemv": lambda a, b: a @ b,
        "gemm": lambda a, b: a @ b,
        "energy": energy,
    }
    function = functions[op]
    reference = function(*(a.astype(np.float64) for a in inputs))
    if op == "energy":
        function = p.compile(function)
    resident_call = lambda: function(*resident)
    resident_call.end_to_end = lambda: p.asnumpy(function(*upload(inputs)))
    return resident_call, reference, {"input_bytes": sum(a.nbytes for a in inputs)}


def measure(backend, function):
    backend.synchronize()
    start = time.perf_counter_ns()
    value = function()
    backend.synchronize()
    return (time.perf_counter_ns() - start) * 1e-9, value


def run(output, repeats=21, quick=False, seed=4781, control="commands"):
    if output.exists() or repeats < 3:
        raise ValueError("Use a new output path and at least three repetitions")
    output.parent.mkdir(parents=True, exist_ok=True)
    engines, states = {}, []
    if control not in ("commands", "descriptors"):
        raise ValueError("control must be commands or descriptors")
    modes = (
        {
            mode: {"_command_mode": mode, "_descriptor_mode": "sets"}
            for mode in ("wrapped", "direct")
        }
        if control == "commands"
        else {
            mode: {"_command_mode": "direct", "_descriptor_mode": mode} for mode in ("sets", "push")
        }
    )
    report = {
        "schema": "openpacho.vulkan-dispatch.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repeats": repeats,
        "seed": seed,
        "control": control,
        "rows": [],
        "errors": [],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vulkan_binding": importlib.metadata.version("vulkan"),
        "source_sha256": source_hashes(),
        "telemetry_start": telemetry(),
        "timing_scope": "synchronized wall, uninstrumented; warmup, CPU validation and separate diagnostic invocation excluded",
    }
    report["source_sha256"][str(Path(__file__).name)] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    try:
        for mode, options in modes.items():
            engines[mode] = VulkanBackend(os.environ.get("PACHO_VULKAN_DEVICE"), **options)
        first = next(iter(engines.values()))
        report["device"] = first.device
        report["driver_version_raw"] = int(first._properties.driverVersion)
        report["push_descriptor_limits"] = {k: v._push_descriptor_limit for k, v in engines.items()}
        for spec in cases(quick):
            states = []
            for mode, engine in engines.items():
                try:
                    fn, expected, metadata = workload(engine, spec)
                    cold, value = measure(engine, fn)
                    del value
                    for _ in range(3):
                        _, value = measure(engine, fn)
                        del value
                    if hasattr(fn, "end_to_end"):
                        _, value = measure(engine, fn.end_to_end)
                        del value
                        fn.end_to_end_samples = []
                    states.append((mode, engine, fn, expected, metadata, cold, []))
                except Exception as exc:  # noqa: BLE001 - preserve failed configurations
                    report["errors"].append({"case": spec, "mode": mode, "error": repr(exc)})
            rng = random.Random(seed)
            for _ in range(repeats):
                order = list(states)
                rng.shuffle(order)
                for mode, engine, fn, expected, metadata, cold, samples in order:
                    elapsed, value = measure(engine, fn)
                    samples.append(elapsed)
                    del value
                    if hasattr(fn, "end_to_end"):
                        elapsed, value = measure(engine, fn.end_to_end)
                        fn.end_to_end_samples.append(elapsed)
                        del value
            outputs = {}
            for mode, engine, fn, expected, metadata, cold, samples in states:
                value = fn()
                actual = p.asnumpy(value)
                outputs[mode] = actual
                if hasattr(fn, "validate"):
                    validation = fn.validate(actual)
                else:
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=8e-4 if spec[2] == "float32" else 2e-10,
                        atol=2e-5 if spec[2] == "float32" else 1e-10,
                    )
                    validation = {"criterion": "elementwise-numpy-reference"}
                del value
                observed, diagnostic = Diagnostic(engine).run(fn)
                np.testing.assert_array_equal(p.asnumpy(observed), actual)
                del observed
                full_diagnostic = None
                if hasattr(fn, "end_to_end"):
                    observed, full_diagnostic = Diagnostic(engine).run(fn.end_to_end)
                    np.testing.assert_array_equal(observed, actual)
                    del observed
                report["rows"].append(
                    {
                        "case": list(spec),
                        "mode": mode,
                        "wall": _sample_stats(samples),
                        "end_to_end_wall": _sample_stats(fn.end_to_end_samples)
                        if hasattr(fn, "end_to_end")
                        else None,
                        "cold_seconds": cold,
                        "iterations": getattr(fn, "iterations", None),
                        "error": error_metrics(
                            actual,
                            expected,
                            reference=f"numpy-{np.asarray(expected).dtype}",
                            precision=str(np.asarray(expected).dtype),
                        ),
                        "validation": validation,
                        "diagnostic": diagnostic,
                        "end_to_end_diagnostic": full_diagnostic,
                        **metadata,
                    }
                )
            if len(outputs) == 2:
                np.testing.assert_array_equal(*outputs.values())
            output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(f"completed {spec}", flush=True)
    except Exception as exc:  # noqa: BLE001 - retain evidence then fail the CLI
        report["errors"].append({"error": repr(exc)})
    finally:
        states.clear()
        for engine in engines.values():
            engine.close()
        report["telemetry_end"] = telemetry()
        report["all_valid"] = not report["errors"] and len(report["rows"]) == 2 * len(cases(quick))
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=4781)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--control", choices=("commands", "descriptors"), default="commands")
    args = parser.parse_args()
    with threadpool_limits(limits=8):
        report = run(args.output, args.repeats, args.quick, args.seed, args.control)
    raise SystemExit(0 if report["all_valid"] else 1)


if __name__ == "__main__":
    main()
