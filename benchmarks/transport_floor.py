"""Measured transport/dispatch controls, without changing OpenPacho's runtime.

Pageable vs OpenCL-allocated mapped host memory exposes avoidable staging costs.
Event time is device time; synchronized wall also includes host API costs.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pyopencl as cl

import openpacho as p


def run(output):
    engine = p.get_backend("opencl")
    queue = cl.CommandQueue(engine.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    rows = []

    def measure(name, nbytes, function):
        function().wait()
        wall, device = [], []
        for _ in range(11):
            queue.finish()
            start = time.perf_counter()
            event = function()
            event.wait()
            wall.append(time.perf_counter() - start)
            device.append((event.profile.end - event.profile.start) * 1e-9)
        rows.append(
            {
                "name": name,
                "bytes": nbytes,
                "wall_seconds": wall,
                "event_seconds": device,
                "wall_median_seconds": statistics.median(wall),
                "event_median_seconds": statistics.median(device),
                "effective_wall_GB_s": nbytes / statistics.median(wall) / 1e9,
            }
        )

    program = cl.Program(engine.context, "__kernel void noop() {}").build()
    kernel = cl.Kernel(program, "noop")
    measure("empty_dispatch", 0, lambda: kernel(queue, (1,), None))
    for size in (1, 1024 * 1024, 16 * 1024 * 1024):
        host = np.arange(size, dtype=np.float32)
        target = cl.Buffer(engine.context, cl.mem_flags.READ_WRITE, host.nbytes)
        other = cl.Buffer(engine.context, cl.mem_flags.READ_WRITE, host.nbytes)
        pinned = cl.Buffer(
            engine.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, host.nbytes
        )
        mapped, _event = cl.enqueue_map_buffer(
            queue,
            pinned,
            cl.map_flags.READ | cl.map_flags.WRITE,
            0,
            host.shape,
            host.dtype,
            is_blocking=True,
        )
        mapped[:] = host
        try:
            measure(
                "pageable_H2D",
                host.nbytes,
                lambda h=host, t=target: cl.enqueue_copy(queue, t, h, is_blocking=False),
            )
            measure(
                "mapped_host_H2D",
                host.nbytes,
                lambda h=mapped, t=target: cl.enqueue_copy(queue, t, h, is_blocking=False),
            )
            measure(
                "device_D2D",
                host.nbytes,
                lambda t=target, o=other, n=host.nbytes: cl.enqueue_copy(queue, o, t, byte_count=n),
            )
            result = np.empty_like(host)
            measure(
                "pageable_D2H",
                host.nbytes,
                lambda h=result, t=target: cl.enqueue_copy(queue, h, t, is_blocking=False),
            )
            np.testing.assert_array_equal(result, host)
            measure(
                "mapped_host_D2H",
                host.nbytes,
                lambda h=mapped, t=target: cl.enqueue_copy(queue, h, t, is_blocking=False),
            )
            np.testing.assert_array_equal(mapped, host)
        finally:
            mapped.base.release(queue).wait()
    # Link may downclock at idle; query after transfer traffic, and keep raw data.
    fields = "name,driver_version,compute_cap,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,clocks.sm,clocks.mem"
    try:
        info = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv"],
            capture_output=True,
            text=True,
            check=False,
        )
        smi_output, smi_error = info.stdout, info.stderr
    except OSError as exc:
        # Transfer controls remain usable on non-NVIDIA OpenCL devices.
        smi_output, smi_error = "", str(exc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "device": engine.device,
                "rows": rows,
                "nvidia_smi": smi_output,
                "nvidia_smi_error": smi_error,
                "note": "D2D bytes count payload once (not read+write traffic). Mapped allocation setup excluded.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in rows:
        print(
            row["name"], row["bytes"], round(row["wall_median_seconds"] * 1e6, 2), "us", flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
