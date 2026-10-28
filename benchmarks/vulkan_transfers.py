"""Public Vulkan transfer costs with exact host/device/software provenance.

Run the same script against a frozen wheel and current source in separate
processes. Warm synchronized wall samples include allocation/pool, mapping,
host copies and GPU transfer. They are not PCIe-only measurements.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

import openpacho as p
from openpacho import _vulkan


def run(output, repeats=11):
    backend = p.get_backend("vulkan")
    props = backend._properties
    rows = []
    for size in (64, 1048576, 16777216, 67108864):
        host = (np.arange(size, dtype=np.float32) % 1021) / np.float32(17)
        resident = p.asarray(host, backend=backend)
        backend.synchronize()
        for operation in ("upload", "download"):
            samples = []
            for repeat in range(repeats + 1):
                backend.synchronize()
                start = time.perf_counter()
                value = (
                    p.asarray(host, backend=backend)
                    if operation == "upload"
                    else p.asnumpy(resident)
                )
                backend.synchronize()
                elapsed = time.perf_counter() - start
                if repeat:
                    samples.append(elapsed)
                actual = p.asnumpy(value) if operation == "upload" else value
                np.testing.assert_array_equal(actual, host)
                del actual, value
            row = {
                "operation": operation,
                "bytes": host.nbytes,
                "dtype": str(host.dtype),
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
                "p25_seconds": float(np.percentile(samples, 25)),
                "p75_seconds": float(np.percentile(samples, 75)),
            }
            rows.append(row)
            print(operation, host.nbytes, round(row["median_seconds"] * 1000, 3), "ms", flush=True)
        del resident
        backend.synchronize()
    source = Path(_vulkan.__file__)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": backend.device,
        "vendor_id": int(props.vendorID),
        "device_id": int(props.deviceID),
        "driver_version_raw": int(props.driverVersion),
        "api_version_raw": int(props.apiVersion),
        "compiler": backend.compiler,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "repeats": repeats,
        "rows": rows,
        "scope": "one tested adapter and host; no architecture-wide performance claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    run(args.output, args.repeats)
