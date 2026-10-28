"""Controlled GEMM experiments; candidates never change runtime selection.

Preallocated output, synchronized launch wall, same context and arithmetic order.
Global uploads/packing are outside this kernel experiment, not claimed as free.
"""

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.numerical_metrics import error_metrics
from benchmarks.stability import nvidia_snapshot
from openpacho._gemm import _source as current_source


@dataclass(frozen=True)
class Tile:
    registers: int = 4
    k: int = 16
    pad_left: int = 0
    pad_right: int = 0
    transpose_left: bool = False
    specialize_bounds: bool = False

    def __post_init__(self):
        if self.registers not in (2, 4) or self.k not in (8, 16, 32):
            raise ValueError("Unmeasured tile configuration")
        if self.pad_left not in (0, 1) or self.pad_right not in (0, 1):
            raise ValueError("Padding must be zero or one")

    @property
    def width(self):
        return self.registers * 8

    @property
    def left_stride(self):
        return (self.width if self.transpose_left else self.k) + self.pad_left

    @property
    def right_stride(self):
        return self.width + self.pad_right

    @property
    def left_size(self):
        return (self.k if self.transpose_left else self.width) * self.left_stride

    @property
    def right_size(self):
        return self.k * self.right_stride

    @property
    def shared_bytes(self):
        return (self.left_size + self.right_size) * 4

    def left_index(self, row, col):
        return (
            f"({col})*{self.left_stride}+({row})"
            if self.transpose_left
            else f"({row})*{self.left_stride}+({col})"
        )


VARIANTS = {
    "current": Tile(),
    "generated-baseline": Tile(),
    "core-bounds": Tile(specialize_bounds=True),
    "pad-left": Tile(pad_left=1),
    "pad-both": Tile(pad_left=1, pad_right=1),
    "transpose-left": Tile(transpose_left=True),
    "transpose-pad-left": Tile(transpose_left=True, pad_left=1),
    "registers-2x2": Tile(registers=2),
    "tile-k32": Tile(k=32),
    "specialized-bounds": Tile(specialize_bounds=True),
}


def source(backend, m, k, n, tile):
    if min(m, k, n) < 1 or max(m * k, k * n, m * n) > 2**31 - 1:
        raise ValueError("Experiment requires nonempty matrices with signed 32-bit indices")
    width, tk, registers = tile.width, tile.k, tile.registers
    nr, nc = (m + width - 1) // width, (n + width - 1) // width
    left = lambda row, col: f"left_tile[{tile.left_index(row, col)}]"
    right = lambda row, col: f"right_tile[({row})*{tile.right_stride}+({col})]"

    def guarded(value, conditions):
        conditions = [
            condition
            for condition, aligned in conditions
            if not tile.specialize_bounds or not aligned
        ]
        return f"({' && '.join(conditions)})?{value}:0.0f" if conditions else value

    load_a = guarded(f"x0[ar*{k}+ak]", [(f"ar<{m}", m % width == 0), (f"ak<{k}", k % tk == 0)])
    load_b = guarded(f"x1[bk*{n}+bc]", [(f"bk<{k}", k % tk == 0), (f"bc<{n}", n % width == 0)])
    init = " ".join(f"float c{r}{c}=0.0f;" for r in range(registers) for c in range(registers))
    inner = " ".join(f"float a{r}={left(f'ly+{r * 8}', 'kk')};" for r in range(registers))
    inner += " ".join(f"float b{c}={right('kk', f'lx+{c * 8}')};" for c in range(registers))
    inner += " ".join(f"c{r}{c}+=a{r}*b{c};" for r in range(registers) for c in range(registers))
    stores = []
    for r in range(registers):
        for c in range(registers):
            conditions = [
                (f"row+{r * 8}<{m}", m % width == 0),
                (f"col+{c * 8}<{n}", n % width == 0),
            ]
            conditions = [
                s for s, aligned in conditions if not tile.specialize_bounds or not aligned
            ]
            prefix = f"if({' && '.join(conditions)}) " if conditions else ""
            stores.append(prefix + f"x2[(row+{r * 8})*{n}+col+{c * 8}]=c{r}{c};")
    body = f"""
        int lx=lane%8, ly=lane/8;
        int row=(i/{nc})*{width}, col=(i%{nc})*{width}; {init}
        for(int start=0;start<{k};start+={tk}) {{
            for(int t=lane;t<{width * tk};t+=64) {{
                int ar=row+t/{tk}, ak=start+t%{tk};
                int bk=start+t/{width}, bc=col+t%{width};
                {left(f"t/{tk}", f"t%{tk}")}={load_a};
                {right(f"t/{width}", f"t%{width}")}={load_b};
            }}
            BARRIER;
            for(int kk=0;kk<{tk};++kk) {{ {inner} }}
            BARRIER;
        }}
        row+=ly; col+=lx; {" ".join(stores)}
    """
    shared = f"float left_tile[{tile.left_size}],right_tile[{tile.right_size}];"
    if backend == "opencl":
        body = (
            "__kernel void compute(__global const float*x0,__global const float*x1,__global float*x2){"
            "int i=get_group_id(0),lane=get_local_id(0);__local "
            + shared
            + body.replace("BARRIER", "barrier(CLK_LOCAL_MEM_FENCE)")
            + "}"
        )
    elif backend == "cuda":
        body = (
            'extern "C" __global__ void compute(const float*x0,const float*x1,float*x2){'
            "int i=blockIdx.x,lane=threadIdx.x;__shared__ "
            + shared
            + body.replace("BARRIER", "__syncthreads()")
            + "}"
        )
    elif backend == "vulkan":
        body = "int lane=int(gl_LocalInvocationID.x);" + body.replace("BARRIER", "barrier()")
    else:
        raise ValueError("Kernel experiment supports OpenCL, Vulkan and CUDA")
    return body, nr * nc, "shared " + shared


def shared_limit(backend):
    if backend.name == "opencl":
        return int(backend.queue.device.local_mem_size)
    if backend.name == "vulkan":
        return int(backend._properties.limits.maxComputeSharedMemorySize)
    from openpacho._cuda import _check

    return int(
        _check(
            backend.driver.cuDeviceGetAttribute(
                backend.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                backend._device,
            )
        )
    )


def run(output, backend_name, repeats=21, seed=941):
    if output.exists() or repeats < 3:
        raise ValueError("Use a new path and at least three repetitions")
    if backend_name not in ("opencl", "vulkan", "cuda-custom"):
        raise ValueError("Select one custom GPU backend")
    output.parent.mkdir(parents=True, exist_ok=True)
    _, backend = engine(backend_name)
    report = {
        "schema": "openpacho.kernel-variants.v1",
        "backend": backend_name,
        "device": backend.device,
        "repeats": repeats,
        "seed": seed,
        "rows": [],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_gemm_sha256": hashlib.sha256(
            Path("src/openpacho/_gemm.py").read_bytes()
        ).hexdigest(),
        "shared_limit": shared_limit(backend),
        "before": nvidia_snapshot(),
        "scope": "preallocated output, synchronized launch wall; preparation, warmup, validation and telemetry excluded; no global packing/transfer speedup claimed",
    }
    try:
        if not backend._register_gemm:
            raise RuntimeError("Baseline register GEMM capability unavailable")
        for m, k, n in (
            (32, 16, 32),
            (33, 17, 65),
            (257, 129, 513),
            (1025, 513, 769),
            (1024, 1024, 1024),
        ):
            # Bound requested logical buffers before allocation. Driver/compiler
            # memory, allocator caches and previous case buffers are not peak VRAM.
            allocated_bytes = (m * k + k * n) * 16 + (m * n + 16) * len(VARIANTS) * 4 + m * n * 12
            if allocated_bytes > 128 * 1024**2:
                raise ValueError("Experiment exceeds its 128 MiB logical-data budget")
            rng = np.random.default_rng(941)
            ah, bh = [
                rng.uniform(-0.5, 0.5, shape).astype(np.float32) for shape in ((m, k), (k, n))
            ]
            a, b = [p.asarray(host, backend=backend) for host in (ah, bh)]
            reference = ah.astype(np.float64) @ bh.astype(np.float64)
            states = []
            for name, tile in VARIANTS.items():
                row = {
                    "shape": [m, k, n],
                    "variant": name,
                    "tile": asdict(tile),
                    "shared_bytes": tile.shared_bytes,
                    "live_data_bytes": allocated_bytes,
                }
                report["rows"].append(row)
                if tile.shared_bytes > report["shared_limit"]:
                    row.update(status="unsupported", reason="shared-memory limit")
                    continue
                body, groups, shared = source(backend.name, m, k, n, tile)
                if name == "current":
                    body, groups = current_source(backend.name, (m, k), (k, n))
                elif name == "core-bounds":
                    body, groups = current_source(
                        backend.name, (m, k), (k, n), specialize_bounds=True
                    )
                row["kernel_sha256"] = hashlib.sha256(body.encode()).hexdigest()
                out = p.full((m * n + 16,), np.nan, dtype=np.float32, backend=backend)

                def launch(body=body, groups=groups, shared=shared, out=out, a=a, b=b):
                    if backend.name == "opencl":
                        backend._run(body, groups * 64, a._data, b._data, out._data, local_size=64)
                    elif backend.name == "cuda":
                        backend._run(body, groups, a._data, b._data, out._data, threads=64)
                    else:
                        backend._run(
                            body,
                            groups,
                            [(v._data, v.dtype) for v in (a, b, out)],
                            cooperative=True,
                            shared=shared,
                        )
                    return out

                first, _, _ = measure(launch, p, backend)
                row["first_seconds"] = first
                row["warmup"] = warmup(launch, p, backend)
                states.append((row, launch, out, []))
            order_rng = random.Random(seed)
            for _ in range(repeats):
                order = list(states)
                order_rng.shuffle(order)
                for row, fn, out, times in order:
                    wall, _, _ = measure(fn, p, backend)
                    times.append(wall)
            for row, fn, out, times in states:
                actual = p.asnumpy(out)
                np.testing.assert_array_equal(np.isnan(actual[-16:]), True)
                actual = actual[:-16].reshape(m, n)
                np.testing.assert_allclose(actual, reference, rtol=8e-4, atol=2e-5)
                row.update(
                    status="ok",
                    guard_unchanged=True,
                    wall=_sample_stats(times),
                    error=error_metrics(actual, reference, reference="numpy-float64"),
                )
            output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(f"Completed {backend_name} {m}x{k}x{n}", flush=True)
        report["all_valid"] = all(r.get("status") == "ok" for r in report["rows"])
        return report
    finally:
        backend.synchronize()
        report["after"] = nvidia_snapshot()
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if hasattr(backend, "close"):
            backend.close()
        else:
            backend.release_cached_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", required=True)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--seed", type=int, default=941)
    args = parser.parse_args()
    with threadpool_limits(limits=8):
        result = run(args.output, args.backends, args.repeats, args.seed)
    raise SystemExit(0 if result["all_valid"] else 1)


if __name__ == "__main__":
    main()
