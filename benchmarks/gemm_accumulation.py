"""Bounded float32 GEMM accumulation research; never changes runtime defaults.

One backend per process. Timings cover synchronized launches into preallocated
resident buffers, including Python dispatch; they are not end-to-end API timings.
Use the standard readiness gate between workers. Numerical failures are evidence,
not a reason to omit a candidate's measurements.
"""

import argparse
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import openpacho as p
from benchmarks.backend_sharing import engine, measure, warmup
from benchmarks.gaps import _sample_stats
from benchmarks.gemm_accumulation_kernels import source_for
from benchmarks.gemm_precision_oracle import evaluate_sampled_errors, sampled_dot_references
from benchmarks.numerical_metrics import error_metrics
from benchmarks.padding_decision import validate_result
from benchmarks.stability import nvidia_snapshot

VARIANTS = ("current", "sequential", "blocked", "kahan", "neumaier", "blocked_neumaier")


def cases():
    """The two historical seeds stay fixed across timing repetitions."""
    result = [
        {"name": "small-tail", "shape": (33, 31, 65), "seed": 981, "index": 0},
        {"name": "historical-972", "shape": (65, 8191, 2051), "seed": 972, "index": 10},
        {"name": "historical-973", "shape": (65, 8191, 2051), "seed": 973, "index": 10},
        {"name": "long-k", "shape": (65, 16383, 257), "seed": 981, "index": 1},
        {"name": "square", "shape": (1024, 1024, 1024), "seed": 981, "index": 2},
    ]
    result += [
        {"name": f"cancellation-{k}", "shape": (33, k, 33), "kind": "cancellation"}
        for k in (15, 16, 17, 31, 32, 33, 8191)
    ]
    result.append({"name": "nonfinite", "shape": (32, 16, 33), "kind": "nonfinite"})
    return result


def make_inputs(case):
    m, k, n = case["shape"]
    if case.get("kind") == "cancellation":
        # The three rows expose ordering and a tile boundary's effect on partial sums.
        patterns = ([1e8, 1, -1e8], [1e8, -1e8, 1], [1, 1e8, -1e8])
        a = np.array([np.resize(patterns[row % 3], k) for row in range(m)], dtype=np.float32)
        return a, np.ones((k, n), dtype=np.float32)
    if case.get("kind") == "nonfinite":
        largest = np.finfo(np.float32).max
        a = np.array(
            [
                [np.nan, 1, 2, 3],
                [np.inf, 1, 2, 3],
                [-np.inf, 1, 2, 3],
                [np.inf, -np.inf, 1, 2],
                [largest, largest, -largest, 0],
                [largest, 0, 0, 0],
                [1, 2, 3, 4],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        padded = np.zeros((m, k), dtype=np.float32)
        padded[:, :4] = a[np.arange(m) % len(a)]
        return padded, np.broadcast_to(
            np.resize(np.array([1, 2, 0], dtype=np.float32), n), (k, n)
        ).copy()
    rng = np.random.default_rng(case["seed"] * 101 + case["index"])
    return tuple(rng.uniform(-0.5, 0.5, shape).astype(np.float32) for shape in ((m, k), (k, n)))


def balanced_orders(repeats, seed):
    """Every variant occupies every position twice per twelve repetitions."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 12 or repeats % 12:
        raise ValueError("Repeats must be a positive multiple of twelve")
    rng = random.Random(seed)
    orders = []
    for _ in range(repeats // 12):
        base = list(VARIANTS)
        rng.shuffle(base)
        cycle = [base[i:] + base[:i] for i in range(len(base))]
        orders.extend(cycle + [list(reversed(order)) for order in cycle])
    return orders


def prepare_kernel(backend, resident, variant):
    a, b = resident
    kernel = source_for(backend, a.shape, b.shape, variant)
    shape = (a.shape[0], b.shape[1])
    data = backend.empty(shape, np.dtype("float32"))
    # Wrapping an already allocated backend buffer never uploads or allocates.
    from openpacho._array import ndarray

    out = ndarray._wrap(data, shape, np.dtype("float32"), backend)
    if backend.name == "opencl":

        def call():
            backend._run(kernel.source, kernel.count * 64, a._data, b._data, data, local_size=64)
            return out
    elif backend.name == "cuda":

        def call():
            backend._run(kernel.source, kernel.count, a._data, b._data, data, threads=64)
            return out
    else:

        def call():
            backend._run(
                kernel.source,
                kernel.count,
                [(a._data, a.dtype), (b._data, b.dtype), (data, out.dtype)],
                cooperative=True,
                shared=kernel.shared,
            )
            return out

    return call, kernel


def strict_classification(a, b):
    """IEEE float32 sequential operations, including intermediate/product overflow."""
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    with np.errstate(all="ignore"):
        for k in range(a.shape[1]):
            result = result + a[:, k, None] * b[None, k, :]
    return result


def classification(value):
    return np.where(
        np.isnan(value), 3, np.where(np.isposinf(value), 1, np.where(np.isneginf(value), -1, 0))
    )


def run(output, backend_name, *, repeats=24, seed=1011, only=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output file")
    orders = balanced_orders(repeats, seed)
    if backend_name not in ("opencl", "vulkan", "cuda-custom"):
        raise ValueError("Choose one custom GPU backend")
    selected = [case for case in cases() if only is None or case["name"] in only]
    if not selected or only is not None and set(only) != {case["name"] for case in selected}:
        raise ValueError("Unknown or empty case selection")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [
        *Path(p.__file__).resolve().parent.glob("*.py"),
        *Path(__file__).parent.glob("gemm_*py"),
    ]
    report = {
        "schema": "openpacho.gemm-accumulation.v1",
        "complete": False,
        "backend": backend_name,
        "seed": seed,
        "repeats": repeats,
        "orders": orders,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
        "timing_scope": "synchronized preallocated resident kernel launch wall time, including Python dispatch; no upload, allocation or download; not public API end-to-end performance",
        "warmup": "first call excluded, then at least 3 calls and 0.10 seconds per variant",
        "reference": "full float64 matmul; selected coordinates additionally use exact dyadic sums and Decimal80 error calculations, both original products and rounded float32 products",
        "before": nvidia_snapshot(),
        "rows": [],
    }

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    try:
        xp, backend = engine(backend_name)
        for case in selected:
            print(f"{backend_name}: {case['name']}", flush=True)
            a, b = make_inputs(case)
            resident = tuple(p.asarray(value, backend=backend) for value in (a, b))
            calls, kernels, outputs = {}, {}, {}
            row = {
                "case": case,
                "variants": {},
                "input_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in (a, b)],
            }
            report["rows"].append(row)
            for variant in VARIANTS:
                calls[variant], kernels[variant] = prepare_kernel(backend, resident, variant)
                first, _, result = measure(calls[variant], xp, backend)
                outputs[variant] = p.asnumpy(result).copy()
                row["variants"][variant] = {
                    "first_call_seconds": first,
                    "tile": kernels[variant].tile,
                    "kernel_sha256": hashlib.sha256(kernels[variant].source.encode()).hexdigest(),
                }
            public = p.asnumpy(resident[0] @ resident[1])
            row["public_uses_register_gemm"] = bool(
                min(a.shape[0], b.shape[1]) >= 32 and a.shape[1] >= 16
            )
            row["current_matches_public_bits"] = bool(
                np.array_equal(public.view(np.uint32), outputs["current"].view(np.uint32))
            )
            if row["public_uses_register_gemm"] and not row["current_matches_public_bits"]:
                raise RuntimeError("Current control does not reproduce the public matmul route")
            if case.get("kind") == "nonfinite":
                expected = strict_classification(a, b)
                row["expected_classification"] = classification(expected).tolist()
                for variant in VARIANTS:
                    actual = classification(outputs[variant])
                    row["variants"][variant].update(
                        classification=actual.tolist(),
                        classification_matches_sequential=bool(
                            np.array_equal(actual, classification(expected))
                        ),
                    )
            else:
                with threadpool_limits(limits=1, user_api="blas"):
                    reference = a.astype(np.float64) @ b.astype(np.float64)
                m, n = reference.shape
                difference = np.abs(outputs["current"].astype(np.float64) - reference)
                worst = np.unravel_index(
                    np.argmax(difference / (5e-5 + 5e-5 * np.abs(reference))), reference.shape
                )
                coordinates = sorted(
                    {
                        (0, 0),
                        (m - 1, n - 1),
                        (m // 2, n // 2),
                        (min(16, m - 1), min(653, n - 1)),
                        tuple(map(int, worst)),
                    }
                )
                refs = sampled_dot_references(a, b, coordinates)
                row["sampled_references"] = refs
                for variant in VARIANTS:
                    actual = outputs[variant]
                    row["variants"][variant].update(
                        validation=validate_result(actual, reference),
                        error=error_metrics(actual, reference),
                        sampled_errors=evaluate_sampled_errors(actual, refs),
                    )
            # Tiny adversarial cases are accuracy probes, not speed evidence.
            if "kind" not in case:
                samples = {variant: [] for variant in VARIANTS}
                for variant in VARIANTS:
                    row["variants"][variant]["warmup"] = warmup(
                        calls[variant], xp, backend, seconds=0.10
                    )
                for order in orders:
                    for variant in order:
                        elapsed, _, _ = measure(calls[variant], xp, backend)
                        samples[variant].append(elapsed)
                for variant in VARIANTS:
                    replay = p.asnumpy(calls[variant]())
                    row["variants"][variant].update(
                        samples_seconds=samples[variant],
                        timing=_sample_stats(samples[variant]),
                        replay_matches_bits=bool(
                            np.array_equal(replay.view(np.uint32), outputs[variant].view(np.uint32))
                        ),
                    )
                    if not row["variants"][variant]["replay_matches_bits"]:
                        raise RuntimeError("Kernel output changed across timing repetitions")
            save()
        report["complete"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["after"] = nvidia_snapshot()
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("opencl", "vulkan", "cuda-custom"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1011)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    run(args.output, args.backend, repeats=args.repeats, seed=args.seed, only=args.cases)
