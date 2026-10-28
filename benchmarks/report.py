"""Generate transparent CSV summaries, static charts and an English report from raw data."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"opencl": "OpenPacho / OpenCL", "cuda": "CuPy / CUDA"}
COLORS = {"opencl": "#167d8d", "cuda": "#bf5b20"}


def summarize(report):
    grouped = defaultdict(list)
    for row in report["rows"]:
        grouped[row["case_id"]].append(row)
    comparisons = []
    for case_id, rows in grouped.items():
        cpu = min((r for r in rows if r["mode"] == "cpu"), key=lambda r: r["median_s"])
        for row in rows:
            if row["mode"] not in ("resident", "end_to_end"):
                continue
            seed = int(
                hashlib.sha256((case_id + row["backend"] + row["mode"]).encode()).hexdigest()[:8],
                16,
            )
            rng = np.random.default_rng(seed)
            c, g = np.array(cpu["samples_s"]), np.array(row["samples_s"])
            indices = rng.integers(0, len(c), (2000, len(c)))
            ratios = np.median(c[indices], axis=1) / np.median(g[indices], axis=1)
            low, high = np.quantile(ratios, [0.025, 0.975])
            ratio = cpu["median_s"] / row["median_s"]
            classification = (
                "faster"
                if ratio >= 1.1 and low > 1
                else "slower"
                if ratio <= 1 / 1.1 and high < 1
                else "unclear"
            )
            comparisons.append(
                {
                    "case_id": case_id,
                    "operation": row["operation"],
                    "shape": row["shape"],
                    "dtype": row["dtype"],
                    "steps": row["steps"],
                    "backend": row["backend"],
                    "mode": row["mode"],
                    "cpu_backend": cpu["backend"],
                    "cpu_s": cpu["median_s"],
                    "gpu_s": row["median_s"],
                    "speedup": ratio,
                    "bootstrap_low": float(low),
                    "bootstrap_high": float(high),
                    "classification": classification,
                    "valid": row["validation"]["valid"],
                }
            )
    return comparisons


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_matmul(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for i, dtype in enumerate(("float32", "float64")):
        for j, mode in enumerate(("resident", "end_to_end")):
            ax = axes[i, j]
            for backend, label in LABELS.items():
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["operation"] == "matmul"
                        and len(set(r["shape"])) == 1
                        and r["dtype"] == dtype
                        and r["mode"] == mode
                        and r["backend"] == backend
                    ),
                    key=lambda r: r["shape"][0],
                )
                if not selected:
                    continue
                x = [r["shape"][0] for r in selected]
                y = [r["speedup"] for r in selected]
                ax.plot(x, y, "o-", color=COLORS[backend], label=label)
                ax.fill_between(
                    x,
                    [r["bootstrap_low"] for r in selected],
                    [r["bootstrap_high"] for r in selected],
                    color=COLORS[backend],
                    alpha=0.12,
                )
            ax.axhline(1, color="#555555", ls="--", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.set_title(
                f"{dtype} · {'resident data' if mode == 'resident' else 'includes upload and download'}"
            )
            ax.set_ylabel("NumPy time / GPU time")
            ax.set_xlabel("Square matrix side length")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Matrix multiplication: above 1× means the GPU wins\nBaseline: fastest NumPy measured across 1, 8, and 16 threads",
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, output, "matmul-speedup")


def plot_workloads(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, op, title in zip(
        axes.flat,
        ("add", "transcendental", "distance", "reuse"),
        (
            "Addition: little computation per byte",
            "sin / exp / cos: more computation",
            "Distance: scalar output",
            "100 steps, one upload and download",
        ),
    ):
        for backend, label in LABELS.items():
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["operation"] == op
                    and r["dtype"] == "float32"
                    and r["mode"] == "end_to_end"
                    and r["backend"] == backend
                    and (op != "reuse" or r["steps"] == 100)
                ),
                key=lambda r: r["shape"][0],
            )
            ax.plot(
                [r["shape"][0] for r in selected],
                [r["speedup"] for r in selected],
                "o-",
                color=COLORS[backend],
                label=label,
            )
        ax.axhline(1, ls="--", color="#555555", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Number of elements")
        ax.set_ylabel("NumPy time / GPU time")
        ax.set_title(title)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("End-to-end cost including transfers · float32", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output, "workload-speedup")


def plot_transfers(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode, title in zip(axes, ("h2d", "d2h"), ("Host → GPU", "GPU → host")):
        for backend, label in LABELS.items():
            selected = sorted(
                (r for r in report["transport"] if r["backend"] == backend and r["mode"] == mode),
                key=lambda r: r["bytes"],
            )
            ax.plot(
                [r["bytes"] / 2**20 for r in selected],
                [r["GB_per_s"] for r in selected],
                "o-",
                color=COLORS[backend],
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Copy size (MiB)")
        ax.set_ylabel("Effective GB/s")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Copies with preallocated buffers and pageable host memory", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output, "transfer-bandwidth")


def plot_solvers(rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for i, dtype in enumerate(("float32", "float64")):
        for j, op in enumerate(("cg", "jacobi", "sd")):
            ax = axes[i, j]
            for backend, label in LABELS.items():
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["operation"] == op
                        and r["dtype"] == dtype
                        and r["mode"] == "end_to_end"
                        and r["backend"] == backend
                    ),
                    key=lambda r: r["shape"][0],
                )
                ax.plot(
                    [r["shape"][0] for r in selected],
                    [r["speedup"] for r in selected],
                    "o-",
                    label=label,
                    color=COLORS[backend],
                )
            ax.axhline(1, ls="--", color="#555555", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            name = {
                "cg": "Conjugate gradient",
                "jacobi": "Jacobi",
                "sd": "Gradient descent",
            }[op]
            ax.set_title(f"{name} · {dtype}")
            ax.set_xlabel("Number of variables")
            ax.set_ylabel("NumPy time / GPU time")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Solvers including upload and download: size changes the outcome\nSame recurrence and tolerance; above 1× means the GPU wins",
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, output, "solver-speedup")


def table_row(case_id, comparisons):
    found = {
        r["backend"]: r
        for r in comparisons
        if r["case_id"] == case_id and r["mode"] == "end_to_end"
    }
    if not found:
        return None
    first = next(iter(found.values()))
    values = [f"`{case_id}`", f"{first['cpu_s'] * 1000:.4f} ({first['cpu_backend']})"]
    for backend in LABELS:
        row = found.get(backend)
        values.append(f"{row['gpu_s'] * 1000:.4f} / {row['speedup']:.2f}×" if row else "—")
    return "| " + " | ".join(values) + " |"


def build_report(raw, comparisons, output, cold=None):
    metadata = raw["metadata"]
    lines = [
        "# Complete OpenPacho, NumPy, and CUDA benchmarks",
        "",
        "Local results for the existing implementation. No kernels were modified for this comparison.",
        "",
        "## Hypothesis and criterion",
        "",
        (
            "The hypothesis under test is that transfer costs prevent OpenPacho from outperforming NumPy, even with CUDA. "
            "The performance claim is separated from its causal explanation. CuPy/CUDA is an independent control with "
            "different kernels, libraries, and allocator; OpenPacho continues to use OpenCL."
        ),
        "",
        (
            "Each configuration uses **the fastest NumPy measured across 1, 8, and 16 threads** as its reference. "
            "GPU/CPU did not run concurrently. The comparison uses synchronized wall time, the same dtype, and validated results. "
            "A ratio above 1 means the GPU takes less time."
        ),
        "",
        "## Test grid summary",
        "",
        "| Backend | Scenario | Clear wins | Clear losses | Small/uncertain difference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for backend, label in LABELS.items():
        for mode in ("resident", "end_to_end"):
            counts = Counter(
                r["classification"]
                for r in comparisons
                if r["backend"] == backend and r["mode"] == mode
            )
            lines.append(
                f"| {label} | {mode} | {counts['faster']} | {counts['slower']} | {counts['unclear']} |"
            )
    lines.extend(
        [
            "",
            (
                "A “clear” result requires a median difference of at least 10% and a descriptive 95% bootstrap "
                "interval that does not cross 1. This is not a probability of winning in real applications: counts "
                "depend on this test grid. The CPU baseline is selected using the same data; "
                "this favors the CPU, and intervals do not correct for selection or multiple comparisons."
            ),
            "",
            "## End-to-end cost examples",
            "",
            (
                "Includes upload, computation, and download, with the runtime already initialized. Times are in milliseconds; "
                "the second value is NumPy time divided by GPU time."
            ),
            "",
            "| Configuration | NumPy ms (threads) | OpenPacho ms / ratio | CuPy ms / ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    selected = [
        "add-16777216-float32-k1",
        "transcendental-16777216-float32-k1",
        "distance-16777216-float32-k1",
        "reuse-4194304-float32-k100",
        "matmul-1024x1024x1024-float32-k1",
        "matmul-2048x2048x2048-float32-k1",
        "matmul-2048x2048x2048-float64-k1",
        "matvec-4096x4096-float32-k1",
        "cg-1024x1024-float32-k1",
        "jacobi-1024x1024-float32-k1",
    ]
    for case in selected:
        row = table_row(case, comparisons)
        if row:
            lines.append(row)
    lines.extend(
        [
            "",
            "## Jacobi and gradient descent",
            "",
            (
                "The table includes resident timings. Losing with resident data cannot be attributed to uploading "
                "and downloading the entire matrix for each solve. Synchronization and small scalar "
                "transfers at each iteration remain."
            ),
            "",
            "| Method and size | NumPy ms | OpenCL resident / end-to-end ms | CUDA resident / end-to-end ms |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for dtype, n in (("float32", 1024), ("float32", 4096), ("float32", 8192), ("float64", 8192)):
        for method in ("cg", "jacobi", "sd"):
            cid = f"{method}-{n}x{n}-{dtype}-k1"
            selected_rows = [r for r in comparisons if r["case_id"] == cid]
            if not selected_rows:
                continue
            values = [f"{method} {n} {dtype}", f"{selected_rows[0]['cpu_s'] * 1000:.3f}"]
            for backend in LABELS:
                modes = {
                    r["mode"]: r["gpu_s"] * 1000 for r in selected_rows if r["backend"] == backend
                }
                values.append(f"{modes['resident']:.3f} / {modes['end_to_end']:.3f}")
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "![Solver scaling](solver-speedup.png)", ""])
    lines.extend(
        [
            "",
            "![Matrix multiplication](matmul-speedup.png)",
            "",
            "![Different workloads](workload-speedup.png)",
            "",
            "## Separate transfer costs from execution costs",
            "",
            (
                "`resident` keeps large arrays on the GPU. It includes Python dispatch, temporaries, allocation/deallocation, "
                "and synchronization. OpenPacho also uploads scalar operands: the 100-step run includes "
                "200 small scalar uploads. This is not kernel-only time."
            ),
            "",
            (
                "`upload` and `download` measure the complete APIs separately. OpenPacho upload includes a contiguous "
                "host copy and buffer creation. The transport test uses existing buffers and measures something different. "
                "Summing phase medians is not assumed to reconstruct the end-to-end median exactly."
            ),
            "",
            (
                "CuPy retains its default memory pool; OpenPacho creates new OpenCL buffers. These differences "
                "are part of the cost of using the current libraries, not proof that the OpenCL standard is inferior."
            ),
            "",
            "![Copy bandwidth](transfer-bandwidth.png)",
            "",
            "## Conditions and reproducibility",
            "",
            f"- System: `{metadata['platform']}`; Python `{metadata['python']}`.",
            f"- CPU: `{metadata['processor']}`; {metadata['logical_cpus']} logical processors. BLAS was controlled with threadpoolctl.",
            f"- OpenCL GPU: `{metadata.get('selected_opencl_device', 'not selected')}`.",
            f"- GPU CUDA: `{metadata.get('cuda_device', {})}`.",
            f"- Versions: `{metadata['versions']}`.",
            f"- Runtime CUDA: `{metadata.get('cuda_runtime')}`; TF32 override: `{metadata.get('cuda_tf32_override')}`.",
            f"- Configurations: {len(metadata['case_order'])}; raw rows: {len(raw['rows'])}; errors: {len(raw['errors'])}.",
            (
                f"- Rounds per row: {sorted({len(r['samples_s']) for r in raw['rows']})}; adaptive blocks of up to {metadata['arguments']['max_batch']} calls. "
                "Each call is synchronized, even within a block; each sample is its mean time."
            ),
            "- Configuration and row order randomized with a recorded seed; validation outside the clock.",
            (
                "- Transport: fixed ascending size order, OpenCL followed by CUDA; useful as a diagnostic, "
                "without attributing statistical significance to small bandwidth differences."
            ),
            f"- Sources unchanged during the run: `{metadata['source_unchanged']}`; hashes in raw.json.",
            f"- Suite duration: {metadata['elapsed_s']:.1f} s; GPU telemetry before, during, and after the run in raw.json.",
            "",
            (
                "Cases: addition, reduction, transcendental functions, distance, 1/10/100-step sequences, "
                "square/rectangular matmul, matvec, materialized transpose, and the three solvers. "
                "Float32 and float64 are compared separately. The largest vector has 16,777,216 elements; "
                "matmul reaches 2048² and matvec/transpose 4096². The solver extension reaches 8192 variables "
                "(512 MiB per float64 matrix) with 7 rounds and one call per sample."
            ),
            "",
            (
                "Solvers use the same recurrence, finiteness checks, and true-residual criterion. "
                "The SPD matrix has diagonal 2.5 and sub/superdiagonal -1, stored densely. "
                "It does not represent all condition numbers or sparse problems. Iterations and residuals are saved "
                "in resident GPU solver rows; end-to-end rows also validate the result. "
                "NumPy uses an equivalent iterative solver, not a timed direct solve."
            ),
            "",
            "## Initialization and first call",
            "",
        ]
    )
    if cold:
        lines.extend(
            [
                (
                    "Fresh processes; the existing CuPy cache is retained and the OpenCL disk cache is disabled in this run. The metric sums imports/initialization "
                    "and the first matmul including transfers; it excludes input generation and validation. "
                    "These are three repetitions and should not be interpreted as stable system startup times."
                ),
                "",
                "| Backend | dtype | Median initialization ms | First operation ms | Median sum ms |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for backend in ("numpy-1", "numpy-8", "numpy-16", "opencl", "cuda"):
            for dtype in ("float32", "float64"):
                rs = [r for r in cold["rows"] if r["backend"] == backend and r["dtype"] == dtype]
                med = lambda key, rows=rs: np.median([r[key] for r in rows]) * 1000
                lines.append(
                    f"| {backend} | {dtype} | {med('initialization_s'):.1f} | {med('first_end_to_end_s'):.1f} | {med('startup_plus_first_s'):.1f} |"
                )
    else:
        lines.append(
            "First use within the suite is recorded in first_call_s; it is not equivalent to a cold process."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `raw.json`: every sample, metadata, validation, hashes, and errors.",
            "- `comparisons.csv`: each GPU/scenario against the best measured NumPy configuration.",
            "- `timings.csv`: medians, dispersion, bytes, and phases for every row.",
            "- PNG and SVG figures: exportable results.",
            "",
            (
                "The result describes only this machine, this code, and these inputs. It does not prove that every "
                "GPU wins or that CUDA is intrinsically superior to OpenCL. Timing design drew on "
                "[the official CuPy documentation](https://docs.cupy.dev/en/stable/user_guide/performance.html), "
                "particularly synchronization, first calls, and memory allocation."
            ),
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold", type=Path)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.raw.resolve() == (args.output / "raw.json").resolve():
        parser.error("use a separate report directory to preserve the original raw measurements")
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    original_metadata = json.loads(json.dumps(raw["metadata"]))
    raw["metadata"]["runs"] = [{"path": str(args.raw), "metadata": original_metadata}]
    for extra_path in args.extra:
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        for path, digest in raw["metadata"]["source_sha256"].items():
            if path.startswith("src/") and extra["metadata"]["source_sha256"].get(path) != digest:
                raise ValueError(f"Runtime source differs between runs: {path}")
        raw["rows"].extend(extra["rows"])
        raw["errors"].extend(extra["errors"])
        raw["metadata"]["case_order"].extend(extra["metadata"]["case_order"])
        raw["metadata"]["elapsed_s"] += extra["metadata"]["elapsed_s"]
        raw["metadata"]["source_unchanged"] &= extra["metadata"]["source_unchanged"]
        raw["metadata"]["runs"].append({"path": str(extra_path), "metadata": extra["metadata"]})
    comparisons = summarize(raw)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(
        json.dumps(raw, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_csv(args.output / "comparisons.csv", comparisons)
    flattened = [
        {k: v for k, v in row.items() if k not in ("samples_s", "validation")}
        for row in raw["rows"]
    ]
    write_csv(args.output / "timings.csv", flattened)
    plot_matmul(comparisons, args.output)
    plot_workloads(comparisons, args.output)
    plot_transfers(raw, args.output)
    plot_solvers(comparisons, args.output)
    cold = json.loads(args.cold.read_text(encoding="utf-8")) if args.cold else None
    if cold:
        (args.output / "cold-start.json").write_text(json.dumps(cold, indent=2), encoding="utf-8")
    build_report(raw, comparisons, args.output, cold)
    print(
        json.dumps(
            {
                f"{b}/{m}": dict(
                    Counter(
                        r["classification"]
                        for r in comparisons
                        if r["backend"] == b and r["mode"] == m
                    )
                )
                for b in LABELS
                for m in ("resident", "end_to_end")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
