"""Render synchronized execution measurements into Markdown, CSV and a static plot."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(inputs, output, artifacts):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    rows = [row for report in reports for row in report["rows"]]
    if any(
        "error" in row or len(row["resident_seconds"]) != 7 or len(row["end_to_end_seconds"]) != 7
        for row in rows
    ):
        raise ValueError("This report requires complete, valid seven-round measurements")
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "measurements.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "case",
                "size",
                "dtype",
                "implementation",
                "mode",
                "median_ms",
                "p25_ms",
                "p75_ms",
                "samples_ms",
            ]
        )
        for row in rows:
            for mode in ("resident", "end_to_end"):
                values = np.array(row[f"{mode}_seconds"]) * 1000
                writer.writerow(
                    [
                        row["case"],
                        row["size"],
                        row["dtype"],
                        row["implementation"],
                        mode,
                        *np.quantile(values, [0.5, 0.25, 0.75]),
                        json.dumps(values.tolist()),
                    ]
                )
    keys = list(dict.fromkeys((r["case"], r["size"], r["dtype"]) for r in rows))
    names = ["opencl", "opencl-clblast", "vulkan", "cupy", "cublas", "nvmath"]
    labels = ["OpenCL custom", "OpenCL CLBlast", "Vulkan", "CuPy", "cuBLAS", "nvmath"]

    def group(key):
        return {r["implementation"]: r for r in rows if (r["case"], r["size"], r["dtype"]) == key}

    def selected(key, name):
        suffix = "-solver" if key[0].endswith("_solve") else "-compiled"
        return group(key).get(name + suffix if name in names[:3] else name)

    def milliseconds(case, size, implementation=None, mode="resident"):
        data = group((case, size, "float32"))
        value = (
            min(
                row[f"{mode}_median_seconds"]
                for name, row in data.items()
                if name.startswith("numpy")
            )
            if implementation is None
            else data[implementation][f"{mode}_median_seconds"]
        )
        return f"{value * 1000:.3f}"

    m = milliseconds
    solver_count = sum(row["case"].endswith("_solve") for row in rows)
    lines = [
        "# Compiler, Vulkan, and CUDA references: local results",
        "",
        (
            f"Platform: {reports[0]['platform']}. Devices: {reports[0]['devices']}. "
            "The original session on 2026-09-06 used Windows 11, RTX 5060 Ti "
            "16 GB (driver 616.64), Ryzen 7 5700X3D, and 32 GB RAM. "
            "AMD/Intel have not been validated."
        ),
        "",
        (
            f"**{len(rows)} configurations, seven rounds per mode, zero validation errors.** "
            f"{len(rows) - solver_count} operation/step configurations and {solver_count} complete solves. "
            "JSON preserves every sample; CSV adds quartiles."
        ),
        "",
        "## What changes and which costs remain",
        "",
        (
            f"100 updates of 1,048,576 float32 values: {m('updates100', 1048576)} ms "
            f"for the best NumPy result, {m('updates100', 1048576, 'opencl-eager')} ms for eager OpenCL, "
            f"{m('updates100', 1048576, 'opencl-compiled')} ms with compile, and "
            f"{m('updates100', 1048576, 'cupy')} ms for CuPy. "
            f"Including transfers: {m('updates100', 1048576, mode='end_to_end')} ms "
            f"NumPy and {m('updates100', 1048576, 'opencl-compiled', 'end_to_end')} ms compiled OpenCL."
        ),
        "",
        (
            f"Matmul 2048×2048 float32: {m('matmul', 2048)} ms NumPy, "
            f"{m('matmul', 2048, 'opencl-clblast-compiled')} ms OpenCL+CLBlast, "
            f"{m('matmul', 2048, 'cublas')} ms cuBLAS and {m('matmul', 2048, 'nvmath')} ms nvmath. "
            "The main advantage here comes from the matrix provider: compiling an isolated "
            "matmul neither fuses it nor necessarily improves its kernel."
        ),
        "",
        (
            f"Complete Jacobi solve at 2048 float32: {m('jacobi_solve', 2048)} ms NumPy, "
            f"{m('jacobi_solve', 2048, 'opencl-solver')} ms custom OpenCL and "
            f"{m('jacobi_solve', 2048, 'cupy')} ms CuPy. Steepest descent: "
            f"{m('sd_solve', 2048)} ms NumPy, {m('sd_solve', 2048, 'opencl-solver')} ms OpenCL and "
            f"{m('sd_solve', 2048, 'cupy')} ms CuPy. These are resident timings: transfers "
            "are not the only source of differences. GEMV/DOT, dispatch, allocations, "
            "and convergence synchronization remain. CLBlast alone does not solve the algorithm."
        ),
        "",
        (
            f"Vulkan: resident matmul 2048 float32 {m('matmul', 2048, 'vulkan-compiled')} ms, "
            f"with transfers {m('matmul', 2048, 'vulkan-compiled', 'end_to_end')} ms; "
            f"100 large updates, {m('updates100', 1048576, 'vulkan-compiled')} ms. "
            "It works; pooling, fusion, and lower submission overhead are still missing. "
            "These numbers do not represent a limit of Vulkan as a technology."
        ),
        "",
        (
            f"100 updates of only 256 values: {m('updates100', 256)} ms NumPy and "
            f"{m('updates100', 256, 'opencl-compiled')} ms compiled OpenCL. Correctness "
            "is preserved; making the GPU win this case is not a priority."
        ),
        "",
        "## Method and comparison limits",
        "",
        (
            "- Synchronized wall time, including Python dispatch and result allocation. "
            "Implementation order is randomized across rounds. Asynchronous launch times are not reported as complete computation."
        ),
        (
            "- Resident: inputs already uploaded, plans prepared, and kernels warmed up. "
            "Each repetition resets the iterations' logical state. "
            "Validation and result download are outside this timing."
        ),
        (
            "- With transfers: new input buffers, computation, and download. "
            "Plans/programs are reused; nvmath rebinds operands without replanning. "
            "NumPy uses existing host arrays: no artificial copy is added."
        ),
        (
            "- JSON separates upload, planning, first execution, and download. "
            "First execution refers to that configuration within the process; "
            "it may reuse caches from earlier configurations. It does not measure "
            "an independent cold process or include device initialization. "
            "OpenPacho capture is included in its first execution; explicit "
            "nvmath planning is reported separately."
        ),
        (
            "- NumPy is measured with BLAS limited to 1, 8, and 16 threads. Tables show "
            "the best median for each case and mode. Changing that limit does not "
            "parallelize non-BLAS operations; small differences there are noise."
        ),
        (
            "- Original float32/float64, TF32 disabled; nvmath uses COMPUTE_32F/64F. "
            "Shape, dtype, and values are checked in every round. Bitwise identity is not required."
        ),
        (
            "- `updates100`: x=(1-rate)*x+rate*b, one hundred steps. `energy`: "
            "sum((2*(a*b+c))²,axis=-1), size×32 arrays. Jacobi20 and gradient20 use "
            "20 steps with a dense diagonally dominant SPD matrix; gradient20 uses a fixed rate."
        ),
        (
            "- `_solve` cases run solvers from examples/solvers.py to convergence, "
            "with the true residual at each iteration, rtol=1e-4/1e-7 for float32/64 and "
            "maxiter=400. Steepest descent uses an exact line search. "
            "NumPy/CuPy execute the same recurrence through native operations "
            "in `benchmarks.suite.native_solver`. Scalar coefficients may use "
            "different paths/intermediate precision. The host residual is also checked "
            "against the tolerance (a 2% margin for reduction order). "
            "Inputs are tridiagonal systems stored densely; this is not a sparse benchmark."
        ),
        (
            "- CUDA is the platform. The cuBLAS reference uses direct bindings; "
            "CuPy matmul may also use cuBLAS, and nvmath uses cuBLASLt. "
            "These are not three independent devices. cuBLAS/nvmath are compared "
            "only for matmul; no high-level solver is invented for those interfaces."
        ),
        (
            "- PyTorch/JAX, peak memory usage, energy, every rectangular shape, "
            "and performance portability across vendors were not measured. "
            "This suite covers the current scope, not every possible workload."
        ),
        "",
    ]
    lines += [
        "## Resident timings",
        "",
        (
            "Milliseconds; lower is better. OpenPacho uses compile for operations/steps; "
            "in `_solve`, the measured solver already compiles its updates. "
            "Complete eager and compiled results are available in JSON/CSV."
        ),
        "",
    ]
    for mode in ("resident", "end_to_end"):
        if mode == "end_to_end":
            lines += [
                "## Upload + compute + download",
                "",
                "Milliseconds; warmed-up plans and programs.",
                "",
            ]
        lines += [
            "| Case / size / dtype | Best NumPy | " + " | ".join(labels) + " |",
            "| --- | " + " | ".join(["---:"] * 7) + " |",
        ]
        for key in keys:
            data = group(key)
            cpu = min(
                row[f"{mode}_median_seconds"]
                for name, row in data.items()
                if name.startswith("numpy")
            )
            values = [f"{1000 * cpu:.3f}"]
            for name in names:
                row = selected(key, name)
                values.append(f"{1000 * row[f'{mode}_median_seconds']:.3f}" if row else "—")
            lines.append(f"| {key[0]} / {key[1]} / {key[2]} | " + " | ".join(values) + " |")
        lines.append("")
    lines += [
        "## Next priorities",
        "",
        (
            "1. Reduce synchronization and execution costs in solvers without silently "
            "changing their check frequency or tolerances; optimize GEMV/DOT."
        ),
        (
            "2. Fuse map-reduce and improve decisions about shared temporaries. "
            "Measure an executor with lower overhead when Python dominates."
        ),
        (
            "3. Vulkan pooling and fusion, descriptor/command reuse, and optimized "
            "matrix kernels. Validate a second vendor."
        ),
        (
            "4. Extend frequent operations, such as batched matmul, based on real use cases. "
            "Keep CLBlast and custom kernels interchangeable under the contract; "
            "a CUDA backend and sparse operations remain future work."
        ),
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYOPENCL_NO_CACHE='1'",
        "$env:PACHO_VULKAN_COMPILER='<path to glslang or glslc>'",
        "$env:OPENPACHO_CLBLAST_LIBRARY='<path to CLBlast>'",
        "python -m benchmarks.execution --repeats 7 --output .artifacts/benchmarks/execution/raw.json",
        "python -m benchmarks.execution --solvers-only --repeats 7 --output .artifacts/benchmarks/execution-solvers/raw.json",
        "python -m benchmarks.execution_report --inputs .artifacts/benchmarks/execution/raw.json .artifacts/benchmarks/execution-solvers/raw.json --output docs/execution-results.md --artifacts .artifacts/benchmarks/execution-report",
        "```",
        "",
        (
            "Additional dependencies: the benchmark extra, CuPy and nvmath-python with "
            "compatible CUDA components, native CLBlast, Vulkan bindings, and a shader compiler. "
            "These are not part of OpenPacho's minimal CPU dependencies."
        ),
        "",
        "Measured versions: "
        + ", ".join(f"{name} {version}" for name, version in reports[0]["versions"].items())
        + ".",
        "Original local artifacts are in "
        + ", ".join(str(p.parent).replace("\\", "/") for p in inputs)
        + ".",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    # Full matrix of measured speedups; log colors expose both wins and losses.
    plot_names = names[:4]
    ratios = []
    for key in keys:
        data = group(key)
        best = min(
            row["resident_median_seconds"] for name, row in data.items() if name.startswith("numpy")
        )
        ratios.append(
            [best / selected(key, name)["resident_median_seconds"] for name in plot_names]
        )
    ratios = np.array(ratios)
    fig, ax = plt.subplots(figsize=(10, 12))
    chart = ax.imshow(np.log2(ratios), cmap="RdYlGn", vmin=-10, vmax=6, aspect="auto")
    ax.set_xticks(range(4), labels[:4], rotation=15)
    ax.set_yticks(range(len(keys)), [f"{c} · {n} · {dt}" for c, n, dt in keys], fontsize=9)
    for i in range(len(keys)):
        for j in range(4):
            ax.text(j, i, f"{ratios[i, j]:.3g}×", ha="center", va="center", fontsize=9)
    ax.set_title(
        "Resident speed relative to the best NumPy result\n>1× wins; <1× loses · 7 rounds · RTX 5060 Ti",
        pad=20,
    )
    fig.colorbar(chart, ax=ax, label="log₂(speed relative to NumPy)", shrink=0.65)
    fig.tight_layout()
    fig.savefig(artifacts / "resident-speedups.png", dpi=160)
    fig.savefig(artifacts / "resident-speedups.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    render(args.inputs, args.output, args.artifacts)
