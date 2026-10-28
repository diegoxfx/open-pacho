"""Export complete gap measurements without selecting only favorable results."""

import argparse
import csv
import json
from pathlib import Path


def run(root, output, paths=None):
    output.mkdir(parents=True, exist_ok=True)
    paths = paths or (
        "gaps-execution-final/raw.json",
        "gaps-solvers-final/raw.json",
        "gaps-final/scale.json",
    )
    reports = [json.loads((root / path).read_text(encoding="utf-8")) for path in paths]
    rows = []
    for report in reports:
        if report.get("unavailable"):
            raise ValueError(f"Unavailable implementations: {report['unavailable']}")
        for row in report["rows"]:
            if "error" in row:
                raise ValueError(f"Failed measurement: {row}")
            label = (
                str(row["size"]) if "size" in row else f"{row['left_shape']} @ {row['right_shape']}"
            )
            for mode in ("resident", "end_to_end"):
                samples = row[f"{mode}_seconds"]
                if len(samples) != report["repeats"]:
                    raise ValueError("Incomplete benchmark samples")
                import numpy as np

                rows.append(
                    {
                        "case": row["case"],
                        "size_or_shapes": label,
                        "dtype": row["dtype"],
                        "implementation": row["implementation"],
                        "mode": mode,
                        "median_ms": row[f"{mode}_median_seconds"] * 1000,
                        "p25_ms": float(np.percentile(samples, 25)) * 1000,
                        "p75_ms": float(np.percentile(samples, 75)) * 1000,
                        "iterations": row.get("iterations", ""),
                        "max_abs_error": row.get("numerical_error", {}).get("max_abs_error", ""),
                        "relative_l2_error": row.get("numerical_error", {}).get(
                            "relative_l2_error", ""
                        ),
                        "reference": row.get("numerical_error", {}).get("reference", ""),
                        "reference_precision": row.get("numerical_error", {}).get(
                            "reference_precision", ""
                        ),
                        "actual_nonfinite_count": row.get("numerical_error", {}).get(
                            "actual_nonfinite_count", ""
                        ),
                        "true_residual_norm": row.get("solver_error", {}).get(
                            "true_residual_norm", ""
                        ),
                        "backward_error": row.get("solver_error", {}).get("backward_error", ""),
                    }
                )
    with (output / "all-results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Mediciones completas",
        "",
        f"{len(rows) // 2} configuraciones, ambos modos, siete muestras por modo. Milisegundos.",
        "",
        (
            "NumPy: mínimo de medianas entre 1/8/16 hilos, independientemente por modo. "
            "OpenPacho: compile en operaciones; solver de la implementación medida en solve; eager en controles scale. "
            "CSV conserva también eager y todos los hilos."
        ),
        "",
    ]
    for mode in ("resident", "end_to_end"):
        text += [
            f"## {mode}",
            "",
            "| Caso / tamaño / dtype | NumPy | OpenCL | CLBlast | Vulkan | CuPy | cuBLAS | nvmath |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        keys = dict.fromkeys((r["case"], r["size_or_shapes"], r["dtype"]) for r in rows)
        for key in keys:
            matches = [
                r
                for r in rows
                if (r["case"], r["size_or_shapes"], r["dtype"]) == key and r["mode"] == mode
            ]
            values = []
            for backend in (
                "numpy",
                "opencl",
                "opencl-clblast",
                "vulkan",
                "cupy",
                "cublas",
                "nvmath",
            ):
                options = [
                    r["median_ms"]
                    for r in matches
                    if (
                        r["implementation"].startswith("numpy-")
                        if backend == "numpy"
                        else r["implementation"]
                        in (backend, backend + "-compiled", backend + "-solver")
                    )
                ]
                values.append(f"{min(options):.3f}" if options else "—")
            text.append("| " + " / ".join(key) + " | " + " | ".join(values) + " |")
        text.append("")
    text += [
        "## Precisión junto al tiempo residente",
        "",
        "Los errores son diagnósticos fuera del cronómetro. Float64 frente a float64 es una comparación de igual precisión; no certifica exactitud. Para solvers se usa la solución generadora y se identifica el redondeo del RHS. CSV conserva la referencia y precisión exactas.",
        "",
        "| Caso / tamaño / dtype | Implementación | ms | Error abs. máx | Error L2 relativo | Residuo verdadero | Error hacia atrás |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def diagnostic(value):
        return (
            "—"
            if value == "" or value is None
            else f"{value:.3e}"
            if isinstance(value, (float, int))
            else str(value)
        )

    for row in rows:
        if row["mode"] == "resident":
            text.append(
                "| "
                + " | ".join(
                    [
                        f"{row['case']} / {row['size_or_shapes']} / {row['dtype']}",
                        row["implementation"],
                        f"{row['median_ms']:.3f}",
                        *(
                            diagnostic(row[key])
                            for key in (
                                "max_abs_error",
                                "relative_l2_error",
                                "true_residual_norm",
                                "backward_error",
                            )
                        ),
                    ]
                )
                + " |"
            )
    (output / "all-results.md").write_text("\n".join(text), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    selectors = [
        ("gemv", "[8192, 8192] @ [8192]", "GEMV 8192² · matriz de 256 MiB"),
        ("jacobi_solve", "2048", "Jacobi completo · 2048 variables"),
        ("matmul", "2048", "Matmul 2048²"),
    ]
    names = ("numpy", "opencl", "opencl-clblast", "vulkan", "cupy")
    for axis, (case, size, title) in zip(axes, selectors):
        for mode, dx, color, label in (
            ("resident", -0.19, "#2563eb", "Datos residentes"),
            ("end_to_end", 0.19, "#ea580c", "Carga + cálculo + descarga"),
        ):
            values = []
            for name in names:
                found = [
                    r["median_ms"]
                    for r in rows
                    if r["case"] == case
                    and r["size_or_shapes"] == size
                    and r["dtype"] == "float32"
                    and r["mode"] == mode
                    and (
                        r["implementation"].startswith("numpy-")
                        if name == "numpy"
                        else r["implementation"] in (name, name + "-compiled", name + "-solver")
                    )
                ]
                values.append(min(found))
            axis.bar(np.arange(len(names)) + dx, values, width=0.36, color=color, label=label)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(
            range(len(names)), ["NumPy", "OpenCL", "CLBlast", "Vulkan", "CuPy"], rotation=35
        )
        axis.set_yscale("log")
        axis.set_ylabel("Mediana (ms, escala logarítmica)")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "OpenPacho: dónde ayuda la residencia y dónde persiste el coste de ejecución", fontsize=13
    )
    fig.savefig(output / "gaps.png", dpi=160)
    fig.savefig(output / "gaps.svg")
    plt.close(fig)
    print(f"{len(rows) // 2} configurations exported", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".artifacts/benchmarks"))
    parser.add_argument("--output", type=Path, default=Path(".artifacts/benchmarks/gaps-report"))
    parser.add_argument("--operations", default="gaps-execution-final/raw.json")
    parser.add_argument("--solvers", default="gaps-solvers-final/raw.json")
    parser.add_argument("--scale", default="gaps-final/scale.json")
    args = parser.parse_args()
    run(args.root, args.output, (args.operations, args.solvers, args.scale))
