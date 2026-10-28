"""Untimed numerical diagnostics, including range-safe norms and JSON values.

These are benchmark helpers, never part of the runtime or its stopping criterion.
An equal-precision reference is a comparison, not a high-precision oracle.
"""

import math
from decimal import Decimal, localcontext

import numpy as np


def number(value):
    """Strict JSON number, or an explicit string when outside binary64 range."""
    result = float(value)
    if result == 0 and value != 0:
        return str(value)
    return result if math.isfinite(result) else str(value)


def norm_parts(value):
    value = np.asarray(value, dtype=np.float64)
    scale = float(np.max(np.abs(value), initial=0))
    if scale == 0:
        return 0.0, 0.0
    return scale, float(np.sqrt(np.sum((value / scale) ** 2, dtype=np.float64)))


def stable_norm(value):
    scale, normalized = norm_parts(value)
    with localcontext() as context:
        context.prec = 80
        return number(Decimal.from_float(scale) * Decimal.from_float(normalized))


def precision_metadata():
    return {
        "float64_mantissa_bits": np.finfo(np.float64).nmant + 1,
        "longdouble_mantissa_bits": np.finfo(np.longdouble).nmant + 1,
        "longdouble_is_wider": bool(np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant),
        "decimal_precision_digits": 80,
    }


def error_metrics(actual, expected, *, reference="numpy-float64", precision="float64"):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"Diagnostic shape mismatch: {actual.shape} != {expected.shape}")
    result = {
        "actual_dtype": str(actual.dtype),
        "reference": reference,
        "reference_precision": precision,
        "reference_is_higher_precision": precision == "decimal80"
        or (precision == "float64" and actual.dtype.itemsize < 8),
        "actual_nonfinite_count": int(np.count_nonzero(~np.isfinite(actual))),
        "reference_nonfinite_count": int(np.count_nonzero(~np.isfinite(expected.astype(float)))),
        "max_abs_error": None,
        "relative_l2_error": None,
    }
    if result["actual_nonfinite_count"] or result["reference_nonfinite_count"]:
        result["status"] = "nonfinite"
        return result
    with localcontext() as context:
        context.prec = 80
        if precision == "decimal80":
            refs = list(expected.flat)
            differences = [Decimal.from_float(float(a)) - e for a, e in zip(actual.flat, refs)]
            absolute = max(map(abs, differences), default=Decimal(0))
            numerator = sum((d * d for d in differences), Decimal(0)).sqrt()
            denominator = sum((e * e for e in refs), Decimal(0)).sqrt()
        else:
            a, e = actual.astype(np.float64), expected.astype(np.float64)
            with np.errstate(over="ignore"):
                difference = a - e
            scale = 1.0
            if not np.all(np.isfinite(difference)):
                scale = max(
                    float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(e), initial=0))
                )
                difference = a / scale - e / scale
            absolute = Decimal.from_float(scale) * Decimal.from_float(
                float(np.max(np.abs(difference), initial=0))
            )
            ds, dn = norm_parts(difference)
            es, en = norm_parts(e)
            numerator = Decimal.from_float(scale) * Decimal.from_float(ds) * Decimal.from_float(dn)
            denominator = Decimal.from_float(es) * Decimal.from_float(en)
        relative = (
            numerator / denominator
            if denominator
            else (Decimal(0) if not numerator else Decimal("Infinity"))
        )
        result.update(
            max_abs_error=number(absolute), relative_l2_error=number(relative), status="finite"
        )
    return result


def solver_metrics(a, b, x, *, reference_solution=None, reference="known-generating-solution"):
    """True residual in scaled float64; normwise backward error uses Frobenius A.

    Scaling avoids overflow in A*x and in squared norms. This diagnostic still
    has binary64 rounding error; the small accuracy suite also has Decimal data.
    """
    original_x = np.asarray(x)
    a, b, x = (np.asarray(v, dtype=np.float64) for v in (a, b, x))
    if not all(np.all(np.isfinite(v)) for v in (a, b, x)):
        return {"status": "nonfinite", "true_residual_norm": None, "backward_error": None}
    sa, na = norm_parts(a)
    sb, nb = norm_parts(b)
    sx, nx = norm_parts(x)
    with localcontext() as context:
        context.prec = 80
        product = Decimal.from_float(sa) * Decimal.from_float(sx)
        total = max(product, Decimal.from_float(sb))
        if not total:
            residual = Decimal(0)
        else:
            ax = (a / sa) @ (x / sx) if sa and sx else np.zeros_like(b)
            scaled = (float(Decimal.from_float(sb) / total) * (b / sb) if sb else 0) - float(
                product / total
            ) * ax
            sr, nr = norm_parts(scaled)
            residual = total * Decimal.from_float(sr) * Decimal.from_float(nr)
        bnorm = Decimal.from_float(sb) * Decimal.from_float(nb)
        denominator = product * Decimal.from_float(na) * Decimal.from_float(nx) + bnorm
        result = {
            "status": "finite",
            "residual_reference_precision": "scaled-float64",
            "backward_error_norm": "Frobenius(A), L2(x), L2(b)",
            "true_residual_norm": number(residual),
            "relative_residual": number(residual / bnorm)
            if bnorm
            else (0.0 if not residual else "Infinity"),
            "backward_error": number(residual / denominator) if denominator else 0.0,
        }
    if reference_solution is not None:
        result["forward_error"] = error_metrics(original_x, reference_solution, reference=reference)
    return result


def record_error(row, actual, expected, *, key="numerical_error", **kwargs):
    """Keep the worst observed diagnostics over warmup and timed-result checks."""
    current = error_metrics(actual, expected, **kwargs)
    previous = row.get(key)
    if previous is None:
        row[key] = current
    else:
        for field in ("max_abs_error", "relative_l2_error", "actual_nonfinite_count"):
            if current[field] is not None and (
                previous[field] is None
                or Decimal(str(current[field])) > Decimal(str(previous[field]))
            ):
                previous[field] = current[field]
        if current["status"] != "finite":
            previous["status"] = current["status"]
    return current
