"""Untimed, CPU-only sampled references for float32 GEMM experiments.

Operands and products are dyadic rationals, so integer accumulation can preserve
the exact sum even across extreme cancellation. Exact decimal strings preserve
these references; error evaluation uses Decimal80. No BLAS result is used.
"""

from decimal import Decimal, localcontext
from numbers import Integral

import numpy as np

from benchmarks.numerical_metrics import number


def _units(value):
    """Exact finite float32 value in units of the smallest subnormal, 2**-149."""
    numerator, denominator = float(value).as_integer_ratio()
    return numerator * ((1 << 149) // denominator)


def _decimal_sum(units, exponent):
    # Decimal string construction is exact, independent of the active context.
    return Decimal(f"{units * 5**exponent}e-{exponent}")


def sampled_dot_references(a, b, coordinates):
    """Build strict-JSON references for selected output ``(row, column)`` pairs.

    ``a`` and ``b`` must be finite, compatible 2D float32 arrays. Reference
    strings contain exact sums, with errors evaluated at 80 decimal digits. The original
    reference sums exact operand products; the second sums products individually
    rounded to float32 (including underflow). Product overflow makes that second
    reference unsupported, with an explicit count and a null value. The original
    sum of absolute products scales errors without cancellation in the divisor.
    The returned object can be reused by ``evaluate_sampled_errors`` for variants.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Operands must be two-dimensional")
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise TypeError("Operands must have float32 dtype")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Operand inner dimensions must match")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Operands must contain only finite values")
    selected = []
    for coordinate in coordinates:
        try:
            row, column = coordinate
        except (TypeError, ValueError) as error:
            raise ValueError("Each coordinate must contain a row and column") from error
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in (row, column)
        ):
            raise ValueError("Coordinates must be integers")
        row, column = int(row), int(column)
        if not (0 <= row < a.shape[0] and 0 <= column < b.shape[1]):
            raise ValueError("Coordinate is outside the output shape")
        selected.append((row, column))
    points = []
    with localcontext() as context:
        context.prec = 80
        for row, column in selected:
            total = absolute = 0
            for left, right in zip(a[row, :], b[:, column]):
                product = _units(left) * _units(right)
                total += product
                absolute += abs(product)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                rounded = np.multiply(a[row, :], b[:, column], dtype=np.float32)
            nonfinite = int(np.count_nonzero(~np.isfinite(rounded)))
            rounded_reference = None
            if not nonfinite:
                rounded_total = sum((_units(value) for value in rounded), 0)
                rounded_reference = str(_decimal_sum(rounded_total, 149))
            points.append(
                {
                    "coordinate": [row, column],
                    "original_products_reference": str(_decimal_sum(total, 298)),
                    "rounded_products_reference": rounded_reference,
                    "rounded_products_status": "nonfinite_products" if nonfinite else "finite",
                    "rounded_product_nonfinite_count": nonfinite,
                    "sum_abs_products": str(_decimal_sum(absolute, 298)),
                }
            )
    return {
        "output_shape": [int(a.shape[0]), int(b.shape[1])],
        "inner_dimension": int(a.shape[1]),
        "decimal_precision_digits": 80,
        "reference_method": "exact dyadic sums; decimal80 error evaluation",
        "points": points,
    }


def _ratio(numerator, denominator):
    return (
        number(numerator / denominator) if denominator else (0.0 if not numerator else "Infinity")
    )


def _errors(value, reference, scale, status):
    result = {"status": status, "abs_error": None, "relative_error": None, "scaled_error": None}
    if status == "finite":
        error = abs(value - Decimal(reference))
        result.update(
            abs_error=number(error),
            relative_error=_ratio(error, abs(Decimal(reference))),
            scaled_error=_ratio(error, scale),
        )
    return result


def evaluate_sampled_errors(actual, references):
    """Compare a 2D float32/float64 result against reusable sampled references.

    Absolute and relative errors are reported for both product models. Scaled
    error divides by the original sum of absolute products. A zero denominator
    yields zero for zero error and the JSON string ``"Infinity"`` otherwise.
    Nonfinite actual values have null errors and explicit status; overflow in
    rounded products leaves only that product model unsupported.
    """
    actual = np.asarray(actual)
    if actual.ndim != 2 or list(actual.shape) != references["output_shape"]:
        raise ValueError("Actual shape must match the reference output shape")
    if actual.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("Actual must have float32 or float64 dtype")
    points = []
    nonfinite_count = 0
    with localcontext() as context:
        context.prec = 80
        for reference in references["points"]:
            value = float(actual[tuple(reference["coordinate"])])
            finite = bool(np.isfinite(value))
            nonfinite_count += not finite
            decimal_value = Decimal.from_float(value) if finite else None
            scale = Decimal(reference["sum_abs_products"])
            original_status = "finite" if finite else "nonfinite_actual"
            rounded_status = (
                original_status
                if reference["rounded_products_status"] == "finite"
                else "unsupported_nonfinite_products"
            )
            points.append(
                {
                    **reference,
                    "actual": number(value),
                    "original_products_error": _errors(
                        decimal_value,
                        reference["original_products_reference"],
                        scale,
                        original_status,
                    ),
                    "rounded_products_error": _errors(
                        decimal_value,
                        reference["rounded_products_reference"],
                        scale,
                        rounded_status,
                    ),
                }
            )
    return {
        "status": "nonfinite_actual" if nonfinite_count else "finite",
        "actual_dtype": str(actual.dtype),
        "sampled_nonfinite_count": nonfinite_count,
        "points": points,
    }
