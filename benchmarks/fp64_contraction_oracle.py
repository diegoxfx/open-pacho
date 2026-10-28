"""CPU-only binary64 operation-order controls and exact GEMM accuracy samples.

The fused control rounds the exact product plus the current binary64 accumulator
once, using Fraction conversion (round to nearest, ties to even). It requires no
hardware FMA or BLAS. Policy equivalence is deliberately separate from accuracy:
neither sequential policy is an exact dot product, and contraction cannot recover
rounding lost by an earlier step. NaN payloads and signs are unspecified.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral

import numpy as np

SEED = 1069
BOUNDARIES = (15, 16, 17, 31, 32, 33)


def _operands(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Operands must be two-dimensional")
    if a.dtype != np.float64 or b.dtype != np.float64:
        raise TypeError("Operands must have float64 dtype")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Operand inner dimensions must match")
    return a, b


def _coordinates(shape, coordinates):
    if coordinates is None:
        return list(np.ndindex(shape))
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
        if not (0 <= row < shape[0] and 0 <= column < shape[1]):
            raise ValueError("Coordinate is outside the output shape")
        selected.append((row, column))
    return selected


def _padding(k, padded_k):
    if padded_k is None:
        return k
    if isinstance(padded_k, (bool, np.bool_)) or not isinstance(padded_k, Integral) or padded_k < k:
        raise ValueError("padded_k must be an integer at least the inner dimension")
    return int(padded_k)


def _round(value):
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _fused_step(x, y, acc):
    """IEEE round-to-nearest binary64 fma, ignoring NaN payload/sign choice."""
    x, y, acc = float(x), float(y), float(acc)
    if math.isnan(x) or math.isnan(y) or math.isnan(acc):
        return math.nan
    if math.isinf(x) or math.isinf(y):
        if x == 0 or y == 0:
            return math.nan
        product = -math.inf if np.signbit(x) != np.signbit(y) else math.inf
        if math.isinf(acc) and np.signbit(product) != np.signbit(acc):
            return math.nan
        return product
    if math.isinf(acc):
        return acc
    product = Fraction.from_float(x) * Fraction.from_float(y)
    total = product + Fraction.from_float(acc)
    if not total:
        # Exact cancellation gives +0. Only -0 + -0 gives -0 in RN mode.
        negative_product_zero = not product and (np.signbit(x) != np.signbit(y))
        return -0.0 if negative_product_zero and acc == 0 and np.signbit(acc) else 0.0
    return _round(total)


def _dot(left, right, policy, padded_k):
    acc = np.float64(0.0)
    with np.errstate(all="ignore"):
        for x, y in zip(left, right):
            acc = (
                _fused_step(x, y, acc)
                if policy == "fused"
                else np.add(acc, np.multiply(x, y, dtype=np.float64), dtype=np.float64)
            )
        for _ in range(padded_k - len(left)):
            acc = (
                _fused_step(0.0, 0.0, acc)
                if policy == "fused"
                else np.add(acc, np.float64(0.0), dtype=np.float64)
            )
    return float(acc)


def reference(a, b, policy="separate", padded_k=None, coordinates=None):
    """Sequential K-order binary64 control, starting at positive zero.

    Separate rounds multiplication and addition independently; fused rounds once
    per product plus accumulator. ``padded_k`` explicitly appends +0 * +0 steps,
    which can change a final negative zero. With coordinates, return a 1D array
    in the requested order; otherwise return the entire M by N float64 result.
    """
    a, b = _operands(a, b)
    if policy not in ("separate", "fused"):
        raise ValueError("policy must be 'separate' or 'fused'")
    k = _padding(a.shape[1], padded_k)
    shape = (a.shape[0], b.shape[1])
    selected = _coordinates(shape, coordinates)
    values = np.array([_dot(a[r], b[:, c], policy, k) for r, c in selected], dtype=np.float64)
    return values.reshape(shape) if coordinates is None else values


def scalar(value):
    """Strict-JSON scalar descriptor, retaining all finite bits and zero signs."""
    value = float(value)
    classification = (
        "nan"
        if math.isnan(value)
        else "positive_inf"
        if value == math.inf
        else "negative_inf"
        if value == -math.inf
        else "negative_zero"
        if value == 0 and np.signbit(value)
        else "positive_zero"
        if value == 0
        else "subnormal"
        if abs(value) < np.finfo(np.float64).tiny
        else "finite"
    )
    return {
        "classification": classification,
        "hex": value.hex(),
        "bits": f"{np.float64(value).view(np.uint64).item():016x}",
    }


def compare(actual, expected):
    """Compare finite bits and NaN / signed-infinity classes, ignoring NaN bits."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != np.float64 or expected.dtype != np.float64:
        return {"equivalent": False, "error": "Shape/float64 dtype mismatch"}
    finite = np.isfinite(actual) & np.isfinite(expected)
    bits = finite & (actual.view(np.uint64) != expected.view(np.uint64))
    classes = (
        (np.isnan(actual) != np.isnan(expected))
        | (np.isposinf(actual) != np.isposinf(expected))
        | (np.isneginf(actual) != np.isneginf(expected))
    )
    zeros = (actual == 0) & (expected == 0) & (np.signbit(actual) != np.signbit(expected))
    return {
        "equivalent": not bool(np.any(bits | classes)),
        "finite_bit_mismatch_count": int(np.count_nonzero(bits)),
        "classification_mismatch_count": int(np.count_nonzero(classes)),
        "signed_zero_mismatch_count": int(np.count_nonzero(zeros)),
        "examples": [
            {
                "coordinate": [int(v) for v in index],
                "actual": scalar(actual[tuple(index)]),
                "expected": scalar(expected[tuple(index)]),
            }
            for index in np.argwhere(bits | classes)[:8]
        ],
    }


def fraction_json(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _fraction_from_json(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def sampled_references(a, b, coordinates=None, padded_k=None):
    """Reusable strict-JSON policy controls and exact sums for selected outputs.

    None selects all outputs (appropriate for the small critical cases). Exact
    original products remain available for finite inputs even if an individual
    binary64 product overflows. The sum of rounded products is unsupported then.
    Nonfinite input dots still have both IEEE policy controls, but no exact sum.
    """
    a, b = _operands(a, b)
    shape = (a.shape[0], b.shape[1])
    selected = _coordinates(shape, coordinates)
    k = _padding(a.shape[1], padded_k)
    points = []
    for row, column in selected:
        left, right = a[row], b[:, column]
        finite_inputs = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
        point = {
            "coordinate": [row, column],
            "separate": scalar(_dot(left, right, "separate", k)),
            "fused": scalar(_dot(left, right, "fused", k)),
            "exact_reference_status": "finite_inputs" if finite_inputs else "nonfinite_inputs",
            "exact_original_products": None,
            "exact_sum_of_rounded_products": None,
            "sum_abs_products": None,
            "rounded_product_nonfinite_count": 0,
            "nearest_binary64_to_exact": None,
        }
        if finite_inputs:
            exact = rounded = absolute = Fraction(0)
            nonfinite = 0
            with np.errstate(all="ignore"):
                for x, y in zip(left, right):
                    product = Fraction.from_float(float(x)) * Fraction.from_float(float(y))
                    exact += product
                    absolute += abs(product)
                    rounded_product = float(np.multiply(x, y, dtype=np.float64))
                    if math.isfinite(rounded_product):
                        rounded += Fraction.from_float(rounded_product)
                    else:
                        nonfinite += 1
            point.update(
                exact_original_products=fraction_json(exact),
                exact_sum_of_rounded_products=fraction_json(rounded) if not nonfinite else None,
                sum_abs_products=fraction_json(absolute),
                rounded_product_nonfinite_count=nonfinite,
                nearest_binary64_to_exact=scalar(_round(exact)),
            )
        points.append(point)
    return {
        "output_shape": list(shape),
        "inner_dimension": int(a.shape[1]),
        "padded_k": k,
        "method": "sequential binary64 policies and exact Fraction original/rounded-product sums",
        "points": points,
    }


def _number(value):
    rounded = _round(value)
    return "Infinity" if rounded == math.inf else "-Infinity" if rounded == -math.inf else rounded


def _ratio(numerator, denominator):
    return (
        _number(numerator / denominator) if denominator else (0.0 if not numerator else "Infinity")
    )


def _errors(value, target, scale):
    error = abs(value - target)
    return {
        "status": "finite",
        "absolute_error": _number(error),
        "absolute_error_exact": fraction_json(error),
        "relative_error": _ratio(error, abs(target)),
        "scaled_error": _ratio(error, scale),
    }


def evaluate(actual, references, policy="separate"):
    """Policy bit/class comparison plus independent mathematical accuracy errors.

    Error summaries include exact Fraction maxima so float underflow or overflow
    in the convenient numeric summaries cannot hide an error. Zero-relative-error
    divisors yield 0 for exact agreement and the JSON string Infinity otherwise.
    """
    actual = np.asarray(actual)
    if actual.ndim != 2 or list(actual.shape) != references["output_shape"]:
        raise ValueError("Actual shape must match the reference output shape")
    if actual.dtype != np.float64:
        raise TypeError("Actual must have float64 dtype")
    if policy not in ("separate", "fused"):
        raise ValueError("policy must be 'separate' or 'fused'")
    points, maxima = [], []
    for point in references["points"]:
        got = float(actual[tuple(point["coordinate"])])
        expected = float.fromhex(point[policy]["hex"])
        match = compare(np.array([got]), np.array([expected]))["equivalent"]
        result = {**point, "actual": scalar(got), "matches_policy": match}
        status = (
            "nonfinite_inputs"
            if point["exact_reference_status"] != "finite_inputs"
            else "nonfinite_actual"
            if not math.isfinite(got)
            else "finite"
        )
        result["original_products_error"] = {"status": status}
        result["rounded_products_error"] = {"status": status}
        if status == "finite":
            value = Fraction.from_float(got)
            target = _fraction_from_json(point["exact_original_products"])
            scale = _fraction_from_json(point["sum_abs_products"])
            result["actual_equals_exact"] = value == target
            result["original_products_error"] = _errors(value, target, scale)
            maxima.append(abs(value - target))
            rounded = point["exact_sum_of_rounded_products"]
            if rounded is not None:
                result["rounded_products_error"] = _errors(
                    value, _fraction_from_json(rounded), scale
                )
        if (
            point["exact_reference_status"] == "finite_inputs"
            and point["rounded_product_nonfinite_count"]
        ):
            result["rounded_products_error"] = {"status": "unsupported_nonfinite_products"}
        points.append(result)
    maximum = max(maxima) if maxima else None
    return {
        "policy": policy,
        "sample_count": len(points),
        "policy_equivalent": all(p["matches_policy"] for p in points),
        "policy_mismatch_count": sum(not p["matches_policy"] for p in points),
        "max_absolute_error": _number(maximum) if maximum is not None else None,
        "max_absolute_error_exact": fraction_json(maximum) if maximum is not None else None,
        "finite_error_sample_count": len(maxima),
        "points": points,
    }


def cases():
    """Named deterministic 8 by K by 8 host float64 critical GEMM cases."""
    rng = np.random.default_rng(SEED)
    for k in BOUNDARIES:
        yield (
            f"random-k{k}",
            (
                rng.uniform(-1, 1, (8, k)).astype(np.float64),
                rng.uniform(-1, 1, (k, 8)).astype(np.float64),
            ),
        )

    def structured(name, left, right, k=17):
        a, b = np.zeros((8, k), dtype=np.float64), np.zeros((k, 8), dtype=np.float64)
        a[:, -len(left) :] = left
        b[-len(right) :, :] = np.asarray(right, dtype=np.float64)[:, None]
        return name, (a, b)

    x, rounded = 1 + 2.0**-52, 1 + 2.0**-51
    for k in BOUNDARIES:
        yield structured(f"fma-residual-k{k}", [-rounded, x], [1, x], k)
    yield structured("reversed-product-rounding", [x, -rounded], [x, 1])
    yield structured("accumulation-cancellation", [2.0**54, 1, -(2.0**54)], [1, 1, 1])
    tiny, largest = float.fromhex("0x0.0000000000001p-1022"), np.finfo(np.float64).max
    yield structured("subnormal-tie", [tiny, tiny], [1, 0.5])
    yield structured("negative-underflow-zero", [-tiny], [0.5])
    yield structured("overflow-avoided-by-fma", [-largest, largest], [1, 2])
    yield structured("product-overflow-cancellation", [largest, -largest], [2, 2])
    a, b = np.zeros((8, 17), dtype=np.float64), np.ones((17, 8), dtype=np.float64)
    a[:, :3] = [
        [np.nan, 1, 0],
        [np.inf, 1, 0],
        [-np.inf, 1, 0],
        [np.inf, -np.inf, 0],
        [0, 1, 0],
        [-0.0, -0.0, -0.0],
        [largest, largest, -largest],
        [tiny, -tiny, 0],
    ]
    b[:, 1] = -1
    b[:, 2] = 0
    b[:, 3] = -0.0
    b[0, 4] = np.inf
    b[0, 5] = -np.inf
    b[0, 6] = np.nan
    b[:, 7] = 0.5
    yield "nonfinite-signed-zero", (a, b)
