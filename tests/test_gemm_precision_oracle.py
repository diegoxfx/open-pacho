import json
from decimal import Decimal

import numpy as np
import pytest

from benchmarks.gemm_precision_oracle import evaluate_sampled_errors, sampled_dot_references


def test_exact_cancellation_preserves_small_term_across_entire_float32_range():
    largest = np.finfo(np.float32).max
    smallest = np.nextafter(np.float32(0), np.float32(1))
    a = np.array([[largest, smallest, -largest]], dtype=np.float32)
    b = np.array([[largest], [1], [largest]], dtype=np.float32)
    references = sampled_dot_references(a, b, [(0, 0)])
    point = references["points"][0]
    assert Decimal(point["original_products_reference"]) == Decimal.from_float(float(smallest))
    assert point["rounded_products_reference"] is None
    assert point["rounded_product_nonfinite_count"] == 2
    assert point["rounded_products_status"] == "nonfinite_products"
    errors = evaluate_sampled_errors(np.array([[smallest]], dtype=np.float32), references)
    assert errors["points"][0]["original_products_error"]["abs_error"] == 0
    assert errors["points"][0]["rounded_products_error"]["status"] == (
        "unsupported_nonfinite_products"
    )
    json.dumps(errors, allow_nan=False)


def test_product_rounding_and_accumulation_error_are_distinguished():
    # (1 + 2**-23)**2 = 1 + 2**-22 + 2**-46; its float32 product loses 2**-46.
    x = np.float32(1 + 2**-23)
    a = np.array([[x, -1]], dtype=np.float32)
    b = np.array([[x], [1 + 2**-22]], dtype=np.float32)
    references = sampled_dot_references(a, b, [(0, 0)])
    point = references["points"][0]
    assert Decimal(point["original_products_reference"]) == Decimal.from_float(2**-46)
    assert Decimal(point["rounded_products_reference"]) == 0
    errors = evaluate_sampled_errors(np.zeros((1, 1), dtype=np.float32), references)["points"][0]
    assert errors["original_products_error"]["abs_error"] == 2**-46
    assert errors["original_products_error"]["relative_error"] == 1
    assert errors["original_products_error"]["scaled_error"] > 0
    assert errors["rounded_products_error"]["abs_error"] == 0
    assert errors["rounded_products_error"]["relative_error"] == 0
    # A separate accumulation error is measured against the rounded-product sum.
    changed = evaluate_sampled_errors(np.ones((1, 1), dtype=np.float32), references)
    assert changed["points"][0]["rounded_products_error"]["abs_error"] == 1
    assert changed["points"][0]["rounded_products_error"]["relative_error"] == "Infinity"


def test_rounded_product_underflow_and_zero_scale_are_explicit():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    references = sampled_dot_references(
        np.array([[tiny]], dtype=np.float32), np.array([[tiny]], dtype=np.float32), [(0, 0)]
    )
    point = references["points"][0]
    assert Decimal(point["original_products_reference"]) > 0
    assert Decimal(point["rounded_products_reference"]) == 0
    zero = np.zeros((1, 1), dtype=np.float32)
    zero_references = sampled_dot_references(zero, zero, [(0, 0)])
    exact = evaluate_sampled_errors(zero, zero_references)["points"][0]
    assert exact["original_products_error"]["scaled_error"] == 0
    perturbed = evaluate_sampled_errors(zero + 1, zero_references)["points"][0]
    assert perturbed["original_products_error"]["scaled_error"] == "Infinity"


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_nonfinite_actual_has_no_finite_error(value):
    a = np.ones((1, 1), dtype=np.float32)
    references = sampled_dot_references(a, a, [(0, 0)])
    errors = evaluate_sampled_errors(np.array([[value]], dtype=np.float32), references)
    assert errors["status"] == "nonfinite_actual"
    assert errors["sampled_nonfinite_count"] == 1
    for model in ("original_products_error", "rounded_products_error"):
        assert errors["points"][0][model] == {
            "status": "nonfinite_actual",
            "abs_error": None,
            "relative_error": None,
            "scaled_error": None,
        }
    json.dumps(errors, allow_nan=False)


def test_coordinates_select_correct_rectangular_entries_and_json_roundtrip():
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
    references = json.loads(json.dumps(sampled_dot_references(a, b, [(1, 2), (0, 1)])))
    assert [Decimal(p["original_products_reference"]) for p in references["points"]] == [61, 24]
    errors = evaluate_sampled_errors(a.astype(np.float64) @ b, references)
    assert all(p["original_products_error"]["abs_error"] == 0 for p in errors["points"])
    with pytest.raises(ValueError, match="shape"):
        evaluate_sampled_errors(np.zeros((3, 2), dtype=np.float32), references)
    with pytest.raises(TypeError, match="dtype"):
        evaluate_sampled_errors(np.zeros((2, 3), dtype=np.int32), references)


@pytest.mark.parametrize("coordinate", [(-1, 0), (0, 1), (0.0, 0), (True, 0), (0,), None])
def test_invalid_coordinates_rejected(coordinate):
    a = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="[Cc]oordinate"):
        sampled_dot_references(a, a, [coordinate])


def test_operand_dimensions_dtype_and_finiteness_validated():
    a = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="two-dimensional"):
        sampled_dot_references(a[0], a, [])
    with pytest.raises(ValueError, match="inner dimensions"):
        sampled_dot_references(a, np.ones((2, 1), dtype=np.float32), [])
    with pytest.raises(TypeError, match="float32"):
        sampled_dot_references(a.astype(np.float64), a, [])
    with pytest.raises(ValueError, match="finite"):
        sampled_dot_references(a * np.inf, a, [])


def test_empty_inner_dimension_is_exact_zero():
    references = sampled_dot_references(
        np.empty((1, 0), dtype=np.float32), np.empty((0, 2), dtype=np.float32), [(0, 1)]
    )
    point = references["points"][0]
    assert Decimal(point["original_products_reference"]) == 0
    assert Decimal(point["rounded_products_reference"]) == 0
    assert Decimal(point["sum_abs_products"]) == 0
