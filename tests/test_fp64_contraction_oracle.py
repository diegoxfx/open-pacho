import json
from fractions import Fraction

import numpy as np
import pytest

from benchmarks.fp64_contraction_oracle import (
    BOUNDARIES,
    _fused_step,
    cases,
    compare,
    evaluate,
    reference,
    sampled_references,
)


def dot(left, right, policy, **kwargs):
    return reference(
        np.array([left], dtype=np.float64),
        np.array(right, dtype=np.float64)[:, None],
        policy,
        **kwargs,
    )[0, 0]


def fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def test_fused_product_residual_and_reversed_order_distinguish_policy_from_truth():
    x, rounded = 1 + 2.0**-52, 1 + 2.0**-51
    assert dot([-rounded, x], [1, x], "fused").hex() == "0x1.0000000000000p-104"
    assert dot([-rounded, x], [1, x], "separate").hex() == "0x0.0p+0"
    for policy in ("fused", "separate"):
        assert dot([x, -rounded], [x, 1], policy) == 0
    a, b = np.array([[x, -rounded]]), np.array([[x], [1]])
    references = sampled_references(a, b)
    report = evaluate(reference(a, b, "fused"), references, "fused")
    assert report["policy_equivalent"]
    point = report["points"][0]
    assert not point["actual_equals_exact"]
    assert fraction(point["exact_original_products"]) == Fraction(1, 2**104)
    assert fraction(point["exact_sum_of_rounded_products"]) == 0
    assert point["original_products_error"]["relative_error"] == 1
    assert point["rounded_products_error"]["absolute_error"] == 0


def test_contraction_cannot_fix_earlier_accumulation_loss():
    for policy in ("fused", "separate"):
        assert dot([2.0**54, 1, -(2.0**54)], [1, 1, 1], policy) == 0
    a, b = np.array([[2.0**54, 1, -(2.0**54)]]), np.ones((3, 1))
    report = evaluate(reference(a, b, "fused"), sampled_references(a, b), "fused")
    assert report["policy_equivalent"]
    assert report["max_absolute_error"] == 1
    assert fraction(report["max_absolute_error_exact"]) == 1


def test_fused_subnormal_halfway_rounds_whole_sum_once_ties_even():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    assert dot([tiny, tiny], [1, 0.5], "separate").hex() == "0x0.0000000000001p-1022"
    assert dot([tiny, tiny], [1, 0.5], "fused").hex() == "0x0.0000000000002p-1022"
    assert _fused_step(tiny, 0.5, 0.0).hex() == "0x0.0p+0"
    assert _fused_step(-tiny, 0.5, 0.0).hex() == "-0x0.0p+0"
    assert _fused_step(tiny, 0.5, 2 * tiny) == 2 * tiny


def test_fused_avoids_product_overflow_and_exact_sum_survives_overflow():
    largest = np.finfo(np.float64).max
    assert dot([-largest, largest], [1, 2], "fused") == largest
    assert np.isposinf(dot([-largest, largest], [1, 2], "separate"))
    a, b = np.array([[-largest, largest]]), np.array([[1.0], [2.0]])
    references = sampled_references(a, b)
    point = references["points"][0]
    assert fraction(point["exact_original_products"]) == Fraction.from_float(float(largest))
    assert point["exact_sum_of_rounded_products"] is None
    assert point["rounded_product_nonfinite_count"] == 1
    report = evaluate(reference(a, b, "fused"), references, "fused")
    assert report["points"][0]["original_products_error"]["absolute_error"] == 0
    assert report["points"][0]["rounded_products_error"]["status"] == (
        "unsupported_nonfinite_products"
    )
    report = evaluate(reference(a, b), references)
    assert report["policy_equivalent"]
    assert report["points"][0]["original_products_error"]["status"] == "nonfinite_actual"
    json.dumps(report, allow_nan=False)


def test_overflow_cancellation_still_has_exact_finite_dot():
    largest = np.finfo(np.float64).max
    a, b = np.array([[largest, -largest]]), np.array([[2.0], [2.0]])
    point = sampled_references(a, b)["points"][0]
    assert fraction(point["exact_original_products"]) == 0
    assert point["rounded_product_nonfinite_count"] == 2
    assert np.isnan(reference(a, b)[0, 0])
    assert np.isposinf(reference(a, b, "fused")[0, 0])


@pytest.mark.parametrize(
    "x,y,acc,expected",
    [
        (0, np.inf, 0, "nan"),
        (np.inf, 0, 0, "nan"),
        (np.inf, 1, -np.inf, "nan"),
        (-np.inf, 1, np.inf, "nan"),
        (np.nan, 1, 0, "nan"),
        (1, np.nan, 0, "nan"),
        (1, 1, np.nan, "nan"),
        (np.inf, -1, -np.inf, "negative_inf"),
        (1, 1, np.inf, "positive_inf"),
    ],
)
def test_fused_nonfinite_classifications(x, y, acc, expected):
    from benchmarks.fp64_contraction_oracle import scalar

    assert scalar(_fused_step(x, y, acc))["classification"] == expected


def test_finite_exact_product_with_infinite_accumulator_is_not_spurious_invalid():
    largest = np.finfo(np.float64).max
    assert _fused_step(largest, 2, -np.inf) == -np.inf
    assert _fused_step(-largest, 2, np.inf) == np.inf


def test_negative_zero_underflow_and_explicit_padding():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    assert dot([-tiny], [0.5], "fused").hex() == "-0x0.0p+0"
    assert dot([-tiny], [0.5], "fused", padded_k=16).hex() == "0x0.0p+0"
    assert dot([-tiny], [0.5], "separate").hex() == "0x0.0p+0"
    assert _fused_step(-0.0, 1, -0.0).hex() == "-0x0.0p+0"
    assert _fused_step(-0.0, 1, 0.0).hex() == "0x0.0p+0"
    assert _fused_step(-1, 1, 1).hex() == "0x0.0p+0"
    for policy in ("fused", "separate"):
        assert dot([-0.0], [1], policy).hex() == "0x0.0p+0"


def test_comparison_uses_finite_bits_and_nonfinite_classes_ignoring_nan_payloads():
    nan = np.array([0xFFF8000000000001], dtype=np.uint64).view(np.float64)[0]
    assert compare(np.array([nan, np.inf, -np.inf]), np.array([np.nan, np.inf, -np.inf]))[
        "equivalent"
    ]
    report = compare(np.array([-0.0, np.inf, 1]), np.array([0.0, -np.inf, np.nextafter(1.0, 2.0)]))
    assert not report["equivalent"]
    assert report["finite_bit_mismatch_count"] == 2
    assert report["classification_mismatch_count"] == 1
    assert report["signed_zero_mismatch_count"] == 1
    json.dumps(report, allow_nan=False)


def test_nonfinite_input_reference_is_explicit_and_strict_json_safe():
    a, b = np.array([[np.nan, 1], [0, 1], [np.inf, -np.inf]]), np.ones((2, 1))
    references = sampled_references(a, b)
    report = evaluate(reference(a, b), json.loads(json.dumps(references, allow_nan=False)))
    assert report["policy_equivalent"]
    assert report["points"][0]["original_products_error"]["status"] == "nonfinite_inputs"
    assert report["points"][0]["exact_original_products"] is None
    assert report["points"][1]["original_products_error"]["absolute_error"] == 0
    json.dumps(report, allow_nan=False)


def test_errors_remain_exact_when_numeric_error_underflows_or_overflows():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    a, b = np.array([[tiny]]), np.array([[tiny]])
    report = evaluate(np.zeros((1, 1)), sampled_references(a, b))
    assert report["max_absolute_error"] == 0
    assert fraction(report["max_absolute_error_exact"]) == Fraction(1, 2**2148)
    largest = np.finfo(np.float64).max
    report = evaluate(
        np.array([[-largest]]), sampled_references(np.array([[largest]]), np.array([[2.0]]))
    )
    assert report["max_absolute_error"] == "Infinity"
    assert fraction(report["max_absolute_error_exact"]) == 3 * Fraction.from_float(float(largest))
    json.dumps(report, allow_nan=False)


def test_zero_relative_divisor_and_empty_inner_dimension():
    a, b = np.empty((1, 0)), np.empty((0, 2))
    expected = reference(a, b, "fused", padded_k=16)
    assert np.array_equal(expected.view(np.uint64), np.zeros((1, 2)).view(np.uint64))
    references = sampled_references(a, b)
    report = evaluate(expected + 1, references)
    assert report["points"][0]["original_products_error"]["relative_error"] == "Infinity"
    assert report["points"][0]["original_products_error"]["scaled_error"] == "Infinity"


def test_coordinate_order_rectangular_values_and_json_roundtrip():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    coordinates = [(1, 2), (0, 1)]
    assert reference(a, b, coordinates=coordinates).tolist() == [61.0, 24.0]
    references = sampled_references(a, b, coordinates)
    report = evaluate(a @ b, json.loads(json.dumps(references, allow_nan=False)))
    assert report["sample_count"] == 2
    assert report["policy_equivalent"]
    assert [p["coordinate"] for p in report["points"]] == [[1, 2], [0, 1]]


@pytest.mark.parametrize("coordinate", [(-1, 0), (0, 1), (0.0, 0), (True, 0), (0,), None])
def test_invalid_coordinates_rejected(coordinate):
    with pytest.raises(ValueError, match="[Cc]oordinate"):
        reference(np.ones((1, 1)), np.ones((1, 1)), coordinates=[coordinate])


@pytest.mark.parametrize("padding", [0, 1.5, True, -1])
def test_invalid_padding_rejected(padding):
    with pytest.raises(ValueError, match="padded_k"):
        reference(np.ones((1, 1)), np.ones((1, 1)), padded_k=padding)


def test_api_validation_and_empty_coordinates():
    a = np.ones((1, 1))
    with pytest.raises(ValueError, match="two-dimensional"):
        reference(a[0], a)
    with pytest.raises(ValueError, match="inner dimensions"):
        reference(a, np.ones((2, 1)))
    with pytest.raises(TypeError, match="float64"):
        reference(a.astype(np.float32), a)
    with pytest.raises(ValueError, match="policy"):
        reference(a, a, "reassociated")
    with pytest.raises(ValueError, match="shape"):
        evaluate(np.ones((2, 1)), sampled_references(a, a))
    with pytest.raises(TypeError, match="float64"):
        evaluate(a.astype(np.float32), sampled_references(a, a))
    assert reference(a, a, coordinates=[]).shape == (0,)
    assert reference(np.empty((0, 1)), a).shape == (0, 1)
    assert evaluate(a, sampled_references(a, a, []))["sample_count"] == 0


def test_cases_reproducible_native_tile_sized_and_boundary_complete():
    first, second = list(cases()), list(cases())
    assert len({name for name, _ in first}) == len(first)
    assert {a.shape[1] for name, (a, _) in first if name.startswith("random-")} == set(BOUNDARIES)
    for (name, (a, b)), (other_name, (aa, bb)) in zip(first, second):
        assert name == other_name
        assert a.shape[0] == b.shape[1] == 8
        assert a.shape[1] == b.shape[0]
        assert a.dtype == b.dtype == np.float64
        assert np.array_equal(a, aa, equal_nan=True)
        assert np.array_equal(b, bb, equal_nan=True)
    for name, (a, b) in first:
        if name.startswith("fma-residual-"):
            assert np.all(reference(a, b, "fused") == 2.0**-104)
            assert np.all(reference(a, b) == 0)
