import json
from decimal import Decimal, localcontext

import numpy as np
import pytest

from benchmarks.accuracy import Case, decimal_array, reference
from benchmarks.numerical_metrics import (
    error_metrics,
    number,
    precision_metadata,
    record_error,
    solver_metrics,
    stable_norm,
)


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_norm_and_relative_error_across_range(scale):
    x = np.array([3.0, 4.0]) * scale
    assert stable_norm(x) == pytest.approx(5 * scale, rel=1e-14, abs=0)
    result = error_metrics(x * 1.25, x)
    assert result["relative_l2_error"] == pytest.approx(0.25)
    assert result["max_abs_error"] == pytest.approx(scale, rel=1e-14, abs=0)


def test_zero_reference_and_nonfinite_are_explicit_strict_json():
    assert error_metrics(np.zeros(3), np.zeros(3))["relative_l2_error"] == 0
    assert error_metrics(np.ones(3), np.zeros(3))["relative_l2_error"] == "Infinity"
    result = error_metrics(np.array([np.inf, np.nan]), np.zeros(2))
    assert result["actual_nonfinite_count"] == 2
    assert result["relative_l2_error"] is None
    assert result["status"] == "nonfinite"
    json.dumps(result, allow_nan=False)


def test_opposite_extremes_do_not_overflow_diagnostics():
    result = error_metrics(np.array([1e308]), np.array([-1e308]))
    assert result["relative_l2_error"] == pytest.approx(2)
    assert isinstance(result["max_abs_error"], str)
    json.dumps(result, allow_nan=False)


def test_decimal_reference_preserves_error_below_float64_rounding():
    with localcontext() as context:
        context.prec = 80
        expected = np.asarray(Decimal(1) + Decimal(2) ** -54)
        result = error_metrics(np.asarray(1.0), expected, precision="decimal80")
    assert result["max_abs_error"] == 2**-54
    assert result["relative_l2_error"] > 0


def test_decimal_metric_below_float64_range_is_not_reported_as_zero():
    assert number(Decimal("1e-400")) == "1E-400"


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_solver_backward_error_scale_invariance(scale):
    a = np.array([[2.0, -1], [-1.0, 2]])
    x = np.array([1.0, 2]) * scale
    b = a @ x
    result = solver_metrics(a, b, x * 1.01, reference_solution=x)
    assert result["relative_residual"] == pytest.approx(0.01)
    assert result["forward_error"]["relative_l2_error"] == pytest.approx(0.01)
    assert 0 < result["backward_error"] < 0.01


def test_solver_diagnostic_avoids_overflowing_matrix_vector_product():
    a = np.array([[1e200, -1e200], [1e200, -1e200]])
    x, b = np.full(2, 1e200), np.zeros(2)
    result = solver_metrics(a, b, x)
    assert result["true_residual_norm"] == 0
    assert result["backward_error"] == 0


def test_worst_error_retains_nonfinite_round():
    row = {}
    record_error(row, np.array([1.1]), np.array([1.0]))
    record_error(row, np.array([1.01]), np.array([1.0]))
    assert row["numerical_error"]["relative_l2_error"] == pytest.approx(0.1)
    record_error(row, np.array([np.inf]), np.array([1.0]))
    assert row["numerical_error"]["status"] == "nonfinite"
    assert row["numerical_error"]["actual_nonfinite_count"] == 1


def test_decimal_dot_and_solve_use_actual_float_inputs():
    values = np.array([1e16, 1, -1e16])
    assert reference(Case("cancel", "dot", (values, np.ones(3)))) == Decimal(1)
    a = np.array([[0.0, 2], [3.0, 1]])
    b = np.array([4.0, 5])
    x = reference(Case("pivot", "solve", (a, b)))
    assert list(x) == [Decimal(1), Decimal(2)]
    assert np.all(decimal_array(a) @ x == decimal_array(b))


def test_precision_metadata_does_not_assume_windows_longdouble():
    metadata = precision_metadata()
    assert metadata["longdouble_is_wider"] == (np.finfo(np.longdouble).nmant > 52)
    assert metadata["decimal_precision_digits"] == 80


def test_shape_mismatch_is_not_broadcast_during_diagnostics():
    with pytest.raises(ValueError, match="shape mismatch"):
        error_metrics(np.ones((2, 1)), np.ones(2))


@pytest.mark.parametrize("solvers_only", [False, True])
def test_execution_report_records_untimed_metrics_without_optional_devices(
    tmp_path, monkeypatch, solvers_only
):
    from benchmarks import execution

    def unavailable(*args, **kwargs):
        raise RuntimeError("CPU-only diagnostic test")

    monkeypatch.setattr(execution.pacho, "get_backend", unavailable)
    monkeypatch.setattr(execution, "available_references", list)
    monkeypatch.setattr(execution, "availability_errors", dict)
    report = execution.run(
        tmp_path / "report.json", repeats=1, quick=True, solvers_only=solvers_only
    )
    assert report["diagnostics_timed"] is False
    assert report["rows"] and not any("error" in row for row in report["rows"])
    for row in report["rows"]:
        assert row["numerical_error"]["status"] == "finite"
        assert len(row["resident_seconds"]) == 1
        if solvers_only:
            assert row["solver_error"]["backward_error"] >= 0
            assert "RHS rounded" in row["numerical_error"]["reference"]
