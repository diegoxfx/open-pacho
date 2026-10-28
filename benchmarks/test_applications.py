"""CPU-only semantic tests for the bounded application workloads."""

import numpy as np

from benchmarks.applications import (
    RIDGE_REGULARIZATION,
    RIDGE_STEPS,
    make_ridge_inputs,
    ridge_gradient_step,
    ridge_objective,
    ridge_regression,
    run,
    sample_covariance,
)


def test_sample_covariance_matches_numpy_covariance_and_preserves_input():
    rng = np.random.default_rng(47)
    values = rng.standard_normal((17, 5)).astype(np.float32)
    original = values.copy()

    actual = sample_covariance(values)
    expected = np.cov(values, rowvar=False, ddof=1)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual.dtype == values.dtype
    np.testing.assert_array_equal(values, original)


def test_ridge_twenty_steps_matches_float64_reference_and_records_diagnostics():
    X, y, _ = make_ridge_inputs(31, 7, "float32", seed=47)
    original_X, original_y = X.copy(), y.copy()
    actual = ridge_regression(X, y)

    X64, y64 = X.astype(np.float64), y.astype(np.float64)
    Xt64 = X64.T.copy()
    expected = np.zeros(X.shape[1], dtype=np.float64)
    rate = 0.25 / X.shape[0]
    for _ in range(RIDGE_STEPS):
        expected = ridge_gradient_step(X64, Xt64, y64, expected, rate, RIDGE_REGULARIZATION)

    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=3e-6)
    assert ridge_objective(X64, y64, expected) < ridge_objective(X64, y64, np.zeros_like(expected))
    assert actual.dtype == X.dtype
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


def test_cpu_quick_report_contains_timing_modes_and_ridge_metadata(tmp_path):
    output = tmp_path / "applications.json"
    report = run(output, repeats=1, quick=True, backends=())

    assert output.exists()
    assert report["unavailable"] == []
    assert len(report["rows"]) == 4  # covariance/ridge x NumPy 1/8 threads
    ridge_rows = [row for row in report["rows"] if row["workload"] == "ridge"]
    assert ridge_rows
    for row in report["rows"]:
        assert row["calculate_seconds"]
        assert row["resident_seconds"]
        assert row["upload_seconds"]
        assert row["preparation_seconds"]
        assert row["download_seconds"]
        assert row["end_to_end_seconds"]
    for row in ridge_rows:
        assert row["steps"] == 20
        assert row["rate"] == 0.25 / 256
        assert row["regularization"] == RIDGE_REGULARIZATION
        assert row["diagnostics"]["reference"] == "numpy-float64"
