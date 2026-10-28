"""Benchmark validation cannot silently accept broken distributions."""

import numpy as np
import pytest

from benchmarks.numerical_metrics import error_metrics
from benchmarks.random_generation import moments, native_numpy


def test_same_dtype_rng_reference_is_not_labelled_higher_precision():
    a = np.array([0.5], dtype=np.float32)
    assert not error_metrics(a, a, precision="float32")["reference_is_higher_precision"]
    assert error_metrics(a, a.astype(np.float64))["reference_is_higher_precision"]


@pytest.mark.parametrize("distribution", ["random", "uniform", "normal", "exponential"])
def test_numpy_comparator_dtype_and_distribution(distribution):
    values = native_numpy(np.random.default_rng(47), distribution, (257, 129), np.dtype("float32"))
    assert values.dtype == np.dtype("float32")
    result = moments(values, distribution)
    assert result["nonfinite"] == 0


@pytest.mark.parametrize(
    "distribution,values",
    [
        ("random", np.ones(33000)),
        ("normal", np.zeros(33000)),
        ("uniform", np.full(33000, 7)),
        ("exponential", np.full(33, -1)),
        ("normal", np.full(33, np.nan)),
    ],
)
def test_bad_values_are_rejected(distribution, values):
    with pytest.raises(AssertionError):
        moments(values, distribution)
