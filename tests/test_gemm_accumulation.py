"""Critical experiment controls: balanced order and reproducible historical inputs."""

from collections import Counter

import numpy as np
import pytest

from benchmarks.gemm_accumulation import (
    VARIANTS,
    balanced_orders,
    cases,
    classification,
    make_inputs,
    strict_classification,
)


def test_timing_order_balances_positions_and_preserves_samples():
    orders = balanced_orders(24, 1011)
    assert orders == balanced_orders(24, 1011)
    assert all(set(order) == set(VARIANTS) for order in orders)
    for position in range(len(VARIANTS)):
        assert Counter(order[position] for order in orders) == dict.fromkeys(VARIANTS, 4)
    for repeats in (0, 11, 13, True):
        with pytest.raises(ValueError, match="multiple of twelve"):
            balanced_orders(repeats, 1011)


def test_historical_input_seed_matches_padding_experiment():
    # Use the same RNG contract on a small shape without allocating the historical 64 MiB input.
    historical = next(case for case in cases() if case["name"] == "historical-972")
    assert historical["shape"] == (65, 8191, 2051)
    small = {**historical, "shape": (2, 3, 4)}
    a, b = make_inputs(small)
    rng = np.random.default_rng(972 * 101 + 10)
    np.testing.assert_array_equal(a, rng.uniform(-0.5, 0.5, (2, 3)).astype(np.float32))
    np.testing.assert_array_equal(b, rng.uniform(-0.5, 0.5, (3, 4)).astype(np.float32))


def test_nonfinite_fixture_preserves_ieee_overflow_and_invalid_products():
    a, b = make_inputs(next(case for case in cases() if case["name"] == "nonfinite"))
    actual = classification(strict_classification(a, b))
    np.testing.assert_array_equal(
        actual[:8, :3],
        [
            [3, 3, 3],
            [1, 1, 3],
            [-1, -1, 3],
            [3, 3, 3],
            [1, 3, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    )
