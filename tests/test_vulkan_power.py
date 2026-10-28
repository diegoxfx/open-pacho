"""GLSL pow domain gaps must not leak into the supported NumPy float32 API."""

import os

import numpy as np
import pytest

import openpacho as p


@pytest.fixture(scope="module")
def backend():
    try:
        return p.get_backend("vulkan")
    except p.BackendUnavailableError as exc:
        if os.environ.get("PACHO_REQUIRE_VULKAN") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.mark.parametrize("compiled", [False, True])
def test_power_real_domain_signed_zero_and_nonfinite(backend, compiled):
    bases = np.array([-np.inf, -3, -1, -0.0, 0.0, 0.5, 1, 3, np.inf, np.nan], np.float32)[:, None]
    exponents = np.array(
        [-np.inf, -16777216, -3, -2, -0.5, 0, 0.5, 2, 3, 16777216, np.inf, np.nan], np.float32
    )
    with np.errstate(all="ignore"):
        expected = np.power(bases, exponents)
    a, b = p.asarray(bases, backend=backend), p.asarray(exponents, backend=backend)
    function = p.compile(lambda a, b: a**b) if compiled else lambda a, b: a**b
    actual = p.asnumpy(function(a, b))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0, equal_nan=True)
    special = (expected == 0) | np.isinf(expected)
    np.testing.assert_array_equal(np.signbit(actual[special]), np.signbit(expected[special]))


def test_negative_square_map_reduce_and_dynamic_scalar_rebinding(backend):
    host = np.linspace(-3, 2, 129 * 17, dtype=np.float32).reshape(129, 17)
    a = p.asarray(host, backend=backend)
    function = p.compile(lambda a, exponent: np.sum(a**exponent, axis=1))
    for exponent in (2.0, 3.0, 0.0):
        expected = np.sum(host**exponent, axis=1)
        np.testing.assert_allclose(p.asnumpy(function(a, exponent)), expected, rtol=1e-4, atol=2e-4)
    assert function.cache_info()["misses"] == 1
