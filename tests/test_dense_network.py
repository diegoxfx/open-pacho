"""Independent gradient checks and execution contracts for the dense client."""

import os

import numpy as np
import pytest

import openpacho as p
from examples.dense_network import loss_and_gradients, predict, sgd_step
from openpacho.exceptions import BackendUnavailableError


def _inputs(dtype=np.float64, shape=(5, 3, 4, 2), seed=47):
    n, d, h, o = shape
    rng = np.random.default_rng(seed)
    sizes = ((n, d), (n, o), (d, h), (h,), (h, o), (o,))
    return tuple((rng.standard_normal(size) * 0.3 + 0.1).astype(dtype) for size in sizes)


def _independent_loss(x, y, w1, b1, w2, b2):
    # Deliberately independent of the example's forward/loss implementation.
    errors = np.tanh(np.matmul(x, w1) + b1) @ w2 + b2 - y
    return np.sum(errors**2, dtype=np.float64) / (2 * y.size)


def _finite_difference_gradients(inputs):
    x, y, *parameters = inputs
    gradients = []
    for parameter in parameters:
        gradient = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            step = 1e-5 * max(1.0, abs(original))
            try:
                parameter[index] = original + step
                upper = _independent_loss(x, y, *parameters)
                parameter[index] = original - step
                lower = _independent_loss(x, y, *parameters)
            finally:
                parameter[index] = original
            gradient[index] = (upper - lower) / (2 * step)
        gradients.append(gradient)
    return gradients


def test_all_gradients_and_update_match_central_differences():
    inputs = _inputs()
    originals = tuple(value.copy() for value in inputs)
    loss, *gradients = loss_and_gradients(*inputs)
    finite_differences = _finite_difference_gradients(inputs)
    np.testing.assert_allclose(loss, _independent_loss(*inputs), rtol=2e-15)
    for actual, expected in zip(gradients, finite_differences):
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-10)
    rate = 0.01
    step_loss, *updated = sgd_step(*inputs, rate)
    np.testing.assert_equal(step_loss, loss)
    for original, actual, derivative in zip(inputs[2:], updated, finite_differences):
        np.testing.assert_allclose(actual, original - rate * derivative, rtol=2e-10, atol=3e-12)
        assert not np.shares_memory(actual, original)
    assert _independent_loss(*inputs[:2], *updated) < loss
    for actual, original in zip(inputs, originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_saturated_hidden_activations_remain_finite(dtype):
    x, y, w1, b1, w2, b2 = _inputs(dtype)
    w1.fill(0)
    b1[:] = np.asarray([100, -100, 100, -100], dtype=dtype)
    loss, dw1, db1, dw2, db2 = loss_and_gradients(x, y, w1, b1, w2, b2)
    for value in (loss, dw1, db1, dw2, db2, *sgd_step(x, y, w1, b1, w2, b2, 0.01)):
        assert np.all(np.isfinite(value))
    np.testing.assert_array_equal(dw1, np.zeros_like(w1))
    np.testing.assert_array_equal(db1, np.zeros_like(b1))


def _assert_outputs(actual, expected, dtype):
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    tolerance = 3e-5 if dtype == np.float32 else 2e-11
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected):
        host = p.asnumpy(value)
        assert host.dtype == np.dtype(dtype)
        assert host.shape == np.shape(reference)
        np.testing.assert_allclose(host, reference, rtol=tolerance, atol=tolerance)


def _check_execution(backend, dtype, shape):
    functions = (predict, loss_and_gradients, sgd_step)
    compiled = tuple(p.compile(function) for function in functions)
    # Both new arrays and a new dynamic rate must rebind the captured plan.
    for seed, rate in ((47, 0.01), (71, 0.07)):
        inputs = _inputs(dtype, shape, seed)
        originals = tuple(value.copy() for value in inputs)
        resident = tuple(p.asarray(value, backend=backend) for value in inputs)
        x, _y, w1, b1, w2, b2 = inputs
        rx, _ry, rw1, rb1, rw2, rb2 = resident
        host_args = ((x, w1, b1, w2, b2), inputs, (*inputs, rate))
        device_args = ((rx, rw1, rb1, rw2, rb2), resident, (*resident, rate))
        for function, captured, args, backend_args in zip(
            functions, compiled, host_args, device_args
        ):
            expected = function(*args)
            _assert_outputs(function(*backend_args), expected, dtype)
            _assert_outputs(captured(*backend_args), expected, dtype)
        for value, original in zip(resident, originals):
            np.testing.assert_array_equal(p.asnumpy(value), original)
        # In particular, inference is still at the original weights after SGD.
        _assert_outputs(predict(rx, rw1, rb1, rw2, rb2), predict(x, w1, b1, w2, b2), dtype)
    for function in compiled:
        assert function.cache_info()["size"] == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("shape", [(5, 3, 4, 2), (1, 7, 3, 5)], ids=["rectangular", "one-row"])
def test_numpy_eager_compiled_values_dtype_and_rebinding(dtype, shape):
    _check_execution(p.get_backend("numpy"), dtype, shape)


@pytest.fixture(params=["opencl", "vulkan", "cuda-custom", "cuda-cublas"])
def dense_native_backend(request):
    selected = request.param
    name = "cuda" if selected.startswith("cuda-") else selected
    options = {"blas": selected.split("-", 1)[1]} if name == "cuda" else {}
    device = os.environ.get(f"PACHO_{name.upper()}_DEVICE")
    if device:
        options["device"] = device
    try:
        return p.get_backend(name, **options)
    except (BackendUnavailableError, ImportError) as exc:
        if os.environ.get(f"PACHO_REQUIRE_{name.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_native_eager_compiled_values_dtype_and_rebinding(dense_native_backend, dtype):
    backend = dense_native_backend
    if dtype == np.float64 and not getattr(backend, "fp64", True):
        pytest.skip("selected device does not support float64")
    if backend.name == "vulkan" and dtype == np.float64:
        x, _y, w1, b1, w2, b2 = (p.asarray(value, backend=backend) for value in _inputs(dtype))
        for function in (predict, p.compile(predict)):
            with pytest.raises(NotImplementedError, match="float64 tanh"):
                function(x, w1, b1, w2, b2)
        return
    _check_execution(backend, dtype, (5, 3, 4, 2))


@pytest.mark.parametrize(
    "function,invalid",
    [
        (function, invalid)
        for function in (predict, loss_and_gradients, sgd_step)
        for invalid in ("integer", "mixed", "empty", "bias", "weights", "targets")
        if invalid != "targets" or function is not predict
    ],
)
def test_metadata_validation(function, invalid):
    x, y, w1, b1, w2, b2 = _inputs()
    if invalid == "integer":
        x = x.astype(np.int32)
    elif invalid == "mixed":
        w2 = w2.astype(np.float32)
    elif invalid == "empty":
        x, y = x[:0], y[:0]
    elif invalid == "bias":
        b1 = b1[None, :]
    elif invalid == "weights":
        w2 = w2[:-1]
    else:
        y = y[:, :1]
    args = (x, w1, b1, w2, b2) if function is predict else (x, y, w1, b1, w2, b2)
    if function is sgd_step:
        args = (*args, 0.01)
    with pytest.raises((TypeError, ValueError)):
        function(*args)
