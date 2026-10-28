"""Resident selection must preserve NumPy truth, promotion and branch values."""

import os

import numpy as np
import pytest

import openpacho as p
from openpacho.backends import NumpyBackend


@pytest.fixture(scope="module", params=("numpy", "opencl", "vulkan", "cuda"))
def where_backend(request):
    name = request.param
    options = (
        {"blas": "custom", "device": os.environ.get("PACHO_CUDA_DEVICE")} if name == "cuda" else {}
    )
    try:
        return p.get_backend(name, **options)
    except p.BackendUnavailableError as exc:
        if os.environ.get(f"PACHO_REQUIRE_{name.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def assert_result(actual, expected, backend):
    assert actual._backend is backend
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(p.asnumpy(actual), expected)


def test_where_three_way_broadcast_and_numpy_dispatch(where_backend):
    condition = np.array([[[True, False, True]], [[False, True, False]]])
    left = np.arange(4, dtype=np.float32).reshape(1, 4, 1)
    right = np.array([-2, -3, -4], dtype=np.float32)
    arrays = [p.asarray(a, backend=where_backend) for a in (condition, left, right)]
    expected = np.where(condition, left, right)
    for operation in (p.where, np.where, p.compile(np.where), p.fuse(np.where)):
        assert_result(operation(*arrays), expected, where_backend)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_where_numeric_truth_preserves_selected_nonfinites_and_zero_sign(where_backend, dtype):
    if dtype == np.float64 and not getattr(where_backend, "fp64", True):
        pytest.skip("device does not support float64")
    condition = np.array([0, -0.0, 0.5, np.nan, np.inf, -np.inf], dtype=dtype)
    left = np.array([np.nan, np.nan, np.nan, -0.0, np.inf, -np.inf], dtype=dtype)
    right = np.array([0, -0.0, 42, 13, np.nan, np.nan], dtype=dtype)
    args = [p.asarray(a, backend=where_backend) for a in (condition, left, right)]
    expected = np.where(condition, left, right)
    for operation in (p.where, p.compile(lambda c, x, y: np.where(c, x, y))):
        result = operation(*args)
        assert_result(result, expected, where_backend)
        np.testing.assert_array_equal(np.signbit(p.asnumpy(result)[[0, 1, 3]]), [False, True, True])


@pytest.mark.parametrize("other", [1, 1.0, np.float64(1), np.array(1, dtype=np.float64)])
def test_where_branch_promotion_keeps_weak_and_strong_scalars_distinct(where_backend, other):
    if not getattr(where_backend, "fp64", True):
        pytest.skip("test uses a float64 condition independent of branch precision")
    condition = np.array([0, 0.5], dtype=np.float64)
    host = np.array([2, 3], dtype=np.float32)
    c, a = [p.asarray(v, backend=where_backend) for v in (condition, host)]
    b = p.asarray(other, backend=where_backend) if isinstance(other, np.ndarray) else other
    expected = np.where(condition, host, other)
    operation = p.compile(lambda c, a, b: p.where(c, a, b))
    assert_result(p.where(c, a, b), expected, where_backend)
    assert_result(operation(c, a, b), expected, where_backend)


def test_where_nested_selection_rebinds_boolean_and_numeric_scalars(where_backend):
    host = np.array([-3, -1, 0, 2, 4], dtype=np.float32)
    a = p.asarray(host, backend=where_backend)

    @p.compile
    def select(a, enabled, threshold, replacement):
        return np.where(enabled, np.where(a >= threshold, a, replacement), -a)

    prepared = select.prepare(a, True, 0.0, -1.0)
    for enabled, threshold, replacement in ((True, 0.0, -1.0), (False, 3.0, 7.0), (True, 3.0, 7.0)):
        expected = np.where(enabled, np.where(host >= threshold, host, replacement), -host)
        assert_result(prepared(a, enabled, threshold, replacement), expected, where_backend)
    assert select.cache_info()["misses"] == 1


def test_where_empty_scalar_and_boolean_outputs(where_backend):
    select = p.compile(lambda c, x: np.where(c, x, False))
    for shape in ((), (0, 3), (2, 0, 3)):
        condition = np.ones(shape, dtype=bool)
        host = np.zeros(shape, dtype=bool)
        c, a = [p.asarray(v, backend=where_backend) for v in (condition, host)]
        expected = np.where(condition, host, False)
        assert_result(p.where(c, a, False), expected, where_backend)
        assert_result(select(c, a), expected, where_backend)


def test_where_large_integer_selection_is_exact(where_backend):
    if where_backend.name == "cuda":
        pytest.skip("CUDA does not support integer storage")
    left = np.array([2**62 + 1, -(2**62) + 3], dtype=np.int64)
    right = np.array([2**62 + 9, -(2**62) + 17], dtype=np.int64)
    c, a, b = [p.asarray(v, backend=where_backend) for v in ([True, False], left, right)]
    for operation in (p.where, p.compile(lambda c, a, b: p.where(c, a, b))):
        assert_result(operation(c, a, b), np.where([True, False], left, right), where_backend)


def test_where_map_reduce_keeps_data_resident_and_fuses(where_backend, monkeypatch):
    host = (np.arange(9003, dtype=np.float32) % 23 - 11).reshape(3, 3001)
    a = p.asarray(host, backend=where_backend)
    select = p.compile(lambda a, cut: np.sum(np.where(a > cut, a * a, 0.0), axis=1))
    launches = []
    if where_backend.name != "numpy":
        run = where_backend._run

        def track(*args, **kwargs):
            launches.append(args[1])
            return run(*args, **kwargs)

        monkeypatch.setattr(where_backend, "_run", track)
        monkeypatch.setattr(where_backend, "upload", lambda *a: pytest.fail("unexpected upload"))
    with monkeypatch.context() as guarded:
        guarded.setattr(where_backend, "download", lambda *a: pytest.fail("unexpected download"))
        first = select(a, 0.0)
        second = select(a, 4.0)
    for result, cut in ((first, 0), (second, 4)):
        assert_result(result, np.sum(np.where(host > cut, host * host, 0.0), axis=1), where_backend)
    if where_backend.name != "numpy":
        assert [s["kind"] for s in select.explain(a, 0.0)["steps"]] == ["map_reduce"]
        assert len(launches) == 2


def test_where_errors_are_explicit_and_unselected_branch_is_validated():
    a = p.ones(3, dtype=np.float32, backend="numpy")
    with pytest.raises(NotImplementedError, match="nonzero|one.argument"):
        p.where(a)
    with pytest.raises(TypeError):
        p.where(a, 1)
    with pytest.raises(ValueError):
        p.where(a, np.ones(2), np.ones(3))
    with pytest.raises(p.UnsupportedDtypeError):
        p.where(True, a, 1j)
    other = p.asarray([1, 2, 3], backend=NumpyBackend())
    with pytest.raises(p.BackendMismatchError):
        p.where(True, a, other)
    with pytest.raises(p.BackendMismatchError):
        p.compile(lambda a, b: np.where(True, a, b))(a, other)
