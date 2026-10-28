"""Resident extrema contracts, chunk merges, mapped producers and normalization."""

import os

import numpy as np
import pytest

import openpacho as p


@pytest.fixture(scope="module", params=("numpy", "opencl", "vulkan", "cuda-custom"))
def extrema_backend(request):
    name, _, provider = request.param.partition("-")
    try:
        return p.get_backend(name, blas=provider or None)
    except p.BackendUnavailableError as exc:
        if os.environ.get(f"PACHO_REQUIRE_{name.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def check(result, expected, backend):
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert p.get_backend(result) is backend
    np.testing.assert_array_equal(p.asnumpy(result), expected)


def reduce_with(function, axis):
    return lambda x: function(x, axis=axis)


def extrema_with(axis, keepdims):
    return p.compile(
        lambda x: (
            np.min(x, axis=axis, keepdims=keepdims),
            np.amax(x, axis=axis, keepdims=keepdims),
        )
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_extrema_axes_and_numpy_dispatch(extrema_backend, dtype):
    if dtype == np.float64 and not getattr(extrema_backend, "fp64", True):
        pytest.skip("selected device lacks float64")
    host = (np.arange(30).reshape(2, 3, 5) - 16).astype(dtype)
    a = p.asarray(host, backend=extrema_backend)
    for axis, keepdims in ((None, False), (0, True), (-1, False), ((2, 0), True), ((), False)):
        calculate = extrema_with(axis, keepdims)
        expected = [function(host, axis=axis, keepdims=keepdims) for function in (np.min, np.max)]
        eager = (a.min(axis=axis, keepdims=keepdims), a.max(axis=axis, keepdims=keepdims))
        for results in (eager, calculate(a)):
            for actual, reference in zip(results, expected):
                check(actual, np.asarray(reference), extrema_backend)
    assert p.min is p.amin and p.max is p.amax
    check(np.amin(a), np.asarray(np.min(host)), extrema_backend)
    check(np.max(a), np.asarray(np.max(host)), extrema_backend)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_extrema_nan_infinity_and_zero_ties_across_chunks(extrema_backend, dtype):
    if dtype == np.float64 and not getattr(extrema_backend, "fp64", True):
        pytest.skip("selected device lacks float64")
    host = np.zeros((5, 8193), dtype=dtype)
    host[0, :4096] = -0.0
    host[1] = -0.0
    host[3, 4097] = np.nan
    host[4, 0], host[4, -1] = -np.inf, np.inf
    a = p.asarray(host, backend=extrema_backend)
    calculate = p.compile(lambda x: (p.min(x, axis=1), p.max(x, axis=1)))
    expected = (
        np.array([-0.0, -0.0, +0.0, np.nan, -np.inf], dtype=dtype),
        np.array([+0.0, -0.0, +0.0, np.nan, +np.inf], dtype=dtype),
    )
    for results in ((p.min(a, axis=1), p.max(a, axis=1)), calculate(a)):
        for actual, reference in zip(results, expected):
            check(actual, reference, extrema_backend)
            np.testing.assert_array_equal(
                np.signbit(p.asnumpy(actual)[:3]), np.signbit(reference[:3])
            )
    # Uniform zero signs survive both scalar and the non-reducing axis=() route.
    zero = p.asarray(np.array(-0.0, dtype=dtype), backend=extrema_backend)
    for function in (p.min, p.max):
        assert np.signbit(function(zero, axis=()).item())


@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.uint32, np.uint64))
def test_extrema_integer_limits_stay_exact(extrema_backend, dtype):
    if extrema_backend.name == "cuda":
        pytest.skip("CUDA has no integer arrays")
    limits = np.iinfo(dtype)
    pattern = np.array([limits.min, limits.min + 1, limits.max - 1, limits.max], dtype=dtype)
    host = np.stack([np.resize(pattern, 4101), np.full(4101, limits.max, dtype=dtype)])
    a = p.asarray(host, backend=extrema_backend)
    calculate = p.compile(lambda x: (p.min(x, axis=1), p.max(x, axis=1)))
    for results in ((p.min(a, axis=1), p.max(a, axis=1)), calculate(a)):
        for actual, function in zip(results, (np.min, np.max)):
            check(actual, function(host, axis=1), extrema_backend)
    # Mapping before reduction must not route integer extrema through float.
    mapped = p.compile(lambda x: p.max(p.where(x >= 0, x, x), axis=1))
    check(mapped(a), np.max(host, axis=1), extrema_backend)


def test_extrema_boolean_identities_and_compiled_cache(extrema_backend):
    host = np.zeros((3, 4097), dtype=np.bool_)
    host[0] = True
    host[1, -1] = True
    a = p.asarray(host, backend=extrema_backend)
    # Same producer, shape, dtype and axes: Vulkan's body cache must include operation.
    calculate_min = p.compile(lambda x: p.min(p.equal(x, True), axis=1))
    calculate_max = p.compile(lambda x: p.max(p.equal(x, True), axis=1))
    calculate_sum = p.compile(lambda x: p.sum(p.equal(x, True), axis=1, dtype=np.float32))
    for function, expected in ((p.min, np.min), (p.max, np.max)):
        check(function(a, axis=1), expected(host, axis=1), extrema_backend)
    for function, expected in (
        (calculate_min, np.min),
        (calculate_max, np.max),
        (calculate_min, np.min),
    ):
        check(function(a), expected(host, axis=1), extrema_backend)
    check(calculate_sum(a), np.sum(host, axis=1, dtype=np.float32), extrema_backend)


def test_materialized_extrema_preserves_shared_public_and_transposed_inputs(extrema_backend):
    @p.compile
    def calculate(x, offset):
        shared = x + offset
        return shared, p.min(shared, axis=0), p.max(shared, axis=1), p.max(shared.T, axis=1)

    host = (np.arange(3 * 257, dtype=np.float32) % 29 - 14).reshape(3, 257)
    first = p.asarray(host, backend=extrema_backend)
    prepared = calculate.prepare(first, 0.5)
    for array, reference, offset in (
        (first, host, 0.5),
        (p.asarray(-host, backend=extrema_backend), -host, -1.5),
    ):
        shared = reference + offset
        expected = shared, shared.min(axis=0), shared.max(axis=1), shared.T.max(axis=1)
        for actual, want in zip(prepared(array, offset), expected):
            check(actual, want, extrema_backend)
    kinds = [step["kind"] for step in prepared.explain()["steps"]]
    selected = "map_reduce" if extrema_backend.name in ("opencl", "cuda") else "reduce"
    assert kinds.count(selected) == 3
    assert kinds.count("gather") == 1


@pytest.mark.parametrize("operation", ("min", "max"))
def test_extrema_map_reduce_rebinds_and_stays_resident(extrema_backend, monkeypatch, operation):
    function = getattr(p, operation)
    host = (np.arange(3 * 8193, dtype=np.float32) % 41 - 20).reshape(3, 8193)
    inputs = [p.asarray(value, backend=extrema_backend) for value in (host, host * 0.5)]
    calculate = p.compile(lambda x, cut: function(p.where(x > cut, x, -x), axis=1, keepdims=True))
    launches = []
    if extrema_backend.name != "numpy":
        run = extrema_backend._run

        def track(*args, **kwargs):
            launches.append(args[1])
            return run(*args, **kwargs)

        monkeypatch.setattr(extrema_backend, "_run", track)
        monkeypatch.setattr(extrema_backend, "upload", lambda *args: pytest.fail("implicit upload"))
    prepared = calculate.prepare(inputs[0], 0.0)
    with monkeypatch.context() as guarded:
        guarded.setattr(extrema_backend, "download", lambda *args: pytest.fail("implicit download"))
        results = (prepared(inputs[0], 0.0), prepared(inputs[1], 4.0))
    for result, h, cut in zip(results, (host, host * 0.5), (0, 4)):
        expected = getattr(np, operation)(np.where(h > cut, h, -h), axis=1, keepdims=True)
        check(result, expected, extrema_backend)
    if extrema_backend.name != "numpy":
        steps = calculate.explain(inputs[0], 0.0)["steps"]
        assert [step["kind"] for step in steps] == ["map_reduce"]
        assert steps[0]["operation"] == operation
        assert len(launches) == 4  # one mapped chunk pass and one partial merge per call


def test_extrema_empty_scalar_and_explicit_errors(extrema_backend):
    for function in (p.min, p.max):
        for shape, axis in (((0,), None), ((0, 0), 0), ((0, 3), 0), ((3, 0), 1)):
            a = p.empty(shape, dtype=np.float32, backend=extrema_backend)
            reduce = reduce_with(function, axis)
            for apply in (reduce, p.compile(reduce)):
                with pytest.raises(ValueError, match="zero.size|empty"):
                    apply(a)
        for shape, axis, expected_shape in (((0, 3), 1, (0,)), ((3, 0), 0, (0,)), ((0,), (), (0,))):
            a = p.empty(shape, dtype=np.float32, backend=extrema_backend)
            for result in (function(a, axis=axis), p.compile(reduce_with(function, axis))(a)):
                check(result, np.empty(expected_shape, dtype=np.float32), extrema_backend)
        a = p.asarray(np.array(-4.5, dtype=np.float32), backend=extrema_backend)
        for axis in (None, 0, -1, ()):
            assert function(a, axis=axis).item() == -4.5
        for kwargs in ({"out": a}, {"initial": 0}, {"where": True}):
            with pytest.raises(NotImplementedError):
                function(a, **kwargs)
        for axis in ((0,), 1):
            with pytest.raises(ValueError):
                function(a, axis=axis)
        with pytest.raises(ValueError, match="duplicate"):
            function(p.ones((2, 3), backend=extrema_backend), axis=(0, 0))
        with pytest.raises(NotImplementedError, match="elementwise"):
            p.fuse(reduce_with(function, None))(a)


def test_stable_normalization_example(extrema_backend):
    from examples.normalization import logsumexp, softmax

    host = np.array([[10000, 10001, 9999], [-10000, -10001, -9999]], dtype=np.float32)
    a = p.asarray(host, backend=extrema_backend)
    shifted = host.astype(np.float64) - np.max(host.astype(np.float64), axis=1, keepdims=True)
    expected_softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    expected_lse = np.max(host.astype(np.float64), axis=1, keepdims=True) + np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )
    for calculate in (
        (lambda x: (softmax(x), logsumexp(x))),
        p.compile(lambda x: (softmax(x), logsumexp(x))),
    ):
        weights, logs = calculate(a)
        np.testing.assert_allclose(p.asnumpy(weights), expected_softmax, rtol=2e-6, atol=0)
        np.testing.assert_allclose(p.asnumpy(logs), expected_lse, rtol=0, atol=5e-4)
        np.testing.assert_allclose(np.sum(p.asnumpy(weights), axis=1), 1, rtol=2e-6)
        assert p.get_backend(weights) is extrema_backend and weights.dtype == a.dtype
