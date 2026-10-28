"""CUDA fusion preserves public semantics, precision and resident lifetimes."""

import gc
import os
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import openpacho as p


@pytest.fixture(scope="module", params=("custom", "cublas"))
def backend(request):
    try:
        result = p.get_backend(
            "cuda", device=os.environ.get("PACHO_CUDA_DEVICE"), blas=request.param
        )
    except p.BackendUnavailableError as exc:
        if os.environ.get("PACHO_REQUIRE_CUDA") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    yield result
    result.synchronize()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape,axes",
    [
        ((9001,), None),
        ((5, 32), (-1,)),
        ((3, 2, 33), (1, 2)),
        ((9, 7, 39), (0, 2)),
        ((67, 3, 69), (0, 2)),
        ((2, 3), ()),
        ((), None),
        ((2, 0, 3), (1,)),
        ((0, 4097), (1,)),
    ],
)
def test_axes_tails_and_empty_reductions(backend, dtype, shape, axes):
    host = (np.arange(int(np.prod(shape))) % 7).astype(dtype).reshape(shape)
    operation = p.compile(lambda a: np.sum(a * a + 2, axis=axes, keepdims=True))
    a = p.asarray(host, backend=backend)
    actual = operation(a)
    assert actual.dtype == host.dtype
    np.testing.assert_array_equal(
        p.asnumpy(actual), np.sum(host * host + 2, axis=axes, keepdims=True)
    )
    assert [step["kind"] for step in operation.explain(a)["steps"]] == ["map_reduce"]


def test_broadcast_mean_rebinds_scalars_and_dtypes(backend):
    host = np.arange(4 * 67 * 9, dtype=np.float32).reshape(4, 67, 9) / 1000
    bias = np.arange(9, dtype=np.float64)
    operation = p.compile(lambda a, b, rate: np.mean((a + b) * rate, axis=(0, 2)))
    a, b = [p.asarray(x, backend=backend) for x in (host, bias)]
    for rate in (0.5, 1.75):
        actual = operation(a, b, rate)
        assert actual.dtype == np.float64
        np.testing.assert_allclose(
            p.asnumpy(actual), np.mean((host + bias) * rate, axis=(0, 2)), rtol=1e-13
        )
    assert operation.cache_info()["misses"] == 1


def test_float32_rounding_precedes_float64_accumulation(backend):
    host = np.linspace(0.7, 3.3, 9001, dtype=np.float32)
    operation = p.compile(lambda a: np.sum(a * a, dtype=np.float64))
    actual = operation(p.asarray(host, backend=backend))
    np.testing.assert_allclose(p.asnumpy(actual), np.sum(host * host, dtype=np.float64), rtol=1e-14)
    # Narrowing is a real materialization of precision even without a buffer.
    x = np.array([1 + 2**-25, 1 - 2**-25, 2**-150], np.float64)
    narrow = p.compile(lambda a: a.astype(np.float32).astype(np.float64) - a)
    np.testing.assert_array_equal(
        p.asnumpy(narrow(p.asarray(x, backend=backend))),
        x.astype(np.float32).astype(np.float64) - x,
    )


def test_shared_outputs_no_upload_and_bounded_partials(backend, monkeypatch):
    host = np.arange(9001, dtype=np.float32) / 1000
    a = p.asarray(host, backend=backend)
    allocations, launches = [], []
    empty, run = backend.empty, backend._run
    monkeypatch.setattr(
        backend, "empty", lambda shape, dtype: (allocations.append(shape), empty(shape, dtype))[1]
    )
    monkeypatch.setattr(
        backend, "_run", lambda *args, **kwargs: (launches.append(args[1]), run(*args, **kwargs))[1]
    )
    monkeypatch.setattr(backend, "upload", lambda *args: pytest.fail("unexpected upload"))
    total = p.compile(lambda a, rate: np.sum((a + rate) * (a + rate)))
    value = total(a, 2.0)
    assert allocations == [(1, 3), (1, 1)]
    assert launches == [3, 1]
    np.testing.assert_allclose(p.asnumpy(value), np.sum((host + 2) ** 2), rtol=2e-6)

    @p.compile
    def shared(a, rate):
        shifted = a + rate
        return shifted, np.sum(shifted * shifted), shifted - 1

    first = shared(a, 2.0)
    second = shared(a, 3.0)
    for values, offset in ((first, 2), (second, 3)):
        for actual, expected in zip(
            values, (host + offset, np.sum((host + offset) ** 2), host + offset - 1)
        ):
            np.testing.assert_allclose(p.asnumpy(actual), expected, rtol=2e-6)
    ref = weakref.ref(a._data)
    del a, first, second, values, actual
    backend.synchronize()
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fuse_transcendentals_nonfinite_and_signed_zero(backend, dtype):
    host = np.array([-0.0, 0.0, 0.3, -0.7, np.nan, np.inf, -np.inf], dtype=dtype)
    a = p.asarray(host, backend=backend)
    f = p.fuse(lambda a: np.isfinite(a) == (a > 0))
    np.testing.assert_array_equal(p.asnumpy(f(a)), np.isfinite(host) == (host > 0))
    with np.errstate(all="ignore"):
        op = lambda a: np.sin(a) * np.exp(-a * a) + np.cos(a)
        np.testing.assert_allclose(p.asnumpy(p.fuse(op)(a)), op(host), rtol=2e-6, equal_nan=True)
    zero = p.asnumpy(p.fuse(lambda a: a * dtype(1))(a))
    np.testing.assert_array_equal(np.signbit(zero[:2]), np.signbit(host[:2]))
    boolean = p.fuse(lambda a: np.isfinite(a > 0))
    np.testing.assert_array_equal(p.asnumpy(boolean(a)), np.ones(host.shape, bool))


def test_no_hidden_fma_and_same_reduction_order(backend):
    a = p.asarray([1 + 2**-23], dtype=p.float32, backend=backend)
    b = p.asarray([1 - 2**-23], dtype=p.float32, backend=backend)
    fused = p.compile(lambda a, b: a * b - 1)
    np.testing.assert_array_equal(p.asnumpy(fused(a, b)), p.asnumpy(a * b - 1))
    host = np.tile(np.array([1e20, 1, -1e20, 3, 1e-38], np.float32), 2001)
    x = p.asarray(host, backend=backend)
    total = p.compile(lambda x: np.sum((x + 1) * 0.5))
    np.testing.assert_array_equal(p.asnumpy(total(x)), p.asnumpy(np.sum((x + 1) * 0.5)))


def test_matmul_stays_with_provider_and_threads_rebind(backend, monkeypatch):
    host = np.arange(33 * 17, dtype=np.float32).reshape(33, 17) / 1000
    a, x, b = [
        p.asarray(v, backend=backend)
        for v in (host, np.ones(17, np.float32), np.ones(33, np.float32))
    ]
    original = backend.matmul
    calls = []
    monkeypatch.setattr(backend, "matmul", lambda *args: (calls.append(1), original(*args))[1])
    operation = p.compile(lambda a, x, b, rate: (b - a @ x) * rate)
    operation(a, x, b, 1.0)
    kinds = [step["kind"] for step in operation.explain(a, x, b, 1.0)["steps"]]
    assert kinds == ["matmul", "fused"]

    def work(rate):
        return p.asnumpy(operation(a, x, b, rate))

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(work, (0.5, 1.5, 2.5)))
    for value, rate in zip(results, (0.5, 1.5, 2.5)):
        np.testing.assert_allclose(
            value, (1 - host @ np.ones(17, np.float32)) * rate, rtol=2e-6, atol=1e-6
        )
    assert len(calls) == 4


def test_bool_reduction_remains_explicitly_unsupported(backend):
    a = p.asarray([1, 2], dtype=np.float32, backend=backend)
    with pytest.raises(NotImplementedError, match="Boolean reduction"):
        p.compile(lambda a: np.sum(a > 0, dtype=np.bool_))(a)
