"""RNG known answers, stream ownership, distributions and actual GPU residency."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import openpacho as p
from openpacho._random import _philox_numpy
from openpacho.exceptions import BackendUnavailableError, UnsupportedDtypeError


@pytest.fixture(params=["numpy", "opencl", "vulkan", "cuda"], scope="module")
def rng_backend(request):
    name = request.param
    try:
        backend = p.get_backend(name)
    except (BackendUnavailableError, ImportError) as exc:
        if os.environ.get(f"PACHO_REQUIRE_{name.upper()}") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    yield backend
    backend.synchronize()


@pytest.mark.parametrize(
    "counter,key,expected",
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_random123_known_answers(counter, key, expected):
    # Published Random123 tests/kat_vectors, Philox4x32, ten rounds.
    assert tuple(map(int, _philox_numpy(counter, key))) == expected


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("distribution", ["random", "uniform", "normal", "exponential"])
def test_irregular_reproducible_chunks_and_cpu_reference(rng_backend, dtype, distribution):
    backend = rng_backend
    try:
        backend.validate_dtype(np.dtype(dtype))
    except UnsupportedDtypeError:
        pytest.skip("fp64 unavailable")
    rng = p.random.default_rng(2**64 - 1, backend=backend)
    if (
        backend.name == "vulkan"
        and dtype == "float64"
        and distribution in ("normal", "exponential")
    ):
        with pytest.raises(NotImplementedError):
            getattr(rng, distribution)(size=1, dtype=dtype)
        assert rng._counter == 0
        return
    kwargs = {"dtype": dtype}
    if distribution == "uniform":
        kwargs.update(low=-3, high=7)
    elif distribution == "normal":
        kwargs.update(loc=2, scale=3)
    elif distribution == "exponential":
        kwargs.update(scale=2)
    actual = getattr(rng, distribution)(size=(33, 17), **kwargs)
    assert p.get_backend(actual) is backend
    expected = getattr(p.random.default_rng(2**64 - 1, backend="numpy"), distribution)(
        size=(33, 17), **kwargs
    )
    host = p.asnumpy(actual)
    np.testing.assert_allclose(
        host,
        p.asnumpy(expected),
        rtol=3e-6 if dtype == "float32" else 1e-13,
        atol=3e-6 if dtype == "float32" else 1e-13,
    )
    if distribution == "random":
        np.testing.assert_array_equal(host, p.asnumpy(expected))
    replay = p.random.default_rng(2**64 - 1, backend=backend)
    split = np.concatenate(
        [p.asnumpy(getattr(replay, distribution)(size=n, **kwargs)) for n in (1, 127, 0, 433)]
    )
    np.testing.assert_array_equal(split, host.ravel())
    assert not np.array_equal(
        host, p.asnumpy(getattr(rng, distribution)(size=host.shape, **kwargs))
    )


def test_counter_carry_exhaustion_and_empty(rng_backend):
    rng = p.random.default_rng(0, backend=rng_backend)
    ref = p.random.default_rng(0, backend="numpy")
    for start in (2**32 - 3, 2**64 - 7):
        rng._counter = ref._counter = start
        np.testing.assert_array_equal(
            p.asnumpy(rng.random(7, dtype="float32")), p.asnumpy(ref.random(7, dtype="float32"))
        )
    with pytest.raises(OverflowError):
        rng.random(1, dtype="float32")
    assert rng.random((0, 17), dtype="float32").shape == (0, 17)


@pytest.mark.parametrize("distribution", ["random", "uniform", "normal", "exponential"])
def test_statistical_smoke_and_finite_range(rng_backend, distribution):
    values = (
        p.asnumpy(
            getattr(p.random.default_rng(47, backend=rng_backend), distribution)(
                size=(513, 513), dtype="float32"
            )
        )
        .astype(float)
        .ravel()
    )
    assert np.all(np.isfinite(values))
    mean, variance = (
        (0.5, 1 / 12)
        if distribution in ("random", "uniform")
        else ((0, 1) if distribution == "normal" else (1, 1))
    )
    assert abs(values.mean() - mean) < 0.015
    assert abs(values.var() - variance) < 0.025
    assert abs(np.corrcoef(values[:-1], values[1:])[0, 1]) < 0.015
    if distribution in ("random", "uniform"):
        assert values.min() >= 0 and values.max() < 1
        counts = np.histogram(values, bins=32, range=(0, 1))[0]
        assert np.max(abs(counts - values.size / 32)) < 6 * np.sqrt(values.size / 32)
    elif distribution == "normal":
        assert abs(np.mean(values < 0) - 0.5) < 0.01
        assert abs(np.mean(abs(values) > 2) - 0.0455) < 0.005
    else:
        assert values.min() >= 0
        assert abs(np.mean(values > 2) - np.exp(-2)) < 0.005


def test_ranges_scalars_and_no_host_upload(rng_backend, monkeypatch):
    backend = rng_backend

    def forbidden(*args, **kwargs):
        raise AssertionError("RNG must not upload a host-generated matrix")

    monkeypatch.setattr(backend, "upload", forbidden)
    rng = p.random.default_rng(17, backend=backend)
    assert rng.random(dtype="float32").shape == ()
    for low, high in [(-3, 7), (1, np.nextafter(np.float32(1), np.float32(2))), (2, 2)]:
        values = p.asnumpy(rng.uniform(low, high, (17, 33), dtype="float32"))
        assert (values >= low).all()
        assert (values < high).all() if high > low else (values == low).all()
    np.testing.assert_array_equal(p.asnumpy(rng.normal(3, 0, 65, dtype="float32")), 3)
    np.testing.assert_array_equal(p.asnumpy(rng.exponential(0, 65, dtype="float32")), 0)


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.5, [1, 2]])
def test_invalid_seed(seed):
    with pytest.raises((ValueError, TypeError)):
        p.random.default_rng(seed)


def test_invalid_parameters_do_not_consume_stream():
    rng = p.random.default_rng(0, backend="numpy")
    for method, kwargs in [
        (rng.random, {"size": -1}),
        (rng.random, {"size": 2**31}),
        (rng.random, {"dtype": "int32"}),
        (rng.uniform, {"low": 3, "high": 2}),
        (rng.uniform, {"high": float("inf")}),
        (rng.normal, {"scale": -1}),
        (rng.exponential, {"scale": float("nan")}),
        (rng.normal, {"loc": [1, 2]}),
        (rng.uniform, {"low": -3e38, "high": 3e38, "dtype": "float32"}),
        (rng.normal, {"scale": -1e-50, "dtype": "float32"}),
        (rng.exponential, {"scale": -1e-50, "dtype": "float32"}),
        (rng.uniform, {"low": 1e-50, "high": 0, "dtype": "float32"}),
        (rng.normal, {"loc": 10**1000}),
    ]:
        with pytest.raises((ValueError, TypeError)):
            method(**kwargs)
    assert rng._counter == 0


def test_capture_rejects_closed_over_generator_without_advancing():
    rng = p.random.default_rng(0, backend="numpy")
    for decorator in (p.compile, p.fuse):

        @decorator
        def sample(a):
            return a + rng.random(a.shape)

        with pytest.raises(NotImplementedError, match="not captured"):
            sample(p.ones(3, backend="numpy"))
        assert rng._counter == 0


def test_concurrent_calls_reserve_distinct_complete_ranges():
    rng = p.random.default_rng(12, backend="numpy")
    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(lambda _: p.asnumpy(rng.random(129)), range(8)))
    expected = p.asnumpy(p.random.default_rng(12, backend="numpy").random(129 * 8)).reshape(8, 129)
    assert sorted(c.tobytes() for c in chunks) == sorted(c.tobytes() for c in expected)


def test_backend_is_bound_once_and_launch_failure_burns_reserved_range(monkeypatch):
    from openpacho.backends import NumpyBackend

    first, second = NumpyBackend(), NumpyBackend()
    with p.use_backend(first):
        rng = p.random.default_rng(47)
    with p.use_backend(second):
        assert p.get_backend(rng.random(3)) is first
    original = first.random_data

    def fail(*args):
        raise RuntimeError("uncertain dispatch")

    monkeypatch.setattr(first, "random_data", fail)
    with pytest.raises(RuntimeError, match="uncertain dispatch"):
        rng.random(7)
    assert rng._counter == 10
    monkeypatch.setattr(first, "random_data", original)
    reference = p.random.default_rng(47, backend="numpy")
    reference.random(10)
    np.testing.assert_array_equal(p.asnumpy(rng.random(33)), p.asnumpy(reference.random(33)))
