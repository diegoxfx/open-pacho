"""CUDA cooperative GEMV source and numerical coverage."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

import openpacho as pacho
from openpacho._cuda_gemv import matvec_data
from openpacho.exceptions import BackendUnavailableError, UnsupportedDtypeError


class _SourceBackend:
    name = "cuda"

    def __init__(self):
        self.allocations = []
        self.dispatches = []
        self.reductions = []

    def empty(self, shape, dtype):
        buffer = SimpleNamespace(shape=tuple(shape), dtype=np.dtype(dtype))
        self.allocations.append((buffer.shape, buffer.dtype))
        return buffer

    def full(self, shape, value, dtype):
        buffer = self.empty(shape, dtype)
        buffer.value = value
        return buffer

    def _run(self, source, blocks, *args, threads=128):
        self.dispatches.append((source, blocks, args, threads))

    def reduce(self, value, axes, keepdims, dtype):
        self.reductions.append((value, axes, keepdims, np.dtype(dtype)))
        return value._data


def _operand(shape, dtype=np.float32, data=None):
    shape = tuple(shape)
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=np.dtype(dtype),
        _data=data if data is not None else object(),
    )


def test_source_has_a_uniform_128_lane_tree_and_flattened_rows():
    backend = _SourceBackend()
    a, b = _operand((2, 3, 67)), _operand((67,))
    result = matvec_data(backend, a, b, (2, 3), np.float32)

    assert result.shape == (2, 3)
    source, blocks, _, threads = backend.dispatches[0]
    assert blocks == 6 and threads == 128
    assert "long long" in source
    assert "a[row*67LL+j]" in source
    assert "__shared__ float sums[128]" in source
    assert source.count("__syncthreads();") == 2
    assert "for(long long j=start+(long long)lane;j<stop;j+=128LL)" in source


def test_single_block_source_covers_the_full_inner_dimension():
    backend = _SourceBackend()
    a, b = _operand((3, 4099)), _operand((4099,))
    matvec_data(backend, a, b, (3,), np.float32)

    source, blocks, _, threads = backend.dispatches[0]
    assert blocks == 3 and threads == 128
    assert "long long start=0LL" in source
    assert "long long stop=4099LL" in source
    assert "min((chunk+1)*1024LL" not in source


def test_long_dot_uses_bounded_partials_and_backend_reduce():
    backend = _SourceBackend()
    a, b = _operand((9001,), np.float64), _operand((9001,), np.float64)

    result = matvec_data(backend, a, b, (), np.float64)

    assert result.shape == (9,)
    assert backend.allocations == [((9,), np.dtype(np.float64))]
    source, blocks, _, threads = backend.dispatches[0]
    assert blocks == 9 and threads == 128
    assert "long long start=chunk*1024LL" in source
    assert "min((chunk+1)*1024LL,9001LL)" in source
    assert len(backend.reductions) == 1
    reduced, axes, keepdims, dtype = backend.reductions[0]
    assert reduced.shape == (9,) and axes == (0,) and not keepdims
    assert dtype == np.dtype(np.float64)


def test_empty_and_zero_inner_inputs_are_safe_without_a_launch():
    backend = _SourceBackend()
    empty = matvec_data(backend, _operand((0, 64)), _operand((64,)), (0,), np.float32)
    zero_inner = matvec_data(backend, _operand((3, 0)), _operand((0,)), (3,), np.float32)

    assert empty.shape == (0,)
    assert zero_inner.shape == (3,) and zero_inner.value == 0
    assert not backend.dispatches


def _cuda_required() -> bool:
    return os.environ.get("PACHO_REQUIRE_CUDA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.fixture(scope="module")
def cuda_custom():
    try:
        backend = pacho.get_backend(
            "cuda", device=os.environ.get("PACHO_CUDA_DEVICE", "0"), blas="custom"
        )
    except (BackendUnavailableError, ImportError) as exc:
        if _cuda_required():
            pytest.fail(str(exc))
        pytest.skip(f"CUDA unavailable{f' ({exc})' if str(exc) else ''}")
    yield backend
    backend.synchronize()


def _run_matvec(backend, left, right):
    a, b = pacho.asarray(left, backend=backend), pacho.asarray(right, backend=backend)
    return a @ b


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "shape",
    (
        (33, 17),
        (129, 67),
        (33, 255),
        (33, 256),
        (33, 257),
        (513, 1023),
        (3, 4099),
        (2, 3, 257),
        (2, 1, 513),
    ),
)
def test_cuda_gemv_irregular_rows_batches_and_boundaries(cuda_custom, dtype, shape):
    try:
        cuda_custom.validate_dtype(dtype)
    except UnsupportedDtypeError as exc:
        pytest.skip(str(exc))
    rng = np.random.default_rng(9101 + len(shape) + shape[-1])
    left = rng.uniform(-1, 1, shape).astype(dtype)
    right = rng.uniform(-1, 1, shape[-1]).astype(dtype)
    actual = _run_matvec(cuda_custom, left, right)
    expected = np.matmul(left.astype(np.float64), right.astype(np.float64))
    tolerance = (5e-5, 5e-6) if dtype == np.dtype(np.float32) else (5e-12, 5e-13)
    np.testing.assert_allclose(
        pacho.asnumpy(actual), expected, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.cuda
def test_cuda_long_rows_keep_the_benchmark_cancellation_case_within_tolerance(cuda_custom):
    # The historical serial float32 sum exceeded this tolerance in row 40.
    # Keep the observed input as a regression without relaxing the error bound.
    rng = np.random.default_rng(941)
    left = rng.uniform(-0.5, 0.5, (129, 8193)).astype(np.float32)
    right = rng.uniform(-0.5, 0.5, 8193).astype(np.float32)
    actual = _run_matvec(cuda_custom, left, right)
    reference = left.astype(np.float64) @ right.astype(np.float64)
    np.testing.assert_allclose(pacho.asnumpy(actual), reference, rtol=5e-5, atol=5e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("length", (63, 64, 65, 4096, 4097, 9001))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cuda_dot_tails_boundaries_and_long_partials(cuda_custom, length, dtype):
    try:
        cuda_custom.validate_dtype(dtype)
    except UnsupportedDtypeError as exc:
        pytest.skip(str(exc))
    rng = np.random.default_rng(9102 + length)
    left = rng.uniform(-1, 1, length).astype(dtype)
    right = rng.uniform(-1, 1, length).astype(dtype)
    actual = _run_matvec(cuda_custom, left, right)
    expected = np.matmul(left.astype(np.float64), right.astype(np.float64))
    tolerance = (5e-5, 5e-6) if dtype == np.dtype(np.float32) else (5e-12, 5e-13)
    np.testing.assert_allclose(
        pacho.asnumpy(actual), expected, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cuda_gemv_cancellation_large_finite_signed_and_nonfinite(cuda_custom, dtype):
    try:
        cuda_custom.validate_dtype(dtype)
    except UnsupportedDtypeError as exc:
        pytest.skip(str(exc))
    left = np.zeros((5, 257), dtype=dtype)
    right = np.zeros(257, dtype=dtype)
    # Every term and intermediate in this cancellation is exactly
    # representable in both supported precisions, so tree order is irrelevant.
    left[0, :4] = np.asarray([4096, -4096, 3, -3], dtype=dtype)
    right[:4] = np.asarray([1, 1, 1, 1], dtype=dtype)
    # Powers of two exercise a very large and a very small finite operand
    # while keeping their product exactly one and safely in range.
    left[1, 20], right[20] = np.asarray(2.0**60, dtype=dtype), np.asarray(2.0**-60, dtype=dtype)
    left[2, 10], right[10] = np.asarray(-0.0, dtype=dtype), np.asarray(1.0, dtype=dtype)
    left[2, 11], right[11] = np.asarray(0.0, dtype=dtype), np.asarray(1.0, dtype=dtype)
    left[3, 60], right[60] = np.asarray(np.inf, dtype=dtype), np.asarray(0.0, dtype=dtype)
    left[4, 61], right[61] = np.asarray(-np.inf, dtype=dtype), np.asarray(1.0, dtype=dtype)
    actual = _run_matvec(cuda_custom, left, right)
    expected = np.asarray([0.0, 1.0, 0.0, np.nan, -np.inf], dtype=np.float64)
    values = pacho.asnumpy(actual)
    np.testing.assert_allclose(values, expected, rtol=5e-5, atol=5e-6, equal_nan=True)
    assert values[0] == 0 and values[1] == 1 and np.isfinite(values[1])
    assert values[2] == 0 and not np.signbit(values[2])
    assert np.isnan(values[3]) and np.isneginf(values[4])
