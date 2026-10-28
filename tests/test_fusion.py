import numpy as np
import pytest

import openpacho as xp


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("shape", [(17, 3), (0, 3), ()])
def test_fused_broadcast_and_dtype(backend, dtype, shape):
    a = np.full(shape, 2, dtype=dtype)
    b = np.asarray(3, dtype=dtype)
    f = xp.fuse(lambda a, b: (a + b) * (a - 1))
    result = f(xp.asarray(a, backend=backend), xp.asarray(b, backend=backend))
    expected = (a + b) * (a - 1)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(xp.asnumpy(result), expected)


def test_transcendentals_and_shared_expression(backend):
    a = np.linspace(-2, 2, 35, dtype=np.float32).reshape(5, 7)
    b = np.linspace(0, 1, 7, dtype=np.float64)

    @xp.fuse
    def f(a, b, scale=0.25):
        t = xp.sin(a) * np.exp(-a * a)
        return (t + t) * scale + xp.cos(b)

    result = f(xp.asarray(a, backend=backend), xp.asarray(b, backend=backend))
    t = np.sin(a) * np.exp(-a * a)
    np.testing.assert_allclose(xp.asnumpy(result), (t + t) * 0.25 + np.cos(b), rtol=2e-6)
    assert result.dtype == np.float64


def test_comparison_nonfinite(backend):
    a = xp.asarray([np.nan, np.inf, -np.inf, -1, 0, 2], dtype=np.float32, backend=backend)
    f = xp.fuse(lambda a: xp.isfinite(a) == (a > 0))
    h = xp.asnumpy(a)
    np.testing.assert_array_equal(xp.asnumpy(f(a)), np.isfinite(h) == (h > 0))


def test_scalar_values_not_baked_into_program(opencl_backend):
    a = xp.asarray([1, 2, 3], dtype=np.float32, backend=opencl_backend)
    f = xp.fuse(lambda a, scale: a * scale + 7.125)
    f(a, 0.25)
    count = len(opencl_backend._kernels)
    np.testing.assert_allclose(xp.asnumpy(f(a, 0.5)), [7.625, 8.125, 8.625])
    assert len(opencl_backend._kernels) == count


def test_fusion_uses_one_launch_and_no_uploads(opencl_backend, monkeypatch):
    a = xp.ones((257,), backend=opencl_backend, dtype=np.float32)
    calls = []
    original = opencl_backend._run
    monkeypatch.setattr(
        opencl_backend, "_run", lambda *args: (calls.append(args), original(*args))[-1]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected upload")

    monkeypatch.setattr(opencl_backend, "upload", forbidden)
    f = xp.fuse(lambda a: 0.99 * a + 0.01 * a)
    np.testing.assert_allclose(xp.asnumpy(f(a)), 1)
    assert len(calls) == 1
    np.testing.assert_allclose(xp.asnumpy(0.5 * a + 1), 1.5)
    assert len(calls) == 3


def test_fusion_rejects_control_flow_and_reductions(backend):
    a = xp.ones((3,), backend=backend)
    with pytest.raises(TypeError, match="control flow"):
        xp.fuse(lambda a: a if a > 0 else -a)(a)
    with pytest.raises((TypeError, NotImplementedError)):
        xp.fuse(lambda a: np.sum(a))(a)
    with pytest.raises((TypeError, NotImplementedError)):
        xp.fuse(lambda a: a @ a)(a)
    with pytest.raises(TypeError, match="one elementwise"):
        xp.fuse(lambda a: (a, a))(a)


def test_fusion_rejects_mixed_backends(opencl_backend):
    a = xp.ones(3, backend="numpy")
    b = xp.ones(3, backend=opencl_backend)
    with pytest.raises(xp.BackendMismatchError):
        xp.fuse(lambda a, b: a + b)(a, b)
    with pytest.raises(xp.BackendMismatchError):
        xp.fuse(lambda a: a + b)(a)


def test_scalar_overflow_and_intermediate_rounding(backend):
    a = xp.asarray([1], dtype=np.int32, backend=backend)
    with pytest.raises(OverflowError):
        xp.fuse(lambda a: a + 2**40)(a)
    a = xp.asarray([2**24], dtype=np.float32, backend=backend)
    np.testing.assert_array_equal(xp.asnumpy(xp.fuse(lambda a: (a + 1) - a)(a)), [0])


def test_pool_preserves_queued_inputs_and_releases_cache(opencl_backend):
    a = xp.ones(513, dtype=np.float32, backend=opencl_backend)
    for _ in range(100):
        a = a + 1
    np.testing.assert_array_equal(xp.asnumpy(a), np.full(513, 101, dtype=np.float32))
    opencl_backend.release_cached_memory()
    assert opencl_backend._pool.held_blocks == 0
    np.testing.assert_array_equal(xp.asnumpy(a), 101)


def test_upload_snapshot_noncontiguous(opencl_backend):
    a = np.arange(100, dtype=np.float32).reshape(10, 10)[:, ::2]
    expected = a.copy()
    result = xp.asarray(a, backend=opencl_backend)
    a[:] = -1
    np.testing.assert_array_equal(xp.asnumpy(result), expected)


def test_backend_configuration_remains_explicit():
    with pytest.raises(ValueError, match="BLAS configuration"):
        xp.get_backend("numpy", blas="clblast")
    with pytest.raises(ValueError, match="blas must"):
        xp.get_backend("opencl", blas="unknown")
    original = xp.get_backend()
    with xp.use_backend("numpy"):
        assert xp.get_backend().name == "numpy"
    assert xp.get_backend() is original
