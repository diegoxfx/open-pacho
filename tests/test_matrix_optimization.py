"""Matrix batch semantics and cooperative vector kernels, including irregular sizes."""

import numpy as np
import pytest

import openpacho as p
from examples import solvers


@pytest.mark.parametrize(
    "left,right",
    [
        ((2, 3, 4), (1, 4, 5)),
        ((2, 1, 3, 4), (1, 5, 4, 2)),
        ((4,), (2, 4, 3)),
        ((2, 3, 4), (4,)),
        ((0, 3, 4), (1, 4, 2)),
        ((2, 3, 0), (1, 0, 2)),
    ],
)
def test_batched_matmul_and_compilation(backend, left, right):
    rng = np.random.default_rng(470)
    a, b = (rng.uniform(-1, 1, shape).astype(np.float32) for shape in (left, right))
    da, db = (p.asarray(x, backend=backend) for x in (a, b))
    expected = a @ b
    for actual in (da @ db, p.compile(lambda x, y: x @ y)(da, db)):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(p.asnumpy(actual), expected, rtol=2e-5, atol=1e-6)
    with pytest.raises(NotImplementedError, match="Higher-rank dot"):
        p.dot(da, db)


@pytest.mark.parametrize("shape", [(37, 129), (1, 4097), (3, 1025), (1, 4096)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_irregular_gemv_and_dot(backend, shape, dtype):
    rng = np.random.default_rng(48)
    a, b = rng.integers(-3, 4, shape).astype(dtype), rng.integers(-3, 4, shape[1]).astype(dtype)
    da, db = (p.asarray(x, backend=backend) for x in (a, b))
    np.testing.assert_allclose(p.asnumpy(da @ db), a @ b, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(p.asnumpy(db @ db), b @ b, rtol=2e-5, atol=1e-6)


def test_cg_reuses_squared_residual_without_extra_host_checks(opencl_backend, monkeypatch):
    a = np.diag(np.full(31, 3, dtype=np.float32))
    a += np.diag(np.ones(30, dtype=np.float32), 1) + np.diag(np.ones(30, dtype=np.float32), -1)
    b = np.arange(31, dtype=np.float32)
    da, db = (p.asarray(x, backend=opencl_backend) for x in (a, b))
    original, reads = opencl_backend.download, []

    def download(value):
        reads.append(value.size)
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(opencl_backend, "download", download)
        result = solvers.conjugate_gradient(da, db, rtol=1e-5)
    assert result.converged
    assert reads == [1] * (5 + 2 * result.iterations)
    assert len(result.residual_history) == result.iterations + 1
    assert np.linalg.norm(b - a @ p.asnumpy(result.x)) <= 1.02e-5 * np.linalg.norm(b)
