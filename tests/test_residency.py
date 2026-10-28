"""Guard the central migration promise: compatible operations keep data resident."""

import numpy as np

import openpacho as p
from examples import solvers


def test_numpy_dispatch_does_not_download_arrays(opencl_backend, monkeypatch):
    host = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = p.asarray(host, backend=opencl_backend)

    def unexpected_download(*args, **kwargs):
        raise AssertionError("Array operation attempted a host download")

    with monkeypatch.context() as patch:
        patch.setattr(opencl_backend, "download", unexpected_download)
        distances = np.sqrt(np.sum((a - 1) ** 2, axis=1))
        product = np.transpose(a) @ a
        sliced = a[::-1, ::2]
        assert distances.backend == product.backend == sliced.backend == "opencl"
    # OpenCL float32 pow/sqrt need not round identically to the CPU math library.
    np.testing.assert_allclose(
        p.asnumpy(distances), np.sqrt(np.sum((host - 1) ** 2, axis=1)), rtol=2e-6
    )
    np.testing.assert_allclose(p.asnumpy(product), host.T @ host)
    np.testing.assert_array_equal(p.asnumpy(sliced), host[::-1, ::2])


def test_solver_downloads_only_scalar_diagnostics(opencl_backend, monkeypatch):
    A = p.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=p.float32, backend=opencl_backend)
    b = p.asarray([1.0, 2.0], dtype=p.float32, backend=opencl_backend)
    original = opencl_backend.download
    sizes = []

    def track_download(a):
        sizes.append(a.size)
        assert a.size == 1, "Solver transferred a vector or matrix during iteration"
        return original(a)

    with monkeypatch.context() as patch:
        patch.setattr(opencl_backend, "download", track_download)
        result = solvers.conjugate_gradient(A, b)
        assert result.converged
    assert sizes
    np.testing.assert_allclose(p.asnumpy(result.x), [1 / 11, 7 / 11], rtol=1e-5)
