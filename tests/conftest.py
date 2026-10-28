"""Shared fixtures for backend conformance tests.

Install the package (editable or wheel) before testing. OpenCL is deliberately
optional: a missing runtime or device skips only the OpenCL cases unless the
caller explicitly requests a required OpenCL run.
"""

from __future__ import annotations

import os

import pytest


def _opencl_required() -> bool:
    return os.environ.get("PACHO_REQUIRE_OPENCL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _opencl_backend():
    from openpacho.backends import get_backend
    from openpacho.exceptions import BackendUnavailableError

    device = os.environ.get("PACHO_OPENCL_DEVICE") or None
    try:
        return get_backend("opencl", device=device, blas=os.environ.get("PACHO_BLAS", "custom"))
    except (BackendUnavailableError, ImportError) as exc:
        message = f"OpenCL unavailable{f' ({exc})' if str(exc) else ''}"
        if _opencl_required():
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture(params=("numpy", "opencl"), ids=("numpy", "opencl"))
def backend(request):
    """Return a concrete backend object for one NumPy or OpenCL test run."""

    from openpacho.backends import get_backend

    if request.param == "numpy":
        return get_backend("numpy")
    return _opencl_backend()


@pytest.fixture
def numpy_backend():
    from openpacho.backends import get_backend

    return get_backend("numpy")


@pytest.fixture
def opencl_backend():
    """Return the selected OpenCL backend, honoring the optional device ID."""

    return _opencl_backend()
