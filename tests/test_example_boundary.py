"""Keep repository applications independent of the installed array runtime."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import openpacho as pacho
from examples import solvers
from openpacho.backends import NumpyBackend


def test_package_import_requires_neither_examples_nor_optional_gpu_libraries():
    script = """
import importlib.abc
import sys
blocked = {"examples", "pyopencl", "vulkan", "cuda", "nvmath", "cupy"}
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise RuntimeError("Forbidden import: " + fullname)
sys.meta_path.insert(0, BlockImports())
import openpacho as p
assert not hasattr(p, "solvers")
assert p.asnumpy(p.ones(3) + 2).tolist() == [3, 3, 3]
assert not any(name.partition(".")[0] in blocked for name in sys.modules)
"""
    # Exercise the same installed/source/wheel package as the parent process.
    # A wheel test may deliberately shadow an older package in this interpreter.
    package_root = str(Path(pacho.__file__).parent.parent)
    script = f"import sys; sys.path.insert(0, {package_root!r})\n" + script
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert importlib.util.find_spec("openpacho.solvers") is None


def test_backend_lookup_and_example_preserve_array_ownership_outside_default_context():
    # Equal names do not imply the same backend instance.
    owner = NumpyBackend()
    with pacho.use_backend("numpy"):
        matrix = pacho.asarray([[4.0, 1.0], [1.0, 3.0]], backend=owner)
        assert pacho.get_backend(matrix) is owner
        assert pacho.get_backend() is not owner
        result = solvers.conjugate_gradient(matrix, [1.0, 2.0])
        assert result.converged
        assert pacho.get_backend(result.x) is owner
        np.testing.assert_allclose(pacho.asnumpy(result.x), [1 / 11, 7 / 11])
        with pytest.raises(pacho.BackendMismatchError):
            solvers.conjugate_gradient(matrix, pacho.ones(2))


@pytest.mark.parametrize("config", [{"device": "0"}, {"blas": "custom"}, {"clblast_library": "x"}])
def test_array_backend_lookup_cannot_reconfigure_storage(config):
    value = pacho.ones(3)
    with pytest.raises(ValueError, match="cannot be reconfigured"):
        pacho.get_backend(value, **config)
