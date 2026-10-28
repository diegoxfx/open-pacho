"""Narrow tests for workload semantics, numerical validation and timing boundaries."""

import numpy as np
import pytest

from benchmarks.suite import Adapter, Case, make_inputs, operation, run_case, timed, validate


@pytest.mark.parametrize("op", ["cg", "jacobi", "sd"])
def test_native_solver_and_iteration_diagnostics(op):
    case = Case(op, (16, 16), "float64")
    inputs, exact = make_inputs(case, 47)
    value, iterations = operation(case, np, inputs, diagnostics=True)
    assert 0 < iterations < 400
    assert validate(case, value, exact, inputs, exact)["valid"]
    np.testing.assert_allclose(value, np.linalg.solve(*inputs), rtol=1e-5, atol=1e-7)


def test_transpose_materializes_and_pipeline_reuses_result():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    transposed = operation(Case("transpose", (2, 3), "float32"), np, (source,))
    assert transposed.flags.c_contiguous
    assert not np.shares_memory(source, transposed)
    reference = source * (0.99**10)
    actual = operation(Case("reuse", (2, 3), "float32", 10), np, (source, np.zeros_like(source)))
    np.testing.assert_allclose(actual, reference, rtol=1e-6)


def test_timing_synchronizes_before_and_after_every_invocation():
    events = []

    class FakeAdapter:
        def sync(self):
            events.append("sync")

    timed(FakeAdapter(), lambda: events.append("compute"), batch=3)
    assert events == ["sync", "compute", "sync", "compute", "sync", "compute", "sync"]


def test_validation_rejects_wrong_shape_dtype_and_values():
    case = Case("add", (2,), "float32")
    reference = np.ones(2, np.float32)
    for wrong in (np.ones(3, np.float32), np.ones(2, np.float64), np.zeros(2, np.float32)):
        assert not validate(case, wrong, reference, (), None)["valid"]


def test_cpu_rows_contain_raw_replicates_and_controlled_threads():
    rows = run_case(Case("add", (16,), "float32"), [Adapter("numpy-1")], 3, 47, 0.0001, 2)
    assert len(rows) == 1
    assert rows[0]["backend"] == "numpy-1"
    assert len(rows[0]["samples_s"]) == 3
    assert rows[0]["median_s"] > 0
