"""CPU-only contract tests for the padding benchmark worker."""

from __future__ import annotations

import json

import numpy as np
import pytest

import openpacho as pacho
from benchmarks import padding
from openpacho.backends import NumpyBackend

_PUBLIC_SCOPES = ("resident", "prepared", "eager", "complete")


def _inputs():
    rng = np.random.default_rng(7721)
    return (
        rng.uniform(-0.5, 0.5, (33, 17)).astype(np.float32),
        rng.uniform(-0.5, 0.5, (17, 65)).astype(np.float32),
    )


def _cheap_warmup(calls):
    def warmup(function, xp, backend):
        del function, xp, backend
        calls.append(True)
        return {"calls": 1, "seconds": 0.0}

    return warmup


def test_build_calls_cpu_exposes_four_scopes_and_preserves_inputs():
    inputs = _inputs()
    snapshots = tuple(value.copy() for value in inputs)
    backend = NumpyBackend()

    calls, explanation, resident = padding.build_calls(backend, inputs, padding=True)

    assert tuple(calls) == _PUBLIC_SCOPES
    assert explanation["backend"] == "numpy"
    assert explanation["padding"] is True
    assert all(value.shape == source.shape for value, source in zip(resident, inputs))

    expected = inputs[0] @ inputs[1]
    for scope, function in calls.items():
        result = function()
        if scope == "complete":
            assert isinstance(result, np.ndarray)
            host = result
        else:
            assert isinstance(result, pacho.ndarray)
            assert result.backend == "numpy"
            assert result.shape == expected.shape
            assert result.dtype == np.dtype("float32")
            host = pacho.asnumpy(result)
        np.testing.assert_allclose(host, expected, rtol=5e-5, atol=5e-5)

    for value, snapshot in zip(inputs, snapshots):
        np.testing.assert_array_equal(value, snapshot)
    for value, snapshot in zip(resident, snapshots):
        np.testing.assert_array_equal(pacho.asnumpy(value), snapshot)


def test_run_cpu_fake_engine_writes_complete_paired_report(tmp_path, monkeypatch):
    warmups = []
    monkeypatch.setattr(padding, "cases", lambda: [((33, 17), (17, 65))])
    monkeypatch.setattr(padding, "engine", lambda name: (pacho, NumpyBackend()))
    monkeypatch.setattr(padding, "warmup", _cheap_warmup(warmups))
    monkeypatch.setattr(padding, "prepacked", lambda *args, **kwargs: None)

    output = tmp_path / "padding.json"
    report = padding.run(output, ["opencl"], repeats=3, seed=7721)

    assert report["complete"] is True
    assert report["backend"] == "opencl"
    assert report["repeats"] == 3
    assert len(report["rows"]) == 2
    assert {row["padding"] for row in report["rows"]} == {False, True}
    assert len(warmups) == len(_PUBLIC_SCOPES) * 2

    for row in report["rows"]:
        assert row["error"]["status"] == "finite"
        assert row["validation"]["float64_reference_close"] is True
        assert row["validation"]["numpy_same_dtype_close"] is True
        for scope in _PUBLIC_SCOPES:
            assert len(row[scope]["samples_seconds"]) == 3
            assert row[f"{scope}_warmup"]["calls"] == 1
        assert "prepacked_wall" not in row

    padded = next(row for row in report["rows"] if row["padding"])
    assert padded["exact_unpadded_equality"] is True

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["complete"] is True
    assert saved["backend"] == "opencl"
    assert len(saved["rows"]) == 2
    assert all(len(row["resident"]["samples_seconds"]) == 3 for row in saved["rows"])


@pytest.mark.parametrize(
    ("backends", "repeats"),
    [
        ([], 3),
        (["opencl", "vulkan"], 3),
        (["not-a-backend"], 3),
        (["opencl"], 2),
    ],
    ids=("empty-backends", "multiple-backends", "unsupported-backend", "too-few-repeats"),
)
def test_run_rejects_invalid_inputs(tmp_path, backends, repeats):
    with pytest.raises(ValueError):
        padding.run(tmp_path / "invalid.json", backends, repeats=repeats)


def test_run_requires_a_new_output_file(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        padding.run(output, ["opencl"], repeats=3)
