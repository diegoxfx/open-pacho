"""CPU-only control isolation, failure and bitwise-equality regression checks."""

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import padding_copy_review as review
from openpacho import _padding_copy


def _wheel(tmp_path):
    source = Path(_padding_copy.__file__).read_bytes()
    path = tmp_path / "control.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("openpacho/_padding_copy.py", source)
        # Loading the control must never import the historical runtime.
        archive.writestr("openpacho/__init__.py", "raise RuntimeError('wrong runtime')")
        archive.writestr("openpacho/_cuda.py", "raise RuntimeError('wrong backend')")
    return path, source


def test_wheel_control_loads_only_copy_module_and_records_exact_provenance(tmp_path):
    wheel, source = _wheel(tmp_path)
    current = _padding_copy.cuda_copy
    old, metadata = review.load_control(wheel)

    assert old["cuda_copy"] is not current
    assert _padding_copy.cuda_copy is current
    assert old["_body"]((2, 3), (3, 4), "uint") == _padding_copy._body((2, 3), (3, 4), "uint")
    assert metadata["copy_source_sha256"] == hashlib.sha256(source).hexdigest()
    assert metadata["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_wheel_without_copy_control_fails_instead_of_substituting_current_code(tmp_path):
    wheel = tmp_path / "wrong.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("openpacho/__init__.py", "")
    with pytest.raises(KeyError, match="_padding_copy"):
        review.load_control(wheel)


@pytest.mark.parametrize("name", ("opencl", "cuda", "vulkan"))
def test_prepared_controls_capture_distinct_copy_factories_and_restore_shared_module(
    monkeypatch, name
):
    key = name + "_copy"
    current_copy = getattr(_padding_copy, key)
    old_copy = object()
    marker = object()
    resident = (object(), object())
    backend = SimpleNamespace(name=name)
    monkeypatch.setattr(review.p, "asarray", lambda value, backend: value)

    class Prepared:
        def __init__(self, padded):
            self.copy = getattr(_padding_copy, key) if padded else None
            self._plan = SimpleNamespace(_execution_scope=marker if padded else None)

        def __call__(self, *values):
            return self.copy, self._plan._execution_scope, values

    def compile_probe(function, *, padding=False):
        assert function is review.product
        return SimpleNamespace(prepare=lambda *values: Prepared(padding))

    monkeypatch.setattr(review.p, "compile", compile_probe)
    calls = review.prepare_calls(backend, resident, {key: old_copy})
    assert getattr(_padding_copy, key) is current_copy

    expected_scoped = None if name == "cuda" else marker
    assert calls["old_copy"]() == (old_copy, expected_scoped, resident)
    assert calls["cached_copy"]() == (current_copy, expected_scoped, resident)
    assert calls["current"]() == (current_copy, marker, resident)
    assert calls["unpadded"]() == (None, None, resident)

    # The measured calls must retain the chosen factory, rather than reading
    # a process-global patched function when the clock is running.
    monkeypatch.setattr(_padding_copy, key, object())
    assert calls["old_copy"]()[0] is old_copy
    assert calls["cached_copy"]()[0] is current_copy
    assert calls["current"]()[0] is current_copy


@pytest.mark.parametrize("failure_index", (1, 2))
def test_copy_factory_patch_is_restored_when_plan_preparation_raises(monkeypatch, failure_index):
    backend = SimpleNamespace(name="cuda")
    original = _padding_copy.cuda_copy
    preparations = 0
    monkeypatch.setattr(review.p, "asarray", lambda value, backend: value)

    def prepare(*values):
        nonlocal preparations
        preparations += 1
        if preparations == failure_index:
            raise RuntimeError("preparation failed")
        return SimpleNamespace(_plan=SimpleNamespace(_execution_scope=object()))

    monkeypatch.setattr(
        review.p, "compile", lambda *args, **kwargs: SimpleNamespace(prepare=prepare)
    )
    with pytest.raises(RuntimeError, match="preparation failed"):
        review.prepare_calls(backend, (object(), object()), {"cuda_copy": object()})
    assert _padding_copy.cuda_copy is original


def test_source_equivalence_checks_real_broadcast_crop_shape():
    old = {key: getattr(_padding_copy, key) for key in ("_opencl_source", "_cuda_source", "_body")}
    crop_source, crop_destination = (2, 3, 96, 96), (2, 3, 65, 67)
    original = old["_cuda_source"]

    def changed_batched_crop(source, target):
        result = original(source, target)
        if source == crop_source and target == crop_destination:
            return result + "// intentionally different batched crop"
        return result

    old["_cuda_source"] = changed_batched_crop
    with pytest.raises(AssertionError):
        review.assert_identical_copy_sources(old, (2, 1, 65, 47), (1, 3, 47, 67))


def _cpu_run(monkeypatch, tmp_path, *, signed_zero_difference=False, fail_measurement=False):
    wheel, _ = _wheel(tmp_path)
    state = SimpleNamespace(closed=False, measurements=0)

    def close():
        state.closed = True

    backend = SimpleNamespace(
        name="cpu-probe", device="test", close=close, synchronize=lambda: None
    )
    monkeypatch.setattr(review, "engine", lambda name: (review.p, backend))
    # A zero inner dimension gives an exact zero reference, so a sign-bit
    # change passes the numerical tolerance but must fail the bitwise gate.
    monkeypatch.setattr(review, "cases", lambda: [((2, 0), (0, 2))])

    def calls(*args):
        result = {}
        for mode in ("old_copy", "cached_copy", "current", "unpadded"):
            value = np.zeros((2, 2), dtype=np.float32)
            if signed_zero_difference and mode == "current":
                value[0, 0] = np.float32(-0.0)
            result[mode] = lambda value=value: value.copy()
        return result

    def measure(function, *args):
        state.measurements += 1
        if fail_measurement:
            raise RuntimeError("measurement failed")
        return 1e-6, 1e-6, function()

    monkeypatch.setattr(review, "prepare_calls", calls)
    monkeypatch.setattr(review, "warmup", lambda *args: {"calls": 0, "seconds": 0})
    monkeypatch.setattr(review, "measure", measure)
    monkeypatch.setattr(review, "source_hashes", dict)
    return wheel, state


def test_complete_cpu_control_report_has_bitwise_gate_and_all_paired_samples(tmp_path, monkeypatch):
    wheel, state = _cpu_run(monkeypatch, tmp_path)
    output = tmp_path / "complete.json"
    report = review.run(output, ["opencl"], repeats=3, control_wheel=wheel)

    assert state.closed and state.measurements == 12
    assert report["complete"] is True
    assert len(report["rows"]) == 4
    for row in report["rows"]:
        assert row["bitwise_control_equality"] is True
        assert len(row["wall"]["samples_seconds"]) == 3
    assert json.loads(output.read_text(encoding="utf-8"))["complete"] is True


@pytest.mark.parametrize("failure", ("signed-zero", "measurement"))
def test_run_closes_backend_and_preserves_incomplete_report_on_failure(
    tmp_path, monkeypatch, failure
):
    wheel, state = _cpu_run(
        monkeypatch,
        tmp_path,
        signed_zero_difference=failure == "signed-zero",
        fail_measurement=failure == "measurement",
    )
    output = tmp_path / "incomplete.json"
    error = AssertionError if failure == "signed-zero" else RuntimeError
    with pytest.raises(error):
        review.run(output, ["opencl"], repeats=3, control_wheel=wheel)

    assert state.closed
    if failure == "signed-zero":
        assert state.measurements == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert report["rows"] == []
