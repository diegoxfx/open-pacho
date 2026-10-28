"""CPU-only checks for balanced padding decisions and visible numerical failures."""

import itertools
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import padding_decision as decision
from openpacho.backends import NumpyBackend


class RecordingBackend(NumpyBackend):
    def __init__(self, *, offset=0, fail_close=False):
        self.closed = False
        self.uploads = 0
        self.offset = np.float32(offset)
        self.fail_close = fail_close

    def upload(self, value):
        self.uploads += 1
        return super().upload(value)

    def matmul(self, *args):
        return super().matmul(*args) + self.offset

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failed")


def setup_cpu(monkeypatch, *, offset=0, fail_close=False):
    backend = RecordingBackend(offset=offset, fail_close=fail_close)
    monkeypatch.setattr(decision, "CASES", (((2, 3), (3, 4)),))
    monkeypatch.setattr(decision, "engine", lambda name: (decision.p, backend))
    monkeypatch.setattr(decision, "warmup", lambda *args: {"calls": 1, "seconds": 0})
    monkeypatch.setattr(decision, "nvidia_snapshot", lambda: {"available": False})
    events, call_names = [], {}
    prepare = decision.prepare_calls

    def prepare_calls(*args):
        calls, explanations = prepare(*args)
        call_names.update(
            (function, (mode, scope))
            for mode, scopes in calls.items()
            for scope, function in scopes.items()
        )
        return calls, explanations

    def measure(function, *args):
        events.append(call_names[function])
        return 1e-6, 1e-6, function()

    monkeypatch.setattr(decision, "prepare_calls", prepare_calls)
    monkeypatch.setattr(decision, "measure", measure)
    return backend, events


def test_complete_report_balances_orders_and_includes_real_uploads(tmp_path, monkeypatch):
    backend, events = setup_cpu(monkeypatch)
    output = tmp_path / "nested" / "report.json"
    report = decision.run(output, ["opencl"], repeats=6, seed=42)

    assert backend.closed
    assert report["complete"] is True and report["all_reference_close"] is True
    assert report["all_bitwise_control_equal"] is True
    assert report["case_order"] == [0]
    assert len(report["rows"]) == 3
    assert report["package_origin"].endswith("__init__.py")
    assert "_gemm.py" in report["runtime_sources"]
    assert len(report["worker_sha256"]) == len(report["measurement_sha256"]) == 64
    modes = tuple(mode for mode, _ in decision.MODES)
    for scope in decision.SCOPES:
        measured_modes = [mode for mode, event_scope in events if event_scope == scope]
        assert measured_modes[:3] == list(modes)  # Untimed replay validations.
        samples = measured_modes[3:]
        assert Counter(tuple(samples[i : i + 3]) for i in range(0, len(samples), 3)) == Counter(
            itertools.permutations(modes)
        )
        for row in report["rows"]:
            assert row[scope + "_replay_bitwise_equal"] is True
            assert len(row[scope]["samples_seconds"]) == 6
    # Two resident uploads, then two uploads per replay/timed complete call.
    assert backend.uploads == 2 + 2 * 3 * (1 + 6)
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(json.dumps(report))


def test_prepare_calls_bind_each_distinct_compilation_and_current_uploads(monkeypatch):
    backend = RecordingBackend()
    host = (np.zeros((2, 3), np.float32), np.zeros((3, 4), np.float32))
    flags = []

    def compile_probe(function, *, padding):
        assert function is decision.product
        identity = len(flags)
        flags.append(padding)
        return SimpleNamespace(
            prepare=lambda *arrays: lambda *replacement: (identity, replacement),
            explain=lambda *arrays: {"identity": identity, "padding": padding},
        )

    monkeypatch.setattr(decision.p, "compile", compile_probe)
    monkeypatch.setattr(decision.p, "asnumpy", lambda value: value)
    calls, explanations = decision.prepare_calls(backend, host)
    assert flags == [False, True, False]
    resident_inputs = []
    for identity, (mode, scopes) in enumerate(calls.items()):
        observed, resident = scopes["prepared"]()
        complete_observed, uploaded = scopes["complete"]()
        assert observed == complete_observed == identity
        assert explanations[mode]["identity"] == identity
        assert all(a is not b for a, b in zip(resident, uploaded))
        resident_inputs.append(resident)
    assert all(
        all(a is b for a, b in zip(arrays, resident_inputs[0])) for arrays in resident_inputs
    )
    assert backend.uploads == 2 + 3 * 2


def test_selected_cases_are_the_only_cases_executed(tmp_path, monkeypatch):
    backend, _ = setup_cpu(monkeypatch)
    monkeypatch.setattr(decision, "CASES", (((2, 3), (3, 4)), ((3, 4), (4, 2)), ((1, 3), (3, 5))))
    report = decision.run(
        tmp_path / "subset.json", ["opencl"], repeats=6, seed=42, case_indices=[2, 0]
    )
    assert backend.closed
    assert set(report["case_order"]) == {0, 2}
    assert [row["case_index"] for row in report["rows"]] == [
        index for index in report["case_order"] for _ in decision.MODES
    ]


def test_finite_reference_failure_remains_visible_and_all_modes_finish(
    tmp_path, monkeypatch, capsys
):
    backend, _ = setup_cpu(monkeypatch, offset=1)
    output = tmp_path / "precision.json"
    report = decision.run(output, ["opencl"], repeats=6)

    assert backend.closed and report["complete"] is True
    assert report["all_reference_close"] is False
    assert report["all_bitwise_control_equal"] is True
    assert "Numerical validation failed" in capsys.readouterr().out
    for row in report["rows"]:
        assert row["validation"]["reference_close"] is False
        assert row["validation"]["violations"] == 8
        assert len(row["validation"]["examples"]) == 3
        assert row["validation"]["rtol"] == row["validation"]["atol"] == 5e-5
        assert row["error"]["max_abs_error"] > 0.9
        assert all(len(row[scope]["samples_seconds"]) == 6 for scope in decision.SCOPES)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["complete"] is True and saved["all_reference_close"] is False


def test_validation_keeps_tolerance_and_records_nonfinite_values_as_strict_json():
    actual = np.array([0, 1e-6, 1e-4, np.nan, np.inf], dtype=np.float32)
    validation = decision.validate_result(actual, np.zeros(5, dtype=np.float64))
    assert validation["reference_close"] is False and validation["finite"] is False
    assert validation["violations"] == 3
    assert [example["index"] for example in validation["examples"]] == [[2], [3], [4]]
    json.dumps(validation, allow_nan=False)


@pytest.mark.parametrize("dtype,shape", [(np.float64, (2,)), (np.float32, (3,))])
def test_validation_rejects_wrong_dtype_or_shape(dtype, shape):
    with pytest.raises(ValueError, match="reference shape and float32"):
        decision.validate_result(np.zeros(shape, dtype=dtype), np.zeros(2))


def test_execution_failure_closes_backend_and_saves_partial_report(tmp_path, monkeypatch):
    backend, _ = setup_cpu(monkeypatch)

    def fail(*args):
        raise RuntimeError("measurement failed")

    monkeypatch.setattr(decision, "measure", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="measurement failed"):
        decision.run(output, ["opencl"], repeats=6)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert backend.closed and saved["complete"] is False
    assert saved["all_reference_close"] is True
    assert len(saved["rows"]) == 3
    assert saved["execution_error"]["message"] == "measurement failed"


def test_cleanup_failure_still_saves_report(tmp_path, monkeypatch):
    backend, _ = setup_cpu(monkeypatch, fail_close=True)
    output = tmp_path / "cleanup.json"
    with pytest.raises(RuntimeError, match="close failed"):
        decision.run(output, ["opencl"], repeats=6)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert backend.closed and saved["complete"] is False
    assert saved["cleanup_error"]["message"] == "close failed"
    assert all("complete" in row for row in saved["rows"])


def test_replay_bitwise_check_detects_signed_zero_changes(tmp_path, monkeypatch):
    backend, _ = setup_cpu(monkeypatch)
    monkeypatch.setattr(decision, "CASES", (((2, 0), (0, 2)),))
    calls = 0

    def measure(function, *args):
        nonlocal calls
        calls += 1
        value = np.zeros((2, 2), dtype=np.float32)
        value[0, 0] = np.float32(-0.0)
        return 1e-6, 1e-6, value

    monkeypatch.setattr(decision, "measure", measure)
    output = tmp_path / "replay.json"
    with pytest.raises(AssertionError, match="Replay changed output bits"):
        decision.run(output, ["opencl"], repeats=6)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert backend.closed and calls == 1
    assert saved["complete"] is False and saved["all_reference_close"] is True
    assert saved["rows"][0]["prepared_replay_bitwise_equal"] is False


def test_nonfinite_validation_is_saved_before_rejecting_execution(tmp_path, monkeypatch):
    backend, _ = setup_cpu(monkeypatch, offset=np.inf)
    output = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="nonfinite"):
        decision.run(output, ["opencl"], repeats=6)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert backend.closed and saved["complete"] is False
    assert saved["all_reference_close"] is False
    assert saved["rows"][0]["error"]["actual_nonfinite_count"] == 8


@pytest.mark.parametrize(
    "options",
    [
        {"backends": []},
        {"backends": ["opencl", "vulkan"]},
        {"backends": ["cuda-cublas"]},
        {"repeats": 5},
        {"repeats": 7},
        {"repeats": True},
        {"seed": -1},
        {"case_indices": []},
        {"case_indices": [0, 0]},
        {"case_indices": [-1]},
        {"case_indices": [16]},
        {"case_indices": [1.5]},
        {"case_indices": [True]},
    ],
)
def test_invalid_configuration_is_rejected_before_backend_creation(tmp_path, monkeypatch, options):
    def unexpected(*args):
        pytest.fail("Invalid configuration initialized a backend")

    monkeypatch.setattr(decision, "engine", unexpected)
    arguments = {"backends": ["opencl"], "repeats": 6, **options}
    with pytest.raises(ValueError):
        decision.run(tmp_path / "invalid.json", **arguments)


def test_existing_report_is_never_overwritten(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="new output"):
        decision.run(output, ["opencl"], repeats=6)
    assert output.read_text(encoding="utf-8") == "existing"


def test_cli_passes_case_selection_and_success_is_execution_not_accuracy(monkeypatch, tmp_path):
    observed = {}

    def run(output, backends, **kwargs):
        observed.update(output=output, backends=backends, **kwargs)
        return {"complete": True, "all_reference_close": False}

    monkeypatch.setattr(decision, "run", run)
    output = tmp_path / "cli.json"
    assert (
        decision.main(
            [
                "--output",
                str(output),
                "--backends",
                "cuda-custom",
                "--repeats",
                "12",
                "--seed",
                "99",
                "--case-indices",
                "10,13",
            ]
        )
        == 0
    )
    assert observed == {
        "output": output,
        "backends": ["cuda-custom"],
        "repeats": 12,
        "seed": 99,
        "case_indices": [10, 13],
    }
