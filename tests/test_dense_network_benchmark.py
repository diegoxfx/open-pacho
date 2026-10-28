"""CPU protocol tests, including a real compiled plan backed only by NumPy."""

import json
from collections import Counter

import numpy as np
import pytest

from benchmarks import dense_network as bench
from benchmarks.applications import NumpyAdapter, PachoAdapter


class RecordingAdapter(NumpyAdapter):
    def __init__(self):
        super().__init__(1)
        self.uploads = []
        self.downloads = []

    def upload(self, values):
        self.uploads.append(tuple(np.array(v, copy=True) for v in values))
        return super().upload(values)

    def host(self, value):
        self.downloads.append(np.asarray(value).shape)
        return super().host(value)


def small_case(workload="sgd", dtype="float32"):
    return bench.Case(workload, 5, 3, 4, 2, dtype)


def test_predeclared_profiles_and_balanced_order():
    assert len(bench.QUICK_CASES) == 4
    assert len(bench.ALL_CASES) == 12
    assert {(c.n, c.d, c.h, c.o) for c in bench.ALL_CASES} == {
        (31, 7, 11, 3),
        (4096, 128, 256, 32),
        (4095, 127, 255, 31),
    }
    orders = bench.balanced_orders(2, 12, 47)
    assert orders == bench.balanced_orders(2, 12, 47)
    assert Counter(tuple(o) for o in orders) == {(0, 1): 6, (1, 0): 6}
    for order, reverse in zip(orders[::2], orders[1::2]):
        assert order == reverse[::-1]


def test_reference_uses_exact_supplied_rounded_inputs():
    case = small_case()
    inputs = list(bench.make_inputs(case))
    inputs[0][0, 0] = np.float32(1.234567)
    expected, gradients, prediction = bench.reference(case, tuple(inputs))
    rounded64 = tuple(a.astype(np.float64) for a in inputs)
    for actual, want in zip(expected, bench.network.sgd_step(*rounded64, bench.RATE)):
        np.testing.assert_array_equal(actual, want)
    for actual, want in zip(gradients, bench.network.loss_and_gradients(*rounded64)):
        np.testing.assert_array_equal(actual, want)
    np.testing.assert_array_equal(prediction, bench.network.predict(rounded64[0], *rounded64[2:]))
    assert prediction[0, 0] != bench.reference(case, bench.make_inputs(case))[2][0, 0]


@pytest.mark.parametrize("workload", ["inference", "sgd"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cpu_eager_and_real_prepared_protocol(tmp_path, numpy_backend, workload, dtype):
    adapter = RecordingAdapter()
    plan_adapter = PachoAdapter("numpy-plan", numpy_backend)
    impls = [
        bench.Implementation(adapter, "numpy-eager"),
        bench.Implementation(plan_adapter, "numpy-compiled", True),
    ]
    path = tmp_path / "evidence.json"
    case = small_case(workload, dtype)
    report = bench.run(path, repeats=2, cases=[case], selected_implementations=impls)
    assert report["execution_complete"] and report["all_numerically_accepted"]
    assert json.loads(path.read_text())["all_numerically_accepted"]
    assert report["provenance"]["runtime_sha256"]
    assert report["provenance"]["package_origin"] == str(bench.Path(bench.p.__file__).resolve())
    assert set(report["provenance"]["measurement_helpers"]) == {"applications", "numerical_metrics"}
    for row in report["rows"]:
        assert row["warmup"]["calls"] >= 3 and row["warmup"]["seconds"] >= 0.1
        assert row["peak_memory"] == "not measured"
        assert row["input_bytes"] == sum(
            a.nbytes for a in bench.runtime_inputs(case, bench.make_inputs(case))
        )
        for stage in ("resident", "complete", "upload"):
            assert len(row["timings"][stage]["samples_seconds"]) == 2
        for stage in ("initial_upload", "prepare", "first_call", "download"):
            assert len(row["timings"][stage]["samples_seconds"]) == 1
        assert any(name.endswith(":dw1") for name in row["diagnostics"]) == (workload == "sgd")
        assert all(value["accepted"] for value in row["diagnostics"].values())
        if dtype == "float64":
            assert all(
                value["reference"] == "sameprecision" for value in row["diagnostics"].values()
            )
    # One initial upload and two standalone+complete uploads; identical parameters each time.
    assert len(adapter.uploads) == 5
    for upload in adapter.uploads:
        expected_inputs = bench.runtime_inputs(case, bench.make_inputs(case))
        assert len(upload) == len(expected_inputs)
        for got, original in zip(upload, expected_inputs):
            np.testing.assert_array_equal(got, original)
    if workload == "sgd":
        # First, gradient diagnostic, warmup, and 2*(resident+complete) each download five outputs.
        assert len(adapter.downloads) == 5 * (1 + 1 + 1 + 4) + 1
        assert report["rows"][0]["output_bytes"] == np.dtype(dtype).itemsize * (
            1 + 3 * 4 + 4 + 4 * 2 + 2
        )


def test_complete_call_rebinds_original_parameters_and_downloads_every_output():
    case, adapter = small_case(), RecordingAdapter()
    impl = bench.Implementation(adapter, "recording")
    inputs = bench.make_inputs(case)
    expected, _, _ = bench.reference(case, inputs)
    row = bench._row(case, impl, inputs)
    bindings = []

    def plan(*args):
        bindings.append(args)
        return bench.network.sgd_step(*args)

    assert bench._round(case, impl, inputs, expected, row, (inputs, plan), 0)
    assert len(bindings) == 2
    assert len(adapter.downloads) == 10
    for args in bindings:
        for got, original in zip(args[:-1], inputs):
            np.testing.assert_array_equal(got, original)
    assert len(row["timings"]["complete"]) == 1


def test_numerical_failure_keeps_all_timing_and_records_wrong_dtype(tmp_path):
    class Wrong(bench.Implementation):
        def calculate(self, plan, case, resident):
            return np.zeros((case.n, case.o), dtype="float64")

    report = bench.run(
        tmp_path / "wrong.json",
        repeats=2,
        cases=[small_case("inference")],
        selected_implementations=[Wrong(NumpyAdapter(1), "wrong")],
    )
    row = report["rows"][0]
    assert report["execution_complete"] and not report["all_numerically_accepted"]
    assert row["validation_failures"] and "error" not in row
    assert len(row["timings"]["resident"]["samples_seconds"]) == 2
    assert len(row["timings"]["complete"]["samples_seconds"]) == 2
    assert row["diagnostics"]["first_call:prediction"]["actual_dtype"] == "float64"


def test_runtime_failure_preserved_and_next_row_continues(tmp_path):
    class Broken(bench.Implementation):
        def prepare(self, case, resident):
            raise RuntimeError("intentional prepare failure")

    report = bench.run(
        tmp_path / "failure.json",
        repeats=2,
        cases=[small_case("inference")],
        selected_implementations=[
            Broken(NumpyAdapter(1), "broken"),
            bench.Implementation(NumpyAdapter(1), "good"),
        ],
    )
    failed, passed = report["rows"]
    assert failed["error"] == {
        "type": "RuntimeError",
        "stage": "prepare",
        "message": "intentional prepare failure",
    }
    assert not failed["execution_complete"] and passed["numerically_accepted"]
    assert not report["execution_complete"]


def test_shape_failure_and_missing_outputs_remain_explicit():
    case, impl = small_case("inference"), bench.Implementation(NumpyAdapter(1), "numpy")
    row = bench._row(case, impl, bench.make_inputs(case))
    bench.diagnose(
        row,
        ("prediction",),
        (np.zeros((1,), dtype="float32"),),
        (np.zeros((case.n, case.o)),),
        case,
        "shape",
    )
    assert "shape mismatch" in row["validation_failures"][0]["message"].lower()
    bench.diagnose(row, ("prediction",), (), (np.zeros((case.n, case.o)),), case, "count")
    assert row["validation_failures"][-1]["message"] == "output count mismatch"


def test_backend_initialization_failure_has_rows_and_no_silent_skip(tmp_path, monkeypatch):
    def unavailable(backend):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(bench, "implementations", unavailable)
    report = bench.run(
        tmp_path / "unavailable.json",
        backend="vulkan",
        repeats=2,
        cases=[small_case(dtype="float64")],
    )
    assert not report["execution_complete"]
    assert [row["mode"] for row in report["rows"]] == ["eager", "compiled"]
    assert all(row["error"]["message"] == "backend unavailable" for row in report["rows"])


def test_exclusive_reservation_and_repeat_contract(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("untouched")
    with pytest.raises(FileExistsError):
        bench.run(path, profile="quick", repeats=2)
    assert path.read_text() == "untouched"
    for repeats in (0, 1, 3):
        with pytest.raises(ValueError, match="even"):
            bench.run(tmp_path / f"invalid-{repeats}.json", repeats=repeats)


def test_fatal_exception_preserved_cleanup_once_and_incremental_report(tmp_path, monkeypatch):
    class Engine:
        def __init__(self):
            self.calls = 0

        def close(self):
            self.calls += 1
            raise RuntimeError("cleanup failure")

    adapter = NumpyAdapter(1)
    adapter.pacho_backend = Engine()
    impls = [bench.Implementation(adapter, "one"), bench.Implementation(adapter, "two")]
    path = tmp_path / "interrupted.json"

    def interrupt(*args):
        raise KeyboardInterrupt("original interruption")

    monkeypatch.setattr(bench, "_round", interrupt)
    with pytest.raises(KeyboardInterrupt, match="original interruption"):
        bench.run(path, repeats=2, cases=[small_case("inference")], selected_implementations=impls)
    assert adapter.pacho_backend.calls == 1
    preserved = json.loads(path.read_text())
    assert preserved["fatal_error"]["type"] == "KeyboardInterrupt"
    assert len(preserved["rows"]) == 2
    assert preserved["cleanup_errors"][0]["message"] == "cleanup failure"
