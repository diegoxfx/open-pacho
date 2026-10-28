"""CPU-only protocol and numerical tests for the bounded oscillator runner."""

import json
from collections import Counter

import numpy as np
import pytest

from benchmarks import oscillators as bench
from benchmarks.applications import NumpyAdapter, PachoAdapter


class RecordingAdapter(NumpyAdapter):
    def __init__(self):
        super().__init__(1)
        self.uploads, self.downloads = [], []

    def upload(self, values):
        self.uploads.append(tuple(np.array(v, copy=True) for v in values))
        return super().upload(values)

    def host(self, value):
        self.downloads.append(np.asarray(value).shape)
        return super().host(value)


def data(dtype="float32", n=5):
    case, certificate = bench.Case(n, dtype), {}
    inputs = bench.make_inputs(case, certificate=certificate)
    return case, inputs, bench.reference(case, inputs, certificate)


def test_profiles_dense_spd_and_balanced_order():
    assert [(c.n, c.dtype) for c in bench.QUICK_CASES] == [(31, "float32"), (31, "float64")]
    assert len(bench.ALL_CASES) == 6
    assert {c.n for c in bench.ALL_CASES} == {31, 257, 2049}
    assert bench.DT * bench.STEPS == 2
    for dtype in bench.TOLERANCES:
        case, inputs, ref = data(dtype, 31)
        k, q, v = inputs
        for got, rerun in zip(inputs, bench.make_inputs(case)):
            np.testing.assert_array_equal(got, rerun)
        assert np.count_nonzero(k) == case.n**2
        np.testing.assert_array_equal(k, k.T)
        assert np.linalg.eigvalsh(k.astype(np.float64))[0] > 1
        assert ref["stability"]["numerical_cholesky_succeeded"]
        assert ref["stability"]["modal_minimum_eigenvalue"] > 1
        assert ref["stability"]["verlet_stability_bound_satisfied"]
        assert q.shape == v.shape == (case.n,)
    orders = bench.shared.balanced_orders(2, 12, bench.SEED)
    assert Counter(tuple(o) for o in orders) == {(0, 1): 6, (1, 0): 6}
    for forward, reverse in zip(orders[::2], orders[1::2]):
        assert forward == reverse[::-1]


def test_discrete_reference_and_continuous_oracle_use_supplied_rounded_inputs():
    case, inputs, ref = data()
    modified = tuple(a.copy() for a in inputs)
    modified[1][0] = np.float32(1.234567)
    certificate = ref["stability"]
    result = bench.reference(case, modified, certificate)
    supplied64 = tuple(a.astype(np.float64) for a in modified)
    for got, expected in zip(
        result["discrete"], bench.oscillators.integrate(*supplied64, bench.DT, bench.STEPS)
    ):
        np.testing.assert_array_equal(got, expected)
    assert not np.array_equal(result["discrete"][0], ref["discrete"][0])
    k, q, v = supplied64
    # Reversing the analytic trajectory recovers the supplied initial conditions.
    eigenvalues, modes = np.linalg.eigh(k)
    omega, t = np.sqrt(eigenvalues), bench.DT * bench.STEPS
    qc, vc, _ = result["continuous"]
    cosine, sine = np.cos(omega * t), np.sin(omega * t)
    np.testing.assert_allclose(
        modes @ (cosine * (modes.T @ qc) - sine / omega * (modes.T @ vc)), q, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        modes @ (omega * sine * (modes.T @ qc) + cosine * (modes.T @ vc)), v, rtol=1e-13, atol=1e-13
    )
    assert (
        max(np.max(np.abs(a - b)) for a, b in zip(result["discrete"], result["continuous"])) < 0.003
    )
    assert bench.relative_energy_change(0.0, 0.0) == 0
    assert bench.relative_energy_change(0.0, 1.0) is None
    assert bench.relative_energy_change(1.0, np.nan) is None


def test_large_reference_avoids_eigendecomposition_and_rounded_checks_reject_bad_input(monkeypatch):
    # n=258 exercises the large branch cheaply; production's large case is n=2049.
    case, certificate = bench.Case(258, "float32"), {}
    inputs = bench.make_inputs(case, certificate=certificate)

    def forbidden(*args):
        raise AssertionError("large reference must not compute eigenvectors")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    assert bench.reference(case, inputs, certificate)["continuous"] is None
    bad = tuple(a.copy() for a in inputs)
    bad[0][0, 1] += 1
    with pytest.raises(ValueError, match="rounded input"):
        bench.reference(case, bad, certificate)
    negative = (-np.eye(case.n, dtype=case.dtype), *inputs[1:])
    certificate["rounded_matrix_sha256"] = bench.hashlib.sha256(negative[0].tobytes()).hexdigest()
    with pytest.raises(ValueError, match="Cholesky positivity"):
        bench.reference(case, negative, certificate)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_real_numpy_eager_and_prepared_protocol(tmp_path, numpy_backend, dtype):
    case, inputs, _ = data(dtype)
    adapter = RecordingAdapter()
    impls = [
        bench.Implementation(adapter, "numpy-eager"),
        bench.Implementation(PachoAdapter("numpy-plan", numpy_backend), "numpy-plan", True),
    ]
    path = tmp_path / "report.json"
    report = bench.run(path, repeats=2, cases=[case], selected_implementations=impls)
    assert report["attempts_complete"] and report["execution_complete"]
    assert report["all_numerically_accepted"]
    assert json.loads(path.read_text())["all_numerically_accepted"]
    assert report["provenance"]["package_origin"] == str(bench.Path(bench.p.__file__).resolve())
    assert set(report["provenance"]["measurement_helpers"]) == {
        "applications",
        "numerical_metrics",
        "dense_network",
    }
    assert report["cases"][0]["cpu_blas_threads"] == 1
    for row in report["rows"]:
        assert row["input_bytes"] == sum(a.nbytes for a in inputs)
        assert row["output_bytes"] == 3 * case.n * np.dtype(dtype).itemsize
        assert row["peak_memory"] == "not measured"
        assert row["gemvs_per_integration"] == 65
        assert row["warmup"]["calls"] >= 3 and row["warmup"]["seconds"] >= 0.1
        assert row["validation_calls"] == row["warmup"]["calls"] + 6
        assert row["energy_replay"]["samples_including_t0"] == 65
        assert row["energy_replay"]["max_absolute_relative_excursion"] < 0.002
        assert row["integration_time_error"]["q"]["relative_l2_error"] < 0.002
        for stage in ("resident", "complete", "upload"):
            assert len(row["timings"][stage]["samples_seconds"]) == 2
        for stage in (
            "initial_upload",
            "acceleration_example",
            "prepare",
            "first_call",
            "download",
        ):
            assert len(row["timings"][stage]["samples_seconds"]) == 1
        assert all(metric["accepted"] for metric in row["diagnostics"].values())
        assert all(
            metric["reference"] == ("sameprecision" if dtype == "float64" else "numpy-float64")
            for metric in row["diagnostics"].values()
        )
    assert len(adapter.uploads) == 5  # initial + two standalone + two complete
    for uploaded in adapter.uploads:
        assert len(uploaded) == 3
        for got, original in zip(uploaded, inputs):
            np.testing.assert_array_equal(got, original)
    eager = report["rows"][0]
    assert len(adapter.downloads) == 3 * (1 + eager["warmup"]["calls"] + 64 + 4)


def test_complete_clock_resets_and_includes_initial_acceleration_and_all_outputs(monkeypatch):
    case, inputs, ref = data()
    calls, downloads, in_clock = [], [], False

    class CountedMatrix(np.ndarray):
        def __matmul__(self, other):
            calls.append((in_clock, np.array(other, copy=True)))
            return np.asarray(self) @ other

    counted = (inputs[0].view(CountedMatrix), *inputs[1:])

    class Adapter(RecordingAdapter):
        def host(self, value):
            downloads.append(in_clock)
            return super().host(value)

    original_measure = bench._measure

    def measured(adapter, function):
        def wrapped():
            nonlocal in_clock
            in_clock = True
            try:
                return function()
            finally:
                in_clock = False

        return original_measure(adapter, wrapped)

    monkeypatch.setattr(bench, "_measure", measured)
    adapter = Adapter()
    impl = bench.Implementation(adapter, "counted")
    row = bench._row(case, impl, counted, ref)
    plan = bench.oscillators.verlet_step
    assert bench._round(case, impl, counted, ref, row, (counted, plan), 0)
    assert len(calls) == 130
    assert all(timed for timed, _ in calls)
    np.testing.assert_array_equal(calls[0][1], inputs[1])
    np.testing.assert_array_equal(calls[65][1], inputs[1])
    assert downloads == [False] * 3 + [True] * 3
    assert len(adapter.uploads) == 2
    for got, original in zip(counted, inputs):
        np.testing.assert_array_equal(got, original)


def test_independent_prepared_steps_survive_another_shape(numpy_backend):
    adapter = PachoAdapter("numpy-plan", numpy_backend)
    impl = bench.Implementation(adapter, "compiled", True)
    plans, states, cases, refs = [], [], [], []
    for dtype, n in (("float32", 5), ("float64", 7)):
        case, inputs, ref = data(dtype, n)
        resident = adapter.upload(inputs)
        plans.append(impl.prepare(case, resident, -(resident[0] @ resident[1])))
        states.append(resident)
        cases.append(case)
        refs.append(ref)
    assert plans[0] is not plans[1]
    for index in (0, 1, 0):
        got = impl.download(impl.calculate(plans[index], cases[index], states[index]))
        for actual, expected in zip(got, refs[index]["discrete"]):
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)


def test_numerical_failure_preserves_timing_and_json_for_nonfinite_outputs(tmp_path):
    class Wrong(bench.Implementation):
        def calculate(self, plan, case, resident):
            return tuple(np.full(case.n, np.nan, dtype="float64") for _ in range(3))

    report = bench.run(
        tmp_path / "wrong.json",
        repeats=2,
        cases=[bench.Case(5, "float32")],
        selected_implementations=[Wrong(NumpyAdapter(1), "wrong")],
    )
    row = report["rows"][0]
    assert report["execution_complete"] and not report["all_numerically_accepted"]
    assert row["final_relative_energy_change"] is None
    assert row["validation_failures"] and "error" not in row
    assert len(row["timings"]["resident"]["samples_seconds"]) == 2
    assert len(row["timings"]["complete"]["samples_seconds"]) == 2


def test_missing_provider_is_explicit_and_existing_report_is_preserved(tmp_path, monkeypatch):
    def unavailable(backend):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(bench, "implementations", unavailable)
    path = tmp_path / "missing.json"
    report = bench.run(path, backend="vulkan", repeats=2, cases=[bench.Case(5, "float64")])
    assert report["attempts_complete"] and not report["execution_complete"]
    assert [r["mode"] for r in report["rows"]] == ["eager", "compiled"]
    assert all(r["error"]["stage"] == "initialize_backend" for r in report["rows"])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        bench.run(path, profile="quick", repeats=2)
    assert path.read_bytes() == before
    for repeats in (0, 1, 3):
        with pytest.raises(ValueError, match="even"):
            bench.run(tmp_path / f"bad-{repeats}.json", repeats=repeats)


def test_runtime_failure_continues_other_rows_and_cleanup_is_once(tmp_path):
    class Engine:
        calls = 0

        def close(self):
            self.calls += 1

    class Broken(bench.Implementation):
        def prepare(self, *args):
            raise RuntimeError("prepare failed")

    adapter = NumpyAdapter(1)
    adapter.pacho_backend = Engine()
    report = bench.run(
        tmp_path / "failed.json",
        repeats=2,
        cases=[bench.Case(5, "float32")],
        selected_implementations=[Broken(adapter, "bad"), bench.Implementation(adapter, "good")],
    )
    assert report["attempts_complete"] and not report["execution_complete"]
    assert report["rows"][0]["error"]["stage"] == "prepare"
    assert report["rows"][1]["numerically_accepted"]
    assert adapter.pacho_backend.calls == 1


def test_interruption_saves_incremental_failure_and_closes_backend(tmp_path, monkeypatch):
    class Engine:
        calls = 0

        def close(self):
            self.calls += 1
            raise RuntimeError("cleanup failure")

    adapter = NumpyAdapter(1)
    adapter.pacho_backend = Engine()

    def interrupt(*args):
        raise KeyboardInterrupt("original interruption")

    monkeypatch.setattr(bench, "_round", interrupt)
    path = tmp_path / "interrupted.json"
    with pytest.raises(KeyboardInterrupt, match="original interruption"):
        bench.run(
            path,
            repeats=2,
            cases=[bench.Case(5, "float32")],
            selected_implementations=[
                bench.Implementation(adapter, "one"),
                bench.Implementation(adapter, "two"),
            ],
        )
    report = json.loads(path.read_text())
    assert not report["attempts_complete"]
    assert report["fatal_error"]["type"] == "KeyboardInterrupt"
    assert report["cleanup_errors"][0]["message"] == "cleanup failure"
    assert adapter.pacho_backend.calls == 1
