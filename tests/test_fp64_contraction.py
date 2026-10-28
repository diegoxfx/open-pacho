"""CPU protocol checks only: no device initialization, compilers or GPU calls."""

import itertools
import json
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import fp64_contraction as benchmark
from benchmarks import fp64_contraction_oracle as oracle
from openpacho.backends import NumpyBackend


def test_balanced_permutations_and_rejected_repetition_counts():
    orders = benchmark.mode_orders(24, 1069)
    assert Counter(orders) == {order: 4 for order in itertools.permutations(benchmark.MODES)}
    for bad in (True, 0, 5, 7, 12.0):
        with pytest.raises(ValueError):
            benchmark.mode_orders(bad, 1069)


@pytest.mark.parametrize(
    "counts,variant,observed,sufficient",
    [
        ((0, 1, 1), "separate", "separate", True),
        ((1, 0, 0), "contracted", "fused", True),
        ((0, 1, 1), "contracted", "separate", False),
        ((1, 1, 1), "native", None, False),
        ((0, 0, 0), "no_contraction", None, False),
    ],
)
def test_only_unmixed_compute_ir_establishes_policy(counts, variant, observed, sufficient):
    metadata = {"f64_instruction_evidence": dict(zip(("fma_f64", "mul_f64", "add_f64"), counts))}
    result = benchmark.policy_evidence(metadata, "cuda-custom", variant)
    assert result["observed_policy"] == observed
    assert result["evidence_sufficient"] is sufficient
    opaque = benchmark.policy_evidence({"f64_instruction_evidence": None}, "opencl", "native")
    assert opaque["observed_policy"] is None and not opaque["evidence_sufficient"]


def test_own_policy_acceptance_does_not_demand_fused_equal_separate():
    _, (a, b) = next(item for item in oracle.cases() if item[0] == "fma-residual-k17")
    refs = oracle.sampled_references(a, b)
    separate = oracle.reference(a, b, "separate")
    fused = oracle.reference(a, b, "fused")
    assert not oracle.compare(separate, fused)["equivalent"]
    for policy, actual in (("separate", separate), ("fused", fused)):
        result = benchmark._validation(
            actual, refs, {"observed_policy": policy, "evidence_sufficient": True}, None
        )
        assert result["accepted"] and result["policy_compliance_verified"]
        other = "fused" if policy == "separate" else "separate"
        assert not result["policy_diagnostics"][other]["policy_equivalent"]
    # Policy compliance and mathematical accuracy remain distinct: neither
    # sequential policy recovers this deliberately lost unit term.
    _, (a, b) = next(item for item in oracle.cases() if item[0] == "accumulation-cancellation")
    refs = oracle.sampled_references(a, b)
    for policy in ("separate", "fused"):
        result = benchmark._validation(
            oracle.reference(a, b, policy),
            refs,
            {"observed_policy": policy, "evidence_sufficient": True},
            None,
        )
        assert result["accepted"]
        assert result["policy_diagnostics"][policy]["max_absolute_error"] == 1.0


@pytest.mark.parametrize("flag", benchmark.UNSAFE_OPENCL_FLAGS)
@pytest.mark.parametrize("source", ("effective", "environment"))
def test_unsafe_opencl_build_flags_disqualify_contraction_causality(flag, source):
    metadata = {"f64_instruction_evidence": {"fma_f64": 1, "mul_f64": 0, "add_f64": 0}}
    if source == "effective":
        metadata["effective_build_options"] = "-cl-std=CL1.2 " + flag
    else:
        metadata["build_environment"] = {"PYOPENCL_BUILD_OPTIONS": flag}
    result = benchmark.policy_evidence(metadata, "opencl", "native")
    assert result["observed_policy"] == "fused"
    assert result["unsafe_build_flags"] == [flag]
    assert not result["evidence_sufficient"]
    assert result["status"] == "unsafe_opencl_build_flags"


@pytest.mark.parametrize("name", ("cuda", "opencl"))
def test_poison_uploads_use_existing_buffers_without_runtime_kernel_or_fill(name):
    calls, cache = [], {"native-source": object()}
    host = np.array([[np.nan, np.inf], [np.inf, np.nan]], dtype=np.float64)
    pointer = SimpleNamespace(ptr=12345)
    out = SimpleNamespace(_data=pointer)

    def cuda_copy(*args):
        calls.append(("copy", args))
        return (0,)

    def stream_sync(*args):
        calls.append(("stream-sync", args))
        return (0,)

    def opencl_copy(*args, **kwargs):
        calls.append(("copy", args, kwargs))

    def forbidden(*args, **kwargs):
        pytest.fail("Poison must not invoke a runtime kernel/fill path")

    backend = SimpleNamespace(
        name=name,
        _kernels=cache,
        _run=forbidden,
        full=forbidden,
        _activate=nullcontext,
        synchronize=lambda: None,
        driver=SimpleNamespace(cuMemcpyHtoD=cuda_copy, cuStreamSynchronize=stream_sync),
        cl=SimpleNamespace(enqueue_copy=opencl_copy),
        queue=object(),
    )
    before = benchmark.runtime_state(backend)
    benchmark.poison_output(backend, out, host)
    assert benchmark.runtime_state(backend) == before
    assert calls[0][0] == "copy"
    if name == "cuda":
        assert calls[0][1] == (pointer.ptr, host.ctypes.data, host.nbytes)
        assert calls[1] == ("stream-sync", (0,))
    else:
        assert calls[0][1][1] is pointer and calls[0][1][2] is host
        assert calls[0][2] == {"is_blocking": True}


def test_float64_reference_tolerance_is_an_independent_acceptance_requirement():
    a = np.ones((1, 1), dtype=np.float64)
    refs = oracle.sampled_references(a, a)
    result = benchmark._validation(
        a, refs, {"observed_policy": "separate", "evidence_sufficient": True}, a * 2
    )
    assert result["policy_compliance_verified"] and not result["accepted"]


@pytest.fixture
def cpu_protocol(monkeypatch):
    """Real ndarray transfer/shape protocol with CPU-owned policy executables."""
    events, owners, poison_history = [], [], []

    class Backend(NumpyBackend):
        blas = "custom"

        def matmul(self, a, b, shape, dtype):
            events.append("native")
            return oracle.reference(a._data.reshape(a.shape), b._data.reshape(b.shape)).reshape(-1)

        def close(self):
            events.append("backend-close")

    backend = Backend()
    backend._kernels = {}

    def capture(device, a, b, shape, geometry):
        events.append("capture")
        return SimpleNamespace(
            accumulation_steps=a.shape[1],
            metadata=lambda: {"geometry": geometry, "accumulation_steps": a.shape[1]},
        )

    class Owner:
        def __init__(self, variant, folder):
            self.variant, self.closed = variant, False
            folder.mkdir(parents=True)
            self.metadata = {
                "f64_instruction_evidence": {
                    "fma_f64": int(variant == "contracted"),
                    "mul_f64": int(variant != "contracted"),
                    "add_f64": int(variant != "contracted"),
                }
            }

        def call(self, a, b, out):
            events.append("owned-call")
            policy = "fused" if self.variant == "contracted" else "separate"
            actual = oracle.reference(a._data.reshape(a.shape), b._data.reshape(b.shape), policy)
            np.copyto(out._data, actual.reshape(-1))
            return out

        def close(self):
            self.closed = True
            events.append("owner-close")

    def prepare(device, descriptor, variant, folder):
        events.append("prepare")
        owner = Owner(variant, folder)
        owners.append(owner)
        return owner

    original_poison = benchmark.poison_output

    def poison(device, out, host):
        events.append("poison")
        poison_history.append((id(out._data), host.copy()))
        original_poison(device, out, host)

    monkeypatch.setattr(benchmark.kernels, "capture", capture)
    monkeypatch.setattr(benchmark.kernels, "prepare", prepare)
    monkeypatch.setattr(benchmark, "poison_output", poison)
    monkeypatch.setattr(benchmark.p, "get_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        benchmark,
        "_inputs",
        lambda *args: iter([("cpu-critical", "ieee", (np.ones((2, 3)), np.ones((3, 2))))]),
    )
    return SimpleNamespace(
        backend=backend,
        events=events,
        owners=owners,
        poison_history=poison_history,
        prepare=prepare,
    )


def test_report_exclusive_output_and_owned_cleanup_order(cpu_protocol, tmp_path):
    path = tmp_path / "report.json"
    result = benchmark.run(path, "cuda-custom", profile="quick", repeats=6)
    assert result["attempts_complete"] and result["execution_complete"]
    assert result["numerically_accepted"] and result["evidence_sufficient"]
    assert result["causal_conclusion_allowed"]
    assert json.loads(path.read_text())["execution_complete"]
    assert all(owner.closed for owner in cpu_protocol.owners)
    assert cpu_protocol.events[-1] == "backend-close"
    assert cpu_protocol.events.index("native") < cpu_protocol.events.index("prepare")
    assert cpu_protocol.events[: cpu_protocol.events.index("prepare")].count("poison") == 6
    for row in result["rows"]:
        assert row["runtime_unchanged_during_owned_work"]
        assert row["owned_resources_closed"]
        assert (
            row["runtime_after_native"]
            == row["runtime_after_setup"]
            == row["runtime_after_prepare"]
        )
        assert len({buffer for buffer, _ in cpu_protocol.poison_history}) >= 3
    # Alternating the poison classes across replays catches an unwritten output
    # even when its expected result is itself NaN or positive infinity.
    first_buffer = cpu_protocol.poison_history[0][0]
    poisons = [host for buffer, host in cpu_protocol.poison_history if buffer == first_buffer]
    assert np.array_equal(np.isnan(poisons[0]), np.isinf(poisons[1]))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.run(path, "cuda-custom", repeats=6)
    assert path.read_bytes() == before


def test_prepare_failure_closes_previously_owned_resources_and_continues(
    monkeypatch,
    cpu_protocol,
    tmp_path,
):
    def broken_prepare(device, descriptor, variant, folder):
        if variant == "contracted":
            raise RuntimeError("deliberate compile failure")
        return cpu_protocol.prepare(device, descriptor, variant, folder)

    monkeypatch.setattr(benchmark.kernels, "prepare", broken_prepare)
    path = tmp_path / "failed.json"
    result = benchmark.run(path, "cuda-custom", repeats=6)
    assert result["attempts_complete"] and not result["execution_complete"]
    assert len(result["rows"]) == 2
    assert all(row["error"]["message"] == "deliberate compile failure" for row in result["rows"])
    assert all(row["owned_resources_closed"] for row in result["rows"])
    assert all(owner.closed for owner in cpu_protocol.owners)
    assert cpu_protocol.events[-1] == "backend-close"
    assert json.loads(path.read_text())["attempts_complete"]


def test_performance_protocol_records_balanced_raw_samples_and_full_reference(
    monkeypatch,
    cpu_protocol,
    tmp_path,
):
    monkeypatch.setattr(
        benchmark,
        "_inputs",
        lambda *args: iter(
            [("cpu-performance", "performance", (np.ones((2, 3)), np.ones((3, 2))))]
        ),
    )
    report = benchmark.run(tmp_path / "performance.json", "cuda-custom", repeats=6)
    assert report["execution_complete"] and report["numerically_accepted"]
    for row in report["rows"]:
        assert len(set(row["orders"])) == 6
        for state in row["modes"].values():
            assert len(state["samples_seconds"]) == 6
            assert state["warmup"]["calls"] >= 3 and state["warmup"]["seconds"] >= 0.1
            assert state["validation"]["numpy_float64_allclose"]
            assert (
                len([check for check in state["checks"] if check["stage"].startswith("sample-")])
                == 6
            )


def test_initialization_failure_is_saved_and_main_returns_nonzero(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise RuntimeError("no device")

    monkeypatch.setattr(benchmark.p, "get_backend", unavailable)
    path = tmp_path / "init-failure.json"
    assert benchmark.main(["--backend", "opencl", "--output", str(path), "--repeats", "6"]) == 1
    report = json.loads(path.read_text())
    assert report["error"]["message"] == "no device"
    assert not report["attempts_complete"] and not report["execution_complete"]


def test_no_overwrite_existing_artifact_directory_before_device_initialization(
    monkeypatch, tmp_path
):
    path = tmp_path / "report.json"
    artifacts = path.with_name(path.name + ".artifacts")
    artifacts.mkdir()
    marker = artifacts / "keep.txt"
    marker.write_text("unchanged")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing artifact directory must reject before device initialization")

    monkeypatch.setattr(benchmark.p, "get_backend", forbidden)
    report = benchmark.run(path, "opencl", repeats=6)
    assert report["error"]["type"] == "FileExistsError"
    assert marker.read_text() == "unchanged"
