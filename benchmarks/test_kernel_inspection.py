"""CPU-only checks for compiler evidence and controlled GEMM source variants."""

import re

import pytest

from benchmarks import kernel_inspection as inspection


def _body(source):
    assert source.startswith("// diagnostic variant ")
    return source.split("\n", 1)[1]


def test_ptx_summary_distinguishes_static_precision_predication_and_virtual_registers():
    report = inspection.ptx_summary(
        """
        .version 8.0
        .target sm_120
        .reg .pred %p<3>;
        .reg .f64 %fd<4>;
        .reg .f32 %f<8>;
        @%p1 cvt.f64.f32 %fd1, %f1;
        cvt.rn.f32.f64 %f2, %fd1;
        mul.rn.f32 %f3, %f1, %f2;
        fma.rn.f32 %f4, %f1, %f2, %f3;
        @!%p2 fma.rn.ftz.f32 %f5, %f1, %f2, %f4;
        @%p0 bra $L_done;
        ret;
        $L_done: @!%p0 exit;
        """
    )

    assert report["static_ir_opcodes"] == {
        "cvt.f64.f32": 1,
        "cvt.rn.f32.f64": 1,
        "mul.rn.f32": 1,
        "fma.rn.f32": 1,
        "fma.rn.ftz.f32": 1,
        "bra": 1,
        "ret": 1,
        "exit": 1,
    }
    assert report["float64_conversions"] == 2
    assert report["fma_f32"] == 2
    assert report["predicated_ir_instructions"] == 4
    assert report["ir_branches"] == 1
    assert report["virtual_register_declarations"] == [
        (".pred", "p", "3"),
        (".f64", "fd", "4"),
        (".f32", "f", "8"),
    ]
    assert "physical_registers" not in report
    assert "executed_instructions" not in report


def test_ptx_summary_ignores_both_comment_styles_in_instructions_and_registers():
    report = inspection.ptx_summary(
        """
        /*
        .reg .f64 %commented<999>;
        @%p9 cvt.f64.f32 %fd9, %f9;
        */
        // .reg .f32 %also_commented<888>;
        // fma.rn.f32 %f1, %f2, %f3, %f4;
        .reg .b32 %r<2>;
        @%p1 ret; // fma.rn.f64 %fd1, %fd2, %fd3, %fd4;
        /* same-line annotation */ trap;
        """
    )

    assert report["static_ir_opcodes"] == {"ret": 1, "trap": 1}
    assert report["predicated_ir_instructions"] == 1
    assert report["float64_conversions"] == 0
    assert report["fma_f32"] == 0
    assert report["virtual_register_declarations"] == [(".b32", "r", "2")]


def test_ptx_summary_does_not_label_float64_loads_or_arithmetic_as_conversions():
    report = inspection.ptx_summary(
        """
        ld.global.f64 %fd1, [%rd1];
        add.f64 %fd2, %fd1, %fd1;
        fma.rn.f64 %fd3, %fd2, %fd1, %fd1;
        cvt.rn.f32.s32 %f1, %r1;
        bra.uni $L_1;
        """
    )
    assert report["float64_conversions"] == 0
    assert report["fma_f32"] == 0
    assert report["ir_branches"] == 1
    assert len(report["static_ir_opcodes"]) == 5


@pytest.mark.parametrize("backend", ("opencl", "cuda", "vulkan"))
@pytest.mark.parametrize(
    "ashape, bshape",
    [((33, 17), (17, 65)), ((2, 1, 65, 47), (1, 3, 47, 67))],
)
def test_float_literal_ablation_changes_only_literal_types(backend, ashape, bshape):
    baseline, count = inspection.variant_source(backend, ashape, bshape, "baseline")
    candidate, candidate_count = inspection.variant_source(
        backend, ashape, bshape, "float_literals"
    )

    assert re.search(r"\b0\.0\b", baseline)
    assert not re.search(r"\b0\.0f\b", baseline)
    assert re.search(r"\b0\.0f\b", candidate)
    assert not re.search(r"\b0\.0\b", candidate)
    assert _body(candidate) == re.sub(r"\b0\.0\b", "0.0f", _body(baseline))
    assert candidate_count == count


@pytest.mark.parametrize("backend", ("opencl", "cuda", "vulkan"))
def test_historical_baseline_survives_runtime_float_literal_fix(monkeypatch, backend):
    current = "float c00=0.0f; left_tile[t]=(ar<33)?x0[ar]:0.0f;"
    monkeypatch.setattr(inspection, "_source", lambda *_args, **_kwargs: (current, 9))

    baseline, count = inspection.variant_source(backend, (33, 17), (17, 65), "baseline")
    candidate, candidate_count = inspection.variant_source(
        backend, (33, 17), (17, 65), "float_literals"
    )

    assert _body(baseline) == current.replace("0.0f", "0.0")
    assert _body(candidate) == current
    assert "0.0ff" not in candidate
    assert count == candidate_count == 9


@pytest.mark.parametrize("backend", ("opencl", "cuda", "vulkan"))
def test_aligned_guard_ablation_preserves_dispatch_and_historical_literal_policy(backend):
    baseline, count = inspection.variant_source(backend, (64, 32), (32, 96), "baseline")
    candidate, candidate_count = inspection.variant_source(
        backend, (64, 32), (32, 96), "aligned_guards_removed"
    )
    assert count == candidate_count == 6
    assert "if(row" not in candidate
    assert "?x0[" not in candidate and "?x1[" not in candidate
    assert not re.search(r"\b0\.0f\b", candidate)
    if backend == "opencl":
        # Runtime OpenCL already specializes complete tiles. This row is a
        # deliberate duplicate control, not a second independent optimization.
        assert _body(candidate) == _body(baseline)
    else:
        assert "if(row" in baseline


@pytest.mark.parametrize("backend", ("opencl", "vulkan"))
def test_no_contraction_variant_changes_only_float_policy_and_contraction_control(backend):
    source, count = inspection.variant_source(backend, (33, 17), (17, 65), "float_literals")
    candidate, candidate_count = inspection.variant_source(
        backend, (33, 17), (17, 65), "no_contraction"
    )
    candidate = _body(candidate)
    if backend == "opencl":
        assert candidate.startswith("#pragma OPENCL FP_CONTRACT OFF\n")
        candidate = candidate.removeprefix("#pragma OPENCL FP_CONTRACT OFF\n")
    else:
        assert len(re.findall(r"precise float c\d\d=", candidate)) == 16
        candidate = candidate.replace("precise float", "float")
    assert candidate == _body(source)
    assert candidate_count == count


def test_opencl_fixed_group_variant_keeps_arithmetic_and_dispatch_count():
    source, count = inspection.variant_source("opencl", (33, 17), (17, 65), "float_literals")
    candidate, candidate_count = inspection.variant_source(
        "opencl", (33, 17), (17, 65), "float_literals_fixed_group"
    )
    attribute = "__attribute__((reqd_work_group_size(64,1,1))) "
    assert candidate.count(attribute) == 1
    assert _body(candidate).replace(attribute, "") == _body(source)
    assert candidate_count == count
