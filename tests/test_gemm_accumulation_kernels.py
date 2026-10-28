"""CPU-only checks of experimental source selection and emitted arithmetic."""

import re
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import gemm_accumulation_kernels as kernels
from openpacho import _gemm, _gemm_rect


@pytest.mark.parametrize("name", ("opencl", "cuda", "vulkan"))
@pytest.mark.parametrize("shapes", [((64, 32), (32, 96)), ((2, 1, 33, 47), (1, 3, 47, 67))])
def test_current_is_exact_production_source_and_geometry(name, shapes):
    a, b = shapes
    result = kernels.source_for(name, a, b, "current")
    expected, count = _gemm._source(
        name, a, b, specialize_bounds=name == "opencl" and a == (64, 32)
    )
    assert result.source == expected
    assert result.count == count
    assert result.tile == (32, 32, 16)
    assert result.shared == (
        "shared float left_tile[512],right_tile[512];" if name == "vulkan" else None
    )


@pytest.mark.parametrize("name,tile", [("opencl", (16, 32, 16)), ("cuda", (16, 16, 32))])
def test_actual_backend_configuration_selects_rectangular_tile(name, tile):
    backend = SimpleNamespace(name=name, _small_row_gemm_config=(tile, 4081))
    a, b = (65, 4081), (4081, 257)
    expected, count = _gemm_rect.source(name, a, b, tile)
    for variant in kernels.VARIANTS:
        result = kernels.source_for(backend, a, b, variant)
        assert result.tile == tile and result.count == count
        if variant == "current":
            assert result.source == expected
    # The aligned fast path still follows production's fallback selection.
    assert kernels.source_for(backend, (64, 4096), (4096, 256)).tile == (32, 32, 16)


@pytest.mark.parametrize("name", ("opencl", "cuda", "vulkan"))
@pytest.mark.parametrize("variant", kernels.VARIANTS[1:])
def test_controls_barriers_tail_updates_and_float32_products(name, variant):
    source = kernels.source_for(name, (33, 17), (17, 35), variant).source
    barrier = {
        "opencl": "barrier(CLK_LOCAL_MEM_FENCE);",
        "cuda": "__syncthreads();",
        "vulkan": "barrier();",
    }[name]
    assert source.count(barrier) == 2
    assert source.count("p=a") == 16
    assert ":0.0f" in source and "double" not in source
    assert ("if(start+kk<17)" in source) == (variant in ("kahan", "neumaier", "blocked_neumaier"))
    if name == "opencl":
        assert source.startswith("#pragma OPENCL FP_CONTRACT OFF\n")
    if name == "vulkan":
        assert source.count("precise float c") == 16
        assert source.count("precise float p=") == (32 if variant.startswith("blocked") else 16)
        assert "precise float a0=" not in source
        assert "precise float b0=" not in source
    if variant.startswith("blocked"):
        # Tile reset follows the load barrier; exactly one merge is located
        # between the complete inner K loop and the reuse barrier.
        inner = source.index("for(int kk=0;")
        reset = source.index("float d00=0.0f;")
        merge = source.index("p=d00;")
        reuse = source.index(barrier, inner)
        assert source.index(barrier) < reset < inner < merge < reuse
        assert source.count("p=d00;") == 1


def test_source_drift_fails_loudly(monkeypatch):
    original = _gemm._source

    def changed(*args, **kwargs):
        source, count = original(*args, **kwargs)
        return source.replace("c00+=a0*b0;", "c00 = c00 + a0*b0;"), count

    monkeypatch.setattr(_gemm, "_source", changed)
    with pytest.raises(ValueError, match="source drift"):
        kernels.source_for("opencl", (33, 17), (17, 35), "kahan")


@pytest.mark.parametrize(
    "name,a,b,variant,error",
    [
        ("numpy", (2, 2), (2, 2), "current", "backend"),
        ("cuda", (2, 2), (2, 2), "unknown", "variant"),
        ("cuda", (2,), (2, 2), "current", "dimensions"),
        ("cuda", (0, 2), (2, 2), "current", "dimensions"),
        ("cuda", (2, 3), (2, 2), "current", "inner"),
        ("cuda", (2, 2, 3), (3, 3, 2), "current", "broadcast"),
        ("cuda", (65536, 65536), (65536, 1), "current", "indexing"),
    ],
)
def test_invalid_requests(name, a, b, variant, error):
    with pytest.raises(ValueError, match=error):
        kernels.source_for(name, a, b, variant)


def _execute_emitted_update(algorithm, values):
    """Execute the emitted scalar C subset with NumPy float32 operations.

    This translates declarations, scopes and conditions, without restating the
    algorithm's arithmetic. It tests the expressions actually inserted in kernels.
    """
    source = kernels._update("opencl", algorithm, "s", "q", "value")
    lines, indent, pending = [], 0, False
    for part in re.split(r"([{};])", source):
        part = part.strip()
        if not part or part == ";":
            continue
        if part == "{":
            if not pending:
                lines.append("    " * indent + "if True:")
            indent += 1
            pending = False
        elif part == "}":
            indent -= 1
        elif part.startswith("if("):
            condition = part[3:-1].replace("&&", " and ")
            lines.append("    " * indent + f"if {condition}:")
            pending = True
        elif part == "else":
            lines.append("    " * indent + "else:")
            pending = True
        else:
            statement = re.sub(r"^float ", "", part).replace("0.0f", "zero")
            if "=" in statement:
                lines.append("    " * indent + statement)
    code = compile("\n".join(lines), "<emitted GEMM update>", "exec")
    state = {
        "s": np.float32(0),
        "q": np.float32(0),
        "zero": np.float32(0),
        "isfinite": np.isfinite,
        "fabs": np.abs,
    }
    with np.errstate(all="ignore"):
        for value in values:
            state["value"] = np.float32(value)
            exec(code, {"__builtins__": {}}, state)  # noqa: S102 - trusted generated scalar code
        result = state["s"] + state["q"] if algorithm == "neumaier" else state["s"]
    return result, state["q"]


@pytest.mark.parametrize("algorithm", ("kahan", "neumaier"))
@pytest.mark.parametrize("sign", (1, -1))
def test_compensated_nonfinite_guard_preserves_infinity(algorithm, sign):
    result, correction = _execute_emitted_update(algorithm, [sign * np.inf, 1, 2])
    assert result == sign * np.inf and correction == 0
    result, correction = _execute_emitted_update(
        algorithm, [sign * np.finfo("float32").max] * 2 + [1]
    )
    assert result == sign * np.inf and correction == 0
    result, _ = _execute_emitted_update(algorithm, [np.inf, -np.inf])
    assert np.isnan(result)


def test_emitted_compensations_recover_small_addends():
    result, _ = _execute_emitted_update("kahan", [2**24, 1, 1, -(2**24)])
    assert result == 2
    result, _ = _execute_emitted_update("neumaier", [2**24, 1, -(2**24)])
    assert result == 1
