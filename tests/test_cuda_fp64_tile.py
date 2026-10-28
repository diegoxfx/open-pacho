"""CPU-only checks of the benchmark candidate's generated indexing and math order."""

import re
from collections import Counter
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from benchmarks.cuda_fp64_tile import source_for


def _eval(expression, values):
    """Interpret the integer expression subset emitted by the source generator."""
    expression = re.sub(r"\((?:long long|int)\)", "", expression)
    expression = re.sub(r"(?<=\d)LL\b", "", expression)
    expression = expression.replace("/", "//").replace("&&", " and ")
    if "?" in expression:
        condition, rest = expression.strip("()").split("?", 1)
        yes, no = rest.split(":", 1)
        return _eval(yes if _eval(condition.rstrip(")"), values) else no, values)
    # Expressions come solely from the local generator and contain integers only.
    return eval(expression, {"__builtins__": {}}, values)


def _declaration(source, name):
    return re.search(rf"const (?:int|long long) {name}=([^;]+);", source)[1]


def _simulate(kernel, a, b):
    """Execute emitted cooperative loads/stores and ordered products on the CPU.

    This deliberately reads the generated expressions: incorrect broadcast
    offsets, tile coordinates, guard conditions, or shared indices affect the
    result and cannot hide behind a separately implemented matrix multiply.
    """
    source = kernel.source
    loads = [
        re.search(rf"{name}\[lane\]=\((.*?)\)\?[ab]\[(.*?)\]:0.0;", source).groups()
        for name in ("left", "right")
    ]
    output_guard, output_address = re.search(r"if\((.*?)\) out\[(.*?)\]=acc;", source).groups()
    left_index, right_index = re.search(r"acc\+=left\[(.*?)\]\*right\[(.*?)\];", source).groups()
    tiles_k = int(re.search(r"tile_k<(\d+)LL", source)[1])
    assert "for(int j=0;j<valid_k;++j)" in source
    result = np.full(kernel.shape, np.nan, dtype=np.float64)
    visits = Counter()
    declarations = {
        name: _declaration(source, name)
        for name in ("ly", "lx", "batch", "row", "col", "abase", "bbase")
    }
    for g in range(kernel.blocks):
        contexts = []
        for lane in range(kernel.threads):
            context = {"g": g, "lane": lane}
            for name, expression in declarations.items():
                context[name] = _eval(expression, context)
            contexts.append(context)
        accumulators = np.zeros(kernel.threads, dtype=np.float64)
        k_visits = []
        for tile_k in range(tiles_k):
            shared = [np.full(256, np.nan), np.full(256, np.nan)]
            for context in contexts:
                context["tile_k"] = tile_k
                context["start"] = _eval(_declaration(source, "start"), context)
                context["valid_k"] = _eval(_declaration(source, "valid_k"), context)
                for array, operand, (guard, address) in zip(shared, (a, b), loads):
                    if _eval(guard, context):
                        offset = _eval(address, context)
                        assert 0 <= offset < operand.size
                        array[context["lane"]] = operand.flat[offset]
                    else:
                        array[context["lane"]] = 0.0
            valid_k = contexts[0]["valid_k"]
            k_visits.extend(contexts[0]["start"] + j for j in range(valid_k))
            for context in contexts:
                for j in range(context["valid_k"]):
                    context["j"] = j
                    accumulators[context["lane"]] += (
                        shared[0][_eval(left_index, context)]
                        * shared[1][_eval(right_index, context)]
                    )
        assert k_visits == list(range(a.shape[-1]))
        for context in contexts:
            if _eval(output_guard, context):
                address = _eval(output_address, context)
                assert 0 <= address < result.size
                visits[address] += 1
                result.flat[address] = accumulators[context["lane"]]
    assert visits == Counter(range(result.size))
    return result


@pytest.mark.parametrize(
    "ashape,bshape",
    [
        ((1, 1), (1, 1)),
        ((15, 15), (15, 17)),
        ((16, 16), (16, 16)),
        ((17, 17), (17, 15)),
        ((2, 1, 3, 19), (3, 19, 5)),
        ((3, 5), (2, 1, 5, 17)),
        ((2, 1, 17, 3), (1, 3, 3, 2)),
        ((1, 33), (33, 1)),
    ],
)
def test_cooperative_geometry_broadcast_and_valid_tail_match_serial_fp64(ashape, bshape):
    rng = np.random.default_rng(742)
    a, b = rng.uniform(-1, 1, ashape), rng.uniform(-1, 1, bshape)
    kernel = source_for(ashape, bshape)
    actual = _simulate(kernel, a, b)
    left = np.broadcast_to(a, kernel.shape[:-2] + a.shape[-2:])
    right = np.broadcast_to(b, kernel.shape[:-2] + b.shape[-2:])
    serial = np.zeros(kernel.shape)
    for j in range(a.shape[-1]):
        serial += left[..., :, j, None] * right[..., None, j, :]
    np.testing.assert_array_equal(actual, serial)
    np.testing.assert_allclose(actual, a @ b, rtol=2e-13, atol=2e-14)


def test_fixed_launch_storage_and_uniform_barrier_contract():
    kernel = source_for((2, 1, 17, 19), (3, 19, 33))
    assert kernel.shape == (2, 3, 17, 33)
    assert kernel.blocks == 6 * 2 * 3
    assert kernel.threads == 256
    assert kernel.shared_bytes == 2 * 16 * 16 * np.dtype(np.float64).itemsize == 4096
    with pytest.raises(FrozenInstanceError):
        kernel.blocks = 1
    assert "const double *a, const double *b, double *out" in kernel.source
    assert "__shared__ double left[256]" in kernel.source
    assert "__shared__ double right[256]" in kernel.source
    assert "double acc=0.0;" in kernel.source
    assert "float" not in kernel.source
    assert "return" not in kernel.source
    assert kernel.source.count("__syncthreads();") == 2
    assert "__syncthreads();\n        for(int j=0;j<valid_k;++j)" in kernel.source
    assert "__syncthreads();\n    }\n    if(row<" in kernel.source


@pytest.mark.parametrize(
    "ashape,bshape",
    [
        ((3,), (3, 2)),
        ((2, 3), (3,)),
        ((0, 3), (3, 2)),
        ((2, 0), (0, 3)),
        ((0, 2, 3), (1, 3, 4)),
        ((2, -1), (-1, 3)),
        ((2, 3), (4, 2)),
        ((2, 3, 4), (3, 4, 2)),
    ],
)
def test_ineligible_shapes_require_original_caller_fallback(ashape, bshape):
    with pytest.raises(ValueError):
        source_for(ashape, bshape)


def test_tile_and_dimensions_are_not_silently_truncated():
    with pytest.raises(ValueError, match="only tile=16"):
        source_for((2, 3), (3, 2), tile=8)
    with pytest.raises(TypeError):
        source_for((2, 3), (3, 2), tile=16.0)
    with pytest.raises(TypeError):
        source_for((2, 3.5), (3, 2))


def test_address_expressions_cross_32bit_without_allocating_buffers():
    k = 2**31 + 17
    kernel = source_for((2, 3, k), (1, k, 5))
    source = kernel.source
    context = {"batch": 1, "g": kernel.blocks - 1, "ly": 2, "lx": 4}
    assert _eval(_declaration(source, "abase"), context) == 3 * k
    assert _eval(_declaration(source, "bbase"), context) == 0
    assert f"row*{k}LL" in source
    for name in ("g", "batch", "row", "col", "abase", "bbase", "start"):
        assert f"const long long {name}=" in source
    assert "for(long long tile_k=" in source
    with pytest.raises(OverflowError):
        source_for((2**61, 1), (1, 1))
    # Each input fits, but the broadcast output's byte extent does not.
    with pytest.raises(OverflowError):
        source_for((2**30, 1, 1, 1), (1, 2**30, 1, 1))
