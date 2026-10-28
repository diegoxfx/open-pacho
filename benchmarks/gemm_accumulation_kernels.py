"""Benchmark-only float32 accumulation alternatives for the selected direct GEMM.

``current`` preserves production source exactly. Other variants disable product/add
contraction and retain its tile, cooperative loads, barriers and output ownership.
``blocked`` rounds an ordinary sum within each K tile before an ordinary merge;
``blocked_neumaier`` uses the same partials with a compensated merge. Kahan uses
the conventional subtractive correction; Neumaier adds its correction at the end.
These experiments do not change runtime dispatch or compilation options.
"""

from dataclasses import dataclass
from math import prod
from numbers import Integral
from types import SimpleNamespace

import numpy as np

from openpacho import _gemm, _gemm_rect

VARIANTS = ("current", "sequential", "blocked", "kahan", "neumaier", "blocked_neumaier")


@dataclass(frozen=True)
class GemmKernel:
    source: str
    count: int
    shared: str | None
    tile: tuple[int, int, int]


def _replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(
            f"Production GEMM source drift: expected one occurrence, found {count}: {old}"
        )
    return source.replace(old, new, 1)


def _shape(value):
    value = tuple(value)
    if len(value) < 2 or any(not isinstance(v, Integral) or v <= 0 for v in value):
        raise ValueError("GEMM shapes must have at least two positive integer dimensions")
    return tuple(int(v) for v in value)


def _finite(name, value):
    if name == "vulkan":
        return f"(!isnan({value}) && !isinf({value}))"
    return f"isfinite({value})"


def _update(name, algorithm, accumulator, correction, value):
    """Emit a scoped scalar update; all arithmetic temporaries are float32."""
    dtype = "precise float" if name == "vulkan" else "float"
    s, q = accumulator, correction
    prefix = f"{{ {dtype} p={value};"
    if algorithm == "sequential":
        return prefix + f" {s}={s}+p; }}"
    if algorithm == "kahan":
        finite = " && ".join(_finite(name, v) for v in (s, "p", "t"))
        return (
            prefix + f" {dtype} y=p-{q}; {dtype} t={s}+y;"
            f" if({finite}) {{ {q}=(t-{s})-y; {s}=t; }}"
            f" else {{ {s}={s}+p; {q}=0.0f; }} }}"
        )
    if algorithm == "neumaier":
        absolute = "abs" if name == "vulkan" else "fabs"
        finite = " && ".join(_finite(name, v) for v in (s, "p", "t"))
        return (
            prefix + f" {dtype} t={s}+p;"
            f" if({finite}) {{ {dtype} e;"
            f" if({absolute}({s})>={absolute}(p)) {{ e=({s}-t)+p; }}"
            f" else {{ e=(p-t)+{s}; }} {q}={q}+e; }}"
            f" else {{ {q}=0.0f; }} {s}=t; }}"
        )
    raise ValueError(f"Unsupported scalar accumulation algorithm: {algorithm}")


def source_for(backend, ashape, bshape, variant="current"):
    """Return launch metadata without constructing a device or executing a kernel.

    Pass the actual backend object to reproduce ``select_tile`` configuration.
    A backend name alone selects the ordinary 32x32x16 fallback. CUDA callers
    must use the runtime's existing NVRTC ``--fmad=false`` option. ``shared`` is
    the separate Vulkan shared declaration; OpenCL/CUDA include it in source.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported accumulation variant: {variant}")
    if isinstance(backend, str):
        backend = SimpleNamespace(name=backend)
    name = backend.name
    if name not in ("opencl", "cuda", "vulkan"):
        raise ValueError(f"Unsupported GEMM backend: {name}")
    ashape, bshape = _shape(ashape), _shape(bshape)
    if ashape[-1] != bshape[-2]:
        raise ValueError("GEMM inner dimensions must match")
    try:
        batch = np.broadcast_shapes(ashape[:-2], bshape[:-2])
    except ValueError as exc:
        raise ValueError("GEMM batch dimensions must broadcast") from exc
    m, k, n = ashape[-2], ashape[-1], bshape[-1]
    if max(prod(ashape), prod(bshape), prod(batch) * m * n) > 2**31 - 1:
        raise ValueError("GEMM exceeds GPU indexing limits")
    selected = _gemm_rect.select_tile(backend, ashape, bshape)
    tile = selected or (32, 32, 16)
    tm, tn, tk = tile
    if selected is not None:
        source, count = _gemm_rect.source(name, ashape, bshape, selected)
    else:
        complete = name == "opencl" and m % 32 == 0 and n % 32 == 0 and k % 16 == 0
        source, count = _gemm._source(name, ashape, bshape, specialize_bounds=complete)
    shared = f"shared float left_tile[{tm * tk}],right_tile[{tk * tn}];"
    shared = shared if name == "vulkan" else None
    if variant == "current":
        return GemmKernel(source, count, shared, tile)

    rows, columns = tm // 8, tn // 8
    indices = [(r, c) for r in range(rows) for c in range(columns)]
    dtype = "precise float" if name == "vulkan" else "float"
    compensated = variant in ("kahan", "neumaier", "blocked_neumaier")
    blocked = variant in ("blocked", "blocked_neumaier")
    init = " ".join(f"float c{r}{c}=0.0f;" for r, c in indices)
    replacement = " ".join(f"{dtype} c{r}{c}=0.0f;" for r, c in indices)
    if compensated:
        replacement += " " + " ".join(f"{dtype} q{r}{c}=0.0f;" for r, c in indices)
    source = _replace_once(source, init, replacement)

    # Match the entire known inner loop so generator changes cannot silently
    # modify an unrelated operation or move a merge across a cooperative barrier.
    loads = " ".join(f"float a{r}=left_tile[(ly+{r * 8})*{tk}+kk];" for r in range(rows))
    loads += " ".join(f"float b{c}=right_tile[kk*{tn}+lx+{c * 8}];" for c in range(columns))
    arithmetic = " ".join(f"c{r}{c}+=a{r}*b{c};" for r, c in indices)
    original = f"for(int kk=0;kk<{tk};++kk) {{ {loads + arithmetic} }}"
    algorithm = "sequential" if blocked else variant
    updates = " ".join(
        _update(name, algorithm, f"{'d' if blocked else 'c'}{r}{c}", f"q{r}{c}", f"a{r}*b{c}")
        for r, c in indices
    )
    if compensated:
        updates = f"if(start+kk<{k}) {{ {updates} }}"
    inner = f"for(int kk=0;kk<{tk};++kk) {{ {loads} {updates} }}"
    if blocked:
        reset = " ".join(f"{dtype} d{r}{c}=0.0f;" for r, c in indices)
        merge_algorithm = "neumaier" if variant == "blocked_neumaier" else "sequential"
        merge = " ".join(
            _update(name, merge_algorithm, f"c{r}{c}", f"q{r}{c}", f"d{r}{c}") for r, c in indices
        )
        inner = f"{reset}\n            {inner}\n            {merge}"
    source = _replace_once(source, original, inner)
    if variant in ("neumaier", "blocked_neumaier"):
        finish = " ".join(f"c{r}{c}=c{r}{c}+q{r}{c};" for r, c in indices)
        source = _replace_once(source, "row+=ly; col+=lx;", finish + " row+=ly; col+=lx;")
    if name == "opencl":
        source = "#pragma OPENCL FP_CONTRACT OFF\n" + source
    return GemmKernel(source, count, shared, tile)
