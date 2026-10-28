"""Small correctness checks for the optional CUDA benchmark references."""

import numpy as np
import pytest

from benchmarks.cuda_references import availability_errors, available_references


@pytest.fixture(scope="module")
def cuda_references():
    references = available_references()
    if not references:
        reasons = availability_errors()
        pytest.skip(f"no CUDA matmul reference is available: {reasons}")
    try:
        yield references
    finally:
        for reference in references:
            reference.close()


def test_available_references_are_cuda_only_and_named(cuda_references):
    names = [reference.name for reference in cuda_references]
    assert len(names) == len(set(names))
    assert all(reference.platform == "CUDA" for reference in cuda_references)
    assert set(names) <= {"cupy", "cublas", "nvmath"}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_references_upload_plan_execute_and_reuse(cuda_references, dtype):
    left = np.asarray(
        [[0.25, -0.5, 0.75, -1.0], [1.25, 0.5, -0.25, 0.125], [-0.75, 0.0, 0.5, 1.0]],
        dtype=dtype,
    )
    right = np.asarray([[1.0, -0.25], [0.5, 0.75], [-1.25, 0.125], [0.25, 0.5]], dtype=dtype)
    expected = left @ right

    for reference in cuda_references:
        resident = reference.upload((left, right))
        plan = reference.setup(resident)
        result = reference.execute(plan)
        reference.sync()
        np.testing.assert_allclose(reference.host(result), expected, rtol=3e-5, atol=3e-6)

        # Reusing a prepared plan must execute the new resident values without
        # rebuilding the plan (especially for nvmath-python's stateful API).
        resident[0][...] = 2
        resident[1][...] = 3
        result = reference.compute(plan)
        reference.sync()
        np.testing.assert_allclose(reference.host(result), np.full((3, 2), 24, dtype=dtype))


def test_references_reject_mixed_precision_and_non_matrix_inputs(cuda_references):
    reference = cuda_references[0]
    with pytest.raises(ValueError, match="same dtype"):
        reference.setup(
            (
                np.ones((2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float64),
            )
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        reference.setup(
            (
                np.ones((1, 2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            )
        )
