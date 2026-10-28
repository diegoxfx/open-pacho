"""Address-width guards and reusable metadata independent of bound values."""

import numpy as np

import openpacho as pacho
from openpacho._opencl import _elementwise_source, _gather_source


def test_index_width_is_independent_of_storage_dtype_and_has_wide_fallback():
    dtype = np.dtype("int64")
    small = _elementwise_source("add", (((33,), dtype, False), ((), dtype, True)), (33,), dtype)
    large = _elementwise_source(
        "add", (((2**31,), dtype, False), ((), dtype, True)), (2**31,), dtype
    )
    assert "long *x0" in small and "int i =" in small
    assert "long i =" in large and "2147483648L" in large
    # A negative stride can make intermediate arithmetic larger than the
    # maximum final address; require the conservative 64-bit route then.
    narrow = _gather_source((33,), 32, (-1,), dtype)
    wide = _gather_source((33,), 2**31 - 2, (-1,), dtype)
    assert "int i =" in narrow and "long i =" in wide


def test_cached_metadata_rebinds_inputs_scalars_and_negative_strides(opencl_backend):
    host = np.arange(33 * 17, dtype=np.int64).reshape(33, 17)
    for shift in (2, 197):
        a = pacho.asarray(host + shift, backend=opencl_backend)
        np.testing.assert_array_equal(
            pacho.asnumpy((a + shift).T[::-1]), (host + 2 * shift).T[::-1]
        )
    assert _elementwise_source.cache_info().currsize <= 128
    assert _gather_source.cache_info().currsize <= 128
