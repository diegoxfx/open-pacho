import numpy as np
import pytest
from openpacho.opencl_backend.opencl_array import OpenCLArray
from openpacho.config import set_device_type
import openpacho as op


@pytest.mark.parametrize('length', [1, 3, 4, 7, 8, 16, 31, 128, 759, 1024, 2048, 3333])
def test_dot_vector_vector(device_type, device_index, length):
    set_device_type(device_type, device_index)
    x_host = np.random.rand(length).astype(np.float32)
    x_gpu = OpenCLArray(x_host)

    gpu_res = op.linalg.norm(x_gpu)
    cpu_res = np.linalg.norm(x_host)

    assert np.allclose(gpu_res, cpu_res, atol=1e-6)