import numpy as np
import pytest

from openpacho.opencl_backend.opencl_array import OpenCLArray

def test_array_creation_from_list():
    data_list = [1, 2, 3, 4]
    gpu_array = OpenCLArray(host_data=data_list)
    np_result = gpu_array.to_host()
    assert np.allclose(np_result, np.array(data_list, dtype=np.float32))

def test_array_creation_from_numpy():
    data_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    gpu_array = OpenCLArray(host_data=data_np)
    np_result = gpu_array.to_host()
    assert np.allclose(np_result, data_np)

def test_array_from_host_update():
    gpu_array = OpenCLArray(host_data=[0, 0, 0])
    gpu_array.from_host([5, 6, 7])
    np_result = gpu_array.to_host()
    assert np.allclose(np_result, np.array([5, 6, 7], dtype=np.float32))
