import numpy as np
import pytest

from openpacho.opencl_backend.opencl_array import OpenCLArray
from openpacho.config import set_device_type


def test_array_creation_from_list(device_type, device_index):
    set_device_type(device_type, device_index)
    data_list = [1, 2, 3, 4]
    gpu_array = OpenCLArray(host_data=data_list)
    np_result = gpu_array.to_host()
    assert np.allclose(np_result, np.array(data_list, dtype=np.float32))

def test_array_creation_from_numpy(device_type, device_index):
    set_device_type(device_type, device_index)
    data_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    gpu_array = OpenCLArray(host_data=data_np)
    np_result = gpu_array.to_host()
    assert np.allclose(np_result, data_np)

def test_array_from_host_update(device_type, device_index):
    set_device_type(device_type, device_index)
    gpu_array = OpenCLArray(host_data=[0, 0, 0])
    gpu_array.from_host([5, 6, 7])
    np_result = gpu_array.to_host()
    assert np.allclose(np_result, np.array([5, 6, 7], dtype=np.float32))

def test_elementwise_operations():
    # Create two host arrays
    host_a = np.array([1, 2, 3, 4], dtype=np.float32)
    host_b = np.array([5, 6, 7, 8], dtype=np.float32)

    # Convert them to OpenCLArray
    cl_a = OpenCLArray(host_a)
    cl_b = OpenCLArray(host_b)

    # Perform operations on GPU
    cl_add = cl_a + cl_b
    cl_sub = cl_a - cl_b
    cl_mul = cl_a * cl_b
    cl_div = cl_a / cl_b

    # Fetch results back to host for comparison
    gpu_add_res = cl_add.to_host()
    gpu_sub_res = cl_sub.to_host()
    gpu_mul_res = cl_mul.to_host()
    gpu_div_res = cl_div.to_host()

    # Compare with NumPy
    np_add_res = host_a + host_b
    np_sub_res = host_a - host_b
    np_mul_res = host_a * host_b
    np_div_res = host_a / host_b

    assert np.allclose(gpu_add_res, np_add_res, atol=1e-6)
    assert np.allclose(gpu_sub_res, np_sub_res, atol=1e-6)
    assert np.allclose(gpu_mul_res, np_mul_res, atol=1e-6)
    assert np.allclose(gpu_div_res, np_div_res, atol=1e-6)