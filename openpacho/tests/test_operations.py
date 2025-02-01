# openpacho/tests/test_operations.py

import numpy as np
import pytest
from openpacho.opencl_backend.opencl_array import OpenCLArray
from openpacho.config import set_device_type
from openpacho.la.operations import dot, matmul

def test_elementwise_operations(device_type, device_index):
    # Create two host arrays
    set_device_type(device_type, device_index)
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

def test_elementwise_operations_matrix(device_type, device_index):
    # Create two host arrays
    set_device_type(device_type, device_index)
    host_a = np.random.rand(64, 64).astype(np.float32)
    host_b = np.random.rand(64, 64).astype(np.float32)

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

@pytest.mark.parametrize('length', [4, 128, 759, 3333])
def test_dot_vector_vector(device_type, device_index, length):
    set_device_type(device_type, device_index)
    x_host = np.random.rand(length).astype(np.float32)
    y_host = np.random.rand(length).astype(np.float32)

    x_gpu = OpenCLArray(x_host)
    y_gpu = OpenCLArray(y_host)

    gpu_res = dot(x_gpu, y_gpu)
    cpu_res = np.dot(x_host, y_host)

    assert np.allclose(gpu_res, cpu_res, atol=1e-6)

def test_dot_matrix_vector(device_type, device_index):
    set_device_type(device_type, device_index)
    A_host = np.random.rand(16, 17).astype(np.float32)
    x_host = np.random.rand(17).astype(np.float32)

    A_gpu = OpenCLArray(A_host)
    x_gpu = OpenCLArray(x_host)


    print("A gpu min: ", A_gpu.to_host().min(), "x gpu min: ", x_gpu.to_host().min())
    print("A cpu min: ", A_host.min(), "x cpu min: ", x_host.min())
    gpu_res = dot(A_gpu, x_gpu).to_host()#.ravel()
    cpu_res = A_host.dot(x_host)
    print("gpu res: ", gpu_res.min(), "shape: ", gpu_res.shape)
    print("cpu res: ", cpu_res.min(), "shape: ", cpu_res.shape)
    print("cpu: ", cpu_res)
    print("gpu", gpu_res)
    assert np.allclose(gpu_res, cpu_res, atol=1e-5)

@pytest.mark.skip(reason="no way of currently testing this")
def test_matmul_matrix_matrix(device_type, device_index):
    set_device_type(device_type, device_index)
    A_host = np.random.rand(32, 64).astype(np.float32)
    B_host = np.random.rand(64, 16).astype(np.float32)

    A_gpu = OpenCLArray(A_host)
    B_gpu = OpenCLArray(B_host)

    C_gpu = matmul(A_gpu, B_gpu).to_host()
    C_cpu = A_host @ B_host  # or np.matmul(A_host, B_host)

    assert np.allclose(C_gpu, C_cpu, atol=1e-5)
