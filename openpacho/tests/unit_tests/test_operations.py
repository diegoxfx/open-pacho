# openpacho/tests/test_operations.py

import numpy as np
import pytest
from openpacho.opencl_backend.opencl_array import OpenCLArray
from openpacho.config import set_device_type
from openpacho.linalg.operations import dot

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

# Test vector·vector dot product with a variety of lengths,
# including very small, odd, power-of-2, and larger sizes.
@pytest.mark.parametrize('length', [1, 3, 4, 7, 8, 16, 31, 128, 759, 1024, 2048, 3333])
def test_dot_vector_vector(device_type, device_index, length):
    set_device_type(device_type, device_index)
    x_host = np.random.rand(length).astype(np.float32)
    y_host = np.random.rand(length).astype(np.float32)

    x_gpu = OpenCLArray(x_host)
    y_gpu = OpenCLArray(y_host)

    gpu_res = dot(x_gpu, y_gpu)
    cpu_res = np.dot(x_host, y_host)

    assert np.allclose(gpu_res, cpu_res, atol=1e-6)

# Test matrix-vector dot product for various sizes.
# Here, we treat A as an (M x N) matrix and x as a vector of length N.
@pytest.mark.parametrize("M, N", [
    (1, 2),        # small rectangular matrix
    (3, 3),        # small square matrix
    (16, 17),      # rectangular with odd columns
    (31, 7),       # odd rows, small number of columns
    (64, 32),      # power-of-2 dimensions
    (128, 257),    # power-of-2 rows with odd columns
    (1023, 512)    # large number of rows (odd) and power-of-2 columns
])
def test_dot_matrix_vector(device_type, device_index, M, N):
    set_device_type(device_type, device_index)
    A_host = np.random.rand(M, N).astype(np.float32)
    x_host = np.random.rand(N).astype(np.float32)

    A_gpu = OpenCLArray(A_host)
    x_gpu = OpenCLArray(x_host)

    gpu_res = dot(A_gpu, x_gpu).to_host()
    cpu_res = A_host.dot(x_host)

    assert np.allclose(gpu_res, cpu_res, atol=1e-5)

# Test matrix·matrix multiplication using various shapes.
# A is (M x K) and B is (K x N) so that C becomes (M x N).
@pytest.mark.parametrize("M, K, N", [
    (2, 2, 2),         # very small square matrices
    (3, 5, 7),         # small odd dimensions
    (16, 32, 15),      # rectangular matrices
    (31, 63, 17),      # odd dimensions with larger sizes
    (64, 64, 128),     # power-of-2 square and rectangular mix
    (128, 256, 333),   # larger matrices with odd output dimension
    (256, 512, 1023)   # high-dimensional matrices
])
def test_matmul_matrix_matrix(device_type, device_index, M, K, N):
    set_device_type(device_type, device_index)
    A_host = np.random.rand(M, K).astype(np.float32)
    B_host = np.random.rand(K, N).astype(np.float32)

    A_gpu = OpenCLArray(A_host)
    B_gpu = OpenCLArray(B_host)

    C_gpu = dot(A_gpu, B_gpu).to_host()
    C_cpu = A_host @ B_host

    assert np.allclose(C_gpu, C_cpu, atol=1e-5)

@pytest.mark.parametrize("M, N", [
    (3, 3),        # small square matrix
    (16, 17),      # rectangular with odd columns
    (31, 7),       # odd rows, small number of columns
    (64, 32),      # power-of-2 dimensions
    (128, 257),    # power-of-2 rows with odd columns
    (1023, 512)    # large number of rows (odd) and power-of-2 columns
])
def test_matrix_transpose(device_type, device_index, M, N):
    set_device_type(device_type, device_index)
    A_host = np.random.rand(M, N).astype(np.float32)

    A_gpu = OpenCLArray(A_host)

    gpu_res = A_gpu.transpose().to_host()
    cpu_res = A_host.transpose()

    print("cpu:", cpu_res)
    print("gpu:", gpu_res)

    assert np.allclose(gpu_res, cpu_res, atol=1e-5)