import numpy as np
import pytest
from openpacho.config import set_device_type

# Configure the OpenPacho backend (for example, using "cpu"; adjust as needed)
set_device_type("cpu", 0)

def setup_single_matrix_data(n=8192, m=8192, p=8192):
    """
    Create two random matrices A and B such that A (n x m) @ B (m x p) is valid.
    """
    A = np.random.randn(n, m).astype(np.float32)
    B = np.random.randn(m, p).astype(np.float32)
    return A, B

def setup_chained_matrix_data(n=4096, m=4096, p=4096, q=4096, r=4096):
    """
    Create four random matrices for a chain multiplication.
    Example chain: (((A @ B) @ C) @ D)
    A: (n x m), B: (m x p), C: (p x q), D: (q x r)
    """
    A = np.random.randn(n, m).astype(np.float32)
    B = np.random.randn(m, p).astype(np.float32)
    C = np.random.randn(p, q).astype(np.float32)
    D = np.random.randn(q, r).astype(np.float32)
    return A, B, C, D

def single_matrix_multiplication(A, B, library):
    """
    Perform a single matrix multiplication: A @ B.
    """
    return A @ B

def chained_matrix_multiplication(A, B, C, D, library):
    """
    Perform a chain of multiplications:
         result = (((A @ B) @ C) @ D)
    """
    result = A @ B
    result = result @ C
    result = result @ D
    return result

def convert_to_library_arrays(arrays, library):
    """
    If the provided library is not NumPy, convert the NumPy arrays into the library's array type.
    For OpenPacho, we assume that wrapping with OpenCLArray gives a drop-in replacement.
    """
    if library.__name__ != "numpy":
        from openpacho.opencl_backend.opencl_array import OpenCLArray
        return tuple(OpenCLArray(a) for a in arrays)
    return arrays

def run_single_matrix_benchmark(benchmark, library, n=8192, m=8192, p=8192):
    A, B = setup_single_matrix_data(n, m, p)
    A, B = convert_to_library_arrays((A, B), library)
    # Benchmark the single matrix multiplication
    return benchmark(lambda: single_matrix_multiplication(A, B, library))

def run_chained_matrix_benchmark(benchmark, library, n=4096, m=4096, p=4096, q=4096, r=4096):
    A, B, C, D = setup_chained_matrix_data(n, m, p, q, r)
    A, B, C, D = convert_to_library_arrays((A, B, C, D), library)
    # Benchmark the chained multiplications
    return benchmark(lambda: chained_matrix_multiplication(A, B, C, D, library))

# --- Pytest Benchmark Tests ---

def test_single_matrix_multiplication_numpy(benchmark):
    """
    Benchmark single matrix multiplication using NumPy.
    """
    result = run_single_matrix_benchmark(benchmark, np)
    # Optionally, you can check the resulting shape:
    # assert result.shape == (1024, 1024)

def test_single_matrix_multiplication_openpacho(benchmark):
    """
    Benchmark single matrix multiplication using OpenPacho.
    """
    import openpacho as op
    result = run_single_matrix_benchmark(benchmark, op)
    # Optionally, verify the shape of the result.

def test_chained_matrix_multiplication_numpy(benchmark):
    """
    Benchmark a chain of matrix multiplications using NumPy.
    """
    result = run_chained_matrix_benchmark(benchmark, np)
    # For example, for A (512x512), B (512x512), C (512x512), D (512x512),
    # the expected shape is (512, 512)
    # assert result.shape == (512, 512)

def test_chained_matrix_multiplication_openpacho(benchmark):
    """
    Benchmark a chain of matrix multiplications using OpenPacho.
    """
    import openpacho as op
    result = run_chained_matrix_benchmark(benchmark, op)
