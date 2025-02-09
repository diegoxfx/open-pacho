import numpy as np
import time
from openpacho.opencl_backend.opencl_array import OpenCLArray
from openpacho.config import set_device_type
set_device_type("cpu", 0)

def gradient_descent(A, b, initial_x, learning_rate=0.0001, num_iterations=1000, library=np):
    """
    Performs gradient descent using NumPy arrays.
    Uses the update rule:
        x = x - learning_rate * A.T @ (A @ x - b)
    """
    np = library
    x = initial_x
    print("A shape:", A.shape)
    print("x shape:", x.shape)
    for i in range(num_iterations):
        Ax = A @ x
        grad = A.transpose() @ (Ax - b)
        x = x - learning_rate * grad
    loss = np.linalg.norm(A @ x - b)
    print("total_iterations:", i)
    return x, loss

def run_numpy_gradient_descent(A, b, x_initial):
    import numpy as np
    start = time.time()
    x, loss = gradient_descent(A, b, x_initial, learning_rate=0.0001, num_iterations=1000, library=np)
    end = time.time()
    
    print("NumPy Gradient Descent:")
    print(f"Final loss: {loss:.4f}")
    print(f"Time taken: {end - start:.4f} seconds\n")

def run_openpacho_gradient_descent(A, b, x_initial):
    # Convert NumPy arrays to OpenPacho OpenCL arrays.
    import openpacho as op
    A = OpenCLArray(A)
    b = OpenCLArray(b)
    x_initial = OpenCLArray(x_initial)
    
    # (Optional) Warm-up the OpenCL context if necessary.
    #_ = A @ b  # Dummy operation to initialize the context.
    
    start = time.time()
    x, loss = gradient_descent(A, b, x_initial, learning_rate=0.0001, num_iterations=1000, library=op)
    end = time.time()
    
    print("OpenPacho Gradient Descent:")
    print(f"Final loss: {loss:.4f}")
    print(f"Time taken: {end - start:.4f} seconds\n")

if __name__ == "__main__":
    print("Running performance tests comparing gradient descent implementations...\n")
    
    n, m = 1000, 500
    A = np.random.randn(n, m).astype(np.float32)
    b = np.random.randn(n, 1).astype(np.float32)
    x_initial = np.zeros((A.shape[1], 1), dtype=np.float32)

    run_numpy_gradient_descent(A, b, x_initial)
    run_openpacho_gradient_descent(A, b, x_initial)
