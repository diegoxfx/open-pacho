"""Same NumPy function, optionally compiled for a selected OpenPacho backend."""

import argparse

import numpy as np

import openpacho as pacho


@pacho.compile
def energy(a, b, c):
    x = a * b + c
    y = 2 * x
    return np.sum(y * y, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "opencl", "vulkan", "cuda"), default="numpy")
    args = parser.parse_args()
    rng = np.random.default_rng(47)
    host = tuple(rng.uniform(-1, 1, (1024, 32)).astype(np.float32) for _ in range(3))
    expected = energy(*host)  # Native NumPy passthrough.
    with pacho.use_backend(args.backend):
        arrays = tuple(pacho.asarray(a) for a in host)
        result = energy(*arrays)
        np.testing.assert_allclose(pacho.asnumpy(result), expected, rtol=2e-5)
        result = energy(*arrays)  # Rebind inputs and execute the cached plan.
        print(energy.explain(*arrays))
        print(energy.cache_info())
        print(f"Validated {result.shape} on {result.device}")


if __name__ == "__main__":
    main()
