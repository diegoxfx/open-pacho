"""Run: python examples/numpy_migration.py --backend opencl (or numpy)."""

import argparse

import numpy as np

import openpacho as pacho


def existing_numpy_function(a, b):
    # This function keeps its original NumPy import. Supported calls dispatch to OpenPacho.
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy", "opencl"), default="opencl")
    args = parser.parse_args()
    with pacho.use_backend(args.backend):
        a = pacho.asarray([[1, 2, 3], [4, 5, 6]], dtype=pacho.float32)
        b = pacho.asarray([1, 1, 1], dtype=pacho.float32)
        distances = existing_numpy_function(a, b)
        print(distances)
        print(pacho.asnumpy(distances))


if __name__ == "__main__":
    main()
