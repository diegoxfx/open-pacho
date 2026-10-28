"""Run from the repository root: python -m examples.solve_system --backend numpy."""

import argparse

import openpacho as pacho
from examples.solvers import conjugate_gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "opencl", "vulkan", "cuda"), default="numpy")
    args = parser.parse_args()
    with pacho.use_backend(args.backend):
        matrix = pacho.asarray([[4, 1], [1, 3]], dtype=pacho.float32)
        rhs = pacho.asarray([1, 2], dtype=pacho.float32)
        result = conjugate_gradient(matrix, rhs)
        print(
            f"CG: converged={result.converged}, iterations={result.iterations}, "
            f"residual={result.residual_norm:g}"
        )
        print(pacho.asnumpy(result.x))


if __name__ == "__main__":
    main()
