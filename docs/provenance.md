# Provenance and recovered objective

Sources consulted on 2026-09-06:

- [azapat47/pypacho](https://github.com/azapat47/pypacho), commit
  `f34da092ad47f94ddd6c94005ecc1b6cd9b22982`. README, OpenCL/CUDA arrays, kernels,
  methods, and labs. Documented objective: parallel numerical operations and
  comparison of iterative methods by platform, size, and precision.
- [IEEE publication 9623197](https://ieeexplore.ieee.org/document/9623197): the title
  “PyPacho: A Python library that implements parallel basic
  operations on GPUs” was verified. The full text was not obtained; its
  tables have not been reconstructed, nor have the paper's experimental details
  been confirmed.

Diego Alejandro Cifuentes defined the main objective as a
hardware-agnostic Python GPU library that makes migration from NumPy simple.
This prioritizes OpenCL and array compatibility over vendor-exclusive backends
or an educational interface.

OpenPacho is developed by Diego Alejandro Cifuentes, who coauthored PyPacho with
Andrés Felipe Zapata Palacio, Esteban Echeverri Jaramillo, and Juan David Arcila Moreno.
The new implementation was written from scratch. No PyPacho kernels or classes
were copied. PyPacho contains no license file at the inspected commit.
It is consulted as historical context and evidence of the objective, not as an
implicit license to redistribute its code.

The original authors acknowledged in PyPacho's README are Andrés Felipe Zapata
Palacio, Esteban Echeverri Jaramillo, Diego Alejandro Cifuentes, and Juan David Arcila Moreno.

Technical references used for the design:

- [NumPy: custom array containers and dispatch](https://numpy.org/doc/stable/user/basics.dispatch.html).
- [PyOpenCL: platforms, devices, and contexts](https://documen.tician.de/pyopencl/runtime_platform.html).
- [PyOpenCL: arrays and operations](https://documen.tician.de/pyopencl/array.html).
