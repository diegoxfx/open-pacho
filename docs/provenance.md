# Procedencia y objetivo recuperado

Consulta realizada el 2026-09-06:

- [azapat47/pypacho](https://github.com/azapat47/pypacho), commit
  `f34da092ad47f94ddd6c94005ecc1b6cd9b22982`. README, arrays OpenCL/CUDA, kernels,
  métodos y laboratorios. Objetivo documentado: operaciones numéricas paralelas y
  comparación de métodos iterativos por plataforma, tamaño y precisión.
- [Publicación IEEE 9623197](https://ieeexplore.ieee.org/document/9623197): se pudo
  verificar el título «PyPacho: A Python library that implements parallel basic
  operations on GPUs». No se obtuvo el texto completo; no se han reconstruido sus
  tablas ni confirmado los detalles experimentales del artículo.

El usuario precisó que el objetivo principal es una biblioteca Python de GPU
agnóstica al hardware y de migración sencilla desde NumPy. Esto fija la prioridad
de OpenCL y de la compatibilidad de arrays por encima de backends exclusivos de un
fabricante o una interfaz educativa.

La nueva implementación se escribió desde cero. No se copiaron kernels ni clases
de PyPacho. PyPacho no contiene un archivo de licencia en el commit inspeccionado.
Se consulta como historia y evidencia del objetivo, no como una licencia
implícita para redistribuir su código.

Autores originales reconocidos en el README de PyPacho: Andrés Felipe Zapata
Palacio, Esteban Echeverri Jaramillo, Diego Alejandro Cifuentes y Juan David Arcila Moreno.

Referencias técnicas utilizadas para el diseño:

- [NumPy: custom array containers y dispatch](https://numpy.org/doc/stable/user/basics.dispatch.html).
- [PyOpenCL: plataformas, dispositivos y contextos](https://documen.tician.de/pyopencl/runtime_platform.html).
- [PyOpenCL: arrays y operaciones](https://documen.tician.de/pyopencl/array.html).
