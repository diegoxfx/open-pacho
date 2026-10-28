# Selección residente con where

Fecha: 2026-10-22. Implementación y verificación local de `0.1.0a1`.

## Utilidad y alcance

Las comparaciones ya producían máscaras residentes, pero faltaba una primitiva
para elegir valores a partir de ellas. `where(condition, x, y)` completa ese
flujo para limpieza de datos, umbrales y funciones por tramos. Se integra en
las operaciones elementwise existentes, sin introducir indexación avanzada,
un framework de dominio ni una dependencia adicional.

```python
import numpy as np
import openpacho as pacho


@pacho.compile
def energia_valida(x):
    limpio = np.where(np.isfinite(x), x, 0.0)
    return np.sum(limpio * limpio)


x = pacho.asarray([1.0, float("nan"), -2.0], dtype=pacho.float32, backend="opencl")
print(energia_valida(x).item())  # 5.0; descarga explícita del escalar
```

La API pública `pacho.where` y el despacho `np.where` admiten broadcasting de los
tres argumentos. La condición numérica es verdadera cuando es distinta de cero;
NaN e infinito son verdaderos. Solo `x` e `y` determinan el dtype de salida,
con la promoción NumPy 2 de escalares Python débiles y escalares NumPy tipados.
Se conserva el backend de los arrays y se rechaza mezclar contextos.

Las expresiones de las dos ramas se evalúan antes de seleccionar: `where` no
es control de flujo perezoso ni oculta una operación no soportada en la rama
descartada. Seleccionar una rama tampoco equivale a multiplicarla por una
máscara: los NaN de la otra rama no contaminan el valor seleccionado.
La forma de un argumento que devuelve índices y los keywords `where=` de
ufuncs/reducciones siguen fuera de alcance.

## Integración

El resolver semántico añade promoción y broadcasting de tres operandos; cada
backend emite una selección condicional nativa. NumPy usa su operación propia.
OpenCL, Vulkan y CUDA la integran en eager, `fuse`, `compile` y map-reduce.
Al fusionar, la conversión de una condición numérica a bool compara con cero:
no trunca un valor fraccionario a un entero de almacenamiento booleano.

Los tests verifican que `sum(where(...))` se ejecuta con un lanzamiento por
llamada para el caso de tres filas de 3001 elementos, sin subir ni descargar
datos durante la operación. Es evidencia de fusión y residencia para esa forma;
no es una medición de aceleración ni una garantía de un único lanzamiento para
reducciones arbitrariamente grandes.

## Verificación

- **44 pruebas nuevas pasan** entre NumPy, OpenCL, Vulkan y CUDA custom. Una se
  omite porque CUDA no admite almacenamiento int64; no se emula en CPU.
- **53 regresiones dirigidas pasan**: promoción, booleanos, redondeos,
  broadcasting, rebinding, map-reduce y outputs compartidos de las rutas
  modificadas. Incluyen los dos proveedores CUDA.
- QA independiente: **APPROVE**, con comprobaciones CPU adicionales de tipos
  y escalares contra NumPy. No hubo pruebas GPU simultáneas durante esta ronda.

Se cubren condiciones fraccionarias y no finitas, selección de NaN/Inf/cero con
signo, enteros mayores de 2**53 donde se admiten, vacíos, escalares, salidas bool,
contextos incompatibles, ramas no soportadas y reutilización de planes con
escalares nuevos. No se ejecutó la suite global ni un benchmark de velocidad.

Entorno: Windows, Python 3.12.14, RTX 5060 Ti, driver 617.14, con los tres
backends GPU obligatorios. Los reportes son `where-tests.xml` y
`where-regression-tests.xml` en `.artifacts/iterations-2026-10-22`.
Se usaron `--capture=sys`, TEMP/TMP accesibles y compilador Vulkan explícito,
como en la [iteración de normas](cuda-norm-results.md).
