# Normas estables CUDA y clientes externos

Fecha: 2026-10-22. Validación local en Windows, Python 3.12.14,
NVIDIA RTX 5060 Ti, driver 617.14. La versión continúa siendo `0.1.0a1`.

## Por qué esta iteración

CUDA podía multiplicar matrices y reducir arrays, pero no calcular una norma
estable. Eso impedía usar una primitiva común para magnitudes, normalización y
residuos, y bloqueaba los solvers externos. Reutilizar el algoritmo escalado
existente aporta más utilidad que otra microoptimización de ganancias marginales.
No añade algoritmos de dominio al paquete ni una dependencia nueva.

## Implementación y contrato

`_norm.py` comparte la combinación de pares `(scale, ssq)` entre OpenCL, Vulkan
y CUDA. CUDA añade únicamente su fuente y lanzamiento: 128 hilos, memoria
compartida y bloques de hasta 4096 términos. Los parciales permanecen en GPU;
no se calcula el cuadrado de una entrada extrema antes de escalarla.

`pacho.linalg.norm` y `np.linalg.norm` admiten el mismo subconjunto euclídeo y
Frobenius en float32/64, tanto eager como con `compile`. La norma conserva dtype
y backend, admite los ejes y `keepdims` existentes y funciona con los dos
proveedores CUDA. La elección `blas` sigue afectando a productos matriciales;
la norma usa el kernel propio compartido. NaN domina a infinito y una reducción
vacía devuelve cero. El rango del dtype y los subnormales siguen siendo límites.

Los solvers de `examples/solvers.py` ya pueden usar CUDA, incluidas sus formas
preparadas. Se conservan recurrencias, tolerancias y chequeo de residuo verdadero
en cada iteración. La validación cuenta entradas no finitas con un acumulador
flotante explícito y exige cero: no necesita contar exactamente todos los
elementos de un array grande ni requiere int64 en CUDA. CPU conserva validación
eager. Los solvers siguen fuera del wheel y de la API de la biblioteca.

## Evidencia numérica

Se ejecutaron los **52 casos existentes de normas y solvers** de
`benchmarks.accuracy` correspondientes a `cuda-custom` y `cuda-cublas`, con
referencia Decimal80 sobre los inputs representados y sin cambiar umbrales.
No hubo excepciones ni exclusiones por falta de operación.

Las ocho normas extremas pasan. El mayor error relativo observado fue
**4,52e-8 en float32** y **6,24e-17 en float64**; ambos proveedores usan el mismo
kernel y dieron los mismos resultados. Los 36 casos de sistemas ordinarios,
RHS pequeño y RHS grande también pasan el diagnóstico y convergen.

Persisten **seis alertas de error hacia la solución** en sistemas de Hilbert:
por proveedor, tamaños 4 y 8 en float32, y tamaño 8 en float64. Todos cumplen
su criterio de residuo, pero el error relativo hacia la solución Decimal alcanza
aproximadamente 0,0109, 0,9663 y 0,000150 respectivamente. Esto demuestra por qué
un residuo pequeño no garantiza una solución precisa en un sistema mal
condicionado. Las normas estables no resuelven ese problema ni las alertas
anteriores de cancelación en GEMM. No se relajan tolerancias para ocultarlas.

Estos casos se usan para diagnóstico numérico. Sus tiempos pequeños, con una
repetición, no fundamentan una afirmación de aceleración ni una comparación de
proveedores.

## Reproducción y límites

Los inputs y referencias ya viven en `benchmarks/accuracy.py`. La ejecución
completa se obtiene con:

```powershell
python -m benchmarks.accuracy --backends cuda-custom,cuda-cublas --repeats 1 --output accuracy.json
```

Ese comando también incluye suma/productos y conserva su salida no cero si hay
alertas diagnósticas. Esta iteración limitó el trabajo a `norm` y `solve`;
el script y JSON correspondientes están en
`.artifacts/iterations-2026-10-22/verify_norm_accuracy.py` y
`accuracy-cuda-{custom,cublas}.json`. Los artefactos privados no se distribuyen.

**257 pruebas dirigidas aprobadas**, con los tres backends GPU obligatorios y
51 avisos del compilador PyOpenCL; no se ejecutó la suite global. El reporte
está en `.artifacts/iterations-2026-10-22/norm-final-sys-tests.xml`.
La revisión independiente final aprobó el lote después de corregir la
regresión de instrumentación de la validación CPU y revisar esos resultados.
Las pruebas funcionales cubren rangos extremos, vacíos, escalares, ejes
irregulares, fronteras 4096/4097/8193, precedencia NaN/Inf entre bloques,
ejecución compilada sin descargas y los flujos ordinario/preparado de los tres
solvers. También se comprueban las rutas compartidas CPU/OpenCL/Vulkan.

La ejecución local encontró permisos insuficientes en el TEMP del sandbox al
compilar Vulkan; se configuró TEMP/TMP en el directorio privado de artefactos y
la ruta explícita al compilador GLSL instalado. También se observó una excepción
nativa Windows `0xc0000008` dentro de NVRTC con captura de descriptores de pytest.
El smoke que mezcla los cuatro backends y las suites CUDA en procesos separados
pasaron. La suite integrada también pasó usando `--capture=sys` en lugar de
capturar descriptores nativos. Se conserva esa opción para la verificación local;
la causa interna de la excepción de NVRTC no está diagnosticada y no se atribuye
a un fallo de aritmética ni se declara corregida en el driver.
