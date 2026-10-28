# Padding opcional para productos matriciales

`padding=False` conserva la ruta habitual. Activarlo permite usar buffers privados
alineados con los tiles del kernel GEMM custom, rellenados con **ceros positivos
exactos**. Se recorta el resultado en GPU antes de entregarlo al siguiente paso.
La forma pública, el broadcasting y el dtype no cambian.

La [reevaluación con la base OpenCL corregida](padding-reassessment.md) conserva
la opción por ganancias concretas en productos rectangulares con pocas filas y
K largo. La mayoría de formas pierde rendimiento y no se encontró una ventaja
útil en Vulkan en el equipo medido. Conviene medir el recorrido completo antes
de activarla; no hay selección automática.

```python
import openpacho as pacho


@pacho.compile(padding=True)
def producto(a, b):
    return a @ b


# También pacho.compile(funcion, padding=True).
# a y b son arrays OpenPacho en el backend elegido por el usuario.
c = producto(a, b)
ejecutar = producto.prepare(a, b)  # hereda padding=True
d = ejecutar(otro_a, otro_b)

# Ejecución directa, sin decorar una función:
e = pacho.matmul(a, b, padding=True)
print(producto.explain(a, b))
```

El argumento acepta `bool` y `numpy.bool_`. El operador `@` adquiere la política
del compilador; fuera de él conserva el valor predeterminado. Una llamada explícita
a `pacho.matmul(..., padding=True)` también se captura dentro de `@pacho.compile`
sin opciones. `fuse` sigue siendo solo elemento a elemento y no admite esta opción.
La función compilada llamada con arrays NumPy nativos ejecuta el cuerpo original.
`padding` es una extensión de OpenPacho, no un keyword de `numpy.matmul`.

## Alcance inicial

| Ruta | Comportamiento con `padding=True` |
| --- | --- |
| OpenCL custom, Vulkan, CUDA custom | GEMM float32 elegible, incluidos lotes y broadcasting |
| OpenCL con CLBlast / CUDA con cuBLAS | Conserva el proveedor y su ruta habitual |
| NumPy, float64, vectores, matrices pequeñas/vacías | Conserva la operación habitual |
| Otras operaciones | Sin padding; consumen las dimensiones lógicas originales |

La elegibilidad mantiene la del GEMM de registros existente: operandos con al
menos dos dimensiones, `M,N >= 32`, `K >= 16` y capacidad suficiente de grupo y
memoria compartida. Se redondean **M y N a múltiplos de 32, K a múltiplos de 16**;
no a la siguiente potencia de dos. Los tamaños de lote no se expanden. Los límites
de índices de 32 bits y, cuando están disponibles, del tamaño máximo de buffer
del dispositivo se comprueban antes de preparar esta ruta. Si se superan, se
conserva la operación habitual. Un fallo real de asignación se propaga; no cambia
el backend o proveedor seleccionado.

El contrato es una opción de ejecución, no una promesa de mayor rendimiento ni
de que toda operación se rellene. `explain` indica la política solicitada y los
pasos `padded_matmul` efectivamente seleccionados. Cada uno muestra dimensiones
lógicas/físicas, alineación, `copy_dispatches` y `scratch_bytes`.

## Memoria y coste

Por ejemplo, `(33,17) @ (17,65)` utiliza físicamente `(64,32) @ (32,96)` y recorta
`(64,96)` a `(33,65)`. Sus buffers privados solicitan **45 056 bytes** adicionales:
las dos entradas rellenadas y el resultado físico. El resultado lógico ocupa su
propio buffer. Solo se copia un operando si cambia de tamaño; si todas las
dimensiones ya están alineadas no hay copias ni buffers privados adicionales.

`scratch_bytes` suma esos buffers privados para **una operación**. Excluye entradas
originales, salida lógica, pools, controlador y otras llamadas pendientes; no
representa el pico de memoria física del proceso. Cada invocación vuelve a copiar
los valores de entrada actuales. El plan cachea metadatos, no buffers de entrada
ni una suposición de inmutabilidad. No hay todavía propagación de layouts con
padding a través de varias operaciones: se recorta en cada frontera GEMM.

El kernel GEMM alineado elimina las comprobaciones por elemento de las colas de
sus tiles. Esto **no significa que todo el pipeline carezca de branches**: hay
bucles, guardas de copia y guardas del ejecutor. El balance depende de la forma,
el backend y el hardware: copiar dos entradas, escribir/recortar la salida y
calcular elementos descartados puede costar más que las guardas eliminadas.

## Precisión y subnormales

Las copias reinterpretan cada palabra de float32 como uint32. Así no hacen
aritmética sobre subnormales, ceros con signo o payloads de NaN. Las celdas nuevas
se escriben con el patrón `0x00000000`. En la dimensión K se rellenan ambos
operandos; las posiciones adicionales no leen datos ajenos al producto original.
Los valores de filas/columnas descartadas, incluidos posibles NaN por `0*Inf`,
no alcanzan reducciones posteriores como `sum` o `mean`.

No se activan fast-math, FTZ ni cambios de precisión. La aritmética conserva la
política del backend existente: preservar bits al copiar no garantiza que todos
los dispositivos conserven subnormales al multiplicar. OpenCL permite que una
implementación sin soporte de denormales los reduzca a cero; CUDA mantiene las
opciones existentes `ftz=false` y `fmad=false`.
Referencias: [OpenCL C, requisitos numéricos](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html)
y [opciones de NVRTC](https://docs.nvidia.com/cuda/nvrtc/).

## Medición reproducible

```bash
python -m benchmarks.isolated --worker padding --backends opencl,vulkan,cuda-custom --rounds 2 --repeats 21 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/padding/measurements
```

El worker empareja `False`/`True` con los mismos datos, alterna el orden, calienta
fuera del reloj y registra todas las muestras, errores contra float64, formas y
memoria solicitada. Mide la función compilada residente, el ejecutor preparado,
`matmul` directo y el recorrido completo con transferencias. **Todos incluyen
relleno y recorte**. El diagnóstico `prepacked_wall` se presenta aparte: reutiliza
entradas ya rellenadas y excluye el recorte; mide tiempo de pared con despacho,
asignación y sincronización, no tiempo puro de GPU ni una ganancia de la API.

Los procesos se ejecutan secuencialmente, con controles de inactividad/temperatura
antes de cambiar de backend y orden invertido en la segunda ronda. Un dispositivo
ocupado o caliente detiene la sesión sin inventar muestras. Sigue siendo necesaria
la validación en otros fabricantes.

Consulta los [resultados locales con errores y coste completo](padding-results.md).
Ese informe es histórico; la [decisión vigente](padding-reassessment.md) utiliza
el kernel OpenCL corregido, dos controles sin padding y formas adicionales:

```bash
python -m benchmarks.isolated --worker padding_decision --backends opencl,vulkan,cuda-custom --rounds 3 --repeats 30 --seed 971 --max-wait-seconds 300 --max-temperature 52 --output .artifacts/padding-decision/reproduction
```

En este worker, `complete` registra ejecución terminada y `all_reference_close`
registra precisión. Un proceso completado puede contener violaciones numéricas
finitas, que permanecen visibles con las tolerancias originales.

## Extensión futura: reducciones

`mean` sí puede beneficiarse de una implementación con padding: sumar incluyendo
ceros y dividir por la cantidad de elementos **lógicos** de los ejes reducidos,
no por su tamaño físico. La división puede ser otro kernel o integrarse en la
última etapa de la reducción; se debe medir cuál conviene. La restricción actual
es de alcance, no una imposibilidad matemática. Queda pendiente de implementar.

Esto requiere conservar los ejes, `keepdims`, dtype de acumulación y el contrato
para reducciones vacías, NaN/Inf y subnormales. El padding debe ser cero neutro de
la suma; no se puede extrapolar automáticamente a otras reducciones. Hay que
comparar el coste completo de relleno, reducción y normalización, sin prometer
que quitar guardas compense las copias o el trabajo adicional.
