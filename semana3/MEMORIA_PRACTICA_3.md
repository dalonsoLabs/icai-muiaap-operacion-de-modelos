# Memoria de la práctica 3 — Inferencia y contratos

Daniel Alonso · Operación de Modelos · 14/10/2026

## Leyenda utilizada

- `->` Inicio, objetivo o tarea planteada.
- `~>` Desarrollo, cambio o ajuste realizado.
- `>` Paso técnico llevado a cabo.
- `>>` Evidencia, archivo o comprobación del paso.
- `=>` Resultado obtenido o estado final de la fecha.

## 14/10/2026 — Objetivo y preparación

`->` Mi objetivo ha sido separar el modelo entrenado de la inferencia y construir un módulo local reutilizable que transforme un CSV en predicciones trazables.

`>` He seguido el [enunciado de la semana 3](https://github.com/ssillerom/icai-muiaap-operacion-de-modelos/tree/main/semana3), el lienzo de la clase 1 y el starter de la clase 2. He conservado `uv`, Pydantic, Joblib, scikit-learn, pytest y Ruff, además del paquete `src/model_inference` que proporciona el profesor.

`~>` He partido del trabajo local que ya tenía en contratos, preprocesado y CLI. He completado la carga e inferencia del modelo y he ajustado la validación del lote. He trabajado en una copia independiente para conservar el checkout anterior.

`>>` La base de los materiales es el commit `b85a4f4f67bf7a1b3a714d53b76a9e42bb6756a6` del repositorio del profesor. Mi entrega está en la rama `tareas-semana3` de [mi fork](https://github.com/dalonsoLabs/icai-muiaap-operacion-de-modelos/tree/tareas-semana3/semana3).

## Clase 1 — Contratos y preprocesado

`>` He leído las cinco filas del CSV y he identificado `sample_id` como metadato. He definido once campos numéricos obligatorios, con los rangos del notebook y sin permitir columnas extra.

`~>` He elegido Pydantic para validar las fronteras de entrada y salida. He comparado su comportamiento con una dataclass: las anotaciones y `frozen=True` no comprueban rangos durante la construcción.

`>` He fijado el orden del vector en `FEATURE_NAMES`, he separado el identificador y he respondido las preguntas de las prácticas A, B y C. No he añadido escalado ni entrenamiento sobre los datos nuevos.

`>>` Entrego el [lienzo con tabla, unidades asumidas y criterios de aceptación](modules/03-inference-contracts/exercises/01-local-inference/01.01-contract-and-preprocess-canvas/problem/LIENZO.md) y el [notebook ejecutado](modules/03-inference-contracts/exercises/01-local-inference/01.01-contract-and-preprocess-canvas/problem/01-inferencia-y-contratos-alumno.ipynb), con siete celdas de código ejecutadas y sin errores.

`=>` He dejado definida la especificación que utiliza el módulo de la clase 2.

## Clase 2 — Implementación del módulo

`>` En `contracts.py` valido la petición y la predicción. Permito convertir el texto numérico del CSV a `float`, pero rechazo campos extra, valores no finitos y mediciones fuera de rango.

`>` En `preprocess.py` construyo once valores en un orden único y registro `wine-red-features-v1`. La dataclass conserva ese vector sin modificarlo.

`>` En `inference.py` cargo el artefacto disponible en el material local, compruebo `estimator`, el número de características, `feature_names`, las categorías y `model_version`. El artefacto es un `RandomForestClassifier` con la versión `wine-quality-rf-v1`.

`~>` Uso `predict_proba` para obtener la probabilidad de la categoría predicha según el orden de `classes_`. Valido que las probabilidades sean finitas, estén entre 0 y 1 y sumen 1, y paso la respuesta por el contrato de salida.

`>` En `predict_file.py` compruebo cabeceras, identificadores y todas las filas antes de cargar el modelo. Después preproceso e infiero y solo abro el fichero de salida cuando ya tengo todas las predicciones válidas.

`>>` El código y sus comentarios están en el [starter completado](modules/03-inference-contracts/exercises/01-local-inference/01.02-wine-quality-inference-module/problem/starter/). Conservo las responsabilidades de los cuatro archivos del ejercicio.

`=>` Puedo reutilizar `load_wine_quality_model`, `preprocess_wine_request` e `infer_wine_quality` sin depender del CLI.

## Ejecución reproducible

`>` Desde la raíz del repositorio, ejecuto:

```bash
cd semana3/modules/03-inference-contracts/exercises/01-local-inference/01.02-wine-quality-inference-module/problem/starter
uv sync --frozen
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -m model_inference.predict_file \
  --input assets/inference_samples.csv \
  --output .tmp/wine_predictions.csv
```

`>>` He utilizado Python 3.13.13, Pydantic 2.13.4, scikit-learn 1.9.0, Joblib 1.5.3, NumPy 2.5.1, pytest 8.4.2 y Ruff 0.16.1. Las dependencias están fijadas por el `uv.lock` proporcionado y he añadido `.python-version` para fijar el intérprete usado.

`>>` El modelo debe colocarse en `starter/models/wine_quality_classifier.joblib`. He utilizado el binario que ya estaba en mi carpeta local; no lo he reentrenado. Su SHA-256 es `55423a4af32faf2f27f9899db6bb04e181e9093b2d5ffc1d4c0923784cb6ec83`. El fichero no incluye procedencia de entrenamiento ni versión de scikit-learn en sus metadatos: el nombre de versión por sí solo no demuestra ese historial.

`~>` He mantenido `models/` y `.tmp/` fuera de Git, como establece el starter. Para repetir el CLI después de clonar hay que obtener el mismo modelo del material de clase y colocarlo en esa ruta, o pasar `--model /ruta/al/modelo.joblib`. Los tests generan sus propios artefactos pequeños temporales y pasan sin ese binario.

## Comprobaciones y resultados

`>>` Han pasado los 11 tests originales del starter y 19 casos adicionales: 30 tests en total. Compruebo el orden del vector, cinco salidas, errores de entrada, columna extra, fila posterior inválida, modelo incompatible, versión vacía y probabilidades inválidas. También compruebo que una entrada inválida se rechaza antes de cargar el modelo y que se conserva una salida anterior.

`>>` Ruff valida el estilo y el formato de `src/` y `tests/`. La salida real del CLI está en [wine_predictions.csv](entrega/evidencias/wine_predictions.csv); las comprobaciones se registran en [verificacion.txt](entrega/evidencias/verificacion.txt).

| Muestra | Categoría | Confianza |
| --- | --- | --- |
| red-001 | `needs_review` | 0.863753 |
| red-002 | `needs_review` | 0.815449 |
| red-003 | `acceptable` | 0.616764 |
| red-004 | `needs_review` | 0.760370 |
| red-005 | `needs_review` | 0.733027 |

`=>` He obtenido cinco predicciones con el identificador original y las versiones `wine-quality-rf-v1` y `wine-red-features-v1`. La tabla redondea las confianzas a seis decimales; el CSV conserva los valores completos.

## Decisiones y límites

`~>` He preferido validar y guardar el lote pequeño en memoria antes de inferir. Esto simplifica el comportamiento del taller y evita salidas parciales ante errores de datos o predicción. Para CSV grandes revisaría el consumo de memoria y una estrategia de escritura temporal; la escritura actual no garantiza atomicidad ante fallos de disco.

`>>` El CSV docente no tiene etiquetas, por lo que no calculo accuracy ni F1 ni interpreto estas cinco predicciones como evaluación del modelo. La confianza tampoco demuestra calibración. Los resultados de las diapositivas son ilustrativos y he documentado los que realmente produce mi artefacto.

`>>` Joblib emite seis `DeprecationWarning` en dos pruebas de artefactos temporales por la interacción con NumPy 2.5. Los 30 tests terminan correctamente. No he silenciado los avisos ni cambiado el lock del profesor para ocultarlos.

`>` Solo cargo artefactos de confianza: validar el diccionario después de `joblib.load` comprueba compatibilidad, pero no protege frente a un pickle malicioso. Esta limitación está descrita en la [documentación oficial de Joblib](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html).

`=>` La práctica queda implementada y comprobada localmente. La ejecución tras clonar depende del modelo distribuido aparte por el profesor; las pruebas automatizadas no dependen de él.

## Relación entre requisitos y entrega

| Requisito | Archivo | Comprobación |
| --- | --- | --- |
| Distinción entre entrenamiento, artefacto y datos nuevos | `LIENZO.md` y notebook | Respuestas de la clase 1. |
| Once campos, unidades, rangos y ejemplos | `LIENZO.md` | Tabla y entrada inválida. |
| Validación y comparación con dataclass | Notebook y `contracts.py` | Celdas ejecutadas y tests de contrato. |
| Orden estable y versión del preprocesado | `preprocess.py` | Test original del vector. |
| Artefacto compatible y predicción trazable | `inference.py` | Tests de carga y salida. |
| CLI CSV sin resultados parciales ante errores | `predict_file.py` | Tests del comando y salida anterior conservada. |
| Comando reproducible y cinco muestras | README del starter y CSV de evidencia | Ejecución real con el artefacto local. |

## Referencias

He utilizado los [materiales de la semana 3 del profesor](https://github.com/ssillerom/icai-muiaap-operacion-de-modelos/tree/main/semana3). La fuente de datos es [Red Wine Quality en Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009), basada en [Wine Quality de UCI](https://archive.ics.uci.edu/dataset/186/wine%2Bquality), de Cortez et al. (2009). He comprobado el comportamiento de validación en la [documentación oficial de Pydantic](https://docs.pydantic.dev/latest/concepts/models/) y la interpretación de las probabilidades en [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
