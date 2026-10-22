# Informe de la práctica 4 — Serialización y validación

Daniel Alonso · Operación de Modelos · 22/10/2026

## Leyenda utilizada

- `->` Objetivo o problema que quiero resolver.
- `~>` Decisión o cambio realizado y su justificación.
- `>` Acción técnica ejecutada.
- `>>` Evidencia verificable: archivo, salida o comprobación.
- `=>` Resultado obtenido y estado de verificación.

## 22/10/2026 — Objetivo y preparación

`->` Mi objetivo ha sido convertir el artefacto de la semana 3 en un bundle inspeccionable y ejecutable, separando los metadatos del estimador y validando las fronteras de entrada, carga y salida.

`>` He partido de mi entrega de la semana 3, commit `ef138dd5d84cfdb2ca35ba2a650881cb9f8aad6c`, y he creado la rama `tareas-semana4` en [mi fork](https://github.com/dalonsoLabs/icai-muiaap-operacion-de-modelos/tree/tareas-semana4/semana4). He comprobado los materiales de esta semana frente al commit `9ae8157e93b28b464b24c90b9bf2242647f84584` del profesor: los archivos de S4 y sus dos PDF no presentan diferencias respecto a mi base.

`~>` Mantengo el caso de calidad de vino y el stack del starter: `uv`, Pydantic, Joblib, scikit-learn, pytest y Ruff. No entreno un modelo nuevo, no duplico el CSV de muestras y no añado una interfaz web ni una API en esta semana.

`>>` Antes de implementar, he ejecutado el starter: seis tests fallaban por los TODO y uno ya pasaba. Conservo esa salida en [estado-inicial-starter.txt](entrega/evidencias/estado-inicial-starter.txt). Después he reutilizado el lock de S3, adaptando únicamente la identidad del proyecto, para mantener el entorno del artefacto serializado.

## Clase 1 — Diseño del contrato del artefacto

`>` He inspeccionado el payload legado: contiene `estimator` (`RandomForestClassifier`), `feature_names` (`list`) y `model_version` (`str`). Le faltan las declaraciones explícitas de esquema, preprocesado, categorías y tipo que incorporo al manifiesto.

`~>` Guardo el objeto aprendido en `model.joblib`, bajo la clave `estimator`. En `manifest.json` guardo las seis claves obligatorias: `schema_version`, `model_version`, `preprocessing_version`, `feature_names`, `output_labels` y `estimator_type`. Los rangos, el preprocesado y las reglas de validación permanecen en código.

`>>` El [lienzo](modules/04-model-packaging/exercises/01-model-packaging/01.01-artifact-contract/problem/LIENZO.md) incluye la tabla de almacenamiento con justificación, seis invariantes, la matriz de fallos y la política de filas. El [notebook ejecutado](modules/04-model-packaging/exercises/01-model-packaging/01.01-artifact-contract/problem/01-artefacto-y-manifiesto-alumno.ipynb) tiene cuatro celdas de código ejecutadas y ninguna salida de error; también entrego su [exportación HTML](entrega/01-artefacto-y-manifiesto.html).

`>` He comprobado dos entradas inválidas: `ph=99` y ausencia de `alcohol`. Ambas fallan en `WineQualityRequest`, antes de cargar el bundle. El notebook muestra los tipos de error `less_than_equal` y `missing`.

`>>` Conservo por separado el [manifiesto propuesto](entrega/manifest_propuesto.json) y la [matriz de compatibilidad](entrega/matriz_compatibilidad.csv). El manifiesto identifica el modelo real como `wine-quality-rf-v1`, el preprocesado como `wine-red-features-v1` y el esquema como `wine-quality-bundle-v1`.

`~>` El notebook puede ejecutarse sin el binario de clase mediante un Dummy pequeño, identificado expresamente como demostración. La ejecución guardada aquí inspecciona el Random Forest real; el fallback no pretende reproducir sus cinco predicciones. El notebook diseña y comprueba el contrato, mientras que las funciones de empaquetado están en la clase 2.

`=>` He definido qué debe viajar con el modelo y qué incompatibilidades deben impedir la publicación del CSV.

## Clase 2 — Implementación del bundle y del CLI

`>` He completado el [starter](modules/04-model-packaging/exercises/01-model-packaging/01.02-serializable-inference-module/problem/starter/), conservando sin cambios los contratos y el preprocesado proporcionados.

| Archivo | Trabajo realizado |
| --- | --- |
| `artifact.py` | Manifiesto, creación, guardado, carga, comprobación del estimador e inferencia validada. |
| `predict_file.py` | Validación de CSV completo, orquestación de inferencia y publicación atómica. |
| `migrate_legacy_artifact.py` | Conversión del payload local de S3 al bundle, sin reentrenar. |
| `tests/test_delivery_validation.py` | Casos adicionales de incompatibilidad, salida inválida, migración y fallos de escritura. |

`>` `ArtifactManifest` exige las seis claves y rechaza extras, identificadores vacíos, versiones de esquema o preprocesado incompatibles, cambios en el orden de las once features y categorías distintas de las permitidas.

`~>` No fijo `model_version` a un único texto: identifica el estimador utilizado y debe ser explícito y no vacío. La compatibilidad se decide con el esquema, el preprocesado, las features, las categorías y la interfaz. Cambiar una versión de modelo no equivale por sí solo a romper esa interfaz.

`>` El guardado valida el estimador, escribe ambos archivos en un directorio temporal y publica el directorio completo en una ruta nueva. Rechazo un destino existente para conservar el bundle anterior. Si se omite el manifiesto en la función de guardado, el helper crea `wine-quality-local-v1`; mi migración pasa siempre la versión original de S3.

`>` La carga valida el JSON **antes de `joblib.load`**. Después comprueba el payload, la clase real y los métodos invocables `predict` y `predict_proba`. Cuando el estimador declara `n_features_in_` o `classes_`, compruebo su coherencia con el contrato.

`>` La inferencia reutiliza `preprocess_wine_request` y construye el vector en el orden de S3. Compruebo una categoría por muestra y probabilidades finitas en [0, 1] que sumen 1. La respuesta pasa por `WineQualityPrediction` e incluye las versiones del manifiesto.

`~>` Obtengo la confianza usando la posición de la categoría predicha en `classes_`. El Random Forest guarda sus clases como `acceptable`, `excellent`, `needs_review`; ese orden difiere del de las categorías del contrato. El protocolo mínimo del taller no exige `classes_`: para estimadores que no lo exponen utilizo la probabilidad máxima, como se detalla en el README del starter.

`>` El CLI valida cabecera, identificadores y todas las filas antes de cargar el bundle. Después obtiene y valida todas las predicciones, escribe un temporal junto al destino y publica el CSV con `os.replace`. Si falla una fila o una respuesta, no publica salida nueva; si falla la escritura, conserva el CSV anterior y limpia su temporal.

`=>` El comando termina con código 0 en éxito y 2 ante errores de validación o archivos. La función Python devuelve el número de filas procesadas, sin convertir ese número en un código de error del proceso.

## Ejecución reproducible

`>` Desde la raíz del repositorio, ejecuto:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/semana4/modules/04-model-packaging/exercises/01-model-packaging/01.02-serializable-inference-module/problem/starter"
uv sync --frozen
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

`>>` He utilizado Python 3.13.13, Pydantic 2.13.4, scikit-learn 1.9.0, Joblib 1.5.3, NumPy 2.5.1, SciPy 1.18.0, pytest 8.4.2 y Ruff 0.16.1. `uv.lock` fija las dependencias y `.python-version` el intérprete utilizado. He comprobado además `uv sync --locked` para validar la coherencia del lock con el proyecto.

`>` Para el caso completo coloco el artefacto de confianza de S3 en `models/wine_quality_classifier.joblib` y ejecuto, desde el mismo directorio:

```bash
uv run python -m model_packaging.migrate_legacy_artifact \
  --input models/wine_quality_classifier.joblib \
  --output models/wine_quality_bundle
uv run python -m model_packaging.predict_file \
  --bundle models/wine_quality_bundle \
  --input "$REPO_ROOT/semana3/assets/03-wine-quality/inference_samples.csv" \
  --output .tmp/wine_predictions.csv
```

`>>` La migración requiere una ruta nueva; si el bundle ya existe utilizo otro nombre en ambos comandos. El directorio local contiene exactamente `manifest.json` y `model.joblib`. Las versiones, tamaños, SHA-256 y metadatos inspeccionados están en [artefacto_y_entorno.json](entrega/evidencias/artefacto_y_entorno.json).

`~>` Conservo `models/` y `.tmp/` fuera de Git, como exigen el README y el PDF. Tras clonar, los tests funcionan sin el binario; para repetir las predicciones reales hay que obtener el mismo modelo del material de clase y generar el bundle. La copia del manifiesto en `entrega/` permite revisar sus metadatos en GitHub.

## Comprobaciones y resultados

`>>` Pasan los siete tests originales y 33 casos adicionales: **40 tests**. Ruff comprueba estilo y formato de nueve archivos Python. Las pruebas incluyen manifiesto corrupto o incompleto, validación antes de deserializar, tipo y métodos del estimador, confidencias inválidas, segunda fila fuera de rango, salida anterior conservada, fallo de escritura y códigos del CLI.

`>>` La ejecución real y sus códigos de salida están en [verificacion.txt](entrega/evidencias/verificacion.txt). He ejecutado también dos fallos del comando: segunda muestra con `ph=99` y manifiesto con preprocesado v2. Ambos terminan con código 2 y sin CSV nuevo. En el segundo caso no hay binario en el directorio: el error del manifiesto ocurre antes de intentar leerlo.

`>>` He comparado el estado aprendido de todos los árboles, los parámetros y las clases antes y después de migrar: se conservan. El [CSV de cinco predicciones](entrega/evidencias/wine_predictions.csv) es idéntico al de S3, incluidas las confianzas completas y las versiones.

| Muestra | Categoría | Confianza |
| --- | --- | --- |
| red-001 | `needs_review` | 0.863753 |
| red-002 | `needs_review` | 0.815449 |
| red-003 | `acceptable` | 0.616764 |
| red-004 | `needs_review` | 0.760370 |
| red-005 | `needs_review` | 0.733027 |

`=>` El empaquetado añade validación y trazabilidad sin alterar las cinco predicciones del modelo existente. La tabla redondea a seis decimales y el CSV conserva la precisión original.

## Ejecución en Visual Studio Code

`>` He abierto el worktree `icai-semana4-tareas` en una ventana de VS Code y he ejecutado desde su terminal integrada `uv sync --frozen`, los 40 tests, Ruff y el CLI con el bundle real. La terminal muestra la rama `tareas-semana4`.

`>>` La [salida de VS Code](entrega/evidencias/ejecucion-vscode.txt) confirma los 40 tests, el formato y las cinco predicciones. Conservo una [captura real de esa ejecución](entrega/evidencias/vscode-ejecucion.png) como evidencia adicional; el enunciado no la exige. He comparado también el CSV generado allí con el de la entrega.

`=>` El comando funciona en VS Code, en un proceso independiente del notebook.

## Nota de compatibilidad y límites

`~>` Bloqueo el intercambio de `density` y `alcohol` en `feature_names`: ambas son numéricas, pero cambiar posiciones altera el significado del vector sin provocar necesariamente un error de tipos. No lo confundo con reordenar las cabeceras del CSV conservando sus nombres; el preprocesado accede por nombre y fija el vector canónico.

`~>` Bloqueo `wine-red-features-v2` aunque declare once features: una transformación distinta puede modificar unidades o escalas, por lo que el número de columnas no garantiza compatibilidad. También rechazo un tipo real diferente del declarado y etiquetas ajenas al contrato.

`>>` El guardado atómico evita publicar un bundle a medio construir y el reemplazo del CSV evita exponer contenido parcialmente escrito. Esta práctica no implementa coordinación entre escritores concurrentes ni garantías de persistencia ante un corte eléctrico; son límites distintos de los errores de lote y escritura comprobados aquí.

`>>` El CSV docente no tiene etiquetas reales. No calculo accuracy ni F1 ni trato estas cinco muestras como evaluación estadística. La confianza del clasificador tampoco demuestra calibración.

`>>` La suite produce 20 `DeprecationWarning` por la lectura de arrays de Joblib con NumPy 2.5. Los 40 tests pasan; no he silenciado esos avisos. Mantengo el entorno del artefacto para no confundir un cambio de empaquetado con una actualización de dependencias.

`>` Cargo únicamente binarios de procedencia confiable. Un manifiesto válido permite rechazar incompatibilidades, pero no demuestra la identidad del archivo ni impide ejecutar código malicioso durante la deserialización. La necesidad de mantener el entorno y las limitaciones de pickle/Joblib están descritas en la [documentación oficial de scikit-learn](https://scikit-learn.org/stable/model_persistence.html).

## Revisión de indicaciones y evidencias

`>` He revisado los 14 `README.md`/`readme.md` versionados de S4, el README general, los dos guiones y los PDF [Assignment 4.1](../assignments/semana04_clase01_assignment.pdf) y [Assignment 4.2](../assignments/semana04_clase02_assignment.pdf). He consultado las instrucciones de los README de orientación y solución para comprobar el alcance; mi código se desarrolla en el starter y no modifica la referencia docente.

`>>` El guion de clase 1, diapositiva 21, admite enlace o captura del notebook. El PDF 4.1 exige tabla, manifiesto, matriz y política; el 4.2 exige bundle local, tests, CSV de cinco predicciones y nota de compatibilidad. No imponen capturas de pantalla obligatorias. Entrego el notebook enlazado y su exportación con outputs; el texto de consola documenta la ejecución del código.

`~>` Uso los comandos desde el starter: el entorno de `semana4/` instala la solución del profesor y no serviría para demostrar mi implementación. He corregido las rutas del CSV en mis instrucciones mediante `REPO_ROOT`; la ruta abreviada del guion no apunta al archivo desde el starter anidado.

`>>` La dinámica de parejas e intercambio describe la actividad presencial. No presento como hecha una revisión por otra pareja que no he verificado.

| Requisito | Entrega | Verificación |
| --- | --- | --- |
| Inspección y clasificación binario/JSON/código | Lienzo y notebook | Claves, tipos y tabla justificada. |
| Seis claves e invariantes | `manifest_propuesto.json` y lienzo | Coincidencia con S3 y tests de incompatibilidad. |
| Matriz, dos inputs inválidos y política de filas | Notebook, lienzo y `matriz_compatibilidad.csv` | Errores por rango y ausencia; lote completo rechazado. |
| Bundle guardado y cargado fuera del notebook | Starter y migración | Dos archivos locales; CLI en proceso independiente. |
| Validación antes del binario y después de cargar | `artifact.py` | Tests con spy de `joblib.load`, tipo e interfaz. |
| Entradas y respuestas válidas, sin CSV parcial | `predict_file.py` | Tests de fila posterior, salida inválida y fallo de escritura. |
| Cinco predicciones con versiones | CSV de evidencia | Igualdad con S3 sin reentrenamiento. |
| Dependencias y comando repetible | README, lock e intérprete | `uv sync --frozen`, pytest y Ruff. |
| Dos cambios bloqueados y su motivo | Nota de compatibilidad | Orden de features y versión de preprocesado. |

## Siguiente paso

`->` En la semana 5 podré consumir este bundle desde la interfaz prevista por la asignatura, reutilizando la carga y la validación del módulo. La presentación no tendrá que copiar las reglas ni volver a entrenar el modelo.

`=>` La práctica queda implementada y comprobada localmente en `tareas-semana4`. La ejecución con el modelo real depende del binario distribuido aparte; los tests de la entrega no dependen de él.
