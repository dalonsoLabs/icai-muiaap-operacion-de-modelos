# Práctica 4.2 — Bundle serializado de calidad de vino

He completado el starter conservando `contracts.py` y `preprocess.py` del
profesor. Mi implementación está en `artifact.py` y `predict_file.py`; añado
una migración del artefacto local de S3 para ejecutar el caso completo sin
reentrenarlo. El [informe de la práctica](../../../../../../../REPORT.md)
recoge las decisiones y las evidencias.

## Instalar y comprobar

Desde la raíz de este repositorio:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/semana4/modules/04-model-packaging/exercises/01-model-packaging/01.02-serializable-inference-module/problem/starter"
uv sync --frozen
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

Estos comandos comprueban **mi starter**, no la solución docente instalada por
el proyecto de `semana4/`. Las pruebas crean modelos pequeños temporales; no
necesitan el Random Forest, credenciales ni un dataset nuevo.

He reutilizado la resolución de dependencias de S3 en `uv.lock`, adaptando el
nombre del proyecto, y he fijado Python 3.13.13 en `.python-version`. Así cargo
el modelo con scikit-learn 1.9.0, NumPy 2.5.1, SciPy 1.18.0 y Joblib 1.5.3,
las versiones usadas en la entrega anterior. `uv sync --locked` comprueba que
el lock es coherente con este `pyproject.toml`.

## Generar y consumir el bundle real

Coloco el mismo artefacto de confianza de S3 en
`models/wine_quality_classifier.joblib`. Se obtiene del material local de
clase: **no se descarga del repositorio ni se incluye en Git**. Su SHA-256 y
las versiones del entorno constan en
[artefacto_y_entorno.json](../../../../../../../entrega/evidencias/artefacto_y_entorno.json).

Desde el starter, tras definir `REPO_ROOT` como en el bloque anterior:

```bash
uv run python -m model_packaging.migrate_legacy_artifact \
  --input models/wine_quality_classifier.joblib \
  --output models/wine_quality_bundle
uv run python -m model_packaging.predict_file \
  --bundle models/wine_quality_bundle \
  --input "$REPO_ROOT/semana3/assets/03-wine-quality/inference_samples.csv" \
  --output .tmp/wine_predictions.csv
```

La migración requiere una **ruta de bundle nueva**. Si ya existe, uso otra
ruta en ambos comandos para conservar el bundle anterior. El resultado local
contiene exactamente `manifest.json` y `model.joblib`. El binario contiene
`{"estimator": ...}`; los metadatos se leen del JSON. `models/` y `.tmp/`
siguen ignorados. Conservo una copia legible del
[manifiesto](../../../../../../../entrega/manifest_propuesto.json) y el
[CSV real de cinco predicciones](../../../../../../../entrega/evidencias/wine_predictions.csv).

## Interfaz y fallos

- `create_manifest(estimator, model_version)` requiere una versión explícita.
  Los seis campos son obligatorios al cargar JSON y se prohíben campos extra.
- `save_model_bundle(path, estimator, manifest)` valida y publica ambos
  archivos juntos en una ruta nueva. Si se omite `manifest`, crea una versión
  local `wine-quality-local-v1`; para el modelo de S3 paso siempre su versión
  original mediante la migración.
- `load_model_bundle(path)` valida el manifiesto antes de `joblib.load` y
  después comprueba tipo, interfaz y metadatos del objeto, cuando existen.
- `infer_wine_quality(bundle, request)` conserva el preprocesado y valida
  categoría, probabilidades y versiones con `WineQualityPrediction`.
- `predict_file(input_path, output_path, bundle_path)` devuelve el número de
  filas. Valida todo el lote y todas las respuestas antes de publicar el CSV
  mediante un temporal y `os.replace`.
- El comando termina con código **0** si funciona y **2** para errores de
  validación o archivos. El error identifica la fila del CSV cuando procede.
  No publica una salida nueva y conserva la anterior si falla el lote.

La confianza del clasificador se obtiene mediante la posición de la categoría
predicha en `classes_`, cuyo orden puede diferir de `output_labels`. El
protocolo mínimo del taller no exige `classes_`; si el estimador no lo expone,
se utiliza la probabilidad máxima. El Random Forest y los Dummy de las pruebas
sí lo exponen.

Los tests incluyen manifiestos incompatibles, tipos y métodos incorrectos,
probabilidades inválidas, fallos de escritura y el comportamiento del CLI.
Joblib emite `DeprecationWarning` al leer arrays con NumPy 2.5; los avisos se
conservan en la evidencia y no se silencian.

Solo cargo binarios de confianza. Un JSON correcto comprueba compatibilidad;
no autentica ni hace seguro el pickle. Véanse las limitaciones de persistencia
en la [documentación oficial de scikit-learn](https://scikit-learn.org/stable/model_persistence.html).
