# Práctica 3 — Módulo local de inferencia de vino

He completado los cuatro módulos del starter manteniendo el contrato del profesor. La explicación de cómo lo he realizado está en [MEMORIA_PRACTICA_3.md](../../../../../../../MEMORIA_PRACTICA_3.md), junto con el lienzo de la clase 1 y los resultados reales.

## Ejecución

Necesito el modelo distribuido en clase en `models/wine_quality_classifier.joblib`. No se versiona ni se reentrena durante esta práctica. Desde esta carpeta:

```bash
uv sync --frozen
uv run python -m model_inference.predict_file \
  --input assets/inference_samples.csv \
  --output .tmp/wine_predictions.csv
```

Puedo pasar `--model /ruta/al/modelo.joblib` para indicar otra ubicación. La salida tiene `sample_id`, `quality_band`, `confidence`, `model_version` y `preprocessing_version`. Si falla una fila, rechazo el lote antes de inferir y no creo un fichero parcial. El CLI devuelve 0 cuando tiene éxito y 2 ante errores esperados.

## Comprobaciones

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

Los tests crean modelos temporales y no necesitan el binario del profesor. He conservado los tests originales y añadido casos de contrato y errores. El notebook de la clase 1 está en la carpeta del ejercicio 01.01.
