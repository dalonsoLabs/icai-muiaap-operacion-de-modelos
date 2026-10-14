# Lienzo de contrato y preprocesado — Práctica 3

Daniel Alonso · Operación de Modelos · 14/10/2026

## Qué recibo y qué hago

Distingo los datos de entrenamiento, con mediciones y etiqueta `quality`, del artefacto `.joblib`, que contiene el estimador ya ajustado, `feature_names` y `model_version`. Recibo cinco muestras nuevas sin `quality` para inferir. No modifico el estimador ni utilizo estas filas para entrenar o ajustar un escalador.

`sample_id` identifica la muestra, pero no es una característica. Las once mediciones forman la entrada del clasificador. La procedencia de las filas está documentada en los [assets de la semana](../../../../../../assets/03-wine-quality/README.md): Red Wine Quality (UCI), distribuido mediante Kaggle.

## Contrato de entrada

| Campo | Tipo validado | Unidad asumida | Límite aceptado |
| --- | --- | --- | --- |
| `fixed_acidity` | `float` | g/L de ácido tartárico | 0–20 |
| `volatile_acidity` | `float` | g/L de ácido acético | 0–2 |
| `citric_acid` | `float` | g/L | 0–2 |
| `residual_sugar` | `float` | g/L | 0–20 |
| `chlorides` | `float` | g/L de NaCl | 0–1 |
| `free_sulfur_dioxide` | `float` | mg/L | 0–100 |
| `total_sulfur_dioxide` | `float` | mg/L | 0–300 |
| `density` | `float` | g/cm³ | 0.98–1.01 |
| `ph` | `float` | adimensional | 2.5–4.5 |
| `sulphates` | `float` | g/L de K₂SO₄ | 0–3 |
| `alcohol` | `float` | % vol. | 5–20 |

Uso los límites orientativos del notebook del profesor como límites inclusivos del contrato, no como una descripción exhaustiva de todos los vinos. Asumo las unidades habituales de Wine Quality indicadas en la tabla; el CSV solo contiene números y no permite comprobar sus unidades automáticamente. No convierto unidades durante la inferencia. En una ampliación pediría que se incluyeran explícitamente en los metadatos de entrada.

Todos los campos son obligatorios. Acepto cadenas numéricas procedentes del CSV y Pydantic las convierte a `float`; rechazo valores no numéricos, `NaN`, infinitos, campos ausentes y campos extra. `sample_id` es un texto no vacío validado por el CLI y lo excluyo del objeto del modelo.

## Ejemplos

Fila válida, tomada del CSV docente:

```json
{
  "sample_id": "red-001",
  "fixed_acidity": "7.4",
  "volatile_acidity": "0.7",
  "citric_acid": "0.0",
  "residual_sugar": "1.9",
  "chlorides": "0.076",
  "free_sulfur_dioxide": "11.0",
  "total_sulfur_dioxide": "34.0",
  "density": "0.9978",
  "ph": "3.51",
  "sulphates": "0.56",
  "alcohol": "9.4"
}
```

Fila inválida: conservo la muestra y añado una columna prohibida.

```json
{
  "sample_id": "red-001",
  "fixed_acidity": "7.4",
  "volatile_acidity": "0.7",
  "citric_acid": "0.0",
  "residual_sugar": "1.9",
  "chlorides": "0.076",
  "free_sulfur_dioxide": "11.0",
  "total_sulfur_dioxide": "34.0",
  "density": "0.9978",
  "ph": "3.51",
  "sulphates": "0.56",
  "alcohol": "9.4",
  "unexpected_field": "not allowed"
}
```

Espero un error que mencione `unexpected_field` antes de cargar o invocar el modelo. También considero inválido `ph=99`, porque queda fuera del rango 2.5–4.5.

## Orden exacto del vector

```text
fixed_acidity → volatile_acidity → citric_acid → residual_sugar →
chlorides → free_sulfur_dioxide → total_sulfur_dioxide → density →
ph → sulphates → alcohol
```

Para `red-001`, el vector es:

```text
[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
```

Si intercambio `density` y `alcohol`, sigo teniendo once números. El estimador no tiene por qué detectar que corresponden a otras variables. Por eso defino un único `FEATURE_NAMES` y lo comparo con el orden del artefacto antes de llamar a `predict`.

## Contrato de salida

| Campo | Tipo y restricción | Uso |
| --- | --- | --- |
| `sample_id` | Texto no vacío | Relaciono la salida con la muestra original. |
| `quality_band` | `needs_review`, `acceptable` o `excellent` | Categoría predicha. |
| `confidence` | `float` finito, 0–1 | Probabilidad de la clase predicha en `predict_proba`. |
| `model_version` | Texto no vacío | Identifico el artefacto utilizado. |
| `preprocessing_version` | Texto no vacío | Identifico la transformación utilizada. |

En el módulo, `WineQualityPrediction` valida los cuatro campos del modelo. El CLI añade el `sample_id` previamente validado. En el notebook incluyo los cinco campos juntos para explicar la salida pública.

No interpreto la confianza como una garantía de acierto ni de calibración. Con estas cinco filas sin etiqueta no puedo calcular accuracy, F1 ni otra métrica de calidad predictiva.

## Pydantic y dataclass

Uso Pydantic en las fronteras de entrada y salida porque comprueba los datos en runtime. Uso `dataclass(frozen=True)` para guardar el vector y su versión después de validar. Una dataclass con anotaciones de tipo permite `ph=99` si no programo validación adicional; `frozen=True` limita las modificaciones, no comprueba rangos.

## Tres criterios de aceptación y plan

1. Un CSV con cinco muestras válidas produce cinco filas con el identificador, la categoría, la confianza y ambas versiones.
2. Una columna extra o una fila posterior inválida provoca un error antes de inferir y no crea un CSV parcial ni sobrescribe una salida anterior.
3. Un artefacto con un orden distinto de características se rechaza antes de predecir.

Implemento primero `contracts.py` para validar, después `preprocess.py` para ordenar, luego `inference.py` para cargar y predecir y, por último, `predict_file.py` para coordinar archivos. Mantengo esas responsabilidades separadas para poder reutilizar la inferencia sin depender de un CSV.

He completado y ejecutado el [notebook de la clase 1](01-inferencia-y-contratos-alumno.ipynb). La explicación de la clase 2 y su ejecución real están en [MEMORIA_PRACTICA_3.md](../../../../../../MEMORIA_PRACTICA_3.md).
