# Lienzo de la práctica 4.1 — Contrato del artefacto

Daniel Alonso · Operación de Modelos · 22/10/2026

`->` Quiero poder inspeccionar la compatibilidad antes de deserializar el modelo de S3. En este lienzo diseño el bundle; las funciones de guardado, carga e inferencia se implementan en el starter de la clase 2.

`~>` Conservo las once features, las categorías y el preprocesado de S3. No entreno un modelo nuevo ni añado una API.

[Informe de la semana](../../../../../../REPORT.md) · [Lienzo y matriz](LIENZO.md)


## Inspección del payload legado

`>` Inspecciono exclusivamente el artefacto local de confianza que utilicé en S3. Compruebo las claves y sus tipos; todavía no hago predicciones.

`~>` El notebook también puede repetirse sin ese binario: en ese caso utiliza un `DummyClassifier` pequeño y declara una versión de demostración. El resultado guardado en esta entrega procede del Random Forest real. El fallback demuestra el contrato y no pretende reproducir sus predicciones.


## Qué guardo y por qué

| Dato | Destino | Justificación y riesgo si falta |
| --- | --- | --- |
| Estimador ajustado | `model.joblib`, clave `estimator` | Conservo el objeto y sus parámetros aprendidos; no vuelvo a entrenarlo al predecir. |
| `schema_version` | `manifest.json` | Puedo rechazar un formato que mi cargador desconoce antes de abrir el binario. |
| `model_version` | `manifest.json` | Identifico qué estimador produjo la respuesta. No deduzco la versión del nombre del archivo. |
| `preprocessing_version` | `manifest.json`; transformación en código | Bloqueo una representación diferente aunque tenga once números. |
| `feature_names` | `manifest.json`; orden canónico en código | Comparo nombres y posiciones con S3; mezclar densidad y alcohol altera el significado del vector. |
| `output_labels` | `manifest.json`; contrato de salida en código | Limito las categorías que el consumidor puede recibir. |
| `estimator_type` | `manifest.json`; clase real en el binario | Declaro el tipo esperado y lo contrasto tras cargarlo. |
| Rangos, ausentes, campos extra e interfaz `predict`/`predict_proba` | Código de contratos y carga | El JSON describe el contrato, pero el código debe hacerlo cumplir. |
| Preprocesado y política de filas | Código | Construyo el mismo vector y publico el resultado únicamente si todo el lote es válido. |

`~>` No mezclo `model_version` (identidad del estimador), `preprocessing_version` (representación) y `schema_version` (forma del bundle).

## Manifiesto propuesto

`>` Declaro seis campos obligatorios, sin valores por defecto al leer un bundle y sin campos extra. Solo el helper de creación compone esos campos desde el contrato actual y una versión de modelo explícita.


## Invariantes que bloquearán la ejecución

1. El JSON tiene las seis claves obligatorias y ninguna desconocida; `schema_version` es `wine-quality-bundle-v1`.
2. `preprocessing_version` coincide con `wine-red-features-v1`; `feature_names` coincide exactamente en nombres, longitud y orden con S3.
3. `output_labels` conserva `needs_review`, `acceptable` y `excellent`; la salida debe pertenecer a ese conjunto y tener confianza finita en [0, 1].
4. `model_version` y `estimator_type` son texto no vacío. Después de deserializar compruebo el tipo real y que `predict` y `predict_proba` son invocables.
5. Si existen `n_features_in_` y `classes_`, son coherentes con once features y las categorías permitidas. No uso el orden de `output_labels` para indexar probabilidades: consulto `classes_`.
6. Todos los inputs y todas las respuestas son válidos antes de publicar el CSV.

`>>` La siguiente matriz describe el fallo esperado. Los tests de la clase 2 comprueban estas fronteras con artefactos temporales; esta celda no implementa el cargador.

| Cambio | Comprobación | Error esperado | Etapa de rechazo | ¿CSV publicado? |
| --- | --- | --- | --- | --- |
| Intercambiar `density` y `alcohol` en `feature_names` del manifiesto | Comparación de la secuencia completa | Orden de features incompatible | Validación del JSON, antes de Joblib | No |
| Declarar `wine-red-features-v2` | Igualdad con el preprocesado soportado | Versión no soportada | Validación del JSON, antes de Joblib | No |
| Declarar `low` en `output_labels` | Categorías del contrato | Etiqueta desconocida | Validación del JSON, antes de Joblib | No |
| Declarar RF y guardar un Dummy | Tipo real frente a `estimator_type` | Tipo de estimador incompatible | Tras deserializar, antes de inferir | No |
| Estimador sin `predict_proba` | Interfaz mínima invocable | Método requerido ausente | Tras deserializar, antes de inferir | No |
| Modelo devuelve `unknown` | Contrato de respuesta | Categoría no permitida | Inferencia, antes de escribir | No |
| Segunda muestra con `ph=99` | Rango [2.5, 4.5] | Campo fuera de rango | Validación de filas, antes de cargar el bundle | No |

`~>` El primer caso modifica el orden semántico del vector declarado en el manifiesto. Reordenar solo las columnas del CSV conservando sus nombres no cambia el vector: el preprocesado accede por nombre y fija las posiciones.


`>>` Manifiesto concreto utilizado con el artefacto de S3:

```json
{
  "schema_version": "wine-quality-bundle-v1",
  "model_version": "wine-quality-rf-v1",
  "preprocessing_version": "wine-red-features-v1",
  "feature_names": [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol"
  ],
  "output_labels": [
    "needs_review",
    "acceptable",
    "excellent"
  ],
  "estimator_type": "RandomForestClassifier"
}
```


## Dos entradas inválidas comprobadas

`>` Pruebo una medición fuera de rango y un dato obligatorio ausente con el contrato preparado por el profesor. No abro ningún binario en esta comprobación.


## Política de errores y paso a la clase 2

`~>` Si la segunda fila falla, rechazo el lote completo, comunico la fila y el campo y no publico un CSV nuevo. Si había una salida anterior, la conservo. Primero valido todas las filas, luego cargo el bundle, infiero y valido todas las respuestas; por último escribo un temporal junto al destino y hago un reemplazo atómico.

`>` Orden previsto: manifiesto → guardado → carga con comprobaciones → salida validada → CLI → caso de fallo adicional. Reutilizo `contracts.py` y `preprocess.py` sin cambiarlos. Los tests constituyen la especificación ejecutable.

`>>` Conservo el [manifiesto propuesto](../../../../../../entrega/manifest_propuesto.json), la [matriz en CSV](../../../../../../entrega/matriz_compatibilidad.csv) y el [lienzo](LIENZO.md). La evidencia ejecutable de las decisiones está en los tests y en el informe de clase 2.

`=>` El diseño permite comprobar la compatibilidad antes de deserializar, pero no autentica el binario ni garantiza su seguridad. Solo cargo Joblib de procedencia confiable. No presento como realizada una revisión por otra pareja.


## Entradas inválidas y resultado observado

`>>` En el notebook, `ph=99` produce `less_than_equal` en `ph` y la ausencia de `alcohol` produce `missing`. Ambos fallan al construir `WineQualityRequest`, antes de cargar el bundle y sin escribir CSV.
