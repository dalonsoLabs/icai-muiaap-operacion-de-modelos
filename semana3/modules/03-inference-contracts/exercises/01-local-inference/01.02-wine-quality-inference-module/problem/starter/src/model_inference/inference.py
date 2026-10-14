"""Cargo el artefacto entregado y predigo sin volver a entrenar el modelo."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import joblib

from model_inference.contracts import WineQualityPrediction
from model_inference.preprocess import FEATURE_NAMES, WineFeatures

# Resuelvo la ruta desde el módulo para no depender del directorio del terminal.
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models/wine_quality_classifier.joblib"
)


@dataclass(frozen=True)
class WineQualityModel:
    """Mantengo juntos el estimador y la versión validada del artefacto."""

    estimator: Any
    model_version: str


def load_wine_quality_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> WineQualityModel:
    """Compruebo los metadatos antes de permitir una predicción."""

    # Joblib solo se debe usar con artefactos de una procedencia confiable.
    artifact = joblib.load(model_path)
    if not isinstance(artifact, Mapping):
        raise ValueError("El artefacto debe ser un diccionario de modelo y metadatos.")

    feature_names = artifact.get("feature_names")
    if (
        not isinstance(feature_names, (list, tuple))
        or tuple(feature_names) != FEATURE_NAMES
    ):
        raise ValueError("feature_names no coincide con el orden del contrato.")

    estimator = artifact.get("estimator")
    if not all(
        callable(getattr(estimator, method, None))
        for method in ("predict", "predict_proba")
    ):
        raise ValueError("El artefacto no contiene un clasificador compatible.")
    if getattr(estimator, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError("El clasificador debe aceptar exactamente 11 características.")

    classes = getattr(estimator, "classes_", None)
    if classes is None or not len(classes):
        raise ValueError("El clasificador no tiene clases aprendidas.")
    if any(
        label not in {"needs_review", "acceptable", "excellent"} for label in classes
    ):
        raise ValueError("quality_band contiene una categoría no permitida.")

    model_version = artifact.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version debe ser una cadena no vacía.")
    return WineQualityModel(estimator=estimator, model_version=model_version)


def infer_wine_quality(
    model: WineQualityModel,
    features: WineFeatures,
) -> WineQualityPrediction:
    """Devuelvo la categoría, su probabilidad y las versiones utilizadas."""

    # El estimador espera una matriz: una muestra con once columnas.
    batch = [features.as_vector()]
    labels = model.estimator.predict(batch)
    probabilities = model.estimator.predict_proba(batch)
    if len(labels) != 1 or len(probabilities) != 1:
        raise ValueError("El clasificador debe devolver una única predicción.")

    classes = list(model.estimator.classes_)
    scores = [float(score) for score in probabilities[0]]
    if len(scores) != len(classes) or any(
        not isfinite(score) or not 0 <= score <= 1 for score in scores
    ):
        raise ValueError("El clasificador ha devuelto probabilidades inválidas.")
    if not isclose(sum(scores), 1.0, abs_tol=1e-6):
        raise ValueError("Las probabilidades deben sumar 1.")

    quality_band = str(labels[0])
    if quality_band not in classes:
        raise ValueError("quality_band no aparece en las clases del clasificador.")
    # Uso la probabilidad de la clase predicha según el orden de classes_.
    confidence = scores[classes.index(quality_band)]
    return WineQualityPrediction(
        quality_band=quality_band,
        confidence=confidence,
        model_version=model.model_version,
        preprocessing_version=features.preprocessing_version,
    )
