"""Bundle del modelo: metadatos legibles, binario y fronteras de validación."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import joblib
from pydantic import BaseModel, ConfigDict, Field, field_validator

from model_packaging.contracts import (
    QualityBand,
    WineQualityPrediction,
    WineQualityRequest,
)
from model_packaging.preprocess import (
    FEATURE_NAMES,
    PREPROCESSING_VERSION,
    preprocess_wine_request,
)

ARTIFACT_SCHEMA_VERSION = "wine-quality-bundle-v1"
DEFAULT_BUNDLE_PATH = Path("models/wine_quality_bundle")
MANIFEST_FILENAME = "manifest.json"
MODEL_FILENAME = "model.joblib"
OUTPUT_LABELS: tuple[QualityBand, ...] = (
    "needs_review",
    "acceptable",
    "excellent",
)


class WineQualityEstimator(Protocol):
    """Interfaz mínima que debe cumplir el estimador cargado."""

    def predict(self, features: list[list[float]]) -> Sequence[str]:
        """Devuelve una etiqueta por fila."""

    def predict_proba(self, features: list[list[float]]) -> Sequence[Sequence[float]]:
        """Devuelve probabilidades por fila."""


class ArtifactManifest(BaseModel):
    """Declaro las seis claves obligatorias y rechazo contratos incompatibles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["wine-quality-bundle-v1"]
    model_version: str = Field(min_length=1, strict=True)
    preprocessing_version: Literal["wine-red-features-v1"]
    feature_names: tuple[str, ...]
    output_labels: tuple[QualityBand, ...]
    estimator_type: str = Field(min_length=1, strict=True)

    @field_validator("model_version", "estimator_type")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError(
                "El identificador no puede estar vacío ni tener bordes blancos"
            )
        return value

    @field_validator("feature_names")
    @classmethod
    def require_feature_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != FEATURE_NAMES:
            raise ValueError("feature_names debe conservar los once nombres y su orden")
        return value

    @field_validator("output_labels")
    @classmethod
    def require_output_labels(
        cls, value: tuple[QualityBand, ...]
    ) -> tuple[QualityBand, ...]:
        if value != OUTPUT_LABELS:
            raise ValueError("output_labels debe conservar el contrato de categorías")
        return value


@dataclass(frozen=True)
class LoadedModelBundle:
    """Bundle cargado; no modificar esta interfaz pública."""

    estimator: WineQualityEstimator
    manifest: ArtifactManifest


def _validate_estimator(
    estimator: WineQualityEstimator, manifest: ArtifactManifest
) -> None:
    """Compruebo el objeto real, además de las promesas del manifiesto."""
    for method in ("predict", "predict_proba"):
        if not callable(getattr(estimator, method, None)):
            raise ValueError(f"El estimador no implementa {method}()")
    if type(estimator).__name__ != manifest.estimator_type:
        raise ValueError("estimator_type no coincide con el objeto serializado")
    feature_count = getattr(estimator, "n_features_in_", None)
    if feature_count is not None and feature_count != len(FEATURE_NAMES):
        raise ValueError("El estimador no admite las once características")
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        labels = list(classes)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("classes_ está vacío o contiene duplicados")
        if any(label not in manifest.output_labels for label in labels):
            raise ValueError("classes_ contiene categorías fuera del contrato")


def create_manifest(
    estimator: WineQualityEstimator, model_version: str
) -> ArtifactManifest:
    """Construyo metadatos del contrato actual sin entrenar ni inferir."""
    manifest = ArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_version=model_version,
        preprocessing_version=PREPROCESSING_VERSION,
        feature_names=FEATURE_NAMES,
        output_labels=OUTPUT_LABELS,
        estimator_type=type(estimator).__name__,
    )
    _validate_estimator(estimator, manifest)
    return manifest


def save_model_bundle(
    bundle_path: Path,
    estimator: WineQualityEstimator,
    manifest: ArtifactManifest | None = None,
) -> ArtifactManifest:
    """Publico los dos archivos juntos en una ruta nueva, sin pisar otro bundle."""
    manifest = manifest or create_manifest(estimator, "wine-quality-local-v1")
    # Revalido incluso si me entregan una instancia construida fuera de Pydantic.
    manifest = ArtifactManifest.model_validate(manifest.model_dump())
    _validate_estimator(estimator, manifest)
    if bundle_path.exists():
        raise FileExistsError(f"El bundle ya existe; usa una ruta nueva: {bundle_path}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bundle-", dir=bundle_path.parent))
    try:
        (temporary / MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        joblib.dump({"estimator": estimator}, temporary / MODEL_FILENAME)
        os.rename(temporary, bundle_path)
    finally:
        # Si falla la serialización elimino únicamente mi directorio temporal.
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def load_model_bundle(bundle_path: Path) -> LoadedModelBundle:
    """Valido JSON antes de deserializar un binario de procedencia confiable."""
    manifest = ArtifactManifest.model_validate_json(
        (bundle_path / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    # Esta validación detecta incompatibilidades; no convierte Joblib en un sandbox.
    payload = joblib.load(bundle_path / MODEL_FILENAME)
    if not isinstance(payload, dict) or set(payload) != {"estimator"}:
        raise ValueError("model.joblib debe contener únicamente la clave estimator")
    estimator = payload["estimator"]
    _validate_estimator(estimator, manifest)
    return LoadedModelBundle(estimator=estimator, manifest=manifest)


def infer_wine_quality(
    bundle: LoadedModelBundle,
    request: WineQualityRequest,
) -> WineQualityPrediction:
    """Preproceso con el contrato existente y valido también la salida del modelo."""
    _validate_estimator(bundle.estimator, bundle.manifest)
    vector = [preprocess_wine_request(request).as_vector()]
    predictions = list(bundle.estimator.predict(vector))
    if len(predictions) != 1 or predictions[0] not in bundle.manifest.output_labels:
        raise ValueError("predict() debe devolver una única categoría permitida")
    probabilities = list(bundle.estimator.predict_proba(vector))
    if len(probabilities) != 1:
        raise ValueError("predict_proba() debe devolver una única fila")
    scores = [float(score) for score in probabilities[0]]
    if not scores or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1")
    if not math.isclose(sum(scores), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Las probabilidades deben sumar 1")
    classes = getattr(bundle.estimator, "classes_", None)
    if classes is not None:
        labels = list(classes)
        if len(labels) != len(scores) or predictions[0] not in labels:
            raise ValueError("Las probabilidades y classes_ no son coherentes")
        confidence = scores[labels.index(predictions[0])]
    else:
        # El protocolo mínimo no exige classes_: en ese caso uso la mayor confianza.
        confidence = max(scores)
    return WineQualityPrediction(
        quality_band=predictions[0],
        confidence=confidence,
        model_version=bundle.manifest.model_version,
        preprocessing_version=bundle.manifest.preprocessing_version,
    )
