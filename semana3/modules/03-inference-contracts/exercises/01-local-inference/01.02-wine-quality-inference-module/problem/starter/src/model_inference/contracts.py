"""Defino los contratos de entrada y salida de la inferencia de vino."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QualityBand = Literal["needs_review", "acceptable", "excellent"]


class WineQualityRequest(BaseModel):
    """Valido las once mediciones que consume el modelo."""

    # Rechazo campos extra y valores NaN o infinitos.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Acepto texto numérico del CSV y lo convierto a float al validar.
    fixed_acidity: float = Field(ge=0, le=20)
    volatile_acidity: float = Field(ge=0, le=2)
    citric_acid: float = Field(ge=0, le=2)
    residual_sugar: float = Field(ge=0, le=20)
    chlorides: float = Field(ge=0, le=1)
    free_sulfur_dioxide: float = Field(ge=0, le=100)
    total_sulfur_dioxide: float = Field(ge=0, le=300)
    density: float = Field(ge=0.98, le=1.01)
    ph: float = Field(ge=2.5, le=4.5)
    sulphates: float = Field(ge=0, le=3)
    alcohol: float = Field(ge=5, le=20)


class WineQualityPrediction(BaseModel):
    """Compruebo la categoría, confianza y versiones de cada predicción."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    quality_band: QualityBand
    confidence: float = Field(ge=0, le=1)
    model_version: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
