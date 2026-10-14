"""Preparo el vector con el mismo orden que espera el modelo."""

from dataclasses import dataclass

from model_inference.contracts import WineQualityRequest

# Fijo el orden una sola vez para evitar diferencias entre scripts.
FEATURE_NAMES = (
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
    "alcohol",
)

PREPROCESSING_VERSION = "wine-red-features-v1"


@dataclass(frozen=True)
class WineFeatures:
    """Conservo el vector y la versión de la transformación."""

    values: tuple[float, ...]
    preprocessing_version: str = PREPROCESSING_VERSION

    def as_vector(self) -> list[float]:
        """Devuelvo una lista nueva con exactamente once valores."""

        if len(self.values) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, received {len(self.values)}."
            )
        return list(self.values)


def preprocess_wine_request(request: WineQualityRequest) -> WineFeatures:
    """Extraigo las mediciones validadas sin incluir sample_id."""

    # No ajusto escaladores ni reentreno con los datos de inferencia.
    request_values = request.model_dump()
    ordered_values = tuple(float(request_values[name]) for name in FEATURE_NAMES)
    return WineFeatures(values=ordered_values)
