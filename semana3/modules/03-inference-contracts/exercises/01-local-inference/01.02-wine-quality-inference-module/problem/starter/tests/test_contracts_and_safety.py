"""Compruebo casos límite y que los errores no dejen una salida parcial."""

from pathlib import Path

import joblib
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier
from test_predict_file import WINE_SAMPLE, write_input_csv, write_instructor_model

from model_inference.contracts import WineQualityPrediction, WineQualityRequest
from model_inference.inference import (
    WineQualityModel,
    infer_wine_quality,
    load_wine_quality_model,
)
from model_inference.predict_file import run_predict_file
from model_inference.preprocess import FEATURE_NAMES, preprocess_wine_request


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "no-numérico", "99"])
def test_request_rejects_invalid_ph(value: str) -> None:
    # Verifico tanto el rango físico como los valores que no son finitos.
    with pytest.raises(ValidationError):
        WineQualityRequest.model_validate({**WINE_SAMPLE, "ph": value})


def test_request_rejects_extra_field() -> None:
    with pytest.raises(ValidationError, match="unexpected_field"):
        WineQualityRequest.model_validate({**WINE_SAMPLE, "unexpected_field": "x"})


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_prediction_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        WineQualityPrediction(
            quality_band="acceptable",
            confidence=confidence,
            model_version="v1",
            preprocessing_version="v1",
        )


def test_invalid_later_row_does_not_load_model_or_overwrite_output(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    write_input_csv(input_path, second_invalid_row=True)
    output_path.write_text("resultado anterior\n", encoding="utf-8")

    # El modelo no existe: espero el error de fila antes de intentar cargarlo.
    with pytest.raises(ValueError, match="red-002"):
        run_predict_file(input_path, output_path, tmp_path / "missing.joblib")
    assert output_path.read_text(encoding="utf-8") == "resultado anterior\n"


def test_loader_rejects_wrong_estimator_width(tmp_path: Path) -> None:
    estimator = DummyClassifier(strategy="constant", constant="acceptable")
    estimator.fit([[0.0] * 10], ["acceptable"])
    model_path = tmp_path / "wrong_width.joblib"
    joblib.dump(
        {
            "estimator": estimator,
            "feature_names": list(FEATURE_NAMES),
            "model_version": "v1",
        },
        model_path,
    )
    with pytest.raises(ValueError, match="11 características"):
        load_wine_quality_model(model_path)


def test_loader_rejects_empty_version(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    write_instructor_model(model_path)
    artifact = joblib.load(model_path)
    artifact["model_version"] = " "
    joblib.dump(artifact, model_path)
    with pytest.raises(ValueError, match="model_version"):
        load_wine_quality_model(model_path)


@pytest.mark.parametrize("scores", [[float("nan")], [1.5], [0.2]])
def test_inference_rejects_invalid_probabilities(scores: list[float]) -> None:
    class InvalidEstimator:
        classes_ = ["acceptable"]

        def predict(self, batch: list[list[float]]) -> list[str]:
            return ["acceptable"]

        def predict_proba(self, batch: list[list[float]]) -> list[list[float]]:
            return [scores]

    features = preprocess_wine_request(WineQualityRequest.model_validate(WINE_SAMPLE))
    with pytest.raises(ValueError, match="probabilidades"):
        infer_wine_quality(WineQualityModel(InvalidEstimator(), "v1"), features)


@pytest.mark.parametrize("header", ["sample_id,ph,ph\n", "", "sample_id,ph\n"])
def test_cli_rejects_invalid_headers_without_output(
    tmp_path: Path, header: str
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError):
        run_predict_file(input_path, output_path, tmp_path / "missing.joblib")
    assert not output_path.exists()
