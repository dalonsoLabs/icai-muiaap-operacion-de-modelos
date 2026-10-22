"""Casos añadidos: fronteras del bundle y publicación completa del CSV."""

import csv
import json
import subprocess
import sys
from unittest.mock import patch

import joblib
import pytest
from sklearn.dummy import DummyClassifier
from test_artifact import make_estimator, make_request
from test_predict_file import write_bundle, write_input

from model_packaging.artifact import (
    LoadedModelBundle,
    create_manifest,
    infer_wine_quality,
    load_model_bundle,
    save_model_bundle,
)
from model_packaging.migrate_legacy_artifact import migrate_legacy_artifact
from model_packaging.predict_file import predict_file
from model_packaging.preprocess import FEATURE_NAMES


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "wine-quality-bundle-v2"),
        ("preprocessing_version", "wine-red-features-v2"),
        ("model_version", " "),
        ("model_version", 1),
        ("estimator_type", ""),
        ("feature_names", list(reversed(FEATURE_NAMES))),
        ("output_labels", ["low", "medium", "high"]),
        ("output_labels", ["needs_review", "acceptable", "acceptable"]),
        ("unknown_field", "extra"),
    ],
)
def test_invalid_manifest_prevents_binary_load(tmp_path, field, value):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    with patch("model_packaging.artifact.joblib.load") as binary_load:
        with pytest.raises(ValueError):
            load_model_bundle(bundle)
        binary_load.assert_not_called()


@pytest.mark.parametrize("missing", ["model_version", "preprocessing_version"])
def test_missing_metadata_is_rejected_before_deserialization(tmp_path, missing):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest[missing]
    manifest_path.write_text(json.dumps(manifest))
    with patch("model_packaging.artifact.joblib.load") as binary_load:
        with pytest.raises(ValueError):
            load_model_bundle(bundle)
        binary_load.assert_not_called()


def test_binary_type_must_match_manifest(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["estimator_type"] = "RandomForestClassifier"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="estimator_type"):
        load_model_bundle(bundle)


def test_legacy_dictionary_is_not_a_new_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    joblib.dump({"model": make_estimator()}, bundle / "model.joblib")
    with pytest.raises(ValueError, match="únicamente la clave estimator"):
        load_model_bundle(bundle)


def test_missing_model_file_is_reported(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    (bundle / "model.joblib").unlink()
    with pytest.raises(FileNotFoundError):
        load_model_bundle(bundle)


def test_serialization_failure_leaves_no_half_bundle(tmp_path):
    estimator = make_estimator()
    with patch("model_packaging.artifact.joblib.dump", side_effect=OSError("disco")):
        with pytest.raises(OSError):
            save_model_bundle(tmp_path / "bundle", estimator)
    assert list(tmp_path.iterdir()) == []


def test_existing_bundle_is_preserved(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    previous = {p.name: p.read_bytes() for p in bundle.iterdir()}
    with pytest.raises(FileExistsError):
        save_model_bundle(bundle, make_estimator())
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == previous


class ProbabilityEstimator:
    classes_ = ["excellent", "acceptable", "needs_review"]

    def __init__(self, scores):
        self.scores = scores

    def predict(self, features):
        return ["acceptable"]

    def predict_proba(self, features):
        return [self.scores]


@pytest.mark.parametrize(
    "scores",
    [
        [float("nan"), 0.7, 0.3],
        [float("inf"), 0.7, 0.3],
        [-0.1, 0.7, 0.4],
        [0.1, 1.1, -0.2],
        [0.1, 0.2, 0.1],
        [0.4, 0.6],
    ],
)
def test_invalid_confidence_is_rejected(scores):
    estimator = ProbabilityEstimator(scores)
    bundle = LoadedModelBundle(estimator, create_manifest(estimator, "test-v1"))
    with pytest.raises(ValueError):
        infer_wine_quality(bundle, make_request())


def test_confidence_uses_predicted_class_position():
    estimator = ProbabilityEstimator([0.6, 0.3, 0.1])
    bundle = LoadedModelBundle(estimator, create_manifest(estimator, "test-v1"))
    result = infer_wine_quality(bundle, make_request())
    assert result.quality_band == "acceptable"
    assert result.confidence == 0.3


def test_incompatible_feature_count_is_rejected():
    estimator = DummyClassifier().fit([[0.0] * 10], ["acceptable"])
    with pytest.raises(ValueError, match="once características"):
        create_manifest(estimator, "bad-features-v1")


def test_second_invalid_row_preserves_previous_output_without_loading(tmp_path):
    input_path, output_path = tmp_path / "input.csv", tmp_path / "out.csv"
    write_input(input_path)
    with input_path.open(newline="") as source:
        row = next(csv.DictReader(source))
    write_input(input_path, [row, {**row, "ph": "99", "sample_id": "red-002"}])
    output_path.write_text("salida anterior\n")
    with patch("model_packaging.predict_file.load_model_bundle") as load:
        with pytest.raises(ValueError, match="Fila 3"):
            predict_file(input_path, output_path, tmp_path / "missing")
        load.assert_not_called()
    assert output_path.read_text() == "salida anterior\n"


def test_write_failure_preserves_previous_output_and_removes_temporary(tmp_path):
    bundle, input_path, output_path = (
        tmp_path / "bundle",
        tmp_path / "input.csv",
        tmp_path / "out.csv",
    )
    write_bundle(bundle)
    write_input(input_path)
    output_path.write_text("salida anterior\n")
    with patch("csv.DictWriter.writerows", side_effect=OSError("disco lleno")):
        with pytest.raises(OSError):
            predict_file(input_path, output_path, bundle)
    assert output_path.read_text() == "salida anterior\n"
    assert list(tmp_path.glob(".predictions-*")) == []


@pytest.mark.parametrize("invalid", [False, True])
def test_cli_exit_code_and_visible_error(tmp_path, invalid):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    input_path, output_path = tmp_path / "input.csv", tmp_path / "out.csv"
    write_input(input_path, extra_field=invalid)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_packaging.predict_file",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--bundle",
            str(bundle),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == (2 if invalid else 0)
    assert output_path.exists() is not invalid
    if invalid:
        assert "Error de inferencia" in result.stderr
        assert "Traceback" not in result.stderr
    else:
        assert "Predicciones guardadas: 1" in result.stdout


def test_migration_preserves_estimator_and_model_version(tmp_path):
    estimator = make_estimator()
    legacy = tmp_path / "legacy.joblib"
    joblib.dump(
        {
            "estimator": estimator,
            "feature_names": list(FEATURE_NAMES),
            "model_version": "legacy-v1",
        },
        legacy,
    )
    migrate_legacy_artifact(legacy, tmp_path / "bundle")
    loaded = load_model_bundle(tmp_path / "bundle")
    assert loaded.manifest.model_version == "legacy-v1"
    assert loaded.estimator.predict([[0.0] * 11]).tolist() == ["acceptable"]


def test_migration_rejects_changed_feature_order(tmp_path):
    legacy = tmp_path / "legacy.joblib"
    joblib.dump(
        {
            "estimator": make_estimator(),
            "feature_names": list(reversed(FEATURE_NAMES)),
            "model_version": "legacy-v1",
        },
        legacy,
    )
    with pytest.raises(ValueError, match="orden de features"):
        migrate_legacy_artifact(legacy, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_missing_estimator_method_is_rejected_after_binary_load(tmp_path, method):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    estimator = make_estimator()
    setattr(estimator, method, None)
    joblib.dump({"estimator": estimator}, bundle / "model.joblib")
    with pytest.raises(ValueError, match=method):
        load_model_bundle(bundle)


def test_malformed_json_is_rejected_without_loading_binary(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    (bundle / "manifest.json").write_text('{"schema_version":')
    with patch("model_packaging.artifact.joblib.load") as load:
        with pytest.raises(ValueError):
            load_model_bundle(bundle)
        load.assert_not_called()
