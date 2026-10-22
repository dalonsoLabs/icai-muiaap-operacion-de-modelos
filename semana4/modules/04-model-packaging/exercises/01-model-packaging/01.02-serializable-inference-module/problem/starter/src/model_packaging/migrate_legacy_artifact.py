"""Migro el artefacto local de S3 sin volver a entrenar el clasificador."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from model_packaging.artifact import create_manifest, save_model_bundle
from model_packaging.preprocess import FEATURE_NAMES


def migrate_legacy_artifact(input_path: Path, output_path: Path) -> None:
    """La entrada debe ser el Joblib de confianza distribuido para la semana 3."""
    payload = joblib.load(input_path)
    required = {"estimator", "feature_names", "model_version"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("El artefacto legacy no contiene los tres campos de S3")
    if tuple(payload["feature_names"]) != FEATURE_NAMES:
        raise ValueError("El orden de features legacy no coincide con el contrato")
    manifest = create_manifest(payload["estimator"], payload["model_version"])
    save_model_bundle(output_path, payload["estimator"], manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    migrate_legacy_artifact(arguments.input, arguments.output)
    print(f"Bundle generado sin reentrenar: {arguments.output}")
