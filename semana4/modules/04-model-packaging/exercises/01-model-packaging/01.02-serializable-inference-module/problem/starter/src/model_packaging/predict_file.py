"""CLI local: valido el lote completo y publico el resultado al terminar."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from model_packaging.artifact import (
    DEFAULT_BUNDLE_PATH,
    infer_wine_quality,
    load_model_bundle,
)
from model_packaging.contracts import WineQualityRequest
from model_packaging.preprocess import FEATURE_NAMES

OUTPUT_COLUMNS = (
    "sample_id",
    "quality_band",
    "confidence",
    "model_version",
    "preprocessing_version",
)


def predict_file(input_path: Path, output_path: Path, bundle_path: Path) -> int:
    """Devuelvo el número de filas; cualquier fallo conserva la salida anterior."""
    requests: list[tuple[str, WineQualityRequest]] = []
    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        expected = {"sample_id", *FEATURE_NAMES}
        if len(columns) != len(expected) or set(columns) != expected:
            raise ValueError("Cabecera inválida: sample_id y once features sin extras")
        for row_number, row in enumerate(reader, start=2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("El número de columnas de la fila es incorrecto")
                sample_id = row.pop("sample_id")
                if not sample_id.strip():
                    raise ValueError("sample_id no puede estar vacío")
                requests.append((sample_id, WineQualityRequest.model_validate(row)))
            except ValueError as error:
                raise ValueError(f"Fila {row_number}: {error}") from error
    if not requests:
        raise ValueError("El CSV debe contener al menos una muestra")

    # Una segunda fila inválida se rechaza antes de abrir el binario del modelo.
    bundle = load_model_bundle(bundle_path)
    results = [
        {"sample_id": sample_id, **infer_wine_quality(bundle, request).model_dump()}
        for sample_id, request in requests
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=".predictions-",
            suffix=".csv",
            dir=output_path.parent,
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(results)
        # El temporal comparte filesystem: el lector ve el CSV completo al reemplazar.
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(results)


def parse_args() -> argparse.Namespace:
    """Mantengo los tres argumentos del comando público del taller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE_PATH, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        count = predict_file(arguments.input, arguments.output, arguments.bundle)
    except (OSError, ValueError, EOFError) as error:
        print(f"Error de inferencia: {error}", file=sys.stderr)
        return 2
    print(f"Predicciones guardadas: {count} -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
