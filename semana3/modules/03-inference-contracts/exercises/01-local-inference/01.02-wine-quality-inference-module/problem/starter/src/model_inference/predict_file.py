"""Valido el CSV completo antes de inferir y escribir las predicciones."""

import argparse
import csv
import sys
from pathlib import Path

from pydantic import ValidationError

from model_inference.contracts import WineQualityRequest
from model_inference.inference import (
    DEFAULT_MODEL_PATH,
    infer_wine_quality,
    load_wine_quality_model,
)
from model_inference.preprocess import FEATURE_NAMES, preprocess_wine_request

OUTPUT_FIELDS = (
    "sample_id",
    "quality_band",
    "confidence",
    "model_version",
    "preprocessing_version",
)
EXPECTED_INPUT_FIELDS = {"sample_id", *FEATURE_NAMES}


def _validate_headers(fieldnames: list[str] | None) -> None:
    """Rechazo cabeceras ausentes, duplicadas, incompletas o inesperadas."""

    if not fieldnames:
        raise ValueError("El CSV de entrada debe tener una fila de cabecera.")
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("El CSV de entrada contiene nombres de columna duplicados.")

    extra_fields = sorted(set(fieldnames).difference(EXPECTED_INPUT_FIELDS))
    if extra_fields:
        raise ValueError(f"Columnas CSV inesperadas: {', '.join(extra_fields)}.")

    required_fields = ("sample_id", *FEATURE_NAMES)
    missing_fields = [name for name in required_fields if name not in fieldnames]
    if missing_fields:
        raise ValueError(
            f"Faltan columnas CSV obligatorias: {', '.join(missing_fields)}."
        )


def _collect_predictions(
    input_path: Path,
    model_path: Path,
) -> list[dict[str, str | float]]:
    """Valido primero todas las filas y después cargo e invoco el modelo."""

    # Guardo peticiones validadas para no inferir sobre un lote inválido.
    requests: list[tuple[str, WineQualityRequest]] = []

    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        _validate_headers(reader.fieldnames)

        for line_number, row in enumerate(reader, start=2):
            sample_id_value = row.get("sample_id")
            sample_id = sample_id_value.strip() if sample_id_value else ""
            if not sample_id:
                raise ValueError(
                    f"Fila {line_number}: sample_id es obligatorio y no puede "
                    "estar vacío."
                )
            if None in row:
                raise ValueError(
                    f"Fila {line_number} (sample_id={sample_id}) contiene valores "
                    "sin una columna correspondiente."
                )

            request_values = {name: row.get(name) for name in FEATURE_NAMES}
            try:
                request = WineQualityRequest.model_validate(request_values)
            except (ValidationError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Fila {line_number} (sample_id={sample_id}): {exc}"
                ) from exc

            requests.append((sample_id, request))

    if not requests:
        raise ValueError("El CSV de entrada no contiene filas de datos.")

    model = load_wine_quality_model(model_path)
    predictions: list[dict[str, str | float]] = []
    for sample_id, request in requests:
        features = preprocess_wine_request(request)
        prediction = infer_wine_quality(model, features)
        predictions.append({"sample_id": sample_id, **prediction.model_dump()})
    return predictions


def _write_predictions(
    output_path: Path,
    predictions: list[dict[str, str | float]],
) -> None:
    """Escribo el resultado completo con el orden acordado para la salida."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(predictions)


def run_predict_file(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> int:
    """Coordino el flujo y devuelvo el número de filas escritas."""

    # No abro la salida hasta tener todas las predicciones validadas.
    predictions = _collect_predictions(Path(input_path), Path(model_path))
    _write_predictions(Path(output_path), predictions)
    return len(predictions)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Defino las rutas de entrada, salida y el modelo opcional."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV de entrada.")
    parser.add_argument("--output", type=Path, required=True, help="CSV de salida.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta del modelo (por defecto, el artefacto entregado por el docente).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Comunico errores por stderr y devuelvo 0 si todo va bien o 2 si falla."""

    arguments = _build_argument_parser().parse_args(argv)
    try:
        written_rows = run_predict_file(
            input_path=arguments.input,
            output_path=arguments.output,
            model_path=arguments.model,
        )
    except (OSError, csv.Error, TypeError, ValueError) as exc:
        print(f"Error de inferencia: {exc}", file=sys.stderr)
        return 2

    print(f"Se escribieron {written_rows} predicciones en {arguments.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
