# Operación de Modelos — MUIAAp

**Mi entrega de la práctica 3:** [DESARROLLO.md](DESARROLLO.md),
con el desarrollo, el lienzo, el notebook ejecutado y las comprobaciones del módulo.

Material reproducible para las clases de Operación de Modelos del Máster
Universitario en Inteligencia Artificial Aplicada. Las prácticas usan `uv`,
Pydantic y pruebas automatizadas para avanzar de un experimento a un módulo de
inferencia reutilizable.

La semana 3 está en
[`modules/03-inference-contracts/`](modules/03-inference-contracts/): trabaja
con un modelo de calidad de vino ya entrenado, datos de inferencia de Kaggle y
dos prácticas en parejas.

El alumnado recibe los proyectos de `exercises/**/problem/starter/`. Las
implementaciones completas sólo se consultan en `solutions/` tras el taller.

## Clase 1 — Inferencia, contratos y preprocesado

Se diferencia entrenamiento, artefacto e inferencia; se diseña el contrato de
entrada y salida; y se fija el orden reproducible de las once características.
La práctica es guiada y no implementa todavía el módulo de producción.

[Material y práctica de la clase 1](modules/03-inference-contracts/sessions/01-inferencia-contratos/README.md)

## Clase 2 — Taller de módulo local de inferencia

Se implementa sobre el `starter` un módulo local con `uv`: contratos Pydantic,
preprocesado, carga segura del `.joblib`, inferencia y CLI CSV. Se prueban los
casos válidos y los errores sin dejar salidas parciales.

[Material y práctica de la clase 2](modules/03-inference-contracts/sessions/02-modulo-inferencia-local/README.md)

## Entregables

- [Guion de prácticas de la clase 1](modules/03-inference-contracts/guides/class-1-practices.md): lienzo de contrato y preprocesado.
- [Guion del taller de la clase 2](modules/03-inference-contracts/guides/class-2-workshop.md): módulo local, CLI y tests.
- [Proyecto starter](modules/03-inference-contracts/exercises/01-local-inference/01.02-wine-quality-inference-module/problem/starter/): implementación del alumnado.

## Tests y validación

Desde la raíz de esta semana, instala las dependencias y ejecuta toda la suite
automatizada de la solución docente:

```bash
cd semana3
uv sync
uv run pytest
uv run ruff check modules/03-inference-contracts/solutions
uv run ruff format --check modules/03-inference-contracts/solutions
```

La suite docente anterior comprueba la referencia del profesor. Para revisar mi
entrega hay que ejecutar los tests desde el starter, con su propio entorno:

```bash
cd modules/03-inference-contracts/exercises/01-local-inference/01.02-wine-quality-inference-module/problem/starter
uv sync --frozen
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

El `pyproject.toml` de `semana3/` configura imports de `solutions/`; ejecutar
los tests del alumnado desde ese entorno podría comprobar la referencia docente.
El entorno del starter importa mi implementación de `src/model_inference`.

La evidencia de la clase 1 está en el notebook y su
[exportación HTML con resultados](entrega/01-inferencia-y-contratos.html).
El enunciado acepta «captura o exportación»; la clase 2 entrega código, tests,
CSV y nota técnica. El proceso está explicado en [DESARROLLO.md](DESARROLLO.md).
