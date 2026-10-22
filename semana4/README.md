# Semana 4 — Serialización y validación de modelos

Material reproducible de Operación de Modelos para la cuarta semana del
MUIAAp. Esta semana continúa el módulo local de inferencia de ../semana3 y
convierte el modelo serializado en un artefacto explícito, inspeccionable y
ejecutable.

## Resultado de la semana

El alumnado termina con un bundle formado por:

~~~text
models/wine_quality_bundle/
├── manifest.json       # contrato legible y validable
└── model.joblib        # estimador serializado
~~~

La primera clase diseña el manifiesto y comprueba qué incompatibilidades deben
rechazarse. La segunda implementa el guardado, la carga, la validación de
inputs/outputs, una CLI local y pruebas básicas.

## Mi entrega

Mi trabajo está en la rama `tareas-semana4` y se documenta en
[REPORT.md](REPORT.md), con el mismo formato de la semana 3. Incluye el
[lienzo completado](modules/04-model-packaging/exercises/01-model-packaging/01.01-artifact-contract/problem/LIENZO.md),
el [notebook ejecutado y exportado](entrega/01-artefacto-y-manifiesto.html),
el manifiesto, la matriz de compatibilidad y las evidencias de la clase 2.
Los comandos de mi implementación están en el
[README del starter](modules/04-model-packaging/exercises/01-model-packaging/01.02-serializable-inference-module/problem/starter/README.md).

## Material

- [README y mapa de la unidad](modules/04-model-packaging/README.md)
- [Notebook guiado de clase 1](modules/04-model-packaging/sessions/01-serializacion-validacion/notebooks/01-serializacion-validacion-guiada.ipynb)
- [Guion de prácticas de clase 1](modules/04-model-packaging/guides/class-1-practices.md)
- [Guion del taller de clase 2](modules/04-model-packaging/guides/class-2-workshop.md)
- [Starter del taller](modules/04-model-packaging/exercises/01-model-packaging/01.02-serializable-inference-module/problem/starter/)
- [Solución docente](modules/04-model-packaging/solutions/01.02-serializable-inference-module/)

Los datos de ejemplo se reutilizan desde
../semana3/assets/03-wine-quality/inference_samples.csv; no se duplica la
fuente de verdad entre semanas.

## Validación docente

Desde esta carpeta:

~~~bash
uv sync
uv run pytest
uv run ruff check modules/04-model-packaging/solutions
uv run ruff format --check modules/04-model-packaging/solutions
~~~

Para revisar el starter como lo recibe el alumnado:

~~~bash
cd modules/04-model-packaging/exercises/01-model-packaging/01.02-serializable-inference-module/problem/starter
uv sync
uv run pytest
~~~

El starter original empieza rojo por los TODO del taller. En esta rama está
completado; las evidencias del estado inicial y la verificación final están
en [REPORT.md](REPORT.md). Los comandos de «Validación docente» prueban la
referencia del profesor; para mi entrega utilizo los del starter.
