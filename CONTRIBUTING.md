# Contributing

This project studies a Python reproduction of the Crane scheduling method.
Before changing algorithms or adding results, read the
[reproduction scope](docs/REPRODUCTION.md) and [architecture](docs/ARCHITECTURE.md).

## Set up and check a change

Follow the [README installation instructions](README.md#quick-start), then:

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
python tools/doctor.py
python -m ruff check .
python -m pytest -q
```

Ruff currently checks syntax and undefined-name errors. There is no requirement
to reformat unrelated legacy files. Place automated tests under `tests/` and
keep large experiment runs out of the default test suite.

For dependency updates, regenerate the constraint snapshot in a clean environment,
run the checks and quick start, and verify the CI Python/platform matrix before
merging the new versions. State which environments were actually tested.
`pip freeze` from an unrelated working environment
is not a suitable dependency record.

## Useful contributions

- Correctness fixes with a small failing example and a regression test that
  checks behavior, such as workload conservation or schedule feasibility.
- Explicit shape/graph imports replacing proxy workload metadata.
- Equation-level audits, small exhaustive comparisons, unit checks and cost
  calibration against documented references.
- Clearer experiment commands, provenance and output interpretation.

A change to a cost assumption or objective can change every reported number.
Explain the before/after behavior, update `docs/REPRODUCTION.md` and
`CHANGELOG.md`, and regenerate any result you use as evidence. Keep paper-reported
numbers, historical outputs and new measurements distinguishable.

## Reporting bugs and reproduction gaps

Use the issue templates. Include the exact commit and command, environment check,
smallest workload/configuration that reproduces the problem, expected behavior,
actual output and relevant traceback. For a paper mismatch, identify the section,
equation or figure and explain the intended comparison and units.

## Pull requests

Keep changes focused and describe what a user can now do. State which checks
actually ran and any large experiments that were not run. Do not claim a paper
result based only on a successful program exit or an uncalibrated proxy run.

Do not commit `.venv*`, bytecode, local editor state, credentials or new generated
experiment directories. Include a small fixture only when it is needed to test
behavior and its provenance is documented. Preserve third-party notices and
record sources for any imported material.
