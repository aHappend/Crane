# Crane Reproduction

This repository contains a reproducible Python implementation of the Crane scheduling workflow for DNN pipeline scheduling experiments.

## What is included

- Layer and DAG modeling
- Hierarchical block construction and refinement
- ScT and MeT based scheduling / memory optimization
- Transformer and official NNS experiment entrypoints
- Standard text, CSV, and HTML outputs under `outputs/experiments/`

## Environment

Python 3.11 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Verify the environment

```powershell
python --version
python -m pip --version
```

## Main experiment entrypoints

### Transformer hierarchical experiment

```powershell
python example\run_transformer_min_layer_block_experiment.py
```

### Transformer training reproduction

```powershell
python example\run_transformer_training_repro.py --verbose-progress
```

### Official NNS layer-level suite

```powershell
python example\run_official_nns_layer_level.py
```

## Output format

Each run writes a timestamped directory under `outputs/experiments/`, typically including:

- `summary.txt`
- `summary.csv`
- `best_detail.txt`
- `best_detail.html`
- network-specific detail files for batch runs

The standard outputs intentionally use repository-relative paths instead of machine-specific absolute paths.

## Notes

- `.venv`, `__pycache__`, and generated experiment outputs are ignored by Git.
- The experiment scripts should be run from the repository root.
- Existing reference materials under `outputs/docs/` are kept in the repository; generated experiment runs under `outputs/experiments/` are not.
