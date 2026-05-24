# Experiment Scripts

This directory contains the executable workflow scripts for the registered 800-run experiment.

## Main Entry Points

| Script | Role |
| --- | --- |
| `generate_splits.py` | Regenerate the stratified repeated random holdout split CSV files from the frozen configuration and local datasets. |
| `hash_splits.py` | Validate split CSV files and compute SHA-256 split hashes. |
| `build_run_manifest.py` | Build and audit the complete primary run manifest from frozen configs and split hashes. |
| `train_one_run.py` | Execute one manifest-defined run and write fixed-schema outputs. |
| `run_smoke_test.py` | Run schema-level smoke validation against the smoke manifest. |
| `validate_outputs.py` | Validate per-run output directories against a manifest. |
| `analysis_smoke_test.py` | Exercise analysis inputs/outputs using smoke-test results. |
| `check_cuda_environment.py` | Probe the local CUDA/PyTorch environment before training. |

## Required Layout

Run commands from the repository root. The scripts expect:

```text
research-lab/experiments/registered-workflow/configs/frozen/
research-lab/experiments/registered-workflow/splits/
research-lab/experiments/registered-workflow/run-manifests/
research-lab/data/sipakmed/raw/
research-lab/data/organamnist/
```

## Single Run Example

```bash
python3 research-lab/experiments/scripts/train_one_run.py \
  --manifest research-lab/experiments/registered-workflow/run-manifests/primary_run_manifest.csv \
  --run-id run_0001 \
  --execution-mode primary_train
```

## Smoke Test Example

```bash
python3 research-lab/experiments/scripts/run_smoke_test.py \
  --manifest research-lab/experiments/registered-workflow/run-manifests/smoke_test_manifest.csv
```
