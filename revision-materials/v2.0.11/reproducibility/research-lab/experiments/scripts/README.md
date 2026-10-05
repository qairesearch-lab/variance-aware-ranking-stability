# Experiment Scripts

This directory contains scripts for generating splits, building run manifests, and running smoke tests for the medical imaging classification experiments.

## Current Status

Todo 1 passed the split freeze gate on 2026-04-27, and Todo 2 passed the split-hash gate on the same date. Todo 3 now generates an audited 800-row `primary_run_manifest_draft.csv` with relative output paths and split hashes copied from the formal `../registered-workflow/splits/split_hashes.csv`.

## Scripts Overview

| Script Name | Description | Usage |
|------------|-------------|-------|
| `generate_splits.py` | Validate frozen split configuration and generate stratified repeated random holdout splits for SIPaKMeD and OrganAMNIST | `python3 generate_splits.py` |
| `hash_splits.py` | Validate split files, compute SHA-256 hashes, and generate the stable split hash manifest | `python3 hash_splits.py` |
| `build_run_manifest.py` | Generate and audit the Todo 3 primary run manifest draft from frozen configs and formal split hashes | `python3 build_run_manifest.py` |
| `train_one_run.py` | Read one manifest row, validate it against frozen configs, and emit fixed-schema run outputs for pre-execution validation | `python3 train_one_run.py --manifest <manifest.csv> --run-id <run_id>` |
| `run_smoke_test.py` | Execute the smoke-test manifest in schema-validation mode and validate per-run outputs | `python3 run_smoke_test.py` |
| `validate_outputs.py` | Validate fixed-schema run output directories against a manifest | `python3 validate_outputs.py --manifest <manifest.csv>` |
| `analysis_smoke_test.py` | Exercise formal analysis entry points against smoke outputs | `python3 analysis_smoke_test.py` |

## Directory Structure

```
scripts/
├── splits/
│   ├── sipakmed/         # Split files for SIPaKMeD
│   ├── organamnist/      # Split files for OrganAMNIST
│   ├── sipakmed_audit_summary.csv
│   ├── organamnist_audit_summary.csv
│   └── split_audit_summary.csv  # Audit summary for all splits
├── run-manifests/        # Run manifest drafts and audit reports
├── outputs/              # Output files
│   └── smoke/            # Smoke test outputs
├── generate_splits.py    # Split generation script
├── hash_splits.py        # Split hashing script
├── build_run_manifest.py # Run manifest generation script
├── train_one_run.py      # Per-run execution entry point
├── run_smoke_test.py     # Smoke manifest runner
├── validate_outputs.py   # Run output schema validator
├── analysis_smoke_test.py # Analysis I/O smoke test script
├── requirements-registered.txt # Dependencies
└── README.md            # This file
```

## Workstation Setup Instructions

### 1. Environment Requirements

- Python 3.9+
- PyTorch 2.0.0+
- torchvision 0.15.0+
- CUDA 11.7+ (for GPU training)

### 2. Data Path Requirements

- SIPaKMeD data: `../data/sipakmed/raw/`
- OrganAMNIST data: `../data/organamnist/`

### 3. Weight Cache Path

Set the `TORCH_HOME` environment variable to specify the cache location for model weights:

```bash
export TORCH_HOME=/path/to/weight/cache
```

### 4. Installation

Install the required dependencies:

```bash
pip3 install -r requirements-registered.txt
```

### 5. Running Experiments

#### Single Run

```bash
python3 train_one_run.py --manifest run-manifests/primary_run_manifest.csv --run-id run_0001
```

#### Smoke Test

```bash
python3 run_smoke_test.py --manifest run-manifests/smoke_test_manifest.csv
```

#### Full Run Manifest Execution

```bash
# Example bash script to run all runs
while IFS=, read -r run_id dataset split_id split_seed training_seed model checkpoint_policy config_hash split_hash output_dir status analysis_role; do
  if [ "$status" = "pending" ]; then
    echo "Running $run_id..."
    python3 train_one_run.py --manifest run-manifests/primary_run_manifest.csv --run-id $run_id
  fi
done < run-manifests/primary_run_manifest.csv
```

### 6. Failure Handling and Status Updates

- If a run fails, update the `status` field in the run manifest to `failed`
- After fixing the issue, update the `status` back to `pending` and re-run
- After a successful run, update the `status` to `completed`

### 7. Output Structure

Each run will generate the following output files:

- `run_config.json` - Run configuration
- `metrics.json` - Performance metrics
- `predictions.csv` - Model predictions
- `history.csv` - Training history
- `checkpoint_manifest.json` - Checkpoint information
- `run_status.json` - Run status and timing

### 8. Analysis Workflow

After completing all runs, use the formal analysis scripts in `../registered-workflow/stats/scripts/` to analyze the results:

1. `01_collect_run_outputs.py` - Collect outputs from all runs
2. `02_compute_ranking_metrics.py` - Compute ranking metrics
3. `03_paired_bootstrap.py` - Perform paired bootstrap analysis
4. `04_probability_of_being_best.py` - Calculate probability of being best
5. `05_prepare_mixed_model_data.py` - Prepare data for mixed effects model
6. `mixed_effects_lme4.R` - Run mixed effects model in R
7. `06_generate_primary_tables.py` - Generate primary tables and figures

## Notes

- `generate_splits.py` resolves paths relative to its own location and validates `datasets.yaml`, `random_seeds.yaml`, `split_plan.yaml`, and `dataset_manifest.md`.
- Split files and run manifests are generated in this directory and copied to `../registered-workflow/` after validation.
- For reproducibility, all random seeds are fixed as specified in the frozen configuration
- SIPaKMeD image traversal is sorted before assigning sample IDs; this keeps regenerated split hashes stable.
- Audit class-count fields are stable JSON strings and should be parsed with `json.loads()` when read back from CSV.
- `hash_splits.py` writes `split_hashes.csv` with fixed column order, dataset/split ordering, UTF-8 encoding, and LF newlines.
- `build_run_manifest.py` writes `run-manifests/primary_run_manifest_draft.csv` and `run-manifests/primary_run_manifest_audit.json`; the draft remains non-final until the smoke-test gates pass.
- Todo 3 config hash: `5e203de9913a10ae47f75aab7d778e21e9a142a0699f4a4a6c374b725d67c1ae`.
