# Frozen Configuration Package

OSF registration: https://osf.io/avnp9  
Frozen configuration date: protocol freeze  
Dataset verification updated: dataset verification  
Package scope: primary-analysis configuration and dataset manifest

## Purpose

This directory contains the frozen execution-level configuration files for the registered primary analysis. These files specify the datasets, model pool, seeds, checkpoint policies, split plan, run manifest schema, training defaults, analysis interface, and dataset integrity records.

This package is intended as the frozen configuration reference. It does not duplicate raw image data.

## Contents

| Path | Purpose |
| ---- | ------- |
| `datasets.yaml` | Primary dataset definitions and dataset-specific constraints |
| `models.yaml` | Frozen model pool |
| `random_seeds.yaml` | Training and split-generation seed sets |
| `checkpoint_policies.yaml` | Frozen checkpoint selection policies |
| `training_common.yaml` | Shared training settings and dataset-specific augmentation overrides |
| `split_plan.yaml` | Repeated random holdout split design |
| `run_manifest.yaml` | Primary run matrix and run manifest schema |
| `analysis_pipeline.yaml` | Statistical analysis pipeline interface |
| `dataset_manifest.md` | Dataset source, version, path, sample-count, group-id, split, and integrity summary records |

## Primary Analysis Design

- Datasets: SIPaKMeD and OrganAMNIST
- Models: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0
- Split design: 10 generated image-level repeated random holdout splits
- Split ratio: 70/15/15 train/validation/test
- Training seeds: 42, 52, 62, 72, 82
- Checkpoint policies: validation-best with early stopping, and fixed last-epoch
- Primary matrix size: 800 runs

## Dataset Integrity Records

The public GitHub package keeps dataset source, version, expected sample counts, and split definitions. Per-image checksum CSV files are not included because they are generated from local copies of third-party datasets and are not required to run the provided split-manifest workflow.

SIPaKMeD uses the isolated single-cell images under `raw/im_*/CROPPED/*.bmp` as the primary image set. Official SIPaKMeD cell cluster images and `.dat` annotation files are retained for traceability but excluded from the current primary classification analysis.

OrganAMNIST uses MedMNIST package version 3.0.2. The current sample count is 58,830, reflecting upstream blank-sample removal documented in MedMNIST release notes.

## Path Convention

The raw image data are not included in this frozen configuration package. Relative data paths in `dataset_manifest.md` and `datasets.yaml` assume the full repository layout:

```text
research-lab/
├── data/
└── experiments/registered-workflow/configs/frozen/
```

From this `frozen/` directory, data paths such as `../../../../data/sipakmed/raw` resolve to `research-lab/data/sipakmed/raw` when the complete repository layout is present.

## Verification Commands

Use `research-lab/experiments/scripts/hash_splits.py` to recompute split hashes for the included formal split CSV files.

## Governance

These files define the frozen primary-analysis execution interface. Any future change to dataset source, dataset version, primary dataset inclusion, split policy, model pool, seed set, checkpoint policy, or analysis interface must be recorded through an amendment or deviation record before use.
