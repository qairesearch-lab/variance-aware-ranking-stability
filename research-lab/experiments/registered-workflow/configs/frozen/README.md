# Frozen Configuration Package for OSF Upload

Project: Medical ML Workflow 2.0 - Variance-Aware Validation Framework  
OSF registration: https://osf.io/avnp9  
Frozen configuration date: 2026-04-25  
Dataset verification updated: 2026-04-26  
Package scope: primary-analysis configuration, dataset manifest, and checksum manifests

## Purpose

This directory contains the frozen execution-level configuration files for the registered primary analysis. These files specify the datasets, model pool, seeds, checkpoint policies, split plan, run manifest schema, training defaults, analysis interface, and dataset integrity records.

This package is intended for OSF/GitHub upload as the frozen configuration reference. It does not duplicate raw image data.

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
| `dataset_manifest.md` | Dataset source, version, path, sample-count, group-id, split, and hash records |
| `checksums/sipakmed_primary_images_sha256.csv` | Per-image SHA-256 checksums for SIPaKMeD primary images |
| `checksums/organamnist_primary_images_sha256.csv` | Per-image SHA-256 checksums for OrganAMNIST PNG images |

## Primary Analysis Design

- Datasets: SIPaKMeD and OrganAMNIST
- Models: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0
- Split design: 10 generated image-level repeated random holdout splits
- Split ratio: 70/15/15 train/validation/test
- Training seeds: 42, 52, 62, 72, 82
- Checkpoint policies: validation-best with early stopping, and fixed last-epoch
- Primary matrix size: 800 runs

## Dataset Integrity Records

| Dataset | Manifest | Rows | Manifest SHA-256 |
| ------- | -------- | ---: | ---------------- |
| SIPaKMeD | `checksums/sipakmed_primary_images_sha256.csv` | 4,050 | `1d2f9e512b3f3e668b255ec293e401b5ce9e4cf52f735db13473daccd972db41` |
| OrganAMNIST | `checksums/organamnist_primary_images_sha256.csv` | 58,831 | `74977d371c804c9fa48db29743e4db2f50d8e980d02d7bd8d06805d06780b437` |

Rows include one header row.

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

From this directory:

```bash
shasum -a 256 checksums/sipakmed_primary_images_sha256.csv
shasum -a 256 checksums/organamnist_primary_images_sha256.csv

tail -n +2 checksums/sipakmed_primary_images_sha256.csv | cut -d, -f1 | LC_ALL=C sort | shasum -a 256
tail -n +2 checksums/organamnist_primary_images_sha256.csv | cut -d, -f1 | LC_ALL=C sort | shasum -a 256
```

Expected path-list SHA-256 values:

| Dataset | Path-list SHA-256 |
| ------- | ---------------- |
| SIPaKMeD | `d64cc5fda6cf8f6681f8c347da3c7c76316592b67d3ac1a8ecfcafd7844b8cb8` |
| OrganAMNIST | `d1bee8a6b058e4f241306a7f5f2742fb7d065209cf1b596bf3d6f8953b581ae4` |

## Governance

These files define the frozen primary-analysis execution interface. Any future change to dataset source, dataset version, primary dataset inclusion, split policy, model pool, seed set, checkpoint policy, or analysis interface must be recorded through an amendment or deviation record before use.
