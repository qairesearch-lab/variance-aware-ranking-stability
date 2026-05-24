# Registered Workflow Area

This directory contains the post-registration execution assets for the formal study workflow.

## Directory Roles

- `configs/`: frozen configuration files and amendment configuration records
- `splits/`: formal split definitions, indexes, hash manifests, and audit summaries
- `run-manifests/`: smoke-test and primary-analysis run manifests
- `runs/`: per-run training outputs
- `outputs/`: primary, exploratory, and amendment-level aggregate outputs
- `stats/`: formal statistical-analysis scripts and outputs
- `reproducibility/`: lightweight reproducibility deliverables

This directory is the unified landing area for the registered formal workflow.

## Frozen v1 Status

Post-registration execution-layer Frozen v1 was completed on protocol freeze:

- Primary datasets: SIPaKMeD and OrganAMNIST. autoPET is excluded from the current primary run manifest.
- Primary model pool: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0.
- Training seeds: `42, 52, 62, 72, 82`.
- Split generation seeds: `1001` to `1010`.
- Checkpoint policies: Policy A and Policy B are included in the primary analysis. Policy C is excluded from the primary matrix.
- Split plan: 10 stratified repeated random holdout splits with a 70/15/15 train/validation/test ratio.
- Augmentation: dataset-specific. SIPaKMeD uses horizontal flip; OrganAMNIST does not.

Before formal training, dataset verification, formal split generation, split-hash verification, smoke-level validation, and R/lme4 runtime interface validation were completed.
