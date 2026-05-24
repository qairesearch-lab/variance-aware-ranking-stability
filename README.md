# Evaluation Design and Model Selection Stability in Medical Image Classification

This repository contains the reproducibility package for:

**Evaluation design and model selection stability in medical image classification benchmarks: a prespecified 800-run repeated evaluation study**

The package is organized around the registered workflow used for the study. It includes the executable training scripts, the OSF-frozen configuration bundle, formal split definitions, run manifests, and statistical analysis scripts. Original image datasets and generated experiment outputs are intentionally not included.

See `OSF_GITHUB_VERSION_NOTE.md` for the relationship between the OSF-frozen configuration bundle, the executable reproduction workflow, and the public redacted convenience copy.

## What Is Included

- `research-lab/experiments/scripts/`
  - Split generation, split hashing, run-manifest construction, smoke testing, output validation, and single-run training entry points.
- `research-lab/experiments/registered-workflow/configs/frozen/`
  - Frozen dataset, model, seed, split, checkpoint, training, analysis, dataset manifest, and checksum files restored from the April 26 OSF pre-execution archive.
- `research-lab/experiments/registered-workflow/splits/`
  - The 10 stratified repeated random holdout split definitions for SIPaKMeD and OrganAMNIST, with audit summaries and SHA-256 split hashes.
- `research-lab/experiments/registered-workflow/run-manifests/`
  - Smoke-test and 800-run primary-analysis manifests.
- `research-lab/experiments/registered-workflow/stats/scripts/`
  - Formal post-training statistical analysis scripts.

## What Is Not Included

- Original image data under `research-lab/data/`
- Per-run training outputs under `research-lab/experiments/registered-workflow/runs/primary/`
- Generated primary analysis outputs under `research-lab/experiments/registered-workflow/outputs/primary/`
- Generated tables, figures, checkpoints, model objects, and local cache files

These exclusions keep the repository lightweight and avoid redistributing third-party datasets.

## Dataset Layout

Download the datasets from their public sources and place them as follows:

```text
research-lab/data/sipakmed/raw/
research-lab/data/organamnist/
```

Expected sources:

- SIPaKMeD: https://www.cs.uoi.gr/~marina/sipakmed.html
- OrganAMNIST / MedMNIST v2: https://medmnist.com/

The frozen dataset manifest is available at:

```text
research-lab/experiments/registered-workflow/configs/frozen/dataset_manifest.md
```

## Environment

Python 3.9 or newer is recommended. Install the registered workflow dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

GPU training is strongly recommended for the full 800-run matrix. The training script requires CUDA by default for primary training; use `--allow-cpu` only for diagnostic runs.

## Reproduction Workflow

From the repository root:

1. Verify that the frozen split files and hashes are present:

```bash
python3 research-lab/experiments/scripts/hash_splits.py
```

This writes a script-side hash manifest for validation. The formal split hashes used by the primary manifest are stored in:

```text
research-lab/experiments/registered-workflow/splits/split_hashes.csv
```

2. Run the schema-level smoke test:

```bash
python3 research-lab/experiments/scripts/run_smoke_test.py \
  --manifest research-lab/experiments/registered-workflow/run-manifests/smoke_test_manifest.csv
```

3. Run one primary training job:

```bash
python3 research-lab/experiments/scripts/train_one_run.py \
  --manifest research-lab/experiments/registered-workflow/run-manifests/primary_run_manifest.csv \
  --run-id run_0001 \
  --execution-mode primary_train
```

4. Run the full primary manifest by dispatching each pending `run_id` in `primary_run_manifest.csv`.

Each primary run writes fixed-schema outputs to the repository-relative `output_dir` recorded in the manifest, usually:

```text
research-lab/experiments/registered-workflow/runs/primary/run_XXXX/
```

5. After all runs complete, run the formal analysis scripts in order:

```text
research-lab/experiments/registered-workflow/stats/scripts/
```

The script order and command examples are documented in:

```text
research-lab/experiments/registered-workflow/stats/scripts/README.md
```

## Reproducibility Notes

- The primary matrix contains 2 datasets, 4 models, 10 splits, 5 training seeds, and 2 checkpoint policies, yielding 800 primary runs.
- Split generation seeds are 1001-1010.
- Training seeds are 42, 52, 62, 72, and 82.
- SIPaKMeD uses horizontal flipping during training; OrganAMNIST does not, according to the frozen dataset-specific augmentation configuration.
- The repository uses image-level repeated random holdout because no patient, slide, case, or volume group identifiers were available in the verified dataset manifests.

## License

This repository is released under the MIT License. See `LICENSE` for details.

The original SIPaKMeD and OrganAMNIST image datasets are not redistributed in
this repository and remain subject to their own source licenses and terms of
use.
