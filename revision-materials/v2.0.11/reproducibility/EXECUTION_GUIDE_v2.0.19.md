# Execution guide (origin v2.0.19; current documentation v2.0.20)

## 1. Existing results: verification only

Unzip into a new directory and use the package directory containing `research-lab/` as root. Numerical source files preserve their relative paths and bytes; documentation and wrappers are normalized as listed in PUBLIC_EXPORT_MAP_v2.0.19.csv. Historical server absolute paths in run records identify the training host; the B1 reader uses package-relative manifest/output paths.

```bash
python3 tools/verify_package.py
```

This uses the standard library and checks the copied files against `FILE_MANIFEST_v2.0.20.json`. It performs no training or statistical analysis. Original training environment versions, actual epochs and timing remain in the included P3a source tables. The original release reference remains `v1.0-submission` at commit `2cc72ccdb0fafe0269a8c890891a779032beac9c`; baseline comparison here is against the local public snapshot, not a new remote-access audit.

## 2. Optional statistical reproduction in a disposable copy

The completed B1 inputs comprise 800 original and 300 extension runs. Common-seed and candidate-pool layers reuse these runs. Saved `results_v0.3` is authoritative for this revision. The analysis script refuses to overwrite a completed B1 manifest.

Use a second extracted copy for recalculation; preserve the original package and its checksum. With Python 3.12.14, install `requirements-analysis.txt`. Rename the completed folder in that disposable copy, then execute:

```bash
mv research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results_v0.3 research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results_v0.3_archived
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/run_b1_revision_analysis_v0.1.py
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/audit_b1_computations_v0.1.py
```

Compare the new CSV hashes/numeric fields with archived results. UTC/runtime manifest fields naturally change. This procedure is documented, **not executed in this public export**. B1 uses saved run metrics and does not require images or checkpoints. Earlier `results_v0.1/v0.2` are dependencies/historical analyses, not competing current results. The frozen resampling specifications are included; q remains a point estimate, without a new outer data CI.

## 3. Optional existing-figure rendering

In a disposable copy, install `matplotlib==3.11.2` with the above NumPy/pandas. The candidate assets are already included; rendering is optional and may change binary font metadata across machines.

```bash
python research-lab/reports/JIIM_Revision_Workspace/07_Candidate_Figures_Tables/scripts/build_candidate_assets_v0.1.py
python tools/render_existing_figure.py --figure CF04
python tools/render_existing_figure.py --figure CF02
```

CF04 is an archived intermediate; its existing results supply Figure2b. The expanded CF02 is the adopted Figure2a/b candidate. `tools/` contains public derivatives of the original rendering sources with a package-relative root. Numerical plotting definitions are unchanged. Adopted layout: one main table/three main figures; nine supplementary tables/two existing supplementary figures/CSV attachments. Package inventory includes reserve assets; inventory does not mean all assets enter the manuscript. Asset IDs and the scientific catalogue preserve source correspondence.

## 4. Optional fresh extension training

Use a **separate new training checkout**, since training does not overwrite completed outputs. Copy the exact included code, configs, sealed manifest, final dataset indices and split files. Acquire image data independently. Restore the relative paths recorded in the dataset indices (`isic2019/`, `mura/`); inspect those index files before moving data. Retain the original selected checkpoint files separately for inference verification.

Recorded environment: Python3.10.21, torch2.4.1+cu121, torchvision0.19.1+cu121, CUDA12.1 runtime; other observed package versions are in `ENVIRONMENT_RECORD_v2.0.19.json`. Use the exact environment rather than generic original-release requirements. Installation example:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-extension-training.txt
```

Pretrained ImageNet V1 files are acquired from torchvision's original sources and their hashes compared with run configs. The runner checks the seal/environment/weights and requires CUDA. All extension runs use RuleA; Swin run suffix `_B` means weight-specific evaluation-input branch, not original fixed-epoch RuleB.

For one sealed row in the fresh training checkout:

```bash
CUDA_VISIBLE_DEVICES=0 python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/dispatch_manifest_run.py \
  --manifest research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/run_manifests/jiim_extension_300_v0.6.csv \
  --run-id jiim_ext_isic2019_split_01_seed_42_resnet18_A
```

Dispatch each row at most once and keep original saved outputs separate. This guide does not run a scheduler. Original800 retraining follows the original-release workflow and its dataset-specific observed environments in S3, separately from extension training.

## Dataset source references

Source addresses retained for acquisition: SIPaKMeD https://www.cs.uoi.gr/~marina/sipakmed.html ; OrganAMNIST https://medmnist.com/ ; ISIC2019 https://challenge.isic-archive.com/landing/2019/ ; MURA https://stanfordmlgroup.github.io/competitions/mura/ . These are acquisition/source pointers, not a new license clearance or live accessibility verification.
