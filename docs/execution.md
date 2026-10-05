# Reproduction guide

Run commands from the repository root. Existing training outputs are bundled separately from training source files.

## Verify and restore saved run inputs

```bash
python3 tools/verify_package.py
python3 tools/unpack_run_records.py
```

The archive restores 800 original and 300 extension records to the paths expected by the analysis. It contains metrics, configurations, completion states and extension timing records. It excludes model weights, per-epoch histories and operational logs. Saved statistical tables and figure data can be inspected without unpacking.

## Recalculate statistics in a disposable copy

Recorded analysis environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3. Install `environment/requirements-analysis.txt`. Preserve the distributed results before recalculating:

```bash
mv research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results_saved
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/run_analysis.py
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/audit_computations.py
```

Compare numeric tables with `results_saved`. This export changes paths and packaging only; timestamps and source hashes change on a new execution. Statistical recalculation and retraining were not performed during package preparation. The historical mixed-effects implementation and its saved outputs are included under `research-lab/experiments/registered-workflow/stats/`; R package requirements appear in those source files.

## Render selected main figures

With matplotlib 3.11.2 in the analysis environment:

```bash
python tools/render_figures.py
python tools/render_figure2.py
```

These scripts render existing data to `figures/rendered/` and do not rerun statistics. Figure2 combines panels a and b. Distributed figures and tables correspond to the selected manuscript materials; unused candidate figures are excluded.

## Repeat extension training in a separate checkout

Acquire the images and pretrained weights first. Recorded environment: Python 3.10.21, torch 2.4.1+cu121, torchvision 0.19.1+cu121, CUDA runtime 12.1. Consult `environment/training_environment.csv` for original and extension software records.

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environment/requirements-training.txt
CUDA_VISIBLE_DEVICES=0 python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/dispatch_manifest_run.py \
  --manifest research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/run_manifests/jiim_extension_300_v0.6.csv \
  --run-id jiim_ext_isic2019_split_01_seed_42_resnet18_A
```

Use a new output checkout rather than the restored completed records. The runner checks the frozen manifest, data, preprocessing, environment and weight identities. A few frozen configuration/split filenames retain their original identifiers because changing them would invalidate those checks. These are execution identifiers, not manuscript revision numbers. Original training follows `research-lab/experiments/scripts/train_one_run.py` and the original manifest/configuration files.
