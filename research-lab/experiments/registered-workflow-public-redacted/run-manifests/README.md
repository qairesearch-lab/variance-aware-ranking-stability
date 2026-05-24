# Run Manifests

This directory stores formal workflow run manifests.

## Status

The run-manifest schema and smoke-test-first strategy were frozen before formal execution. Formal split generation, split-hash verification, and smoke-test validation were completed before primary training.

`smoke_test_manifest.csv` is provided for interface validation. The full `primary_run_manifest.csv` defines the 800 primary-analysis runs.

## Required Columns

```text
run_id,dataset,split_id,split_seed,training_seed,model,checkpoint_policy,config_hash,split_hash,output_dir,status,analysis_role
```

## Files

- `smoke_test_manifest.csv`: interface validation manifest before the full primary matrix; excluded from primary analysis.
- `primary_run_manifest.csv`: locked primary training execution manifest.
- `primary_run_manifest_template.csv`: schema template for the primary-analysis run manifest.

The `split_hash` values trace back to `registered-workflow/splits/split_hashes.csv`, and `output_dir` values use repository-relative paths under `research-lab/experiments/registered-workflow/runs/primary/`.

## Primary Matrix Size

```text
2 datasets x 10 splits x 5 seeds x 4 models x 2 checkpoint policies = 800 runs
```

## Rules

- Smoke-test outputs must not enter primary analysis.
- Runs must not be deleted or altered based on model performance.
- Any manifest change after formal execution begins should be recorded as an amendment or deviation.
