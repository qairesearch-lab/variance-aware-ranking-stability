# Run Manifests

This directory stores formal workflow run manifests.

## Status

The run-manifest schema and smoke-test-first strategy were frozen with the registered workflow. Formal split generation, split-hash verification, primary run-manifest draft validation, and smoke-test manifest hash validation were completed before primary training.

`smoke_test_manifest.csv` contains the current SIPaKMeD `split_01` hash and the execution config hash used by the primary run-manifest draft. The full `primary_run_manifest.csv` has been created and locked after training/schema smoke and analysis I/O smoke validations passed.

## Required Columns

```text
run_id,dataset,split_id,split_seed,training_seed,model,checkpoint_policy,config_hash,split_hash,output_dir,status,analysis_role
```

## Files

- `smoke_test_manifest.csv`: interface validation manifest before the full primary matrix; excluded from primary analysis.
- `primary_run_manifest.csv`: locked primary training execution manifest.
- `primary_run_manifest_template.csv`: schema template for the primary-analysis run manifest.

The audited development draft exists at:

```text
research-lab/experiments/scripts/run-manifests/primary_run_manifest_draft.csv
research-lab/experiments/scripts/run-manifests/primary_run_manifest_audit.json
```

The formal `primary_run_manifest.csv` was locked from that audited draft after smoke-level validation. Its `split_hash` values all trace back to `registered-workflow/splits/split_hashes.csv`, and its `output_dir` values use repository-relative paths under `research-lab/experiments/registered-workflow/runs/primary/`.

Execution config hash:

```text
5e203de9913a10ae47f75aab7d778e21e9a142a0699f4a4a6c374b725d67c1ae
```

Formal primary manifest SHA-256:

```text
7a39f39180c94f7ea0cf2b26693644144befed13455f18c77816bd2774d6009f
```

## Primary Matrix Size

```text
2 datasets x 10 splits x 5 seeds x 4 models x 2 checkpoint policies = 800 runs
```

## Rules

- Smoke-test outputs must not enter primary analysis.
- After the primary manifest is generated, runs must not be deleted or altered based on model performance.
- Any manifest change after formal execution begins must be recorded as an amendment or deviation.
