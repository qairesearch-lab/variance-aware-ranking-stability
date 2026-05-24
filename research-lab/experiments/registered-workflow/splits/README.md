# Split Definitions

This directory stores formal split definitions, indexes, hash manifests, and audit summaries.

## Current Status

Formal split generation and split-hash verification were completed before primary training.

Split-generation validation result:

```text
errors 0
SPLIT_GENERATION_VALIDATION_OK
```

Split-hash validation result:

```text
errors 0
hash_manifest_rows 20
smoke_rows 8 primary_draft_rows 800
SPLIT_HASH_VALIDATION_OK
```

## Frozen Rules

- Split design: 10 stratified repeated random holdout splits.
- Split ratio: 70% train, 15% validation, 15% test.
- Split generation seeds: `1001` to `1010`.
- Split generation seeds are separate from training seeds.
- If patient, case, slide, or volume group IDs are available, group-aware splitting should be preferred.
- If group IDs are unavailable, the image-level split limitation must be recorded in `configs/frozen/dataset_manifest.md`.

## Split CSV Schema

Each split file contains:

```text
dataset,split_id,sample_id,relative_path,label,subset,split_seed
```

Allowed `subset` values:

```text
train,validation,test
```

## Files

- `sipakmed/`: 10 SIPaKMeD split files.
- `organamnist/`: 10 OrganAMNIST split files.
- `split_hashes.csv`: SHA-256 manifest for the 20 formal split CSV files.
- `split_audit_summary.csv`: combined split audit summary for both datasets.
- `sipakmed_audit_summary.csv`: SIPaKMeD split audit summary.
- `organamnist_audit_summary.csv`: OrganAMNIST split audit summary.

## Verification Summary

- SIPaKMeD: each split has 4049 rows with `train=2833`, `validation=608`, and `test=608`.
- OrganAMNIST: each split has 58830 rows with `train=41180`, `validation=8825`, and `test=8825`.
- No split contains duplicate `sample_id` or duplicate `relative_path` values.
- The 20 SHA-256 values in `split_hashes.csv` have been recomputed and verified.
- `split_hashes.csv` uses fixed columns, fixed dataset/split ordering, UTF-8 encoding, and LF newlines.
- Audit class-count fields are stable JSON strings: `class_counts`, `train_class_counts`, `validation_class_counts`, and `test_class_counts`.

After formal training begins, split files must not be regenerated or replaced based on model performance. Any change must be recorded as an amendment or deviation.
