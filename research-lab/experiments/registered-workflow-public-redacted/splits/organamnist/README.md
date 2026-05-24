# OrganAMNIST Split Files

This directory stores the formal OrganAMNIST repeated random holdout splits.

Status: generated and validated before primary training on pre-training validation.

Expected files:

```text
split_01.csv
split_02.csv
split_03.csv
split_04.csv
split_05.csv
split_06.csv
split_07.csv
split_08.csv
split_09.csv
split_10.csv
```

Rules:

- Stratified by class label.
- Split ratio is 70/15/15.
- Split seeds are 1001 to 1010.
- MedMNIST official train/validation/test splits are retained only as source records and are not used as primary-analysis splits.
- No usable patient, case, or volume group ID is available; image-level splitting is documented in the frozen dataset manifest.

Verification summary:

- Each split has 58830 rows.
- Each split has `train=41180`, `validation=8825`, and `test=8825`.
- No split contains duplicate `sample_id` or duplicate `relative_path` values.
- Identical total `class_counts` across the 10 splits are expected because all splits start from the same unified sample pool.
- Current SHA-256 for `split_01.csv`: `6745775b43b1567c0b64483b0b4cf88e435d7c726d70fc1446c6b14af0471d5f`.
