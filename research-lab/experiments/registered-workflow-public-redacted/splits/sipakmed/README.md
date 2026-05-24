# SIPaKMeD Split Files

This directory stores the formal SIPaKMeD repeated random holdout splits.

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
- No usable patient, slide, case, or volume group ID is available; image-level splitting is documented in the frozen dataset manifest.

Verification summary:

- Each split has 4049 rows.
- Each split has `train=2833`, `validation=608`, and `test=608`.
- No split contains duplicate `sample_id` or duplicate `relative_path` values.
- Current SHA-256 for `split_01.csv`: `b60e4b883ce818e8035c976b18eb7a39bc4697eca98f6b8c885322c1912120da`.
