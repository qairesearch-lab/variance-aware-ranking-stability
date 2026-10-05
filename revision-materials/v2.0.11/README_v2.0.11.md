# JIIM revision processing and reproducibility materials v2.0.11

Study: Evaluation design and model selection stability in medical image classification benchmarks: a multi-dataset study.

Published material snapshot: 2026-10-05. Manuscript target v2.0 remains in preparation. This is a processing/reproducibility material version, not a completed revised manuscript.

## Contents and provenance

`reproducibility/` contains the frozen P3b payload prepared on 2026-10-04: 4701 files, including 300 extension run records (configuration, metrics, status, history, timing, checkpoint manifest), the original 800-run statistical inputs, executed statistical outputs, scripts, split definitions, configurations, and candidate figure/table sources. Original file bytes, source-relative paths, protocol versions and SHA-256 hashes are retained.

The v2.0.9 local-preparation notes inside this snapshot describe its status **at preparation on 2026-10-04**. This outer v2.0.11 entry records its GitHub publication on 2026-10-05. They are historical provenance rather than the current upload status.

- [Execution and statistical reproduction guide](reproducibility/EXECUTION_GUIDE_v2.0.9.md)
- [Frozen payload file manifest](reproducibility/FILE_MANIFEST_v2.0.9.json)
- [SHA-256 checksums](reproducibility/SHA256SUMS_v2.0.9.txt)
- [Source input closure](reproducibility/SOURCE_CLOSURE_CHECK_v2.0.9.csv)
- [Observed environments](reproducibility/ENVIRONMENT_RECORD_v2.0.9.json)
- [Checkpoint retention and hashes](reproducibility/CHECKPOINT_RETENTION_v2.0.9.csv)
- [Verified source links](DATASET_SOURCE_LINKS_v2.0.11.csv)
- [Publication scope](PUBLICATION_SCOPE_v2.0.11.md)

## Using the material

Download the repository snapshot for the tag `v2.0.11-revision-materials`, then use `revision-materials/v2.0.11/reproducibility/` as the working root. Existing generated results can be read and verified without running training. The execution guide separately describes optional statistical recomputation, figure rendering and new training. This publication did not rerun training or statistics.

Read-only byte verification: `python3 tools/verify_package.py` from the reproducibility root. The original 800-run design and 300-run extension remain distinct; revision-stage analysis decisions are identified as such. Archive candidate figures are not all selected for the manuscript.

## Source access and file scope

Original dataset images/download archives are acquired independently from the four official source entries. This snapshot supplies processing code and author-generated records, including derived benchmark identifiers and split assignments. It does not redistribute original images, pretrained weights, checkpoint binaries, individual prediction arrays, or private manuscript/reviewer correspondence.

The existing repository MIT notice applies to authored code under its existing scope. The snapshot does not grant new rights over the source benchmarks; their citation/access terms and provenance are retained. No additional broad data license is assigned.

The original submission release `v1.0-submission` remains unchanged. This revision-material tag and its commit permalink are recorded in the local revision index.
