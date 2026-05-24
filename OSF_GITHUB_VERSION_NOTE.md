# OSF and GitHub Workflow Versions

This repository keeps the OSF-frozen configuration bundle separate from the
post-freeze execution workflow.

## OSF-Frozen Configuration Bundle

`research-lab/experiments/registered-workflow/configs/frozen/`

These files are restored from the OSF-uploaded archive:

`frozen_workflow_parameters_pre_execution_2026-04-26.zip`

This archive is the pre-execution freeze associated with OSF DOI
`10.17605/OSF.IO/5PB4F`. The registration page records creation and
registration on April 26, 2026. The associated project file was uploaded before
registration as `frozen_workflow_parameters_pre_execution_2026-04-26.zip`.

The restored configuration bundle includes the frozen YAML/Markdown files and
the checksum manifests that were present in the OSF archive. These files should
be preserved as-is for provenance.

Do not silently edit this directory after OSF upload. If a correction is
unavoidable, document the change as an amendment or release note and keep the
previous frozen copy recoverable.

The broader `research-lab/experiments/registered-workflow/` directory is the
execution workflow used for reproducibility. It contains split files, run
manifests, training outputs placeholders, and statistical analysis scripts that
support reproducing the study, but it is not itself identical to the April 26
OSF zip.

The `config_hash` values in `run-manifests/primary_run_manifest.csv` and
`run-manifests/smoke_test_manifest.csv` are preserved from the locked execution
manifest. Do not replace them with hashes recomputed after documentation
cleanup, post-run notes, or public-redaction edits. If a regenerated config
bundle is ever needed, publish it as a distinct amendment rather than
overwriting the frozen manifest.

## Public Redacted Convenience Copy

`research-lab/experiments/registered-workflow-public-redacted/`

This directory is a public convenience copy. It removes or generalizes internal process traces such as local validation dates, machine-specific notes, per-image checksum manifests, and host-name recording. It is useful for public GitHub browsing, but it is not the frozen OSF protocol package.

Because this copy is intentionally redacted, its files and hashes are expected to differ from the OSF-frozen mirror.

## Practical Rule

Use `registered-workflow/` when citing or auditing the OSF-frozen protocol. Use `registered-workflow-public-redacted/` only when a cleaned public-facing view is needed, and state that it differs from the frozen OSF copy.
