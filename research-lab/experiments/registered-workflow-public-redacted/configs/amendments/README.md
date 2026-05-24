# Amendments and Deviations

This directory stores post-registration amendment or deviation configuration records after Frozen v1.

## Current Status

As of pre-training validation, there is no formal amendment or deviation record in this directory.

The completed decisions on model pool, seeds, checkpoint policies, split design, `training_common`, and run-manifest schema are execution-level specifications within the registered protocol scope. They are not amendments.

Current non-amendment implementation details:

- The model pool contains four models, within the registered range of three to four representative models.
- The attention-based variant was optional in the registered protocol; excluding it from the current primary matrix is not a deviation.
- Policy C was optional in the registered protocol; excluding it from the current primary matrix is not a deviation.
- OrganAMNIST official splits are retained for traceability only and are not used for primary analysis.
- The primary split definitions were generated and hash-verified before formal training.

Future changes that should be recorded here include:

- Changing the primary dataset list or adding autoPET to the current primary matrix
- Changing the number or identity of primary models
- Changing training seeds, split seeds, split ratio, or checkpoint policy
- Changing the formal analysis-script interface, primary outcome construct, or SAP mixed-model skeleton
- Excluding or rerunning primary-analysis runs after formal execution begins because of data, code, convergence, or compute limitations
