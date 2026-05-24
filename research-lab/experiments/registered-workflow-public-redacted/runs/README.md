# Formal Run Outputs

This directory stores per-run outputs from the registered formal workflow.

Recommended organization is by `split`, `seed`, `checkpoint_policy`, and `model`, or by an equivalent structure that preserves one-to-one traceability to the run manifest.

Each completed run should be traceable to:

- the run manifest row
- the split CSV
- the split hash
- the frozen configuration hash
- the checkpoint policy
- the analysis role
