# Formal Execution Configurations

This directory stores configuration files for the registered formal workflow.

- `frozen/`: first frozen execution-level configuration package
- `amendments/`: documented amendment or deviation configuration records

## Frozen v1

The following execution-layer configuration files were frozen on protocol freeze:

- `frozen/models.yaml`: four primary-analysis models
- `frozen/datasets.yaml`: SIPaKMeD and OrganAMNIST in the primary matrix; autoPET as a future extension
- `frozen/dataset_manifest.md`: dataset source, version, hash, group-id, split-limit, and sample-count records
- `frozen/random_seeds.yaml`: training seeds and split-generation seeds
- `frozen/checkpoint_policies.yaml`: Policy A/B for primary analysis; Policy C excluded from the primary matrix
- `frozen/training_common.yaml`: input size, normalization, optimizer, fine-tuning, and dataset-specific augmentation
- `frozen/split_plan.yaml`: 10 stratified repeated random holdout splits with a 70/15/15 ratio
- `frozen/run_manifest.yaml`: 800-run primary matrix, manifest schema, and generation constraints
- `frozen/analysis_pipeline.yaml`: formal analysis-script interface and mixed-model implementation plan

These files are post-registration execution-level specifications. They do not change the registered research questions, primary outcome construct, or SAP model skeleton.
