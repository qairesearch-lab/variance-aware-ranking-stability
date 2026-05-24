# Statistical Analysis Assets

This directory stores formal statistical-analysis scripts and outputs for the registered workflow.

## Frozen Interface

The formal analysis-script interface is defined in:

```text
configs/frozen/analysis_pipeline.yaml
```

## Primary-Analysis Requirements

- Paired bootstrap, probability-of-being-best, and mixed-model analyses should each have formal script versions.
- The default mixed-model implementation is R/lme4.
- External tables, figure titles, report text, and formal script output columns should use `stability of model rankings`, `selection_frequency`, or `selection_probability` terminology.
- `winner consistency` should be avoided in formal output naming except when quoting the registered protocol language.
- The SAP-aligned model skeleton is:

```text
metric ~ model + (1 | split) + (1 | seed) + (1 | model:split)
```

## Implementation Status

✅ **All formal analysis scripts have been fully implemented**:
- `01_collect_run_outputs.py` - Collects metrics and predictions from 800 runs
- `02_compute_ranking_metrics.py` - Computes rankings, selection frequency, and stability
- `03_paired_bootstrap.py` - Implements paired bootstrap comparisons (2000 resamples)
- `04_probability_of_being_best.py` - Estimates probability of being top-ranked
- `05_prepare_mixed_model_data.py` - Prepares stratified input for mixed-effects model
- `mixed_effects_lme4.R` - Fits SAP-aligned mixed-effects model with lme4
- `06_generate_primary_tables.py` - Generates final summary tables

✅ **Primary analysis has been executed successfully**:
- All 800 primary runs analyzed
- Mixed-effects model fitted across 4 strata (dataset × checkpoint_policy)
- Final tables generated in `stats/tables/`

✅ **Methodology follow-up analyses have been generated**:
- Evaluation-factor comparison across single split-seed, split-aggregated, seed-aggregated, and full repeated-evaluation evidence levels
- Checkpoint-rule B-minus-A paired summaries
- Single-context fragility examples
- Alternative mixed-effects model specification sensitivity records
- Multiplicity / comparison-family strategy record
- Manuscript-oriented figures in `stats/figures/`

✅ **RQ evidence-enhancement analyses have been generated and written back to the results interpretation**:
- Agreement-rate and top-model discordance intervals
- Top-model entropy / selection dispersion
- Top-two margin instability
- Subsampling recovery summaries
- Calibration missingness record

✅ **Reviewer-response supplement tables have been generated and written back to the results interpretation**:
- Dataset-feature comparison summary
- Performance-gap boundary summary for close-vs-large model comparisons

✅ **Reviewer-requested revision analyses have been generated and written back to the CMPB manuscript manuscript**:
- Paired bootstrap selection-frequency sensitivity with 10,000 replicates
- Reference-definition sensitivity, including majority-vote and leave-one-split-out checks
- Tie-handling audit for exact metric ties, full-reference mean ties, and selection-count ties
- Seed-level variability, checkpoint-rule paired effects, selected-epoch distributions, model-level performance table, and model parameter/environment records

✅ **Main-text manuscript figures have been generated and inserted into the CMPB manuscript manuscript**:
- Figure 1: top-ranked model selection frequency across single split-seed contexts
- Figure 2: top-two balanced-accuracy margin distribution
- Figure 3: subsampling recovery of the full repeated-evaluation reference

✅ **Supplementary tables and figures have been mapped back to manuscript locations**:
- Supplementary Table S1-S29 are indexed in `tables/manuscript_table_inventory.csv`
- Supplementary Fig. S1-S11 are indexed in `figures/supplementary_figure_inventory.csv`
- Main text contains inline references to the corresponding supplementary tables/figures in Methods, Results, and Discussion
- The Supplementary Information draft is `research-lab/reports/CMPB_Workflow_Ranking_Stability_Supplementary_Information_Draft.md`

✅ **CMPB manuscript numeric-consistency audit has been generated**:
- `numeric_consistency_audit.csv` records manuscript-facing values against source CSVs for agreement rates, discordance rates, selection frequencies, top-two margins, checkpoint-rule effects, selected epochs, variance components, and subsampling recovery
- The audit distinguishes full repeated-evaluation reference top-two margins from context-level top-two margin summaries

## Interpretation Principle

The statistical outputs should be interpreted as evidence about benchmark evaluation design and model-selection stability, not as a universal model architecture ranking. Candidate models provide the empirical ranking context used to evaluate split/seed/checkpoint-rule sensitivity, paired bootstrap uncertainty, and the credibility of top-ranked model claims.

## Directory Structure

```
stats/
├── scripts/           # Formal analysis scripts
│   ├── analysis_utils.py
│   ├── 01_collect_run_outputs.py
│   ├── 02_compute_ranking_metrics.py
│   ├── 03_paired_bootstrap.py
│   ├── 04_probability_of_being_best.py
│   ├── 05_prepare_mixed_model_data.py
│   ├── mixed_effects_lme4.R
│   ├── 06_generate_primary_tables.py
│   ├── 07_methodology_followup_analyses.py
│   ├── 08_rq_evidence_enhancement.py
│   ├── 09_review_response_supplement.py
│   ├── 10_review_revision_analyses.py
│   ├── 11_manuscript_figure_panels.py
│   ├── 12_numeric_consistency_audit.py
│   ├── 13_build_supplementary_information.py
│   └── supplementary_mixed_effects_sensitivity.R
├── tables/            # Final lightweight summary tables
├── figures/           # Optional figures
└── model-objects/     # R model objects (not tracked by Git)
```

## Output Tables

The following summary tables are generated in `stats/tables/`:
- `primary_performance_summary.csv`
- `ranking_stability_summary.csv`
- `rank_distribution_summary.csv`
- `selection_probability_summary.csv`
- `paired_bootstrap_comparison_summary.csv`
- `mixed_effects_variance_attribution_summary.csv`
- `mixed_effects_model_status_summary.csv`
- `primary_analysis_audit_manifest.json`

The following methodology follow-up tables are also generated in `stats/tables/`:
- `experimental_workflow_matrix.csv`
- `workflow_factor_comparison_summary.csv`
- `workflow_factor_comparison_aggregate.csv`
- `single_context_fragility_examples.csv`
- `checkpoint_policy_paired_comparison_table.csv`
- `checkpoint_policy_ranking_shift_summary.csv`
- `multiplicity_comparison_family_record.csv`
- `manuscript_table_inventory.csv`
- `mixed_effects_sensitivity_model_status.csv`
- `mixed_effects_sensitivity_fixed_effects.csv`
- `mixed_effects_sensitivity_variance_components.csv`
- `mixed_effects_sensitivity_interpretation_record.csv`
- `methodology_followup_analysis_summary.json`

The following RQ evidence-enhancement tables are generated in `stats/tables/`:
- `calibration_missingness_summary.csv`
- `workflow_stability_interval_summary.csv`
- `top_model_entropy_summary.csv`
- `top_two_context_gap_table.csv`
- `top_two_gap_instability_summary.csv`
- `subsampling_resource_stability_detail.csv`
- `subsampling_resource_stability_summary.csv`
- `rq_evidence_enhancement_summary.json`

The following reviewer-response supplement tables are generated in `stats/tables/`:
- `dataset_feature_comparison_summary.csv`
- `performance_gap_boundary_summary.csv`
- `review_response_supplement_summary.json`

The following reviewer-requested revision tables are generated in `stats/tables/`:
- `paired_context_bootstrap_selection_probability.csv`
- `reference_definition_sensitivity.csv`
- `tie_handling_audit.csv`
- `tie_handling_sensitivity_summary.csv`
- `seed_variability_summary.csv`
- `checkpoint_policy_paired_effects.csv`
- `checkpoint_policy_epoch_distribution.csv`
- `model_level_performance_table.csv`
- `model_parameter_environment_summary.csv`
- `review_revision_analyses_summary.json`
- `numeric_consistency_audit.csv`

## Output Figures

The following manuscript-oriented figures are generated in `stats/figures/`:
- `figure1_selection_frequency.png` / `.svg` / `.pdf`
- `figure2_top_two_margin.png` / `.svg` / `.pdf`
- `figure3_subsampling_recovery.png` / `.svg` / `.pdf`
- `supplementary_figure_inventory.csv`
- `selection_probability_by_workflow_context.png`
- `ranking_flip_rate_by_workflow_context.png` (legacy filename; interpreted as top-model discordance rate)
- `paired_uncertainty_resnet18_vs_resnet50.png`
- `mixed_effects_variance_attribution.png`
- `workflow_match_flip_intervals.png`
- `top_model_entropy_by_workflow_context.png`
- `top_two_gap_distribution_by_workflow_context.png`
- `subsampling_resource_stability_curves.png`
- `experimental_matrix_flow.png`
- `seed_level_variability_by_model.png`
- `checkpoint_policy_paired_effects.png`
- `paired_uncertainty_resnet18_vs_resnet50.png`
- `selection_probability_by_workflow_context.png`
- `ranking_flip_rate_by_workflow_context.png` (legacy filename; interpreted as top-model discordance rate)
- `top_two_gap_distribution_by_workflow_context.png`
- `subsampling_resource_stability_curves.png`

## Publication Notes

- `primary_run_manifest.csv` is a locked pre-training execution manifest; its `status=pending` records the locked pre-run state. Actual execution completion is documented by per-run `run_status.json` files and by `run_metrics_table.csv` containing `run_status=completed`.
- `runs/primary/`, `outputs/primary/`, and `stats/model-objects/` are generated local artifacts and are ignored by Git.
- `stats/tables/`, `stats/figures/`, frozen configs, split definitions, run manifests, scripts, reproducibility notes, and experiment logs are lightweight publication/review artifacts.
- Calibration intercept and calibration slope are unavailable for main interpretation because all 800 primary-run values are null.
- RQ evidence-enhancement outputs have been integrated into the manuscript analysis notes.
- Reviewer-response supplement outputs have been integrated into the manuscript analysis notes.
- Reviewer-requested revision outputs have been integrated into the manuscript revision notes.
- The 24 pairwise model comparisons are interpreted as estimation-oriented effect size / CI / direction-consistency evidence. Formal dichotomous inferential claims require a declared comparison family and multiplicity adjustment.

## Note

Scripts under `research-lab/experiments/scripts/` may be used as development or support scripts. They must not be treated as formal post-registration primary-analysis scripts unless reviewed and copied into the formal analysis-script area with the expected version record.
