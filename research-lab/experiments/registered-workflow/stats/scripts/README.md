# Formal Statistical Analysis Scripts

This directory stores formal post-registration primary-analysis scripts.

## Frozen Script Order

1. `01_collect_run_outputs.py` - Collect per-run metrics and metadata
2. `02_compute_ranking_metrics.py` - Compute model rankings and stability
3. `03_paired_bootstrap.py` - Run paired bootstrap comparisons
4. `04_probability_of_being_best.py` - Estimate selection frequency for the top ranked model
5. `05_prepare_mixed_model_data.py` - Prepare data for mixed-effects model
6. `mixed_effects_lme4.R` - Fit SAP-aligned mixed-effects model using lme4
7. `06_generate_primary_tables.py` - Generate final primary tables

Supplementary methodology follow-up scripts:

- `07_methodology_followup_analyses.py` - Generate evaluation-factor comparison, checkpoint selection rule paired summaries, fragility examples, multiplicity records, and figures
- `08_rq_evidence_enhancement.py` - Generate RQ1/RQ2/RQ3 evidence-enhancement intervals, entropy, top-two margin, subsampling, and calibration-missingness records
- `09_review_response_supplement.py` - Generate reviewer-response supplement tables for dataset features and performance-gap boundary interpretation
- `10_review_revision_analyses.py` - Generate reviewer-requested revision tables for paired-context bootstrap, empirical comparator definition sensitivity, tie audit, seed variability, checkpoint effects/epochs, model-level performance, parameter counts, and revision figures
- `11_manuscript_figure_panels.py` - Generate main-text Figure 1-3 panels using harmonized manuscript terminology
- `12_numeric_consistency_audit.py` - Generate the CMPB v0.3 manuscript numeric-consistency audit table
- `13_build_supplementary_information.py` - Build the CMPB v0.3 Supplementary Information Markdown draft from the table and figure inventories
- `14_top_two_set_stability.py` - Compute top two set Jaccard similarity and exact top two set agreement from existing top two context records
- `supplementary_mixed_effects_sensitivity.R` - Run exploratory alternative mixed-effects model specifications

## Current Status

- ✅ **Interface frozen**: Script order and responsibilities are locked.
- ✅ **Implementation complete**: All scripts have been fully implemented.
- ✅ **Smoke tests passed**: All scripts executed successfully against smoke-test outputs.
- ✅ **Primary analysis executed**: All 800 primary runs have been analyzed.
- ✅ **Output tables generated**: Final summary tables are in `stats/tables/`.
- ✅ **Methodology follow-up generated**: Supplementary workflow-methodology tables and figures are available in `stats/tables/` and `stats/figures/`.
- ✅ **RQ evidence enhancement generated and reported**: `08_rq_evidence_enhancement.py` produced RQ1/RQ2/RQ3 evidence-enhancement tables/figures and the results interpretation file has been updated.
- ✅ **Reviewer-response supplement generated and reported**: `09_review_response_supplement.py` produced dataset-feature and performance-gap boundary tables, and the results interpretation file has been updated.
- ✅ **Reviewer revision analyses generated and reported**: `10_review_revision_analyses.py` produced paired-context bootstrap, reference sensitivity, tie-handling, seed-variability, checkpoint-effect, epoch-distribution, model-performance, parameter/environment, and figure outputs for the CMPB v0.2 revision.
- ✅ **Main-text figures generated and inserted**: `11_manuscript_figure_panels.py` produced Figure 1-3 in PNG/SVG/PDF formats for the CMPB v0.2 manuscript.
- ✅ **Numeric consistency audit generated**: `12_numeric_consistency_audit.py` produced `numeric_consistency_audit.csv` for the CMPB v0.3 manuscript fork.
- ✅ **Supplementary Information draft generated and extended**: `13_build_supplementary_information.py` produced the Supplementary Information draft, and the current v0.4 Supplementary Information now tracks Supplementary Table S1-S32 and Supplementary Fig. S1-S12.
- ✅ **Top two set stability generated**: `14_top_two_set_stability.py` produced top two set Jaccard similarity and exact top two set agreement outputs for Supplementary Table S32.

## Shared Utilities

- `analysis_utils.py` - Shared helper functions for path handling, input validation, and output generation.

## Execution Commands

```bash
# 1. Collect run outputs
python3 01_collect_run_outputs.py \
  --manifest research-lab/experiments/registered-workflow/run-manifests/primary_run_manifest.csv \
  --output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io

# 2. Compute ranking metrics
python3 02_compute_ranking_metrics.py \
  --metrics-table research-lab/experiments/registered-workflow/outputs/primary/analysis_io/run_metrics_table.csv \
  --output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io \
  --metric balanced_accuracy

# 3. Run paired bootstrap
python3 03_paired_bootstrap.py \
  --metrics-table research-lab/experiments/registered-workflow/outputs/primary/analysis_io/run_metrics_table.csv \
  --output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io

# 4. Compute selection frequency for the top ranked model
python3 04_probability_of_being_best.py \
  --model-ranking-table research-lab/experiments/registered-workflow/outputs/primary/analysis_io/model_ranking_table.csv \
  --output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io

# 5. Prepare mixed model data
python3 05_prepare_mixed_model_data.py \
  --metrics-table research-lab/experiments/registered-workflow/outputs/primary/analysis_io/run_metrics_table.csv \
  --output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io \
  --metric balanced_accuracy

# 6. Fit mixed-effects model
Rscript mixed_effects_lme4.R \
  research-lab/experiments/registered-workflow/outputs/primary/analysis_io/mixed_model_input.csv \
  research-lab/experiments/registered-workflow/stats/model-objects

# 7. Generate primary tables
python3 06_generate_primary_tables.py \
  --analysis-output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io \
  --tables-dir research-lab/experiments/registered-workflow/stats/tables

# Supplementary methodology follow-up
python3 07_methodology_followup_analyses.py \
  --analysis-output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io \
  --tables-dir research-lab/experiments/registered-workflow/stats/tables \
  --figures-dir research-lab/experiments/registered-workflow/stats/figures \
  --metric balanced_accuracy

Rscript supplementary_mixed_effects_sensitivity.R \
  research-lab/experiments/registered-workflow/outputs/primary/analysis_io/mixed_model_input.csv \
  research-lab/experiments/registered-workflow/stats/tables

# RQ evidence enhancement
python3 08_rq_evidence_enhancement.py \
  --analysis-output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io \
  --tables-dir research-lab/experiments/registered-workflow/stats/tables \
  --figures-dir research-lab/experiments/registered-workflow/stats/figures \
  --metric balanced_accuracy \
  --resamples 5000 \
  --seed 2026051331

# Reviewer-response supplement
python3 09_review_response_supplement.py \
  --splits-dir research-lab/experiments/registered-workflow/splits \
  --tables-dir research-lab/experiments/registered-workflow/stats/tables

# Reviewer-requested revision analyses
python3 10_review_revision_analyses.py \
  --analysis-output-dir research-lab/experiments/registered-workflow/outputs/primary/analysis_io \
  --runs-dir research-lab/experiments/registered-workflow/runs/primary \
  --tables-dir research-lab/experiments/registered-workflow/stats/tables \
  --figures-dir research-lab/experiments/registered-workflow/stats/figures \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 2026051510

# Manuscript numeric consistency audit
python3 12_numeric_consistency_audit.py \
  --tables-dir research-lab/experiments/registered-workflow/stats/tables

# Supplementary Information draft
python3 13_build_supplementary_information.py

# Top two set stability
python3 14_top_two_set_stability.py \
  --tables-dir research-lab/experiments/registered-workflow/stats/tables
```

## Output Naming

Formal output naming uses `selection_frequency` and `ranking_stability` terminology where current outputs can be named directly. Legacy filenames containing `selection_probability` are retained only for source-file continuity. `winner consistency` is avoided in formal output naming except when quoting registered protocol language.

Supplementary methodology outputs should be interpreted as workflow-reliability evidence rather than model-superiority evidence.
