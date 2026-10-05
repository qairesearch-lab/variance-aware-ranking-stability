# Scientific figure and table captions

Public export v2.0.19. Historical asset identifiers remain traceable; they do not prescribe manuscript placement.

## CT01 · Model_selection_stability_Table1


**English caption / note:** The reference top model is defined by full-grid mean balanced accuracy (BA). Agreement/discordance intervals use 10,000 paired split-then-seed percentile resamples with the reference re-estimated per draw; discordance is 1 minus agreement. q is a point estimate from 10,000 paired crossed split–seed ranking resamples. The 5×3 column uses finite-grid agreement with a held-out five-split reference block, averaged over 252 oriented partitions; source images may overlap across splits. Original seed subsets come from five seeds; extension subsets from three.


**算法及统计对象：** agreement/discordance: existing nested T36; q: adopted crossed T37; budget: finite exact T40

**文件：** [CT01_Model_selection_stability_Table1_v0.1](01_Main_Candidates/CT01_Model_selection_stability_Table1_v0.1.pdf)；[数据](01_Main_Candidates/CT01_Model_selection_stability_Table1_v0.1.csv)。

## CF01 · Observed_selection_frequency_Figure1


**English caption / note:** Observed top-model selection frequencies across 50 original or 30 extension split–seed contexts per stratum. R18/R50/D121/E-B0 denote ResNet-18, ResNet-50, DenseNet-121 and EfficientNet-B0. Error bars are paired crossed split–seed percentile 95% intervals (10,000 replicates) for f, not for q. Zero observed frequencies may yield degenerate nonparametric intervals.


**算法及统计对象：** f point: T31; f data CI: crossed T37; no q CI

**文件：** [CF01_Observed_selection_frequency_Figure1_v0.1](01_Main_Candidates/CF01_Observed_selection_frequency_Figure1_v0.1.pdf)；[数据](01_Main_Candidates/CF01_Observed_selection_frequency_Figure1_v0.1.csv)。

## CF02 · Context_top_two_BA_difference_Figure2


**English caption / note:** Distribution of the difference between the two highest BA values within each observed context. BA differences are shown in percentage points (pp). Boxes show the median and interquartile range; whiskers extend to 1.5 interquartile ranges, and points show all contexts. The leading pair is reselected within each context. Panel y-ranges differ to preserve readability. These distributions differ from the difference between full-grid model means.


**算法及统计对象：** Observed context order statistic; box/whiskers are not confidence intervals

**文件：** [CF02_Context_top_two_BA_difference_Figure2_v0.1](01_Main_Candidates/CF02_Context_top_two_BA_difference_Figure2_v0.1.pdf)；[数据](01_Main_Candidates/CF02_Context_top_two_BA_difference_Figure2_v0.1.csv)。

## CF03 · Aggregate_evaluation_budget_Figure3


**English caption / note:** Top-model agreement as a function of discovery splits and training seeds. Each point averages all eligible discovery subsets within 252 oriented five/five split-ID partitions. The reference is mean BA on the five held-out split IDs and all available seeds. Shading for the three-seed line is the 2.5th–97.5th percentile spread across the finite set of partitions, not a population confidence interval. Training-run budget is four times splits times seeds.


**算法及统计对象：** Finite grid exact aggregation; split IDs disjoint, source images can overlap; no population CI

**文件：** [CF03_Aggregate_evaluation_budget_Figure3_v0.1](01_Main_Candidates/CF03_Aggregate_evaluation_budget_Figure3_v0.1.pdf)；[数据](01_Main_Candidates/CF03_Aggregate_evaluation_budget_Figure3_v0.1.csv)。

## ST01 · Absolute_BA_uncertainty


**English caption / note:** Mean BA and context-level sample SD (ddof=1). Mean intervals use paired crossed split–seed resampling. The 16 strata include reused subsets and candidate/input sensitivities; rows must not be summed as independent training runs.


**算法及统计对象：** Mean/SD T31; crossed mean CI T37

**文件：** [ST01_Absolute_BA_uncertainty_v0.1](02_Supplement_Candidates/ST01_Absolute_BA_uncertainty_v0.1.pdf)；[数据](02_Supplement_Candidates/ST01_Absolute_BA_uncertainty_v0.1.csv)。

## ST02 · Fixed_model_paired_BA_contrasts


**English caption / note:** Six alphabetically oriented fixed-model contrasts per primary stratum. Intervals retain the existing paired split-then-seed procedure. Exact sign-flip p-values use split-average contrasts under conditional joint sign symmetry; Holm adjustment is within each six-comparison stratum. These are separate from the context-wise top-two order statistic.


**算法及统计对象：** T33 existing nested paired CI; conditional exact sign-flip; within-stratum Holm

**文件：** [ST02_Fixed_model_paired_BA_contrasts_v0.1](02_Supplement_Candidates/ST02_Fixed_model_paired_BA_contrasts_v0.1.pdf)；[数据](02_Supplement_Candidates/ST02_Fixed_model_paired_BA_contrasts_v0.1.csv)。

## ST03 · Checkpoint_policy_BA_sensitivity


**English caption / note:** Original ten-split/five-seed paired B-minus-A effects. Paired nested intervals and conditional split-mean sign-flip tests retain the B1 algorithm; Holm family contains eight dataset-model comparisons. Rule A and rule B also differ in training duration/trajectory.


**算法及统计对象：** Existing T34, not recomputed or relabeled crossed

**文件：** [ST03_Checkpoint_policy_BA_sensitivity_v0.1](02_Supplement_Candidates/ST03_Checkpoint_policy_BA_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST03_Checkpoint_policy_BA_sensitivity_v0.1.csv)。

## ST04 · Checkpoint_selected_identity_sensitivity


**English caption / note:** Paired A/B comparisons within identical split–seed contexts. ALL rows report a changed-winner fraction; model rows report B-minus-A observed selection-frequency differences. Both preserve the existing nested interval algorithm.


**算法及统计对象：** T35 paired nested, two explicitly identified targets

**文件：** [ST04_Checkpoint_selected_identity_sensitivity_v0.1](02_Supplement_Candidates/ST04_Checkpoint_selected_identity_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST04_Checkpoint_selected_identity_sensitivity_v0.1.csv)。

## ST05 · Candidate_pool_sensitivity


**English caption / note:** Four-CNN versus five-model candidate pools at the same five splits and three seeds. Intervals are existing paired nested sensitivity intervals. Selected-identity changes reflect candidate-pool dependence.


**算法及统计对象：** Existing candidate-pool sensitivity; no architecture-superiority inference

**文件：** [ST05_Candidate_pool_sensitivity_v0.1](02_Supplement_Candidates/ST05_Candidate_pool_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST05_Candidate_pool_sensitivity_v0.1.csv)。

## ST06 · Swin_input_recipe_sensitivity


**English caption / note:** Swin weight-evaluation recipe minus shared recipe, on the same 15 contexts per dataset. Existing paired nested intervals and conditional five-split sign-flip tests; the minimum two-sided exact p is 0.0625, and Holm adjusts two dataset comparisons.


**算法及统计对象：** Existing T38 paired nested / exact conditional p

**文件：** [ST06_Swin_input_recipe_sensitivity_v0.1](02_Supplement_Candidates/ST06_Swin_input_recipe_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST06_Swin_input_recipe_sensitivity_v0.1.csv)。

## ST07 · Common_training_seed_sensitivity


**English caption / note:** All five original seeds compared with the shared subset 42/52/62. Common-seed results reuse original runs. Budget agreement uses the same five/five split partition definition with the indicated seed pool.


**算法及统计对象：** Existing T44 nested LSO point summaries / exact budget

**文件：** [ST07_Common_training_seed_sensitivity_v0.1](02_Supplement_Candidates/ST07_Common_training_seed_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST07_Common_training_seed_sensitivity_v0.1.csv)。

## ST08 · Reference_definition_sensitivity


**English caption / note:** LSO excludes every occurrence of the held-out original split identity. Crossed LSO intervals use the existing identity-correct sampler. Full-grid agreement intervals retain the existing nested sampler with reference re-estimation; targets and algorithms are distinct.


**算法及统计对象：** Full-grid T36 nested; LSO point T36 and CI T37 crossed

**文件：** [ST08_Reference_definition_sensitivity_v0.1](02_Supplement_Candidates/ST08_Reference_definition_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST08_Reference_definition_sensitivity_v0.1.csv)。

## ST09 · Evaluation_budget_details


**English caption / note:** All primary budget cells. Partition percentiles describe the 252 finite oriented partitions, not population CIs. Mean-rank reference is an alternative ranking rule; bootstrap-reference expected agreement uses the existing nested reference resampling.


**算法及统计对象：** Existing exact finite-grid budget; nested bootstrap reference remains labeled

**文件：** [ST09_Evaluation_budget_details_v0.1](02_Supplement_Candidates/ST09_Evaluation_budget_details_v0.1.pdf)；[数据](02_Supplement_Candidates/ST09_Evaluation_budget_details_v0.1.csv)。

## ST10 · Original_mixed_effects_variance_structure


**English caption / note:** Original four-stratum reported fallback: BA ~ model + (1|split) + (1|model:split). Variance shares describe these fitted models and are not causal attribution. This is a display of existing original outputs, not a new extension mixed-model fit.


**算法及统计对象：** Existing original mixed-effects outputs; no re-fit

**文件：** [ST10_Original_mixed_effects_variance_structure_v0.1](02_Supplement_Candidates/ST10_Original_mixed_effects_variance_structure_v0.1.pdf)；[数据](02_Supplement_Candidates/ST10_Original_mixed_effects_variance_structure_v0.1.csv)。

## ST11 · Mixed_model_full_and_fallback_status


**English caption / note:** Existing original full and reported fallback fits. Full models include a seed random intercept and are singular; the reported split and model×split fallback fits are nonsingular. These are diagnostics of the existing fits.


**算法及统计对象：** Original stored status; no independent-person review claimed

**文件：** [ST11_Mixed_model_full_and_fallback_status_v0.1](02_Supplement_Candidates/ST11_Mixed_model_full_and_fallback_status_v0.1.pdf)；[数据](02_Supplement_Candidates/ST11_Mixed_model_full_and_fallback_status_v0.1.csv)。

## ST12 · Original_selected_epoch_distribution


**English caption / note:** Original submitted runs only. Values describe the selected checkpoint epoch, not necessarily the total number of completed training epochs. Rule B selected epoch is fixed at 60.


**算法及统计对象：** Existing epoch records, not reconstructed missing metadata

**文件：** [ST12_Original_selected_epoch_distribution_v0.1](02_Supplement_Candidates/ST12_Original_selected_epoch_distribution_v0.1.pdf)；[数据](02_Supplement_Candidates/ST12_Original_selected_epoch_distribution_v0.1.csv)。

## ST13 · Recorded_per_run_wall_clock


**English caption / note:** Reported wall-clock within each recorded hardware and timing scope. Original: run-entry through export; extension: after model/data-loader initialization through export/hashing. Different scopes and parallelism prevent direct hardware-speed comparison. Exact source scope is recorded in T46.


**算法及统计对象：** Existing B2 operational record; no speed-ranking analysis

**文件：** [ST13_Recorded_per_run_wall_clock_v0.1](02_Supplement_Candidates/ST13_Recorded_per_run_wall_clock_v0.1.pdf)；[数据](02_Supplement_Candidates/ST13_Recorded_per_run_wall_clock_v0.1.csv)。

## ST14 · Original_model_parameter_counts


**English caption / note:** Original-dataset classifier configurations only. Parameter counts are architectural records; neither analysis-host package versions nor unrecorded extension parameter counts are inferred from them.


**算法及统计对象：** Existing original model record; training and analysis software not conflated

**文件：** [ST14_Original_model_parameter_counts_v0.1](02_Supplement_Candidates/ST14_Original_model_parameter_counts_v0.1.pdf)；[数据](02_Supplement_Candidates/ST14_Original_model_parameter_counts_v0.1.csv)。

## RT01 · Bootstrap_scheme_comparison


**English caption / note:** Frequencies refer to the same reported full-grid reference model per stratum. Crossed is the adopted scheme; nested and split-only are sensitivities, flat is original replay. Original flat values are not available for the two new datasets.


**算法及统计对象：** Existing frequencies; q data CI not computed

**文件：** [RT01_Bootstrap_scheme_comparison_v0.1](03_Response_Reserve/RT01_Bootstrap_scheme_comparison_v0.1.pdf)；[数据](03_Response_Reserve/RT01_Bootstrap_scheme_comparison_v0.1.csv)。

## RT02 · Observed_and_aggregate_selection_evidence


**English caption / note:** f is observed single-context frequency; q is a resampled aggregate ranking frequency. Full-grid top-two mean difference is distinct from the within-context top-two difference in Figure2.


**算法及统计对象：** Existing point estimates, no extra inference

**文件：** [RT02_Observed_and_aggregate_selection_evidence_v0.1](03_Response_Reserve/RT02_Observed_and_aggregate_selection_evidence_v0.1.pdf)；[数据](03_Response_Reserve/RT02_Observed_and_aggregate_selection_evidence_v0.1.csv)。

## RT03 · Existing_difference_instability_association


**English caption / note:** The reported dataset-adjusted logistic association uses standardized negative log10 of the context-wise difference plus 1e-6. The interval contains 1. This is an existing supplementary association, not a causal or threshold estimate.


**算法及统计对象：** Existing identity-correct D01 replay; not a new association fit

**文件：** [RT03_Existing_difference_instability_association_v0.1](03_Response_Reserve/RT03_Existing_difference_instability_association_v0.1.pdf)；[数据](03_Response_Reserve/RT03_Existing_difference_instability_association_v0.1.csv)。

## RT04 · Analysis_stratum_inventory


**English caption / note:** Original completed primary matrix: 800 runs. Extension matrix: 300 runs. The 16 analysis strata overlap; their context/run counts are not additive independent sample sizes.


**算法及统计对象：** Existing B1 inventory and reuse labels

**文件：** [RT04_Analysis_stratum_inventory_v0.1](03_Response_Reserve/RT04_Analysis_stratum_inventory_v0.1.pdf)；[数据](03_Response_Reserve/RT04_Analysis_stratum_inventory_v0.1.csv)。

## SF01 · Absolute_BA_intervals


**English caption / note:** Model mean BA and paired crossed percentile 95% intervals (10,000 resamples). Panel x-ranges differ. Alphabetical/model-family display order does not identify a universal best architecture.


**算法及统计对象：** Existing crossed mean intervals T37

**文件：** [SF01_Absolute_BA_intervals_v0.1](02_Supplement_Candidates/SF01_Absolute_BA_intervals_v0.1.pdf)；[数据](02_Supplement_Candidates/SF01_Absolute_BA_intervals_v0.1.csv)。

## SF02 · Paired_BA_contrast_intervals


**English caption / note:** Six fixed-model paired BA differences per primary stratum. Intervals retain the existing nested algorithm. The zero line indicates equal BA for the specified fixed contrast; significance and multiplicity are reported separately in ST02.


**算法及统计对象：** Existing T33 paired intervals; no new tests

**文件：** [SF02_Paired_BA_contrast_intervals_v0.1](02_Supplement_Candidates/SF02_Paired_BA_contrast_intervals_v0.1.pdf)；[数据](02_Supplement_Candidates/SF02_Paired_BA_contrast_intervals_v0.1.csv)。

## SF03 · Checkpoint_policy_effect_intervals


**English caption / note:** Original paired B-minus-A BA effects with existing nested 95% intervals. Checkpoint rules also differ in training duration/optimization trajectory. Holm-adjusted tests are in ST03.


**算法及统计对象：** Existing T34 nested paired CI

**文件：** [SF03_Checkpoint_policy_effect_intervals_v0.1](02_Supplement_Candidates/SF03_Checkpoint_policy_effect_intervals_v0.1.pdf)；[数据](02_Supplement_Candidates/SF03_Checkpoint_policy_effect_intervals_v0.1.csv)。

## SF04 · Candidate_pool_and_input_sensitivity


**English caption / note:** Left: selected-identity changes when adding Swin to the four-CNN pool, at the same 15 contexts. Right: same-Swin weight-evaluation minus shared input-recipe BA difference. Intervals retain existing paired nested methods; the two panels use different targets and units.


**算法及统计对象：** Existing paired sensitivities, not crossed relabeling

**文件：** [SF04_Candidate_pool_and_input_sensitivity_v0.1](02_Supplement_Candidates/SF04_Candidate_pool_and_input_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/SF04_Candidate_pool_and_input_sensitivity_v0.1.csv)。

## SF05 · Common_seed_pool_sensitivity


**English caption / note:** Original all-five seed pool versus shared seeds 42/52/62. Lines link estimates from reused original runs. Left: identity-correct LSO agreement; right: five-split/three-seed agreement with a held-out split block.


**算法及统计对象：** Existing descriptive estimates; no new intervals

**文件：** [SF05_Common_seed_pool_sensitivity_v0.1](02_Supplement_Candidates/SF05_Common_seed_pool_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/SF05_Common_seed_pool_sensitivity_v0.1.csv)。

## SF06 · Original_fallback_variance_structure


**English caption / note:** Variance proportions from the original reported nonsingular fallback mixed model in four original strata. Split, model×split and residual components sum to one within fit. The full model with a seed random intercept was singular. These proportions describe the fitted variance structure and do not establish causal attribution.


**算法及统计对象：** Original stored mixed-model output, no refit

**文件：** [SF06_Original_fallback_variance_structure_v0.1](02_Supplement_Candidates/SF06_Original_fallback_variance_structure_v0.1.pdf)；[数据](02_Supplement_Candidates/SF06_Original_fallback_variance_structure_v0.1.csv)。

## SF07 · Budget_reference_sensitivity


**English caption / note:** Three-seed budget sensitivity to reference construction. Mean-BA and mean-rank references select a top model in the held-out block; bootstrap expected agreement averages against the held-out model-selection distribution from the existing nested reference bootstrap (2,000 resamples). These are different reference-specific quantities.


**算法及统计对象：** Finite exact discovery enumeration; nested bootstrap reference expectation

**文件：** [SF07_Budget_reference_sensitivity_v0.1](02_Supplement_Candidates/SF07_Budget_reference_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/SF07_Budget_reference_sensitivity_v0.1.csv)。

## RF01 · Bootstrap_design_comparison


**English caption / note:** Reference-model q across three resampling schemes at the same evaluation results. Scheme choice follows the frozen crossed-factor design, not the largest numerical frequency. No data confidence interval for q is depicted.


**算法及统计对象：** Existing point frequencies; no new uncertainty analysis

**文件：** [RF01_Bootstrap_design_comparison_v0.1](03_Response_Reserve/RF01_Bootstrap_design_comparison_v0.1.pdf)；[数据](03_Response_Reserve/RF01_Bootstrap_design_comparison_v0.1.csv)。

## RF02 · Single_context_and_aggregate_frequencies


**English caption / note:** Observed frequency f and crossed-bootstrap frequency q for the full-grid reference model. f uses 50 or 30 observed contexts; q uses 10,000 resampled aggregated rankings. Connecting lines aid comparison of distinct quantities and are not paired-effect estimates.


**算法及统计对象：** Existing descriptive f/q; no statistical test between them

**文件：** [RF02_Single_context_and_aggregate_frequencies_v0.1](03_Response_Reserve/RF02_Single_context_and_aggregate_frequencies_v0.1.pdf)；[数据](03_Response_Reserve/RF02_Single_context_and_aggregate_frequencies_v0.1.csv)。

## RF03 · Existing_difference_instability_association


**English caption / note:** Existing dataset-adjusted logistic association between identity-correct LSO discordance and standardized negative log10 of the context-wise top-two BA difference plus 1e-6. OR=2.022 with interval [0.597,5.763]. The interval includes 1; this is associational evidence, not a threshold or causal estimate.


**算法及统计对象：** Existing D01 replay and data interval, not newly fitted

**文件：** [RF03_Existing_difference_instability_association_v0.1](03_Response_Reserve/RF03_Existing_difference_instability_association_v0.1.pdf)；[数据](03_Response_Reserve/RF03_Existing_difference_instability_association_v0.1.csv)。

## 章节编号核对

本版按原M目标修正候选材料与章节对应；图、表、数字及原英文图注均保留。具体旧新映射见RQ论证核对记录，正式稿件采用继续分别追踪。


## 本轮已采用的Figure2a/b图注（替代上文原CF02独立分布图注）

# Figure2扩展候选：2a与2b v2.0.5

Figure 2: context top-two BA difference (a) and rule/pool sensitivity (b).

**English caption:** Performance separation and sensitivity of model selection to evaluation conditions. (a) Distribution of the difference between the two highest balanced-accuracy (BA) values within each observed split–seed context, in percentage points (pp). The leading pair is reselected within each context. Boxes show the median and interquartile range; whiskers extend to 1.5 interquartile ranges, and points show all observed contexts. Subplot y-ranges differ. (b) Fractions of matched split–seed contexts selecting a different model under alternative evaluation conditions. The checkpoint selection rule comparison uses 10 splits and five seeds per original dataset. The candidate-pool comparison adds shared-input Swin-T to four CNNs using five splits and three seeds per extension dataset. Horizontal bars retain existing paired nested percentile 95% intervals. The two condition comparisons are interpreted separately; their fractions describe changes in selected model identity, not changes in discordance rates.

来源：2a逐行复制原CF02数据/T52；2b逐行复制原CF04数据/T83（T57/T58重排）。不增加实验、重采样或统计检验。所有字段、原始数值和区间保持来源版本。2a的数据位置分布与2b的条件对照比例分别定义；固定模型显著性分析在ST02，A/B BA差在ST03。

CF04独立Figure4方案为历史候选，其四行数据已纳入Figure2b，不再作为独立正文图编号。ST05完整补充数据增列已有同15contexts一致率，版式待补充取舍；RT03完整关联按Q01进入补充。
