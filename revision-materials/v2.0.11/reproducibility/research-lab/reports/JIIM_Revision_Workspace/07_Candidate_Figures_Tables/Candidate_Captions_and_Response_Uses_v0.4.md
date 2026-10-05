# Candidate captions and reviewer uses v0.4

日期2026-10-03。所有材料为候选，未修改原稿；正文候选不等于最终採用。

## CT01 · Model_selection_stability_Table1

建议位置：正文候选；意见：E.1;R3.5;R3.4;R2.10;R1.1;R2.2；修订：M23;M24;M30;M35;M40;M41。

**English caption / note:** The reference top model is defined by full-grid mean balanced accuracy (BA). Agreement/discordance intervals use 10,000 paired split-then-seed percentile resamples with the reference re-estimated per draw; discordance is 1 minus agreement. q is a point estimate from 10,000 paired crossed split–seed ranking resamples. The 5×3 column uses finite-grid agreement with a held-out five-split reference block, averaged over 252 oriented partitions; source images may overlap across splits. Original seed subsets come from five seeds; extension subsets from three.

**用途说明：** 六列六主层；扩展并入原表；候选模型名称只定位参考，不作模型优胜结论。

**算法及统计对象：** agreement/discordance: existing nested T36; q: adopted crossed T37; budget: finite exact T40

**文件：** [CT01_Model_selection_stability_Table1_v0.1](01_Main_Candidates/CT01_Model_selection_stability_Table1_v0.1.pdf)；[数据](01_Main_Candidates/CT01_Model_selection_stability_Table1_v0.1.csv)。

## CF01 · Observed_selection_frequency_Figure1

建议位置：正文候选；意见：E.1;R2.6;R2.10;R3.5；修订：M30;M41。

**English caption / note:** Observed top-model selection frequencies across 50 original or 30 extension split–seed contexts per stratum. R18/R50/D121/E-B0 denote ResNet-18, ResNet-50, DenseNet-121 and EfficientNet-B0. Error bars are paired crossed split–seed percentile 95% intervals (10,000 replicates) for f, not for q. Zero observed frequencies may yield degenerate nonparametric intervals.

**用途说明：** 直接回应单次选择稳定性，保留Figure1统计对象。

**算法及统计对象：** f point: T31; f data CI: crossed T37; no q CI

**文件：** [CF01_Observed_selection_frequency_Figure1_v0.1](01_Main_Candidates/CF01_Observed_selection_frequency_Figure1_v0.1.pdf)；[数据](01_Main_Candidates/CF01_Observed_selection_frequency_Figure1_v0.1.csv)。

## CF02 · Context_top_two_BA_difference_Figure2

建议位置：正文候选；意见：E.1;R1.1;R2.10；修订：M28;M31;M41。

**English caption / note:** Distribution of the difference between the two highest BA values within each observed context. BA differences are shown in percentage points (pp). Boxes show the median and interquartile range; whiskers extend to 1.5 interquartile ranges, and points show all contexts. The leading pair is reselected within each context. Panel y-ranges differ to preserve readability. These distributions differ from the difference between full-grid model means.

**用途说明：** 沿用前二性能差的原图用途，帮助解释排名变化；不检验非负差来宣称模型优胜。

**算法及统计对象：** Observed context order statistic; box/whiskers are not confidence intervals

**文件：** [CF02_Context_top_two_BA_difference_Figure2_v0.1](01_Main_Candidates/CF02_Context_top_two_BA_difference_Figure2_v0.1.pdf)；[数据](01_Main_Candidates/CF02_Context_top_two_BA_difference_Figure2_v0.1.csv)。

## CF03 · Aggregate_evaluation_budget_Figure3

建议位置：正文候选；意见：E.1;R2.3;R3.4;R2.10；修订：M27;M33;M41;M45。

**English caption / note:** Top-model agreement as a function of discovery splits and training seeds. Each point averages all eligible discovery subsets within 252 oriented five/five split-ID partitions. The reference is mean BA on the five held-out split IDs and all available seeds. Shading for the three-seed line is the 2.5th–97.5th percentile spread across the finite set of partitions, not a population confidence interval. Training-run budget is four times splits times seeds.

**用途说明：** 用实际聚合预算回答RQ3；不把q改名为预算一致率。

**算法及统计对象：** Finite grid exact aggregation; split IDs disjoint, source images can overlap; no population CI

**文件：** [CF03_Aggregate_evaluation_budget_Figure3_v0.1](01_Main_Candidates/CF03_Aggregate_evaluation_budget_Figure3_v0.1.pdf)；[数据](01_Main_Candidates/CF03_Aggregate_evaluation_budget_Figure3_v0.1.csv)。

## ST01 · Absolute_BA_uncertainty

建议位置：补充候选；意见：R2.7;E.1；修订：M29;M35;M38。

**English caption / note:** Mean BA and context-level sample SD (ddof=1). Mean intervals use paired crossed split–seed resampling. The 16 strata include reused subsets and candidate/input sensitivities; rows must not be summed as independent training runs.

**用途说明：** 直接回应绝对性能变异；同模型跨评估条件的统计描述，不用于最好架构排名。

**算法及统计对象：** Mean/SD T31; crossed mean CI T37

**文件：** [ST01_Absolute_BA_uncertainty_v0.1](02_Supplement_Candidates/ST01_Absolute_BA_uncertainty_v0.1.pdf)；[数据](02_Supplement_Candidates/ST01_Absolute_BA_uncertainty_v0.1.csv)。

## ST02 · Fixed_model_paired_BA_contrasts

建议位置：补充候选；意见：E.1;R2.7；修订：M25;M29;M31。

**English caption / note:** Six alphabetically oriented fixed-model contrasts per primary stratum. Intervals retain the existing paired split-then-seed procedure. Exact sign-flip p-values use split-average contrasts under conditional joint sign symmetry; Holm adjustment is within each six-comparison stratum. These are separate from the context-wise top-two order statistic.

**用途说明：** 为编辑显著性要求提供明确配对问题的证据；不把Table1变成优胜模型榜。

**算法及统计对象：** T33 existing nested paired CI; conditional exact sign-flip; within-stratum Holm

**文件：** [ST02_Fixed_model_paired_BA_contrasts_v0.1](02_Supplement_Candidates/ST02_Fixed_model_paired_BA_contrasts_v0.1.pdf)；[数据](02_Supplement_Candidates/ST02_Fixed_model_paired_BA_contrasts_v0.1.csv)。

## ST03 · Checkpoint_policy_BA_sensitivity

建议位置：补充候选；意见：R2.4;E.1；修订：M20;M34;M44。

**English caption / note:** Original ten-split/five-seed paired B-minus-A effects. Paired nested intervals and conditional split-mean sign-flip tests retain the B1 algorithm; Holm family contains eight dataset-model comparisons. Rule A and rule B also differ in training duration/trajectory.

**用途说明：** 回应checkpoint复合因素及策略敏感性；不声称分离出纯checkpoint因果效应。

**算法及统计对象：** Existing T34, not recomputed or relabeled crossed

**文件：** [ST03_Checkpoint_policy_BA_sensitivity_v0.1](02_Supplement_Candidates/ST03_Checkpoint_policy_BA_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST03_Checkpoint_policy_BA_sensitivity_v0.1.csv)。

## ST04 · Checkpoint_selected_identity_sensitivity

建议位置：补充候选；意见：R2.4;R2.6；修订：M20;M34;M44。

**English caption / note:** Paired A/B comparisons within identical split–seed contexts. ALL rows report a changed-winner fraction; model rows report B-minus-A observed selection-frequency differences. Both preserve the existing nested interval algorithm.

**用途说明：** 区分性能改变与选择身份改变。

**算法及统计对象：** T35 paired nested, two explicitly identified targets

**文件：** [ST04_Checkpoint_selected_identity_sensitivity_v0.1](02_Supplement_Candidates/ST04_Checkpoint_selected_identity_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST04_Checkpoint_selected_identity_sensitivity_v0.1.csv)。

## ST05 · Candidate_pool_sensitivity

建议位置：补充候选；意见：R1.3;R2.5；修订：M09;M18;M37;M44。

**English caption / note:** Four-CNN versus five-model candidate pools at the same five splits and three seeds. Intervals are existing paired nested sensitivity intervals. Selected-identity changes reflect candidate-pool dependence.

**用途说明：** 回答候选池改变是否影响稳定性；同15个配对条件，不宣称Swin代表所有现代架构。

**算法及统计对象：** Existing candidate-pool sensitivity; no architecture-superiority inference

**文件：** [ST05_Candidate_pool_sensitivity_v0.1](02_Supplement_Candidates/ST05_Candidate_pool_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST05_Candidate_pool_sensitivity_v0.1.csv)。

## ST06 · Swin_input_recipe_sensitivity

建议位置：补充候选；意见：R3.3;R2.5;R1.3；修订：M19;M38;M44。

**English caption / note:** Swin weight-evaluation recipe minus shared recipe, on the same 15 contexts per dataset. Existing paired nested intervals and conditional five-split sign-flip tests; the minimum two-sided exact p is 0.0625, and Holm adjusts two dataset comparisons.

**用途说明：** 回应预处理一致性与输入方案差异；不是新模型优劣比较。

**算法及统计对象：** Existing T38 paired nested / exact conditional p

**文件：** [ST06_Swin_input_recipe_sensitivity_v0.1](02_Supplement_Candidates/ST06_Swin_input_recipe_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST06_Swin_input_recipe_sensitivity_v0.1.csv)。

## ST07 · Common_training_seed_sensitivity

建议位置：补充候选；意见：R3.1;R3.3;R2.3；修订：M12;M35;M43;M44。

**English caption / note:** All five original seeds compared with the shared subset 42/52/62. Common-seed results reuse original runs. Budget agreement uses the same five/five split partition definition with the indicated seed pool.

**用途说明：** 说明原五seed与扩展三seed配置差异的统计敏感性；复用原run，不作为额外独立实验。

**算法及统计对象：** Existing T44 nested LSO point summaries / exact budget

**文件：** [ST07_Common_training_seed_sensitivity_v0.1](02_Supplement_Candidates/ST07_Common_training_seed_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST07_Common_training_seed_sensitivity_v0.1.csv)。

## ST08 · Reference_definition_sensitivity

建议位置：补充候选；意见：R2.3;R3.4；修订：M23;M27;M30。

**English caption / note:** LSO excludes every occurrence of the held-out original split identity. Crossed LSO intervals use the existing identity-correct sampler. Full-grid agreement intervals retain the existing nested sampler with reference re-estimation; targets and algorithms are distinct.

**用途说明：** 作为有限参考稳健性补充；不增加主表LSO列。

**算法及统计对象：** Full-grid T36 nested; LSO point T36 and CI T37 crossed

**文件：** [ST08_Reference_definition_sensitivity_v0.1](02_Supplement_Candidates/ST08_Reference_definition_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/ST08_Reference_definition_sensitivity_v0.1.csv)。

## ST09 · Evaluation_budget_details

建议位置：补充候选；意见：R2.3;R3.4;E.1；修订：M27;M33;M45。

**English caption / note:** All primary budget cells. Partition percentiles describe the 252 finite oriented partitions, not population CIs. Mean-rank reference is an alternative ranking rule; bootstrap-reference expected agreement uses the existing nested reference resampling.

**用途说明：** 保留完整预算而非挑阈值；完整CSV可作为附件，正文只择用Figure3。

**算法及统计对象：** Existing exact finite-grid budget; nested bootstrap reference remains labeled

**文件：** [ST09_Evaluation_budget_details_v0.1](02_Supplement_Candidates/ST09_Evaluation_budget_details_v0.1.pdf)；[数据](02_Supplement_Candidates/ST09_Evaluation_budget_details_v0.1.csv)。

## ST10 · Original_mixed_effects_variance_structure

建议位置：补充候选；意见：R2.8;R2.1；修订：M26;M32;M44。

**English caption / note:** Original four-stratum reported fallback: BA ~ model + (1|split) + (1|model:split). Variance shares describe these fitted models and are not causal attribution. This is a display of existing original outputs, not a new extension mixed-model fit.

**用途说明：** 解释原RQ2因子结构，保留原四层fallback结果；不作为扩展混合模型新拟合。

**算法及统计对象：** Existing original mixed-effects outputs; no re-fit

**文件：** [ST10_Original_mixed_effects_variance_structure_v0.1](02_Supplement_Candidates/ST10_Original_mixed_effects_variance_structure_v0.1.pdf)；[数据](02_Supplement_Candidates/ST10_Original_mixed_effects_variance_structure_v0.1.csv)。

## ST11 · Mixed_model_full_and_fallback_status

建议位置：补充候选；意见：R2.8；修订：M26;M32。

**English caption / note:** Existing original full and reported fallback fits. Full models include a seed random intercept and are singular; the reported split and model×split fallback fits are nonsingular. These are diagnostics of the existing fits.

**用途说明：** 直接解释full模型singular与fallback；不扩展新的方差研究。

**算法及统计对象：** Original stored status; no independent-person review claimed

**文件：** [ST11_Mixed_model_full_and_fallback_status_v0.1](02_Supplement_Candidates/ST11_Mixed_model_full_and_fallback_status_v0.1.pdf)；[数据](02_Supplement_Candidates/ST11_Mixed_model_full_and_fallback_status_v0.1.csv)。

## ST12 · Original_selected_epoch_distribution

建议位置：补充候选；意见：R2.4;R2.9;R3.3；修订：M20;M34;M39。

**English caption / note:** Original submitted runs only. Values describe the selected checkpoint epoch, not necessarily the total number of completed training epochs. Rule B selected epoch is fixed at 60.

**用途说明：** 解释RuleA/B包含训练时长因素；原数据selected epoch不是全部训练完成epoch。

**算法及统计对象：** Existing epoch records, not reconstructed missing metadata

**文件：** [ST12_Original_selected_epoch_distribution_v0.1](02_Supplement_Candidates/ST12_Original_selected_epoch_distribution_v0.1.pdf)；[数据](02_Supplement_Candidates/ST12_Original_selected_epoch_distribution_v0.1.csv)。

## ST13 · Recorded_per_run_wall_clock

建议位置：补充候选；意见：R2.9；修订：M21;M39。

**English caption / note:** Reported wall-clock within each recorded hardware and timing scope. Original: run-entry through export; extension: after model/data-loader initialization through export/hashing. Different scopes and parallelism prevent direct hardware-speed comparison. Exact source scope is recorded in T46.

**用途说明：** 回应实际per-run时长；不同设备计时范围分别记录，不推断GPU优劣或新的预测时长。

**算法及统计对象：** Existing B2 operational record; no speed-ranking analysis

**文件：** [ST13_Recorded_per_run_wall_clock_v0.1](02_Supplement_Candidates/ST13_Recorded_per_run_wall_clock_v0.1.pdf)；[数据](02_Supplement_Candidates/ST13_Recorded_per_run_wall_clock_v0.1.csv)。

## ST14 · Original_model_parameter_counts

建议位置：补充候选；意见：R2.9;R3.3；修订：M18;M39。

**English caption / note:** Original-dataset classifier configurations only. Parameter counts are architectural records; neither analysis-host package versions nor unrecorded extension parameter counts are inferred from them.

**用途说明：** 保留已记录模型参数作为复现补充；不将不同数据集classifier数量混为同一个参数值。

**算法及统计对象：** Existing original model record; training and analysis software not conflated

**文件：** [ST14_Original_model_parameter_counts_v0.1](02_Supplement_Candidates/ST14_Original_model_parameter_counts_v0.1.pdf)；[数据](02_Supplement_Candidates/ST14_Original_model_parameter_counts_v0.1.csv)。

## RT01 · Bootstrap_scheme_comparison

建议位置：回信或备存；意见：R3.5；修订：M24;M40;M42。

**English caption / note:** Frequencies refer to the same reported full-grid reference model per stratum. Crossed is the adopted scheme; nested and split-only are sensitivities, flat is original replay. Original flat values are not available for the two new datasets.

**用途说明：** 本表只备存/回信使用；不恢复已撤销的稿件bootstrap补表。

**算法及统计对象：** Existing frequencies; q data CI not computed

**文件：** [RT01_Bootstrap_scheme_comparison_v0.1](03_Response_Reserve/RT01_Bootstrap_scheme_comparison_v0.1.pdf)；[数据](03_Response_Reserve/RT01_Bootstrap_scheme_comparison_v0.1.csv)。

## RT02 · Observed_and_aggregate_selection_evidence

建议位置：回信或备存；意见：R3.5;R1.1;R2.6；修订：M23;M31;M43。

**English caption / note:** f is observed single-context frequency; q is a resampled aggregate ranking frequency. Full-grid top-two mean difference is distinct from the within-context top-two difference in Figure2.

**用途说明：** 便于回信说明单次与聚合稳定性；指标合并会重复主表/原图，默认不另入正文。

**算法及统计对象：** Existing point estimates, no extra inference

**文件：** [RT02_Observed_and_aggregate_selection_evidence_v0.1](03_Response_Reserve/RT02_Observed_and_aggregate_selection_evidence_v0.1.pdf)；[数据](03_Response_Reserve/RT02_Observed_and_aggregate_selection_evidence_v0.1.csv)。

## RT03 · Existing_difference_instability_association

建议位置：回信或备存；意见：R1.1；修订：M28;M36;M44。

**English caption / note:** The reported dataset-adjusted logistic association uses standardized negative log10 of the context-wise difference plus 1e-6. The interval contains 1. This is an existing supplementary association, not a causal or threshold estimate.

**用途说明：** 已有关联结果可用于回应分差问题；区间包含1，不包装为普遍阈值或因果机制。

**算法及统计对象：** Existing identity-correct D01 replay; not a new association fit

**文件：** [RT03_Existing_difference_instability_association_v0.1](03_Response_Reserve/RT03_Existing_difference_instability_association_v0.1.pdf)；[数据](03_Response_Reserve/RT03_Existing_difference_instability_association_v0.1.csv)。

## RT04 · Analysis_stratum_inventory

建议位置：回信或备存；意见：R3.1;R3.3；修订：M11;M12;M13;M42。

**English caption / note:** Original completed primary matrix: 800 runs. Extension matrix: 300 runs. The 16 analysis strata overlap; their context/run counts are not additive independent sample sizes.

**用途说明：** 16分析层含复用子集，供回信核对设计；不把各层run相加。

**算法及统计对象：** Existing B1 inventory and reuse labels

**文件：** [RT04_Analysis_stratum_inventory_v0.1](03_Response_Reserve/RT04_Analysis_stratum_inventory_v0.1.pdf)；[数据](03_Response_Reserve/RT04_Analysis_stratum_inventory_v0.1.csv)。

## SF01 · Absolute_BA_intervals

建议位置：补充候选；意见：R2.7;E.1；修订：M29。

**English caption / note:** Model mean BA and paired crossed percentile 95% intervals (10,000 resamples). Panel x-ranges differ. Alphabetical/model-family display order does not identify a universal best architecture.

**用途说明：** 性能区间可回应R2.7，但与ST01重复；选图或表即可，不两者均入稿。

**算法及统计对象：** Existing crossed mean intervals T37

**文件：** [SF01_Absolute_BA_intervals_v0.1](02_Supplement_Candidates/SF01_Absolute_BA_intervals_v0.1.pdf)；[数据](02_Supplement_Candidates/SF01_Absolute_BA_intervals_v0.1.csv)。

## SF02 · Paired_BA_contrast_intervals

建议位置：补充候选；意见：E.1；修订：M25;M29。

**English caption / note:** Six fixed-model paired BA differences per primary stratum. Intervals retain the existing nested algorithm. The zero line indicates equal BA for the specified fixed contrast; significance and multiplicity are reported separately in ST02.

**用途说明：** 用于显示编辑所问统计比较；ST02有完整p值，优先选表而非把此图当模型榜。

**算法及统计对象：** Existing T33 paired intervals; no new tests

**文件：** [SF02_Paired_BA_contrast_intervals_v0.1](02_Supplement_Candidates/SF02_Paired_BA_contrast_intervals_v0.1.pdf)；[数据](02_Supplement_Candidates/SF02_Paired_BA_contrast_intervals_v0.1.csv)。

## SF03 · Checkpoint_policy_effect_intervals

建议位置：补充候选；意见：R2.4;E.1；修订：M20;M34;M44。

**English caption / note:** Original paired B-minus-A BA effects with existing nested 95% intervals. Checkpoint rules also differ in training duration/optimization trajectory. Holm-adjusted tests are in ST03.

**用途说明：** 与ST03二选一；显示复合策略差异，不作因果分解。

**算法及统计对象：** Existing T34 nested paired CI

**文件：** [SF03_Checkpoint_policy_effect_intervals_v0.1](02_Supplement_Candidates/SF03_Checkpoint_policy_effect_intervals_v0.1.pdf)；[数据](02_Supplement_Candidates/SF03_Checkpoint_policy_effect_intervals_v0.1.csv)。

## SF04 · Candidate_pool_and_input_sensitivity

建议位置：补充候选；意见：R1.3;R2.5;R3.3；修订：M37;M38;M44。

**English caption / note:** Left: selected-identity changes when adding Swin to the four-CNN pool, at the same 15 contexts. Right: same-Swin weight-evaluation minus shared input-recipe BA difference. Intervals retain existing paired nested methods; the two panels use different targets and units.

**用途说明：** 不同对象分面展示；新架构只支持候选池/输入敏感性，不代表架构优劣。

**算法及统计对象：** Existing paired sensitivities, not crossed relabeling

**文件：** [SF04_Candidate_pool_and_input_sensitivity_v0.1](02_Supplement_Candidates/SF04_Candidate_pool_and_input_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/SF04_Candidate_pool_and_input_sensitivity_v0.1.csv)。

## SF05 · Common_seed_pool_sensitivity

建议位置：补充候选；意见：R3.1;R3.3;R2.3；修订：M35;M43;M44。

**English caption / note:** Original all-five seed pool versus shared seeds 42/52/62. Lines link estimates from reused original runs. Left: identity-correct LSO agreement; right: five-split/three-seed agreement with a held-out split block.

**用途说明：** 与ST07二选一；说明相同run子集的敏感性，非独立复现实验。

**算法及统计对象：** Existing descriptive estimates; no new intervals

**文件：** [SF05_Common_seed_pool_sensitivity_v0.1](02_Supplement_Candidates/SF05_Common_seed_pool_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/SF05_Common_seed_pool_sensitivity_v0.1.csv)。

## SF06 · Original_fallback_variance_structure

建议位置：补充候选；意见：R2.8;R2.1；修订：M26;M32;M44。

**English caption / note:** Variance proportions from the original reported nonsingular fallback mixed model in four original strata. Split, model×split and residual components sum to one within fit. The full model with a seed random intercept was singular. These proportions describe the fitted variance structure and do not establish causal attribution.

**用途说明：** 与ST10/11配合解释fallback，优先表；原RQ2已有因子结构展示，不新增扩展拟合。

**算法及统计对象：** Original stored mixed-model output, no refit

**文件：** [SF06_Original_fallback_variance_structure_v0.1](02_Supplement_Candidates/SF06_Original_fallback_variance_structure_v0.1.pdf)；[数据](02_Supplement_Candidates/SF06_Original_fallback_variance_structure_v0.1.csv)。

## SF07 · Budget_reference_sensitivity

建议位置：补充候选；意见：R2.3;E.1；修订：M27;M33。

**English caption / note:** Three-seed budget sensitivity to reference construction. Mean-BA and mean-rank references select a top model in the held-out block; bootstrap expected agreement averages against the held-out model-selection distribution from the existing nested reference bootstrap (2,000 resamples). These are different reference-specific quantities.

**用途说明：** 回应mean-rank/bootstrap参考敏感性；不同参考定义分别标明，不替换主Figure3对象。

**算法及统计对象：** Finite exact discovery enumeration; nested bootstrap reference expectation

**文件：** [SF07_Budget_reference_sensitivity_v0.1](02_Supplement_Candidates/SF07_Budget_reference_sensitivity_v0.1.pdf)；[数据](02_Supplement_Candidates/SF07_Budget_reference_sensitivity_v0.1.csv)。

## RF01 · Bootstrap_design_comparison

建议位置：回信或备存；意见：R3.5；修订：M24;M40。

**English caption / note:** Reference-model q across three resampling schemes at the same evaluation results. Scheme choice follows the frozen crossed-factor design, not the largest numerical frequency. No data confidence interval for q is depicted.

**用途说明：** 用于回信解释频率为何变化；不恢复新增正文或bootstrap补表。

**算法及统计对象：** Existing point frequencies; no new uncertainty analysis

**文件：** [RF01_Bootstrap_design_comparison_v0.1](03_Response_Reserve/RF01_Bootstrap_design_comparison_v0.1.pdf)；[数据](03_Response_Reserve/RF01_Bootstrap_design_comparison_v0.1.csv)。

## RF02 · Single_context_and_aggregate_frequencies

建议位置：回信或备存；意见：R3.5;R2.6；修订：M23;M43。

**English caption / note:** Observed frequency f and crossed-bootstrap frequency q for the full-grid reference model. f uses 50 or 30 observed contexts; q uses 10,000 resampled aggregated rankings. Connecting lines aid comparison of distinct quantities and are not paired-effect estimates.

**用途说明：** 帮助审稿人理解两个分母与聚合步骤；不新增核心指标，不重复主表。

**算法及统计对象：** Existing descriptive f/q; no statistical test between them

**文件：** [RF02_Single_context_and_aggregate_frequencies_v0.1](03_Response_Reserve/RF02_Single_context_and_aggregate_frequencies_v0.1.pdf)；[数据](03_Response_Reserve/RF02_Single_context_and_aggregate_frequencies_v0.1.csv)。

## RF03 · Existing_difference_instability_association

建议位置：回信或备存；意见：R1.1；修订：M28;M36;M44。

**English caption / note:** Existing dataset-adjusted logistic association between identity-correct LSO discordance and standardized negative log10 of the context-wise top-two BA difference plus 1e-6. OR=2.022 with interval [0.597,5.763]. The interval includes 1; this is associational evidence, not a threshold or causal estimate.

**用途说明：** 为margin意见提供已有模型结果；区间含1，保留备存，不扩展成新机制结论。

**算法及统计对象：** Existing D01 replay and data interval, not newly fitted

**文件：** [RF03_Existing_difference_instability_association_v0.1](03_Response_Reserve/RF03_Existing_difference_instability_association_v0.1.pdf)；[数据](03_Response_Reserve/RF03_Existing_difference_instability_association_v0.1.csv)。

## 章节编号核对

本版按原M目标修正候选材料与章节对应；图、表、数字及原英文图注均保留。具体旧新映射见RQ论证核对记录，正式稿件采用继续分别追踪。


## 本轮已采用的Figure2a/b图注（替代上文原CF02独立分布图注）

# Figure2扩展候选：2a与2b v2.0.5

作者确认Figure2合并展示；正文仍1表3图，不另列Figure4。原前二BA差为2a，新增规则/候选池敏感性为2b。当前是候选排图，正文v2.0未整合。

**English caption:** Performance separation and sensitivity of model selection to evaluation conditions. (a) Distribution of the difference between the two highest balanced-accuracy (BA) values within each observed split–seed context, in percentage points (pp). The leading pair is reselected within each context. Boxes show the median and interquartile range; whiskers extend to 1.5 interquartile ranges, and points show all observed contexts. Subplot y-ranges differ. (b) Fractions of matched split–seed contexts selecting a different model under alternative evaluation conditions. The checkpoint selection rule comparison uses 10 splits and five seeds per original dataset. The candidate-pool comparison adds shared-input Swin-T to four CNNs using five splits and three seeds per extension dataset. Horizontal bars retain existing paired nested percentile 95% intervals. The two condition comparisons are interpreted separately; their fractions describe changes in selected model identity, not changes in discordance rates.

来源：2a逐行复制原CF02数据/T52；2b逐行复制原CF04数据/T83（T57/T58重排）。不增加实验、重采样或统计检验。所有字段、原始数值和区间保持来源版本。2a的数据位置分布与2b的条件对照比例分别定义；固定模型显著性分析在ST02，A/B BA差在ST03。

CF04独立Figure4方案为历史候选，其四行数据已纳入Figure2b，不再作为独立正文图编号。ST05完整补充数据增列已有同15contexts一致率，版式待补充取舍；RT03完整关联按Q01进入补充。
