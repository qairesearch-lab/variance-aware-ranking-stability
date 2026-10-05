# D03 图表统计覆盖说明 v0.1

## Material Passport

- ID：JIIM-D03-COVERAGE-v0.1；日期：2026-10-03
- 状态：方法与输出定位完成；实际图/稿/回信整合待B4
- 本说明对应运行前登记的D03覆盖CSV；不虚构实际修回稿页行。

## Table 1 — RQ1/RQ3

- 关联意见：E.1;R2.7;R3.5
- 科学问题：Within-stratum selection stability and budget/reference sensitivity
- 单位：split x seed within dataset/policy
- 效应：full-grid and LSO agreement; paired bootstrap selection frequency; disjoint aggregate budget agreement
- 区间：10000 hierarchical replicates for agreement; finite partition spread for budget, not CI
- 零假设适用性：No single meaningful null for entire summary table
- 多重性：Pairwise BA tests separately registered; no fabricated omnibus stability p value
- 输出：D02_ranking_stability_v0.1.csv;D04_budget_summary_v0.1.csv
- 限制：Fixed datasets; few split clusters; overlapping source images; reference definitions differ

## Figure 1 — RQ1

- 关联意见：E.1;R2.7
- 科学问题：How often does each candidate rank first across observed contexts?
- 单位：paired context indicators resampled by split then seed
- 效应：observed selection frequency
- 区间：paired hierarchical percentile 95%
- 零假设适用性：No prespecified scientific basis for uniform winner frequency
- 多重性：No p-value family for descriptive proportions
- 输出：D02_model_performance_selection_v0.1.csv
- 限制：Zero observed frequencies give degenerate nonparametric intervals, not structural impossibility

## Figure 2 — RQ1/RQ2

- 关联意见：E.1;R2.7;R3.5
- 科学问题：Distribution of BA difference between the two highest-ranked candidates
- 单位：paired candidate vector per context; top two reselected in each context
- 效应：nonnegative context order-statistic difference
- 区间：hierarchical interval for mean context difference plus distribution plot
- 零假设适用性：Testing gap > 0 is not a model-superiority test
- 多重性：All six prespecified pairwise model BA differences per primary stratum have Holm; secondary global36 shown
- 输出：D02_contexts_v0.1.csv;D02_paired_model_BA_differences_v0.1.csv;D01_margin_implementation_replay_v0.1.csv
- 限制：Candidate pool dependent; association not causal; overlapping intervals not an equality test

## Figure 3 — RQ3

- 关联意见：E.1;R2.3;R3.4
- 科学问题：Does aggregating more split/seed evaluations recover the held-out block ranking?
- 单位：252 oriented 5/5 split partitions; aggregate candidate means per budget subset
- 效应：aggregate winner agreement and full-rank/tau sensitivity
- 区间：exact finite-grid partition ranges/quantiles; no population CI
- 零假设适用性：No binomial test on reused subsets or partitions
- 多重性：All budget cells reported, no post hoc threshold selection
- 输出：D04_budget_summary_v0.1.csv;D04_paired_budget_changes_v0.1.csv
- 限制：Split IDs disjoint but source images can overlap; maximum discovery5 does not establish universal convergence

## Policy supplementary table — RQ2

- 关联意见：E.1;R2.7;R2.4
- 科学问题：Checkpoint-rule sensitivity within identical split/seed/model
- 单位：paired A/B runs; split-average contrasts for conditional sign flip
- 效应：BA B-minus-A and selected-identity change fraction
- 区间：paired hierarchical percentile 95%
- 零假设适用性：Symmetric sign exchangeability of split-mean BA differences under no policy difference
- 多重性：8 original dataset-model policy BA comparisons: Holm
- 输出：D02_policy_B_minus_A_v0.1.csv;D02_policy_selection_sensitivity_v0.1.csv
- 限制：No causal attribution of all observed ranking instability

## Swin supplementary sensitivity — RQ1/RQ2 generalization boundary

- 关联意见：R1.1;R2.2;R3.3
- 科学问题：Candidate pool and input-recipe dependence at the same15 contexts
- 单位：5 splits x3 seeds, shared contexts/labels
- 效应：selected-identity switch fraction and paired input-recipe BA difference
- 区间：paired hierarchical percentile 95%
- 零假设适用性：Conditional sign symmetry for input-recipe BA contrast only
- 多重性：2 dataset input-recipe contrasts: Holm; pool comparison is descriptive
- 输出：D02_candidate_pool_sensitivity_v0.1.csv;D02_swin_input_sensitivity_v0.1.csv
- 限制：Only5 split clusters; smallest two-sided exact p=.0625; not an architecture contest

