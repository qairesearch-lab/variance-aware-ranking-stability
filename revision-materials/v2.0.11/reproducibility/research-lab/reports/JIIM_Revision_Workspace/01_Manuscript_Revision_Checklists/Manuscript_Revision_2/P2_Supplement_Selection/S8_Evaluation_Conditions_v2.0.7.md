# S8候选：评价条件敏感性 v2.0.7

候选组合供Q05审阅；尚未采用或移入正式补充材料。仅整理已有数据，不新增统计。

## Panel A. Paired Rule B minus Rule A BA

| Dataset | Model | B − A ΔBA (pp) | 95% paired CI (pp) | Holm p (eight pairs) |
| --- | --- | --- | --- | --- |
| OrganAMNIST | DenseNet-121 | 0.167 | [0.102, 0.236] | 0.0156 |
| OrganAMNIST | EfficientNet-B0 | 0.078 | [-0.004, 0.172] | 0.1875 |
| OrganAMNIST | ResNet-18 | 0.122 | [0.069, 0.178] | 0.0156 |
| OrganAMNIST | ResNet-50 | 0.197 | [0.130, 0.260] | 0.0156 |
| SIPaKMeD | DenseNet-121 | 0.409 | [0.095, 0.748] | 0.0469 |
| SIPaKMeD | EfficientNet-B0 | 0.140 | [-0.110, 0.357] | 0.2148 |
| SIPaKMeD | ResNet-18 | 0.269 | [-0.047, 0.558] | 0.1875 |
| SIPaKMeD | ResNet-50 | 0.522 | [0.219, 0.814] | 0.0195 |

## Panel B. Rule-related identity and reference sensitivity

| Dataset | Target | Statistic | Estimate | 95% paired CI |
| --- | --- | --- | --- | --- |
| OrganAMNIST | DenseNet-121 | B − A observed f | 0.000 | [0.000, 0.000] |
| OrganAMNIST | EfficientNet-B0 | B − A observed f | 0.000 | [0.000, 0.000] |
| OrganAMNIST | ResNet-18 | B − A observed f | -0.220 | [-0.440, 0.000] |
| OrganAMNIST | ResNet-50 | B − A observed f | 0.220 | [0.000, 0.440] |
| OrganAMNIST | Any winner change | Changed-context fraction | 0.460 | [0.280, 0.640] |
| SIPaKMeD | DenseNet-121 | B − A observed f | 0.020 | [-0.140, 0.180] |
| SIPaKMeD | EfficientNet-B0 | B − A observed f | 0.000 | [0.000, 0.000] |
| SIPaKMeD | ResNet-18 | B − A observed f | -0.080 | [-0.380, 0.200] |
| SIPaKMeD | ResNet-50 | B − A observed f | 0.060 | [-0.200, 0.320] |
| SIPaKMeD | Any winner change | Changed-context fraction | 0.560 | [0.360, 0.740] |

## Panel C. Candidate pools on the same 15 contexts

| Dataset | Swin input | Contexts | Selection changed | 95% paired CI | Mean top-two ΔBA: four / five (pp) | four_CNN_full_grid_agreement | five_model_full_grid_agreement | four_CNN_LSO_agreement | five_model_LSO_agreement | source_strata | LSO_source | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC2019 | shared | 15 | 0.400 | [0.067, 0.733] | 2.833 / 2.527 | 0.8666666666666667 | 0.40000000000000002 | 0.8666666666666667 | 0.20000000000000001 | isic2019_A_cnn4_pool_contexts;isic2019_A_pool5_shared | research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results_v0.3/D02_ranking_stability_v0.1.csv | same 15 contexts; identity change and within-pool agreement are different objects; no new contrast CI or test |
| ISIC2019 | swin_weight_eval | 15 | 0.667 | [0.400, 0.933] | 2.833 / 2.726 | 0.8666666666666667 | 0.66666666666666663 | 0.8666666666666667 | 0.66666666666666663 | isic2019_A_cnn4_pool_contexts;isic2019_A_pool5_swin_weight_eval | research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results_v0.3/D02_ranking_stability_v0.1.csv | same 15 contexts; identity change and within-pool agreement are different objects; no new contrast CI or test |
| MURA | shared | 15 | 0.800 | [0.533, 1.000] | 0.986 / 1.088 | 0.66666666666666663 | 0.80000000000000004 | 0.66666666666666663 | 0.80000000000000004 | mura_A_cnn4_pool_contexts;mura_A_pool5_shared | research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results_v0.3/D02_ranking_stability_v0.1.csv | same 15 contexts; identity change and within-pool agreement are different objects; no new contrast CI or test |
| MURA | swin_weight_eval | 15 | 0.867 | [0.600, 1.000] | 0.986 / 1.084 | 0.66666666666666663 | 0.8666666666666667 | 0.66666666666666663 | 0.8666666666666667 | mura_A_cnn4_pool_contexts;mura_A_pool5_swin_weight_eval | research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/analysis/results_v0.3/D02_ranking_stability_v0.1.csv | same 15 contexts; identity change and within-pool agreement are different objects; no new contrast CI or test |

## Panel D. Paired Swin-T input recipes

| Dataset | Paired contexts | Input ΔBA (pp) | 95% paired CI (pp) | Holm p (two pairs) |
| --- | --- | --- | --- | --- |
| ISIC2019 | 15 | 1.960 | [1.165, 3.226] | 0.1250 |
| MURA | 15 | -0.077 | [-0.562, 0.500] | 0.7500 |

## Candidate caption / note

Each panel identifies its own estimand and matched contexts. Rule contrasts use the separate eight-comparison Holm family. Candidate-pool comparisons report selected-identity changes and both completed-grid and leave-one-split-out agreement on the same five splits and three seeds. Shared-input and weight-specific-input analyses remain distinguishable. Input comparisons retain their two-comparison adjustment. Existing nested intervals and conditional tests are preserved; these analyses describe selection sensitivity and do not rank architecture families universally.
