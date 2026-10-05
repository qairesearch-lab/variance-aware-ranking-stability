# S7候选：B/C/D参考与种子敏感性 v2.0.7

候选组合供Q05审阅；尚未采用或移入正式补充材料。仅整理已有数据，不新增统计。

## Panel B. Completed-grid and leave-one-split-out agreement

| Stratum | Full-grid agreement (nested CI) | Identity-correct LSO agreement (crossed CI) |
| --- | --- | --- |
| OrganAMNIST / A | 0.620 [0.440, 0.800] | 0.620 [0.220, 0.840] |
| OrganAMNIST / B | 0.600 [0.400, 0.780] | 0.600 [0.180, 0.800] |
| SIPaKMeD / A | 0.520 [0.360, 0.740] | 0.520 [0.100, 0.760] |
| SIPaKMeD / B | 0.440 [0.380, 0.680] | 0.220 [0.100, 0.700] |
| ISIC2019 / A | 0.900 [0.733, 1.000] | 0.900 [0.700, 1.000] |
| MURA / A | 0.600 [0.300, 0.800] | 0.600 [0.067, 0.900] |

## Panel C. Common training-seed subset

| Dataset / rule | Reference: all5 / common3 | LSO agreement: all5 / common3 | 5×3 agreement: all5 / common3 pool |
| --- | --- | --- | --- |
| OrganAMNIST / A | ResNet-18 / ResNet-18 | 0.620 / 0.533 | 0.890 / 0.460 |
| OrganAMNIST / B | ResNet-50 / ResNet-50 | 0.600 / 0.667 | 0.587 / 0.587 |
| SIPaKMeD / A | ResNet-18 / ResNet-18 | 0.520 / 0.467 | 0.694 / 0.635 |
| SIPaKMeD / B | ResNet-50 / ResNet-18 | 0.220 / 0.367 | 0.204 / 0.381 |

## Panel D. Representative budgets and reference sensitivity

| Stratum | Budget s×k | Four-model runs | Top agreement | Partition 2.5–97.5% spread | Complete-rank agreement | Mean-rank ref / bootstrap ref |
| --- | --- | --- | --- | --- | --- | --- |
| OrganAMNIST / A | 1×1 | 4 | 0.609 | [0.440, 0.760] | 0.571 | 0.583 / 0.574 |
| OrganAMNIST / A | 3×3 | 36 | 0.813 | [0.448, 0.990] | 0.813 | 0.768 / 0.713 |
| OrganAMNIST / A | 5×3 | 60 | 0.890 | [0.427, 1.000] | 0.890 | 0.838 / 0.770 |
| OrganAMNIST / B | 1×1 | 4 | 0.541 | [0.280, 0.709] | 0.541 | 0.569 / 0.520 |
| OrganAMNIST / B | 3×3 | 36 | 0.530 | [0.083, 0.830] | 0.530 | 0.609 / 0.503 |
| OrganAMNIST / B | 5×3 | 60 | 0.587 | [0.000, 1.000] | 0.587 | 0.684 / 0.521 |
| SIPaKMeD / A | 1×1 | 4 | 0.463 | [0.200, 0.640] | 0.279 | 0.406 / 0.449 |
| SIPaKMeD / A | 3×3 | 36 | 0.604 | [0.000, 0.970] | 0.575 | 0.467 / 0.544 |
| SIPaKMeD / A | 5×3 | 60 | 0.694 | [0.000, 1.000] | 0.693 | 0.521 / 0.598 |
| SIPaKMeD / B | 1×1 | 4 | 0.370 | [0.240, 0.480] | 0.228 | 0.370 / 0.399 |
| SIPaKMeD / B | 3×3 | 36 | 0.311 | [0.023, 0.547] | 0.309 | 0.326 / 0.389 |
| SIPaKMeD / B | 5×3 | 60 | 0.204 | [0.000, 0.600] | 0.204 | 0.230 / 0.339 |
| ISIC2019 / A | 1×1 | 4 | 0.900 | [0.800, 1.000] | 0.288 | 0.900 / 0.900 |
| ISIC2019 / A | 3×3 | 36 | 1.000 | [1.000, 1.000] | 0.547 | 1.000 / 1.000 |
| ISIC2019 / A | 5×3 | 60 | 1.000 | [1.000, 1.000] | 0.675 | 1.000 / 1.000 |
| MURA / A | 1×1 | 4 | 0.491 | [0.133, 0.667] | 0.319 | 0.419 / 0.471 |
| MURA / A | 3×3 | 36 | 0.513 | [0.000, 0.900] | 0.433 | 0.376 / 0.484 |
| MURA / A | 5×3 | 60 | 0.571 | [0.000, 1.000] | 0.508 | 0.373 / 0.490 |

## Candidate caption / note

Full-grid agreement intervals retain the existing nested procedure; identity-correct leave-one-split-out intervals use crossed resampling. Common-seed summaries retain their source procedures. Panel D lists 1×1, 3×3 and 5×3 for each primary stratum without selecting rows by outcome. Partition percentiles summarize the finite enumeration, not a population confidence interval. Mean-rank and the existing nested bootstrap reference remain separately labeled. All 130 budget rows are in the CSV attachment; the original S7 comparator-identity panel remains.
