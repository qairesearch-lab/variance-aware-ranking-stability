# S9候选：方差组成及拟合状态 v2.0.7

候选组合供Q05审阅；尚未采用或移入正式补充材料。仅整理已有数据，不新增统计。

## Panel A. Original stored variance summaries

| Stratum | Split variance share | Model×split share | Residual share | Model |
| --- | --- | --- | --- | --- |
| OrganAMNIST / A | 0.103 | 0.166 | 0.730 | Reported nonsingular fallback |
| OrganAMNIST / B | 0.128 | 0.248 | 0.625 | Reported nonsingular fallback |
| SIPaKMeD / A | 0.118 | 0.362 | 0.520 | Reported nonsingular fallback |
| SIPaKMeD / B | 0.159 | 0.341 | 0.500 | Reported nonsingular fallback |

## Panel B. Original full/fallback fit status

| Dataset / rule | Specification | Converged | Singular | Splits / seeds |
| --- | --- | --- | --- | --- |
| SIPaKMeD / A | sap_full_formula_attempt | True | True | 10 / 5 |
| SIPaKMeD / A | primary_reported_fallback_model | True | False | 10 / 5 |
| SIPaKMeD / B | sap_full_formula_attempt | True | True | 10 / 5 |
| SIPaKMeD / B | primary_reported_fallback_model | True | False | 10 / 5 |
| OrganAMNIST / A | sap_full_formula_attempt | True | True | 10 / 5 |
| OrganAMNIST / A | primary_reported_fallback_model | True | False | 10 / 5 |
| OrganAMNIST / B | sap_full_formula_attempt | True | True | 10 / 5 |
| OrganAMNIST / B | primary_reported_fallback_model | True | False | 10 / 5 |

## Candidate caption / note

All entries are original stored mixed-effects outputs. Variance shares summarize BA variation under the specified fit; they are not causal proportions of model-selection changes. Full and fallback specifications and their fitting status are reported together. This table contains no new mixed-effects fit for the extension datasets.
