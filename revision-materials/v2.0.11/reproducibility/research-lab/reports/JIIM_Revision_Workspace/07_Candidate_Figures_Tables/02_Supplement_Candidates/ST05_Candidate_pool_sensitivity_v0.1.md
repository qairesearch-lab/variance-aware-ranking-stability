# ST05: Candidate pool sensitivity

Archived table rendering v0.1.

| Dataset | Swin input | Contexts | Selection changed | 95% paired CI | Mean top-two ΔBA: four / five (pp) |
| --- | --- | --- | --- | --- | --- |
| ISIC2019 | shared | 15 | 0.400 | [0.067, 0.733] | 2.833 / 2.527 |
| ISIC2019 | swin_weight_eval | 15 | 0.667 | [0.400, 0.933] | 2.833 / 2.726 |
| MURA | shared | 15 | 0.800 | [0.533, 1.000] | 0.986 / 1.088 |
| MURA | swin_weight_eval | 15 | 0.867 | [0.600, 1.000] | 0.986 / 1.084 |

Four-CNN versus five-model candidate pools at the same five splits and three seeds. Intervals are existing paired nested sensitivity intervals. Selected-identity changes reflect candidate-pool dependence.
