# ST03: Checkpoint policy BA sensitivity

Archived table rendering v0.1.

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

Original ten-split/five-seed paired B-minus-A effects. Paired nested intervals and conditional split-mean sign-flip tests retain the B1 algorithm; Holm family contains eight dataset-model comparisons. Rule A and rule B also differ in training duration/trajectory.
