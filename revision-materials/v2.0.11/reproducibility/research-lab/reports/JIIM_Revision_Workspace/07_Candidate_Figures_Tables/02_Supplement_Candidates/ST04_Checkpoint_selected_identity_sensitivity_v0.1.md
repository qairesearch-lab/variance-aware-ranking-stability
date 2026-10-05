# ST04: Checkpoint selected identity sensitivity

Archived table rendering v0.1.

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

Paired A/B comparisons within identical split–seed contexts. ALL rows report a changed-winner fraction; model rows report B-minus-A observed selection-frequency differences. Both preserve the existing nested interval algorithm.
