# CT01: Model selection stability Table1

Archived table rendering v0.1.

| Dataset / rule | Reference top model | Single-context agreement (95% CI) | Discordance (95% CI) | Paired bootstrap frequency q | 5×3 budget agreement |
| --- | --- | --- | --- | --- | --- |
| OrganAMNIST / A (10×5; 4 models) | ResNet-18 | 0.620 [0.440, 0.800] | 0.380 [0.200, 0.560] | 0.9088 | 0.890 |
| OrganAMNIST / B (10×5; 4 models) | ResNet-50 | 0.600 [0.400, 0.780] | 0.400 [0.220, 0.600] | 0.7825 | 0.587 |
| SIPaKMeD / A (10×5; 4 models) | ResNet-18 | 0.520 [0.360, 0.740] | 0.480 [0.260, 0.640] | 0.8494 | 0.694 |
| SIPaKMeD / B (10×5; 4 models) | ResNet-50 | 0.440 [0.380, 0.680] | 0.560 [0.320, 0.620] | 0.5377 | 0.204 |
| ISIC2019 / A (10×3; 4 models) | ResNet-50 | 0.900 [0.733, 1.000] | 0.100 [0.000, 0.267] | 1.0000 | 1.000 |
| MURA / A (10×3; 4 models) | ResNet-50 | 0.600 [0.300, 0.800] | 0.400 [0.200, 0.700] | 0.7687 | 0.571 |

The reference top model is defined by full-grid mean balanced accuracy (BA). Agreement/discordance intervals use 10,000 paired split-then-seed percentile resamples with the reference re-estimated per draw; discordance is 1 minus agreement. q is a point estimate from 10,000 paired crossed split–seed ranking resamples. The 5×3 column uses finite-grid agreement with a held-out five-split reference block, averaged over 252 oriented partitions; source images may overlap across splits. Original seed subsets come from five seeds; extension subsets from three.
