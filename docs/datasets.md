# Dataset acquisition

Acquire image data from the original providers and follow their terms of use. This repository distributes split/metadata records and analysis results, not source images.

| Dataset | Original source | Evaluation unit |
|---|---|---|
| SIPaKMeD | https://www.cs.uoi.gr/~marina/sipakmed.html | Image |
| OrganAMNIST / MedMNIST | https://medmnist.com/ | Image |
| ISIC2019 | https://challenge.isic-archive.com/landing/2019/ | Image; lesion-grouped splits |
| MURA | https://stanfordmlgroup.github.io/competitions/mura/ | Study; patient-grouped splits |

Restore images to the relative paths in the dataset indices and split CSVs. OrganAMNIST input processing follows the original recorded image export and resize pipeline. ISIC2019 and MURA metadata include the frozen grouping/exclusion and duplicate-screen decisions. Pretrained weights are obtained through torchvision; checkpoints and image archives are not bundled.
