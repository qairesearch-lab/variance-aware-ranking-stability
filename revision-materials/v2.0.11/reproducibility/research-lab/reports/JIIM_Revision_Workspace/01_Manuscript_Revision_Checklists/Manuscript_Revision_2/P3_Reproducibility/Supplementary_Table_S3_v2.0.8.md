# Supplementary Table S3：复现规格素材 v2.0.8

## Metadata provenance

Recorded specifications from 1100 formal runs, parameter-shape checks of 300 extension checkpoints, 12 representative complete checkpoint hashes and the archived original parameter records. Training and statistical-analysis runtimes are reported separately.

## Recorded specifications

### Supplementary Table S3. Model definitions, parameter counts, training settings and execution records

**Panel A. Model definitions and trainable scope.**

| Model | Torchvision_call | Weights | Trainable_scope |
| --- | --- | --- | --- |
| ResNet-18 | torchvision.models.resnet18 | IMAGENET1K_V1 | layer4 + fc |
| ResNet-50 | torchvision.models.resnet50 | IMAGENET1K_V1 | layer4 + fc |
| DenseNet-121 | torchvision.models.densenet121 | IMAGENET1K_V1 | features.denseblock4 + features.norm5 + classifier |
| EfficientNet-B0 | torchvision.models.efficientnet_b0 | IMAGENET1K_V1 | features[-1] + classifier |
| Swin-T | torchvision.models.swin_t | IMAGENET1K_V1 | features[-1] + norm + head |

**Panel B. Parameter counts after classifier replacement.**

| Stage | Dataset | Model | Classes | Total | Trainable | Fraction |
| --- | --- | --- | --- | --- | --- | --- |
| original | SIPaKMeD | ResNet-18 | 5 | 11,179,077 | 8,396,293 | 0.751 |
| original | OrganAMNIST | ResNet-18 | 11 | 11,182,155 | 8,399,371 | 0.751 |
| original | SIPaKMeD | ResNet-50 | 5 | 23,518,277 | 14,974,981 | 0.637 |
| original | OrganAMNIST | ResNet-50 | 11 | 23,530,571 | 14,987,275 | 0.637 |
| original | SIPaKMeD | DenseNet-121 | 5 | 6,958,981 | 2,165,253 | 0.311 |
| original | OrganAMNIST | DenseNet-121 | 11 | 6,965,131 | 2,171,403 | 0.312 |
| original | SIPaKMeD | EfficientNet-B0 | 5 | 4,013,953 | 418,565 | 0.104 |
| original | OrganAMNIST | EfficientNet-B0 | 11 | 4,021,639 | 426,251 | 0.106 |
| extension | ISIC2019 | DenseNet-121 | 8 | 6,962,056 | 2,168,328 | 0.311 |
| extension | ISIC2019 | EfficientNet-B0 | 8 | 4,017,796 | 422,408 | 0.105 |
| extension | ISIC2019 | ResNet-18 | 8 | 11,180,616 | 8,397,832 | 0.751 |
| extension | ISIC2019 | ResNet-50 | 8 | 23,524,424 | 14,981,128 | 0.637 |
| extension | ISIC2019 | Swin-T | 8 | 27,525,506 | 14,191,544 | 0.516 |
| extension | MURA | DenseNet-121 | 2 | 6,955,906 | 2,162,178 | 0.311 |
| extension | MURA | EfficientNet-B0 | 2 | 4,010,110 | 414,722 | 0.103 |
| extension | MURA | ResNet-18 | 2 | 11,177,538 | 8,394,754 | 0.751 |
| extension | MURA | ResNet-50 | 2 | 23,512,130 | 14,968,834 | 0.637 |
| extension | MURA | Swin-T | 2 | 27,520,892 | 14,186,930 | 0.515 |

**Panel C. Recorded GPU training environments.**

| Stage | Dataset | Runs | Python | PyTorch | torchvision | CUDA | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| extension | ISIC2019 | 150 | 3.10.21 | 2.4.1+cu121 | 0.19.1+cu121 | 12.1 | NVIDIA GeForce RTX 4090 D |
| extension | MURA | 150 | 3.10.21 | 2.4.1+cu121 | 0.19.1+cu121 | 12.1 | NVIDIA GeForce RTX 4090 D |
| original | OrganAMNIST | 400 | 3.12.7 | 2.10.0+cu128 | 0.25.0+cu128 | 12.8 | NVIDIA GeForce RTX 5070 Ti |
| original | SIPaKMeD | 400 | 3.10.14 | 2.4.1+cu121 | 0.19.1 | 12.1 | NVIDIA GeForce RTX 3060 |

The original analysis-host record (Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0) is distinct from these training hosts. The completed revised B1 analysis was recorded as Python 3.12.14, NumPy 2.3.5 and pandas 2.2.3. Its manifest records the executable, timestamp, input hashes and analysis outputs. Package entries recorded as unavailable are not assigned a fabricated version. Other training package versions remain available in the run configuration records and the prior S3 v0.2 source.

**Panel D. Common optimization settings and actual workers.**

| Stage | Dataset | Optimizer | LR | Weight_decay | Batch | Workers | Maximum_epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| extension | ISIC2019 | AdamW | 0.0001 | 0.0001 | 32 | 4 | 60 |
| extension | MURA | AdamW | 0.0001 | 0.0001 | 32 | 4 | 60 |
| original | OrganAMNIST | AdamW | 0.0001 | 0.0001 | 32 | 24 | 60 |
| original | SIPaKMeD | AdamW | 0.0001 | 0.0001 | 32 | 0 | 60 |

**Panel E. Epochs actually trained and epochs selected for evaluation.**

| Stage | Dataset | Model | Rule | Input | Runs | Trained_mean_SD | Trained_range | Selected_mean_SD | Selected_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extension | ISIC2019 | DenseNet-121 | A | shared | 30 | 9.63 (0.72) | 9–11 | 2.63 (0.72) | 2–4 |
| extension | ISIC2019 | EfficientNet-B0 | A | shared | 30 | 23.53 (3.33) | 17–33 | 16.53 (3.33) | 10–26 |
| extension | ISIC2019 | ResNet-18 | A | shared | 30 | 8.57 (0.57) | 8–10 | 1.57 (0.57) | 1–3 |
| extension | ISIC2019 | ResNet-50 | A | shared | 30 | 8.73 (0.58) | 8–10 | 1.73 (0.58) | 1–3 |
| extension | ISIC2019 | Swin-T | A | shared | 15 | 9.80 (0.56) | 9–11 | 2.80 (0.56) | 2–4 |
| extension | ISIC2019 | Swin-T | A | swin_weight_eval | 15 | 10.20 (0.41) | 10–11 | 3.20 (0.41) | 3–4 |
| extension | MURA | DenseNet-121 | A | shared | 30 | 12.60 (1.45) | 10–16 | 5.60 (1.45) | 3–9 |
| extension | MURA | EfficientNet-B0 | A | shared | 30 | 33.70 (5.31) | 23–45 | 26.70 (5.31) | 16–38 |
| extension | MURA | ResNet-18 | A | shared | 30 | 10.53 (1.11) | 9–13 | 3.53 (1.11) | 2–6 |
| extension | MURA | ResNet-50 | A | shared | 30 | 11.13 (0.86) | 9–13 | 4.13 (0.86) | 2–6 |
| extension | MURA | Swin-T | A | shared | 15 | 14.13 (1.55) | 12–16 | 7.13 (1.55) | 5–9 |
| extension | MURA | Swin-T | A | swin_weight_eval | 15 | 13.93 (1.62) | 12–16 | 6.93 (1.62) | 5–9 |
| original | OrganAMNIST | DenseNet-121 | A | submitted | 50 | 35.26 (7.10) | 23–53 | 28.26 (7.10) | 16–46 |
| original | OrganAMNIST | DenseNet-121 | B | submitted | 50 | 60.00 (0.00) | 60–60 | 60.00 (0.00) | 60–60 |
| original | OrganAMNIST | EfficientNet-B0 | A | submitted | 50 | 56.66 (5.62) | 39–60 | 52.48 (7.66) | 32–60 |
| original | OrganAMNIST | EfficientNet-B0 | B | submitted | 50 | 60.00 (0.00) | 60–60 | 60.00 (0.00) | 60–60 |
| original | OrganAMNIST | ResNet-18 | A | submitted | 50 | 23.68 (5.81) | 12–38 | 16.68 (5.81) | 5–31 |
| original | OrganAMNIST | ResNet-18 | B | submitted | 50 | 60.00 (0.00) | 60–60 | 60.00 (0.00) | 60–60 |
| original | OrganAMNIST | ResNet-50 | A | submitted | 50 | 24.70 (5.89) | 12–43 | 17.70 (5.89) | 5–36 |
| original | OrganAMNIST | ResNet-50 | B | submitted | 50 | 60.00 (0.00) | 60–60 | 60.00 (0.00) | 60–60 |
| original | SIPaKMeD | DenseNet-121 | A | submitted | 50 | 21.66 (5.62) | 14–36 | 14.66 (5.62) | 7–29 |
| original | SIPaKMeD | DenseNet-121 | B | submitted | 50 | 60.00 (0.00) | 60–60 | 60.00 (0.00) | 60–60 |
| original | SIPaKMeD | EfficientNet-B0 | A | submitted | 50 | 54.70 (6.68) | 37–60 | 49.76 (8.60) | 30–60 |
| original | SIPaKMeD | EfficientNet-B0 | B | submitted | 50 | 60.00 (0.00) | 60–60 | 60.00 (0.00) | 60–60 |
| original | SIPaKMeD | ResNet-18 | A | submitted | 50 | 18.06 (5.59) | 9–36 | 11.06 (5.59) | 2–29 |
| original | SIPaKMeD | ResNet-18 | B | submitted | 50 | 60.00 (0.00) | 60–60 | 60.00 (0.00) | 60–60 |
| original | SIPaKMeD | ResNet-50 | A | submitted | 50 | 15.84 (4.15) | 9–25 | 8.84 (4.15) | 2–18 |
| original | SIPaKMeD | ResNet-50 | B | submitted | 50 | 60.00 (0.00) | 60–60 | 60.00 (0.00) | 60–60 |

**Panel F. Observed per-run wall-clock times (seconds).**

| Stage | Dataset | Model | Rule | Input | Runs | Seconds_mean_SD | Median_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| extension | ISIC2019 | DenseNet-121 | A | shared | 30 | 652.26 (50.77) | 647.37 |
| extension | ISIC2019 | EfficientNet-B0 | A | shared | 30 | 1551.94 (203.14) | 1545.13 |
| extension | ISIC2019 | ResNet-18 | A | shared | 30 | 528.06 (56.00) | 525.84 |
| extension | ISIC2019 | ResNet-50 | A | shared | 30 | 535.04 (59.41) | 524.93 |
| extension | ISIC2019 | Swin-T | A | shared | 15 | 588.97 (38.93) | 589.70 |
| extension | ISIC2019 | Swin-T | A | swin_weight_eval | 15 | 701.35 (32.21) | 692.47 |
| extension | MURA | DenseNet-121 | A | shared | 30 | 685.85 (89.55) | 675.38 |
| extension | MURA | EfficientNet-B0 | A | shared | 30 | 1727.78 (267.85) | 1753.36 |
| extension | MURA | ResNet-18 | A | shared | 30 | 568.52 (83.64) | 548.50 |
| extension | MURA | ResNet-50 | A | shared | 30 | 588.35 (58.70) | 591.70 |
| extension | MURA | Swin-T | A | shared | 15 | 735.20 (89.26) | 735.46 |
| extension | MURA | Swin-T | A | swin_weight_eval | 15 | 780.60 (91.96) | 752.69 |
| original | OrganAMNIST | DenseNet-121 | A | submitted | 50 | 1696.74 (292.70) | 1645.50 |
| original | OrganAMNIST | DenseNet-121 | B | submitted | 50 | 2712.18 (12.98) | 2706.00 |
| original | OrganAMNIST | EfficientNet-B0 | A | submitted | 50 | 1385.50 (116.02) | 1458.50 |
| original | OrganAMNIST | EfficientNet-B0 | B | submitted | 50 | 1454.60 (44.04) | 1467.00 |
| original | OrganAMNIST | ResNet-18 | A | submitted | 50 | 654.12 (102.54) | 647.00 |
| original | OrganAMNIST | ResNet-18 | B | submitted | 50 | 1280.48 (15.75) | 1283.00 |
| original | OrganAMNIST | ResNet-50 | A | submitted | 50 | 1200.30 (226.51) | 1171.50 |
| original | OrganAMNIST | ResNet-50 | B | submitted | 50 | 2561.60 (10.54) | 2558.00 |
| original | SIPaKMeD | DenseNet-121 | A | submitted | 50 | 375.14 (96.96) | 350.50 |
| original | SIPaKMeD | DenseNet-121 | B | submitted | 50 | 1036.90 (59.80) | 1044.00 |
| original | SIPaKMeD | EfficientNet-B0 | A | submitted | 50 | 692.16 (97.49) | 711.50 |
| original | SIPaKMeD | EfficientNet-B0 | B | submitted | 50 | 767.02 (69.35) | 752.00 |
| original | SIPaKMeD | ResNet-18 | A | submitted | 50 | 227.80 (72.37) | 209.50 |
| original | SIPaKMeD | ResNet-18 | B | submitted | 50 | 747.82 (54.83) | 742.00 |
| original | SIPaKMeD | ResNet-50 | A | submitted | 50 | 277.84 (76.68) | 279.00 |
| original | SIPaKMeD | ResNet-50 | B | submitted | 50 | 1042.12 (60.55) | 1052.00 |

**Panel G. Dataset and input-branch preprocessing.**

| Dataset | Training | Validation_test | Input |
| --- | --- | --- | --- |
| SIPaKMeD | 224 random resized crop, scale 0.8–1.0; horizontal flip p=0.5; rotation ±10° | Direct resize to 224×224, bilinear | RGB |
| OrganAMNIST | 224 random resized crop, scale 0.8–1.0; no horizontal flip; rotation ±10° | Direct resize to 224×224, bilinear | Grayscale repeated to 3 channels |
| ISIC2019 / shared | 224 random resized crop, scale 0.8–1.0; horizontal flip p=0.5; rotation ±10° | Direct resize to 224×224, bilinear | RGB |
| MURA / shared | Luminance→RGB, pad to square; bilinear 224×224 resize; rotation ±10°; no flip/random crop | Pad to square then direct resize to 224×224, bilinear | Luminance repeated to 3 channels |
| Swin-T / weight-specific evaluation | Same dataset-specific training transform as shared branch | Official V1: bicubic short edge 232, center crop 224; MURA pad-square first | Dataset-specific decoding as above |


### Table notes

1. Parameter counts describe model size and the specified fine-tuning scope. Original counts retain their saved source. Extension counts were reconstructed from named parameter shapes in all 300 formal checkpoints, excluding BatchNorm running buffers and Swin relative-position indices, and matched the historical pilot trainable counts. The exact freeze-module definitions and checkpoint-shape provenance are recorded in the accompanying CSV and JSON files. Swin input branches have the same parameter counts. This metadata reconstruction does not instantiate, fit or retrain a model.
2. Actual training epochs are the contiguous epoch rows written to each run history. Selected epochs identify the checkpoint used for evaluation and do not measure the full training duration. All 400 original Rule B runs trained and selected epoch 60; Rule A used validation-loss checkpoint selection with early stopping. All 300 extension runs used Rule A. The `_B` suffix in 30 Swin run names identifies the weight-specific evaluation-input branch, not original Rule B.
3. Original elapsed times use recorded UTC run entry and end timestamps. Extension times use the saved performance-counter interval after model/data-loader initialization through checkpoint hashing and metric export. The different timing boundaries and recorded GPUs remain visible; no cross-hardware speed ranking is derived. SD in Panels E/F is the descriptive sample SD across recorded runs.
4. Core optimizer, learning rate, weight decay, batch size and maximum epoch settings match the recorded configurations. Worker overrides were 0 for original SIPaKMeD, 24 for original OrganAMNIST and 4 for extensions. Training software versions are fixed within each dataset's formal runs and reported as recorded; the common training YAML does not imply identical package installations across all datasets. CUDA denotes the PyTorch runtime-reported version.
5. ImageNet mean/std normalization was used throughout. Preprocessing and augmentation are dataset- and branch-specific as listed; the shared Swin branch uses the dataset transform, whereas the weight-specific branch changes Swin validation/test processing. Model selection is repeated under the applicable branch. Further exact interpolation, decoder and worker-seed details remain in the source/configuration files.

## Supporting metadata

- [18行完整参数记录](P3a_Model_Parameter_Counts_v2.0.8.csv)
- [300 checkpoint形状核算](P3a_Parameter_Checkpoint_By_Run_v2.0.8.csv)
- [28行实际/选中epoch汇总](P3a_Epoch_Summary_v2.0.8.csv)
- [28行带硬件及计时范围记录](P3a_Recorded_Timing_Summary_v2.0.8.csv)
- [1100 run事实核对](P3a_Run_Metadata_Check_v2.0.8.csv)
- [已完成B1分析运行时](P3a_Revised_Analysis_Runtime_v2.0.8.json)
- [源hash与核对详情](P3a_Metadata_Audit_v2.0.8.json)
