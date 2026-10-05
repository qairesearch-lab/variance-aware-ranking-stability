# Analysis methods

The study evaluates model selection stability across repeated medical image classification evaluations. The revision analysis uses 800 original and 300 extension training runs. It was specified after the earlier results were available and does not replace the original training protocol.

## Evaluation design

The original datasets, OrganAMNIST and SIPaKMeD, use four ImageNet-initialized CNNs, ten splits, five seeds and two checkpoint policies. ISIC2019 and MURA use the same four CNNs, ten grouped splits, three seeds and early stopping. Additional Swin-T runs use five matched splits and three seeds per extension dataset, with shared-input and weight-specific evaluation-input conditions. The latter is an input sensitivity comparison, not fixed-epoch checkpoint policy B. ISIC2019 is evaluated at image level and MURA at study level. Subset analyses reuse completed runs.

## Paired resampling and statistical comparisons

Model and checkpoint comparisons preserve identical split–seed contexts. The analysis uses 10,000 bootstrap replicates and percentile 95% intervals. Crossed resampling samples split and seed levels independently and reuses the seed indices across sampled splits. It is used for the reported aggregate bootstrap selection frequency, model mean-BA intervals and observed selection-frequency intervals. Split-then-seed resampling is retained for the existing paired BA contrasts, checkpoint effects and selection sensitivity intervals; split-only results provide sensitivity comparisons. The empirical reference model is re-estimated when the estimand requires it.

Observed selection frequency describes single observed contexts. Aggregate bootstrap selection frequency describes the fraction of resampled mean-performance rankings selecting the reported reference model; it is reported as a point estimate. Its simulation precision is distinct from uncertainty in the experimental data. Leave-one-split-out calculations exclude every resampled occurrence of the held-out original split identity.

Exact two-sided sign-flip tests use split-mean paired BA differences under conditional sign symmetry. Holm correction is applied to the six fixed-model contrasts within each primary stratum, the eight original dataset-by-model checkpoint comparisons, and the two input-recipe comparisons as separate families. The all-36 contrast adjustment is also supplied.

## Evaluation budget

Each ten-split grid is partitioned into five discovery and five reference split IDs in all 252 oriented partitions. Model BA is first averaged within each discovery subset before ranking; the reference uses the five held-out splits and all available seeds. The analysis enumerates eligible split and seed subsets, with a common reference block for paired budget comparisons. Alternative references use mean rank and 2,000 reference bootstrap draws. Partition ranges and quantiles describe the finite evaluated grid, rather than population confidence intervals.

Source images may recur across different split IDs. The inference concerns the fixed datasets, candidate pools and observed evaluation designs.
