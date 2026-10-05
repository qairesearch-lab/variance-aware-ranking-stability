# Objective data quality and grouped split protocol (v0.6)

Public documentation export v2.0.19. The operative records are `configs/data_quality_amendment_v0.6.yaml` and `configs/analysis_scope_decision_v0.7.yaml`. They supersede the visual adjudication proposals in earlier design records. No manual candidate-pair adjudication or pHash-based test-set exclusion was performed.

## 研究边界与数据来源

研究问题是重复 split、seed、模型池下的模型排序稳定性，不是诊断标签重标、图像匹配算法开发或患者级临床外部验证。ISIC 2019 使用官方八类训练 ground truth 和已提供的 lesion ID；MURA 使用官方检查级 normal/abnormal 标签以及路径中的 patient/study ID。我们不根据图像外观改变任何诊断标签，也不让无资质的单人目测建立新的“同源”参考标准。

## 正式索引规则

1. ISIC 2019 的 25,331 张均有官方训练标签；其中 2,084 张缺 lesion ID，按照已确认主分析人群决定排除，其类别为 NV 1,549、MEL 337、BKL 198，需单独报告。已知 ID 的 23,247 张中，完全相同像素却带 MEL/NV 冲突标签的两个完整 lesion ID `BCN_0003560`、`BCN_0000237`（各 2 张）按预先决定全部排除，不修改标签。正式候选为 23,243 张。
2. MURA 使用已验收的 official train+valid 公开语料 40,005 张、14,656 studies；保持官方检查级标签及患者分组。未公开竞赛 test 不使用。
3. 用文件 SHA-256 或含尺寸的解码 RGB 像素 SHA-256 找出跨原始 ID 的**完全相同图像**。相同标签的跨 ID 图片将其原始组链接为同一技术 split 单位，避免同一图片同时进入训练、验证和测试。这不是宣称两个 ID 必为同一病灶或患者。未获预先批准的冲突标签 exact pair 必须停止封存；不能猜测、重标或静默删除。
4. pHash Hamming 距离≤4 只生成外观相似**候选**。全量既有候选 ISIC 1,150 对、MURA 1,870 对；它们不是确认重复。pHash 结果不决定纳排、标签或分组，不要求人工逐对判定，也不允许批量默认同源/不同源。每个正式 split 报告这些候选中跨 train/validation/test 的对数；数字是可疑相似性的风险指标，不是已证实的数据泄漏率。
5. 只可声称“按官方 ID 分组且完全相同图像没有跨分区”；**不可声称排除所有可能同源或近重复图像**。近重复和缺失 ID 的残余风险要在 Methods/Limitations 写明。


## Sealed inputs and reproducibility

The final cohort contains 23,243 ISIC images and 40,005 MURA images. Exact-image checks identified 8 ISIC and 4 MURA cross-ID pairs and reassigned 5 original groups as technical split units. Across ten grouped splits, 470–590 ISIC and 775–936 MURA pHash candidate pairs crossed partitions; these are similarity candidates, not confirmed duplicates. The 300-run matrix comprises 270 shared-preprocessing runs and 30 Swin-T weight-specific evaluation-input runs. All extension runs use Rule A.

Source audits are in `candidate_indices/final_objective_v0.6/data_quality_audit.json`, `formal_splits_v0.6/split_generation_audit.json`, split CSVs and the manifest seal. Earlier proposed standalone filenames are not required evidence. Reproduction requires independently acquired images and a disposable copy with matching source hashes; preserve the saved outputs.

```bash
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/finalize_duplicate_audit.py --candidate-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/duplicate_candidate_audit_full_v1 --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices --output-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/generate_grouped_splits.py --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6 --audit-file research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6/data_quality_audit.json --output-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/formal_splits_v0.6
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/generate_extension_manifest.py --split-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/formal_splits_v0.6 --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6 --audit-file research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6/data_quality_audit.json --output research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/run_manifests/jiim_extension_300_v0.6.csv
```
