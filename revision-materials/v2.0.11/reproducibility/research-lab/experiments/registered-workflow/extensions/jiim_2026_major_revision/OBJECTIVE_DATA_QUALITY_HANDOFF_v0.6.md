# JIIM 审稿后扩展实验数据协议与交接 v0.6

作者于 2026-09-28 在正式 ISIC/MURA 模型结果产生前，确认以客观审计替代 v0.4 的 3,020 对图片人工“同源”判定。本文件与 `configs/data_quality_amendment_v0.6.yaml` 是现行数据质量规则；2026-09-29 另以 `configs/analysis_scope_decision_v0.7.yaml` 冻结分析范围：**不做 pHash 筛除测试集再排名分析**。原始 800-run 投稿配置、v0.3 中其余研究设计决定及 300-run 主分析加 Swin-T 敏感性范围保持不变。v0.4 文件保留作历史记录，**不得再作为上机步骤或论文 Methods 的事实描述**。v0.3 设计锁中的人工复核、近重复合并和“已判定近重复”闸门均由 v0.6 覆盖，不是现行执行条件。

**本机验收状态（2026-09-28）：**最终客观索引 `candidate_indices/final_objective_v0.6/`、20 个正式 split `formal_splits_v0.6/`、300-run 清单 `run_manifests/jiim_extension_300_v0.6.csv` 及 seal 均已实际生成；原始 ZIP 与解压图片均保留。ISIC 正式索引 23,243 张，MURA 40,005 张。客观 exact-image 审计发现最终队列中 ISIC 8 对、MURA 4 对跨原始 ID 完全相同图像；技术分组重指派共 5 个原始组。每组 split 的 pHash≤4 相似候选跨分区对数为 ISIC **470–590**、MURA **775–936**；不能解释为相同数量的真实重复或泄漏。**尚未进行目标 4090 的完整 Rule-A 校准或正式训练。**

## 研究边界与数据来源

研究问题是重复 split、seed、模型池下的模型排序稳定性，不是诊断标签重标、图像匹配算法开发或患者级临床外部验证。ISIC 2019 使用官方八类训练 ground truth 和已提供的 lesion ID；MURA 使用官方检查级 normal/abnormal 标签以及路径中的 patient/study ID。我们不根据图像外观改变任何诊断标签，也不让无资质的单人目测建立新的“同源”参考标准。

## 正式索引规则

1. ISIC 2019 的 25,331 张均有官方训练标签；其中 2,084 张缺 lesion ID，按照已确认主分析人群决定排除，其类别为 NV 1,549、MEL 337、BKL 198，需单独报告。已知 ID 的 23,247 张中，完全相同像素却带 MEL/NV 冲突标签的两个完整 lesion ID `BCN_0003560`、`BCN_0000237`（各 2 张）按预先决定全部排除，不修改标签。正式候选为 23,243 张。
2. MURA 使用已验收的 official train+valid 公开语料 40,005 张、14,656 studies；保持官方检查级标签及患者分组。未公开竞赛 test 不使用。
3. 用文件 SHA-256 或含尺寸的解码 RGB 像素 SHA-256 找出跨原始 ID 的**完全相同图像**。相同标签的跨 ID 图片将其原始组链接为同一技术 split 单位，避免同一图片同时进入训练、验证和测试。这不是宣称两个 ID 必为同一病灶或患者。未获预先批准的冲突标签 exact pair 必须停止封存；不能猜测、重标或静默删除。
4. pHash Hamming 距离≤4 只生成外观相似**候选**。全量既有候选 ISIC 1,150 对、MURA 1,870 对；它们不是确认重复。pHash 结果不决定纳排、标签或分组，不要求人工逐对判定，也不允许批量默认同源/不同源。每个正式 split 报告这些候选中跨 train/validation/test 的对数；数字是可疑相似性的风险指标，不是已证实的数据泄漏率。
5. 只可声称“按官方 ID 分组且完全相同图像没有跨分区”；**不可声称排除所有可能同源或近重复图像**。近重复和缺失 ID 的残余风险要在 Methods/Limitations 写明。

## 冻结与上机顺序

`finalize_duplicate_audit.py` 已改为客观数据封存器，校验原始索引、指纹和 exact pair 对应关系，生成最终索引、exact 链接、排除记录、pHash 风险候选及 SHA-256 审计；不接受人工 review CSV。之后 `generate_grouped_splits.py` 从最终索引生成 10 组正式 split 并统计 pHash 候选跨分区对数。最后 `generate_extension_manifest.py` 校验审计和索引，再生成 300-run 封存清单。所有输出用新版本路径，不覆盖旧诊断/试验输出；正式运行时复制配置、审计、索引、split、manifest 及 seal。技术冒烟和 CUDA 校准闸门仍需通过。

迁移到 AutoDL 时，可只传原始 ZIP、官方元数据、实验代码及已封存的小文件，再在服务器按 `research-lab/data/{isic2019,mura}/extracted/` 的相对路径解压；无需同时上传原始 ZIP 和本机解压树。正式 runner 需要解压后的图片，缺失时清单验证会失败。图像数据与索引/清单的传输、存储应遵守各数据集的许可证和平台使用条件。

下列为复现命令；本机对应 v0.6 输出已存在，**不要原样再次执行或覆盖**。在另一台机器复现时须先核对原始 ZIP 与元数据哈希，并保持相同项目相对路径：

```bash
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/finalize_duplicate_audit.py --candidate-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/duplicate_candidate_audit_full_v1 --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices --output-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/generate_grouped_splits.py --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6 --audit-file research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6/data_quality_audit.json --output-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/formal_splits_v0.6
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/generate_extension_manifest.py --split-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/formal_splits_v0.6 --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6 --audit-file research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_objective_v0.6/data_quality_audit.json --output research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/run_manifests/jiim_extension_300_v0.6.csv
```

## 修改稿 Methods 与给审稿人的如实口径

以下数据质量与 split 数字已由本机正式索引审计支持；模型结果尚未产生，不能添加性能结论：

> In the reviewer-requested extension, we used the published ISIC 2019 diagnostic ground truth and MURA study labels without relabeling. ISIC images lacking a lesion identifier (n=2,084) were excluded from the lesion-grouped primary cohort. Two complete ISIC lesion-ID groups (four images) containing pixel-identical images with conflicting published MEL/NV labels were excluded before model evaluation. We split ISIC by the supplied lesion identifiers and MURA by the supplied patient identifiers. Cross-ID pixel-identical images with concordant labels were linked as technical split groups to prevent exact-image overlap between partitions; this did not assert shared biological identity. A prespecified pHash Hamming-distance threshold of ≤4 flagged visually similar cross-ID pairs for descriptive leakage-risk reporting only; it did not change labels, inclusion, or split groups, and no new clinical image annotations were created. Across ten grouped splits, 470–590 ISIC and 775–936 MURA pHash candidate pairs per split crossed partitions. Because visual similarity does not establish common patient or lesion identity, residual unverified related-image leakage remains possible. These procedures assess the stability of benchmark model-selection conclusions under grouped evaluation and do not constitute an image-deduplication method or clinical validation.

审稿回复应明确：分组实验回答 image-level 原研究的外推局限；不会以 pHash≤4 人工判定表来声称零泄漏。不得将 pHash 跨分区计数解释为已确认的生物学同源计数，也不得让 300-run 扩展替代原 800-run 结果重分析、统计推断及其他编辑要求。
若审稿人询问为何未以 pHash 候选筛除测试集，可如实说明：pHash 是外观相似性筛查，不提供患者/病灶同源真值；据此删减测试样本会改变评价人群，且无法解决训练与验证之间的潜在相关性。因此预先选择保留完整预定测试集、报告候选跨分区数量和残余风险，而非将未经验证的筛查结果作为排除标准。

## 2026-09-29 分析范围决策与剩余闸门

作者已在正式扩展模型结果产生前决定：**不加入 pHash≤4 候选相关测试图像/study 筛除后的指标重算或模型再排名**。此决定记录在 `configs/analysis_scope_decision_v0.7.yaml`，绑定已封存的 v0.6 manifest 与数据质量配置哈希。理由是外观相似不等于同病灶/同患者，筛除后既不能证明零泄漏，也可能改变测试人群解释。pHash 仅用于每个 split 的跨分区候选数描述及残余风险披露，不更改任何测试成员、诊断标签、split 或模型结果。未来如确需新增该分析，应另行版本化并标记为探索性，不能在看过模型结果后作为预先规定的确证分析。

**不要误删另一项已经确认的敏感性分析：**300-run 清单仍为 270 个共同预处理主分析 runs 加 30 个 Swin-T 权重专属评估预处理 runs。此次“不加”仅指 pHash 测试集筛除分析，不影响任何已封存的训练行，也不增加 GPU runs。

目前没有待作者决定的研究范围选项；但“设计决策已冻结”不等于“立即可在 AutoDL 正式开跑”。技术交接仍需在目标 4090/CUDA 环境验证依赖与预训练权重、完成每数据集至少一个完整 Rule-A run 的内存/耗时/检查点校准，复核传输后的 ZIP、元数据、索引、split、manifest/seal 哈希，且确认正式输出审计通过。现有扩展代码已具备运行与输出审计，但完整的跨 300 runs 统计汇总、分层 bootstrap、原 800-run 重分析及审稿回复表图尚未作为本次数据封存的完成项；上机后须单独实施和验收，不得将训练完成误称为论文补充实验全部完成。

v0.3 设计锁曾列出三个独立导出文件（`split_hashes.csv`、`source_and_output_sha256_manifest.txt`、`zero_group_leakage_report.json`），v0.6 实际封存没有生成这三个同名文件。其校验信息分别保存在 `formal_splits_v0.6/split_generation_audit.json`、`candidate_indices/final_objective_v0.6/data_quality_audit.json`、split CSV 组 ID 与 manifest seal 中；当前有效证据合同按 v0.7 记录，不能在交接时声称三个旧文件已经存在。若期刊归档要求这些单独格式，再从已封存数据无损导出，不改动 split 或 manifest。
