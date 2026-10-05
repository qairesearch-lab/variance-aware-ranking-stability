# JIIM 扩展实验：数据质量修订、上机交接与写作口径（v0.4）

> **历史版本，已被 `OBJECTIVE_DATA_QUALITY_HANDOFF_v0.6.md` 和 `configs/data_quality_amendment_v0.6.yaml` 取代。**不得再使用下文关于 3,020 对人工复核的上机门槛或 Methods/审稿回复措辞。

记录日期：2026-09-27。状态：**作者已确认决策；人工近重复复核尚未完成；正式队列/结果尚未生成。**适用范围仅为审稿后 ISIC 2019 / MURA 扩展实验，不追溯改写最初提交的 800-run 实验。

补充决定：作者后续确认从首个正式 run 起使用 **300 行**封存清单：共用预处理的 E1 主分析 270 runs，加 Swin-T 权重专属验证/测试预处理的 B 敏感性 30 runs。机器可读记录为 `configs/swin_sensitivity_decision_v0.5.yaml`；B 不改变主分析结论的定义，也不是新的图像处理算法。

## 必须传递给上机执行者的决定

1. ISIC 主分析先排除缺失 lesion ID 的 2,084 张，保留 23,247 张/11,847 病灶组的候选群体。发现跨 MEL/NV 标签的完全相同图像后，按完整病灶 ID 排除 `BCN_0003560` 与 `BCN_0000237`，各 2 张，分别原标 MEL 2 张与 NV 2 张。具体样本为 `ISIC_0067502`、`ISIC_0071017`、`ISIC_0067980`、`ISIC_0069013`。不修改原始诊断标签。最终待复核合并前主队列应为 **23,243 张/11,845 病灶组**。
2. 用 pHash 的 Hamming 距离 **≤4（含 4）**筛查跨原始组的近重复候选。当前全量候选 **ISIC 1,150 对、MURA 1,870 对，共 3,020 对**；这四张排除图像不涉及这些候选对。候选对不是确认重复，也不是待删除图像数。先用盲法图像复核和可追踪的审阅人/时间/理由做判定；同源才合并分组、确保同一 split；不确定者暂停、追加复核，不得默认忽略或自动合并。发现跨标签冲突，不能以 pHash 自动改标签。
3. `configs/data_quality_amendment_v0.4.yaml` 是作者确认的机器可读记录；`configs/extension_protocol_design_locked_v0.3.yaml` 保持不变。正式索引、split、manifest、运行配置与事后审计须以两个文件的哈希共同追溯。冒烟结果只说明技术可运行，不是研究结果。不得在看到模型表现后重选阈值、保留冲突样本或调整复核标准。
4. 研究问题是**原论文方法学结论在不同分组数据/固定模型池下的稳健性与模型排序稳定性**，不是开发、验证或比较去重算法。pHash 是防止分组泄漏的常规数据质量控制；不可宣称提出新算法，也不可宣称排除了所有残余近重复风险。

## 冻结门槛与留痕

目前 3,020 对人工复核尚未完成，所以不得称“最终数据已经封存”。用 `candidate_indices/duplicate_candidate_audit_full_v1/blinded_review_gallery_v3/index.html` 人工审阅；每次导出 CSV 留备份，跨浏览器可导入上次导出的 CSV 恢复。完成后才执行 `finalize_duplicate_audit.py`；其输出应含 `isic2019_excluded_records.csv`、最终双数据集索引、`group_merges.csv` 与带 v0.4 SHA-256 的 `data_quality_audit.json`。核对排除前后样本/病灶计数、所有复核记录、冲突处置、split 无组间泄漏，然后生成 10 组正式 split 和 **300 行** E1+B manifest。合成样本正向封存器测试已通过，但不能充当真实复核。租多卡全量跑前还须完成单 4090 的目标环境和完整 Rule-A 校准。交接时复制两个修订 YAML、复核 CSV、画廊/候选审计、最终审计与索引、splits、manifest seal、环境锁和代码版本；不要只交数据图片或诊断 split。

## 论文 Methods 写作草稿

以下仅是**待人工复核及正式索引验收后**可用于投稿修改稿的措辞；方括号需以实际审计输出核实后填写，当前不得写成已经执行完毕：

> In the reviewer-requested ISIC 2019 extension, images without a lesion identifier were excluded from the primary lesion-grouped analysis (2,084 images). Among the remaining 23,247 images, two complete lesion-ID groups (`BCN_0003560` and `BCN_0000237`; two images per group) were excluded before formal model evaluation because exact duplicate images carried conflicting MEL/NV labels and inconsistent metadata. Original labels were not altered, leaving 23,243 images in the eligible cohort before duplicate-group consolidation. In ISIC 2019 and MURA, cross-group near-duplicate candidates were screened using perceptual hashes with a Hamming distance of ≤4. Candidate pairs were visually adjudicated [report reviewer procedure and final counts]; confirmed same-source groups were linked before grouped train/validation/test splitting. This procedure was a data-quality safeguard for the fixed comparative experiment, not a proposed image-matching algorithm. Residual undetected duplicates remain possible.

复核结束后须补充：[实际人工复核数与判定数]、[合并组数]、[最终每集大小/类别分布]、[研究主要指标所用单位]；Methods/Supplement 最好附纳排流程与完整审计规则。不能把 3,020 对候选直接写成 3,020 个重复或 3,020 张剔除。

## 给审稿人的逐点回复口径（待完成后使用）

> We thank the reviewer for raising the possibility of data leakage in the extension analyses. We have made the cohort definition and grouping audit explicit. Before evaluating formal ISIC 2019/MURA model outcomes, we recorded a data-quality amendment excluding two complete ISIC lesion-ID groups containing exact-image pairs with contradictory MEL/NV labels, rather than relabeling the published ground truth. We separately report the 2,084 images without lesion IDs and the four images excluded for the label conflict. We used pHash Hamming distance ≤4 only to flag cross-group candidates for blinded human review; it did not itself determine inclusion, labels, or model performance. Confirmed same-source groups were kept within the same train/validation/test partition, and group leakage was audited [insert completed counts and audit reference]. These steps support the intended comparison of model-ranking stability under grouped data, and we make no claim to a new deduplication method or to perfect elimination of all residual duplicate risk.

在复核未完成、正式训练未完成前，回复信只能使用“we have prespecified/recorded”描述**已记录的决定**，不可使用上述 “we used / were kept / was audited” 作为已完成事实。结果与统计结论应待 E1 验收后另行撰写。
