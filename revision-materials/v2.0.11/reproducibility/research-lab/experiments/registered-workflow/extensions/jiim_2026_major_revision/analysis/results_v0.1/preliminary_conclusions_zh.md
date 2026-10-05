# JIIM 修订后扩展实验：初步统计分析

本报告仅分析冻结的审稿后扩展与原始 800-run Rule-A 对照子集；扩展实验是 post hoc reviewer-requested extension，不并入原预注册矩阵。研究问题是原有“排名/选择不稳定”结论在新增数据集与评估情境中是否仍成立。绝对 balanced accuracy 仅用于描述任务饱和程度，不以模型性能排序寻找优胜架构。

## 数据完整性

- 封存 manifest：300 行；SHA-256 与 seal 一致。
- 输出审计：300/300 通过运行状态、执行方案/环境哈希、冻结测试单位与标签、BA 复算、概率和 checkpoint 哈希核验；原始 800-run Rule-A 中 400 个配对比较 run 也与此前采集表逐项一致。
- 主扩展：270 次 shared-preprocessing Rule-A runs；另 30 次 Swin 权重专属预处理敏感性 runs。
- AutoDL 记录的训练环境一致：Python 3.10.21、PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121、RTX 4090 D；batch size 32、最多 60 epochs、AdamW。
- 每个 run 的 balanced accuracy 均从保存的预测重新复算；相同 split × seed 的模型预测覆盖相同测试单位与标签。

## 主要结果

### 四数据集的排名稳定性

| 数据集 | splits × seeds | LSO agreement (95% CI) | LSO discordance | 平均 top-two BA gap | gap <0.01 contexts |
|---|---:|---:|---:|---:|---:|
| isic2019 | 10 × 3 | 0.900 (0.733–1.000) | 0.100 | 0.0326 | 6 / 30 |
| mura | 10 × 3 | 0.600 (0.133–0.800) | 0.400 | 0.0084 | 19 / 30 |
| organamnist | 10 × 5 | 0.620 (0.320–0.800) | 0.380 | 0.0015 | 50 / 50 |
| sipakmed | 10 × 5 | 0.520 (0.140–0.740) | 0.480 | 0.0067 | 39 / 50 |

### Margin 与排名不一致

跨四个数据集、以 dataset fixed effects 调整的 split-cluster bootstrap logistic model：top-two BA gap 每减少 1 SD，leave-one-split-out comparator discordance OR = 2.022（95% cluster-bootstrap CI 0.747–6.493；40 dataset×split clusters，160 contexts）。该关系是关联性分析；discordance 本身由排名反转定义，不能解读为因果效应。

### 独立 split-block 参照

| 数据集 | 5/5 split-block cross-fitted agreement | scored contexts |
|---|---:|---:|
| isic2019 | 0.900 (partition range 0.800–1.000) | 7560 |
| mura | 0.491 (partition range 0.067–0.733) | 7560 |

### 候选池敏感性

四 CNN 与扩展候选池只在相同 5 splits × 3 seeds contexts 配对比较。下表报告扩充候选池后选择身份改变的比例；它衡量结论对候选池的敏感性。Swin B 与 Swin A 比较只反映 Swin 权重专属输入预处理变化，不能归因于架构本身。

| 数据集 | 配对 contexts | 扩充候选池后选择身份改变率 |
|---|---:|---:|
| isic2019 | 15 | 0.400 |
| mura | 15 | 0.800 |

### 推断性补充（非模型优劣分析）

补充表列出每个新增数据集内四 CNN 的六组配对 balanced-accuracy 差异、split-cluster bootstrap 95% CI、split-block exact sign-flip p 值及 Holm 校正。这些统计是回应审稿人推断性要求的辅助结果，不用于寻找或宣称最优架构。详见 `extension_paired_model_differences.csv`。

Swin 权重专属输入 recipe 与 shared recipe 的配对估计和区间单独见 `swin_preprocessing_sensitivity_paired.csv`。

### 初步结论

扩展实验对原结论提供的是有边界、依赖数据集的支持。原 OrganAMNIST 与 SIPaKMeD 的平均 BA 分别约为 0.985、0.958，显示明显饱和；新增 ISIC 2019、MURA 的平均 BA 约为 0.471、0.781，整体较不饱和。因此新增实验检验原发现能否离开饱和任务，而非比较架构谁更强。

在共同四 CNN 候选池下，ISIC 的 LSO agreement 与 split-block cross-fit agreement 均约为 0.90，显示此评估情境中的选择相对稳定；MURA 分别为 0.60 与 0.491，仍有较多不一致。新增数据并未复现统一方向：ISIC 不支持‘所有任务都不稳定’，MURA 则表明不稳定现象并未局限于原有饱和数据集。最稳妥的结论是排名/选择稳定性受数据集和评估情境影响，原结论不能无条件推广到所有医学影像 benchmark。

近零 margin 是否解释不稳定仍未解决：合并模型 OR=2.02，95% cluster-bootstrap CI 0.75–6.49，包含 1；40 个 dataset×split clusters 的证据不足以支持稳健关联。Swin 扩充候选池后，15 个配对 contexts 中 winner 身份改变比例为 ISIC 40%、MURA 80%；这只说明候选池会影响选择结论，是稳定性分析的一部分，不表示任何模型更优。

两个新增数据集不足以证明医学影像任务总体或患者层级临床泛化。绝对 BA、每模型选择频率、配对统计及 MURA 七部位描述统计均放在补充 CSV 中供审稿核查，不作为主结论。MURA subgroup 仅作估计与不确定性描述。数据质量审计中的 pHash 候选只按冻结方案作风险披露，未据此筛除测试样本或重排。

## 输出表

- `run_integrity_audit.csv`：300 rows
- `original_rule_A_raw_metrics_audit.csv`：400 rows
- `extension_run_metrics.csv`：300 rows
- `four_dataset_context_stability.csv`：160 rows
- `four_dataset_model_performance_and_selection.csv`：16 rows
- `four_dataset_ranking_stability_summary.csv`：4 rows
- `extension_paired_model_differences.csv`：12 rows
- `margin_discordance_logistic_model.csv`：1 rows
- `isic_mura_split_block_crossfit.csv`：2 rows
- `swin_candidate_pool_and_input_recipe_sensitivity.csv`：2 rows
- `swin_preprocessing_sensitivity_paired.csv`：2 rows
- `mura_body_region_model_performance.csv`：28 rows
- `mura_body_region_stability_summary.csv`：7 rows
