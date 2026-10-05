# JIIM 扩展实验上机闸门（2026-09-27）

> **历史版本，已由 `OBJECTIVE_DATA_QUALITY_HANDOFF_v0.6.md` 取代数据质量与上机步骤。**作者 2026-09-28 决定不再逐对人工判定 pHash 候选；本文件以下的人工复核/`--review-file` 指令均不得执行。原始 800-run 实验不受影响。

状态：**NOT READY FOR FORMAL TRAINING**。本文件记录可执行的下一步，不声称已完成任何未完成的人审或 CUDA 训练。研究目标是检验原方法学结论和模型排序的稳健性，不是开发新的图像去重算法。

| 遗留项 | 已完成 | 仍需完成／放行证据 |
|---|---|---|
| 数据质量与分组 | 已确认排除两个完整 ISIC 病灶 ID、pHash≤4；候选数量与排除样本核对通过；离线 v3 画廊可导出/导入复核进度 | 当前 0/3,020 对人工复核；须逐对判定，疑难留空再复核；导出带审阅人/时间的完整 CSV；真实最终审计、双数据集索引与 10 组 split 尚不存在 |
| 训练矩阵 | 作者确认 E1 共用预处理 270 runs + Swin-T B 敏感性 30 runs；代码强制初始清单 300 行，且将选择文件 SHA-256 纳入运行审计 | 待正式 split/索引生成后创建并封存 300 行 manifest；B 只能按独立敏感性解释 |
| 目标环境与算力 | 本地小样本 MPS 冒烟通过；正式入口严格核对 CUDA/版本/batch 32/4 workers | 在目标 4090 上验证 Python 3.10、PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121 等准确版本；10 个 dataset×model 组合的技术冒烟及每数据集至少一个完整 Rule-A run；记录显存、每 epoch、I/O、总时间 |
| 整链路验收 | 合成数据正向封存器测试与未复核拒绝测试通过；当前契约测试 18/18 | 真实人审结束后完成数据审计→split→manifest→CUDA run→只读结果审计；未完成前不得将合成结果或短冒烟作为论文结果 |

## 人工复核交接

用本目录下 `candidate_indices/duplicate_candidate_audit_full_v1/blinded_review_gallery_v3/index.html` 离线审阅；它只显示匿名候选 ID、pHash 距离和图片，不显示标签/病灶/患者 ID。填写审阅人，再逐对选“同源”或“不同源”；仅相似部位、不能确认时留空并写理由。每次复核会话导出 CSV 并妥善备份；换浏览器/机器时，打开同一 v3 画廊，先导入最近的 CSV，再继续。不要在 CSV 中批量默认填“不同源”。若同源判定牵涉标签冲突，封存器会停止，需要单独裁决并做版本化记录。

全部 3,020 对**真实**复核完成后，在项目根目录按下述顺序运行。下方仅 `--review-file` 的示例路径需替换成实际导出 CSV 的绝对路径；三个目标路径必须尚不存在，不要覆盖已有审计：

```bash
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/finalize_duplicate_audit.py --candidate-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/duplicate_candidate_audit_full_v1 --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices --review-file "/absolute/path/to/near_pair_blinded_review_export.csv" --output-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_adjudicated_v0.5
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/generate_grouped_splits.py --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_adjudicated_v0.5 --audit-file research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_adjudicated_v0.5/data_quality_audit.json --output-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/formal_splits_v0.5
python research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/generate_extension_manifest.py --split-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/formal_splits_v0.5 --index-root research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_adjudicated_v0.5 --audit-file research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/candidate_indices/final_adjudicated_v0.5/data_quality_audit.json --output research-lab/experiments/registered-workflow/extensions/jiim_2026_major_revision/run_manifests/jiim_extension_300.csv
```

检查最终 ISIC 图像数 **23,243**（其它近重复只合并分组，不删图）、MURA 图像数 **40,005**、每 split 无组间泄漏、封存清单 **300 行**（主分析 270、B 30），并保存全部 SHA-256。不能把诊断 split 改名当正式 split。

## 4090 校准和配置口径

目标环境文件是 `configs/environment_target_v1.yaml`，其精确执行约束为 `torchvision 0.19.1+cu121`；旧 v0.3 设计文本中的 `0.19.1` 是简写，不是允许随意替换 build。正式 runner 固定 `num_workers=4`，因此 8-worker 比较只能作为技术诊断；如确要切换，须在首个正式 run 前另做版本化配置/代码修订、重跑 schema smoke，并重新冻结全部运行哈希。不得因观察到模型准确率或排名而调整 worker、阈值、epoch 或矩阵。

本机 M5 的 300-run 粗估为：270-run E1 约 617 小时基准，加 B 的约 122 小时基准，合计约 **739 小时连续运行**；加 20% 排期缓冲约 **887 小时／37 天**。这是随机初始化、小缓存短跑的线性外推，不是正式性能或交付承诺。AutoDL 300-run 当前容量规划约 **252 GPU 小时含缓冲**，尚无 4090 实测；只有目标机完整 run 的时间才可决定 3–5 天排期。
