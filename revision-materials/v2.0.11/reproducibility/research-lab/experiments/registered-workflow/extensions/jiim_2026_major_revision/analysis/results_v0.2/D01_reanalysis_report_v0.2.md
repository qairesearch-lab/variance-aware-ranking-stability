# D01 重分析：按原 split 身份排除重复抽样

## Material Passport

- ID：JIIM-D01-v0.2
- 类型：split-cluster bootstrap sensitivity reanalysis
- 状态：ANALYZED；代码成功运行，未独立重训
- 范围：四数据集的共同四CNN；原数据使用Rule A 10×5，扩展数据使用Rule A 10×3
- 输入及SHA-256：见 `d01_reanalysis_manifest.json`
- 输出：LSO区间表、分差模型区间表和manifest

## 方法修正

每次抽取10个split cluster并在cluster内对seed重抽样。对某个held-out原split，参考集排除本次样本中所有具有相同原split身份的抽样位置。若一次draw只含一个不同原split身份，则整次draw重抽并计数。点估计按全部原split identity排除后重新计算。

## LSO结果

| Dataset | contexts | identity-aware LSO agreement | 95% bootstrap interval | degenerate draws resampled |
|---|---:|---:|---:|---:|
| isic2019 | 30 | 0.900 | 0.733–1.000 | 0 |
| mura | 30 | 0.600 | 0.133–0.800 | 0 |
| organamnist | 50 | 0.620 | 0.280–0.800 | 0 |
| sipakmed | 50 | 0.520 | 0.120–0.760 | 0 |

## 分差关联模型

OR = 2.022; identity-aware cluster-bootstrap 95% CI 0.597–5.763; 40 dataset×split clusters and 160 contexts. This is an associational estimate conditional on these datasets; the interval is not evidence of causation or a universal threshold.

## 验证边界

这次修正避免同一原split因cluster bootstrap重复抽样而同时进入held-out位置与其参考集。它没有产生新的患者或机构样本，也不修复重叠holdout对总体外推的限制。只关闭D01所覆盖的LSO/分差区间算法问题；完整Rule B统一不确定性(D02)、聚合预算(D04)及其他D项需另行核验。


## 可重复性核对记录

以固定脚本和seed再次运行一次，四个结果文件（LSO CSV、margin CSV、manifest及本报告生成前的报告文件）逐文件SHA-256相同。该核对证明本地固定环境下结果可复现，不等于独立人员对算法/统计假设的审查；CK14仍待独立复核。
