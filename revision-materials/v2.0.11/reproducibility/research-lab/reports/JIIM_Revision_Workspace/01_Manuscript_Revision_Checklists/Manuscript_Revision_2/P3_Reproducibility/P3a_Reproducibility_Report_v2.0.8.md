# P3a复现规格核查结果 v2.0.8

## Metadata audit scope

日期2026-10-04；仅已有模型/执行元数据核查和修订素材，未训练、未做新推断统计、未改原稿。参数核算为CPU上受限读取checkpoint元数据，不读取张量数值、不调用torch，也不下载权重。

## 已完成

- 1100正式run逐个核对连续history、实际训练epoch、selected epoch、完成状态、环境、核心优化配置和计时；与既有D07/B2记录一致。
- 300扩展checkpoint逐个核对身份、epoch及命名参数形状；10个模型/类别组合在各正式run内一致。12个模型/输入组合代表文件完整SHA-256与checkpoint manifest一致。
- 8原参数记录＋10扩展参数记录，共18行；扩展trainable counts与相同preprocessing脚本的历史pilot一致。未把pilot性能当正式结果。
- 实际训练和selected epoch的28行汇总、28行计时汇总已经按原run核对；SD只是描述性记录。
- 训练环境4行保留事实，已完成B1分析环境单独回填。新[S3完整英文素材](Supplementary_Table_S3_v2.0.8.md)覆盖模型、参数、执行设置、epoch、计时和输入处理。

## Recorded execution facts

1. 原Rule B的400个run均训练60 epoch；扩展300个run全部Rule A。Swin名称`_B`表示weight-specific input branch，不能解释成固定epoch Rule B。
2. actual trained epoch不是selected checkpoint epoch；两列分别给出。
3. 软件版本仍按正式run记录；硬件和workers用于复现说明，不构成研究比较对象。
4. 原训练与扩展计时起止范围不同，原始范围一并保留。没有合并GPU速度排名或补做成本预测。

Metadata audit completed with zero reported errors.
