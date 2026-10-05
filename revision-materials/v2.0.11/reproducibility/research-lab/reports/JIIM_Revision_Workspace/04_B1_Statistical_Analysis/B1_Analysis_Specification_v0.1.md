# B1 分析执行说明

## Analysis scope

Revision-stage reanalysis of 800 original and 300 extension runs, specified after earlier results were available; it is not the original preregistration. The original RQ1–RQ3 and frozen training design are retained. Numerical cross-checks used different implementations in the same workflow, not an independent personnel review. See the package research-tool-use statement.


## 分层与比较单位

1. 原实验：两数据集、A/B分别分析，四CNN，10×5；完整800个run保留。
2. 扩展四CNN：ISIC2019/MURA，Rule A，共同预处理，10×3；240个run。
3. 原实验共同三seed：42/52/62、A/B均保留，10×3；作为480个原run的子集敏感性，不新增样本。
4. 候选池敏感性：扩展split01–05、三seed，在相同15个context比较四CNN与加入共享输入Swin的五模型；权重专用输入Swin另作同context输入方案敏感性，不当作checkpoint Rule B。
5. ISIC主endpoint为image BA、MURA为study BA；原数据为image BA。跨数据集描述不作为单一因素因果比较。

## D01：来源及算法复核

重新从raw metric构建四数据集Rule A四CNN张量，核对160个context的原split身份排除点估计。按原v0.2的seed和随机调用顺序，用向量化身份掩码实现重放10000次LSO bootstrap；每个held-out原split的全部重复抽样副本均排除。参考集不足两个不同原split时重抽并记录。用独立编写的批量IRLS重算既有分差关联模型，记录收敛/失败情况并与v0.2比较。This is an implementation cross-check, not an independent personnel review.

## D02：统一不确定性

- 主要重采样：10000次；先抽split，再在每个抽样split内抽seed；模型和policy配对保持不变。
- 点估计：各模型平均BA、观察选择频率、排名分布、前两名BA差、full-grid参考一致率，以及身份正确的LSO一致率。
- 区间：95% percentile重采样区间。观察选择频率和平均表现重采样后的选择频率是不同量；后者的Monte Carlo误差不当作原实验的不确定性区间。
- 原A/B差：同split/seed/model配对，B−A的BA差和选择频率变化；不将两个policy当独立组。
- 敏感性：split-only（seed固定）与crossed split/seed（全局seed抽样索引在所有split中复用），避免隐藏nested-bootstrap与原交叉seed设计的区别。LSO区间继续排除相同原split身份。
- 非参数bootstrap无法对未观察到的模型胜出产生正频率；零频率退化区间不证明真实胜出概率为零。
- 所有区间条件于当前固定数据集和观察到的评价过程；10个split及复用资料限制总体外推，10000次计算不是10000个独立实验。

## D03：图表与统计问题对应

先按Table 1、Figures 1–3建立问题、比较单位、效应/区间、检验条件和多重性family矩阵。模型BA差及policy BA差提供配对效应与区间；split平均差的双侧exact sign-flip仅作对称/交换性假设下的条件性敏感性检验。

- 四CNN模型对比：每dataset/policy stratum的六比较作Holm校正；同时输出36个主比较的跨stratum校正供透明核查。
- 原policy BA差：八个dataset×model比较作一个Holm family。
- Swin输入方案：两个dataset比较单独Holm family；仅五split，低功效及最小可达p值如实记录。
- 观察频率、LSO区间和order-statistic分差不机械检验“是否超过零”，也不设无理论依据的均匀模型胜出零假设。
- Figure 3以有限网格描述及参考敏感性报告；复用的partition/subset计分数不作为独立N，不给其套用二项分布显著性检验。

## D04：真实聚合预算及参考敏感性

每stratum以10个split分成5个discovery和5个reference，穷举252个有方向的5/5划分；这些划分已包含互为补集的两个方向，不再重复计数。对每个预算(s,k)，从discovery选择s=1…5个split、选择k个seed，先将该s×k中全部模型BA分别平均，再形成一个排名，与固定reference块（五split及全部该stratum seed）的排名比较。所有模型共用同一子集。

- 每个划分内穷举split及seed子集。跨预算保留同一个reference块，按划分配对报告预算差。
- 主参考：平均BA形成排名；敏感性：平均context rank形成排名，以及2000次split/seed bootstrap后的reference排名分布。
- 完成每个预算的模型选择一致率、完整排名相同率、Kendall排名一致性、partition分布及参考敏感性；partition范围/分位数不是总体置信区间。
- 保留原全网格in-sample recovery来源表及点值作为历史estimand复现；明确其参考包含discovery。10×5的自比较1.0不作为收敛证据。
- 共同三seed敏感性采用相同5/5定义。有限网格最多验证五个discovery split，不能宣称发现普遍最低预算或真实总体最佳模型。

## Output provenance

Saved machine outputs are in `analysis/results_v0.3/`. Recorded source hashes describe the original analysis inputs; the public export mapping identifies documentation changes. Numerical inputs and statistical CSVs are unchanged.

## 方法来源及适用边界

- [SciPy paired permutation test说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)：配对交换与差值sign-flip的检验定义；本研究的交换性条件仍需单独说明。
- [Bengio与Grandvalet，JMLR 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html)：交叉验证方差的一般限制；这是对复用样本依赖的相关方法警示，不是对本研究repeated-holdout区间有效性的证明。
