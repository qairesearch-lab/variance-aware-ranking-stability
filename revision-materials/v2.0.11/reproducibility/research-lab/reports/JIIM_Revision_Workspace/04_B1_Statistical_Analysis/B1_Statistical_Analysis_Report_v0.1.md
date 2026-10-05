# B1统计计算与来源核对报告 v0.1

## Analysis provenance

Completed revision-stage analysis of the original 800 and extension 300 runs. Numerical summaries below describe the original nested implementation; the subsequent design-aligned crossed resampling results are separately recorded in `D02_resampling_design_sensitivity_v0.1.csv`. These schemes are not interchangeable.


本轮主分析保留800原run的全部A/B层；扩展240个共同四CNN run、30个共享输入Swin run及30个权重专用输入Swin run均进入对应分层。16个分析stratum包括复用的共同seed与候选池子集，不能把stratum行数相加当作新的独立样本数。

## D01：算法问题与既有结果的复核

四数据集Rule A的160个context重新按原split身份排除，向量化另一实现重放10000次抽样，点估计及LSO区间与v0.2的差异均小于1e-12。新写批量IRLS重放关联模型，10000次均收敛，未丢弃拟合；OR和区间最大差异小于1e-6。
分差关联模型 OR=2.022，95%重采样区间[0.597, 5.763]。区间包含1，不能声称该关联已达到明确的统计支持；也不能据此断言不存在关联。它是补充关联分析，不能成为新的普遍分差阈值或取代原RQ。

Implementation cross-checks do not constitute independent personnel review.

## D02：原A/B与扩展数据的统一不确定性

| 数据集/策略 | 实际context | 全网格参考一致率 [95%层级区间] | 身份正确LSO一致率 [95%层级区间] | 层级paired bootstrap选择频率¹ |
|---|---:|---|---|---:|
| organamnist/A | 50 | 0.620 [0.440, 0.800] | 0.620 [0.300, 0.800] | 0.9518 |
| organamnist/B | 50 | 0.600 [0.400, 0.780] | 0.600 [0.260, 0.760] | 0.7723 |
| sipakmed/A | 50 | 0.520 [0.360, 0.740] | 0.520 [0.120, 0.740] | 0.8688 |
| sipakmed/B | 50 | 0.440 [0.380, 0.680] | 0.220 [0.100, 0.680] | 0.5346 |
| isic2019/A | 30 | 0.900 [0.733, 1.000] | 0.900 [0.733, 1.000] | 1.0000 |
| mura/A | 30 | 0.600 [0.300, 0.800] | 0.600 [0.133, 0.800] | 0.7824 |

¹ 指在完成数据的层级配对重采样中，原全网格参考模型再次获得最高平均BA的频率；不是观察context的胜出频率，也不是“该模型在真实世界最好”的概率。

原四stratum的模型平均BA、原参考身份和观察一致率与历史结果一致。旧flat-context bootstrap的四层共40000次抽样完整重放，计数逐模型一致；新层级值改变重采样单位，没有改动raw BA。旧CSV的逐stratum seed标签并不表示每层重新初始化RNG，复现依照原脚本的连续随机流；新manifest明确按stratum/purpose派生seed。

SIPaKMeD/B的full-grid一致率0.44，而LSO一致率0.22，表明参考身份较敏感。这是两个参考定义的结果，不是数据错误，也不应任取其中较有利的一个。ISIC2019的共同四CNN在本设计下更稳定；MURA仍存在较多选择变化，扩展结论不能写成所有数据集必然同样不稳定。

nested、split-only和crossed split/seed的区间与选择频率全部另表保存。crossed方案在所有split中复用同一个全局seed抽样向量，回应原交叉seed设计。均值及选择频率bootstrap保留全部抽样；只对需要LSO参考的单一原split退化draw进行重抽，次数另表登记。

区间条件在当前固定数据集与观察评价过程内，不能外推为新患者/机构总体置信区间。只有10或5个split层级，复用图像、重复seed及非参数区间的有限样本覆盖限制仍存在。零观察胜出模型的退化区间不证明其真实胜出概率为零。

## RQ2：checkpoint、候选池及输入方案的信号

| 原数据集 | A/B改变context首位模型的比例 | 95%层级区间 |
|---|---:|---|
| organamnist | 0.460 | [0.280, 0.640] |
| sipakmed | 0.560 | [0.360, 0.740] |

policy的每模型BA差和选择频率差保存为配对输出；不能据此比较两个独立训练组或宣称checkpoint规则是全部不稳定性的单一原因。原lme4方差结构证据保持历史身份，本轮没有用Python替代注册R模型。

| 扩展数据集 / Swin输入方案 | 同15个context加入Swin后，首位身份变化比例 | 95%层级区间 |
|---|---:|---|
| isic2019 / shared | 0.400 | [0.067, 0.733] |
| isic2019 / swin_weight_eval | 0.667 | [0.400, 0.933] |
| mura / shared | 0.800 | [0.533, 1.000] |
| mura / swin_weight_eval | 0.867 | [0.600, 1.000] |

这些结果支持“选择结论依赖所评候选集合”的限定，不作为Swin胜过CNN的论证。输入方案BA差是相同Swin权重/架构、相同context的配对敏感性。ISIC输入差的percentile区间不跨零，但仅五split的exact sign-flip经两比较Holm后p=0.125；两种程序不是同一检验的区间反演，不能挑选较有利者声称显著。MURA输入方案差的区间跨零；跨零不等于证明等效。

## D03：Table1/Figures1–3的证据覆盖

六行覆盖矩阵在运行之前登记，明确各图表科学问题、单位、效应/区间、零假设适用性、多重性family和对应输出。Table1、Figure1的选择频率、Figure2的order-statistic分差、Figure3的聚合预算分别有完整证据定位。配对BA检验按stratum全对比Holm，原policy八比较另作一个family，并保留36主模型比较的全局Holm敏感性。

不机械为每个柱/曲线附p值：Figure2的context前两名差由选择产生且非负，检验它是否大于零不能证明某个指定模型更优；Figure3的反复subset/partition计分也不是独立二项试验。适当使用描述、配对效应及区间。科学问题、比较单位与输出定位见覆盖矩阵。

## D04：RQ3真实聚合预算与参考敏感性

穷举252个有方向5/5 split划分，每预算先汇总discovery中s×k个BA形成一个排名，再比较固定reference块。互补方向已经在252划分中，不重复再算。190个budget-cell、47880个partition-budget记录落盘，150个单元用另一直接循环实现核对，全部一致。计分总数和partition数均非独立样本量。

| 数据集/策略，完整原seed池 | disjoint参考下1×1一致率 | disjoint参考下5×3一致率 | mean-rank参考5×3 | bootstrap参考5×3期望一致率 |
|---|---:|---:|---:|---:|
| organamnist/A | 0.609 | 0.890 | 0.838 | 0.770 |
| organamnist/B | 0.541 | 0.587 | 0.684 | 0.521 |
| sipakmed/A | 0.463 | 0.694 | 0.521 | 0.598 |
| sipakmed/B | 0.370 | 0.204 | 0.230 | 0.339 |
| isic2019/A | 0.900 | 1.000 | 1.000 | 1.000 |
| mura/A | 0.491 | 0.571 | 0.373 | 0.490 |

原全网格in-sample 5×3 recovery仍保留为历史估计对象：organamnist/A=0.910；organamnist/B=0.751；sipakmed/A=0.837；sipakmed/B=0.542。不得与上述disjoint参考值混称同一种“恢复率”；本轮没有修改这些旧值。

多数层中聚合预算提高与held-out参考的一致率，但SIPaKMeD/B在当前有限网格中并未单调提高。互补split块对非常接近的候选可能给出不同排序；聚合更多discovery结果不保证贴近另一参考块。因此不能写“增加重复次数必然收敛”或“5×3普遍足够”，也不能反过来把该下降解释为增加训练有害。

ISIC四CNN的5×3观察值为1.0，只说明在当前完成split网格及候选池中一致，不是普遍稳定性、外部验证或收敛定理。reference bootstrap和rank aggregation敏感性均报告，参考不是已知真值。disjoint仅指split ID分离，两个块仍可能包含同一来源图像；不称独立数据验证。

## D05：共同三seed敏感性

| 数据集/策略 | LSO：全5seed→共同3seed | 全网格参考身份改变 | 5×3 disjoint一致率：全5seed池→共同3seed池 |
|---|---|---|---|
| organamnist/A | 0.620 → 0.533 | 否 | 0.890 → 0.460 |
| organamnist/B | 0.600 → 0.667 | 否 | 0.587 → 0.587 |
| sipakmed/A | 0.520 → 0.467 | 否 | 0.694 → 0.635 |
| sipakmed/B | 0.220 → 0.367 | 是 | 0.204 → 0.381 |

原10×5保留为主分析，共同42/52/62作为既定敏感性。SIPaKMeD/B在子集中的参考身份改变，说明seed选择也影响有限网格的排序；不能用三seed结果替代原实验或解释为新的独立复验。原两数据集与扩展数据的endpoint、grouping及适配差异在跨数据集描述中保留。

## 对原RQ的结论性回答与修订建议

**RQ1：** 原高BA数据中的排名变化仍成立，完整A/B得到保留。扩展数据支持排名稳定性随所评数据和候选集合变化的条件性泛化；ISIC四CNN较稳定，MURA仍有明显变化。不能把原结论扩成所有任务都不稳定。

**RQ2：** split/seed选择、checkpoint规则与候选集合均提供可观察的评估因素信号。原方差结构证据、配对checkpoint结果与本轮敏感性各有对应范围，不能作单一原因或精确因果贡献分解。分差关联的区间跨1，保留为补充关联结果。

**RQ3：** 预算可在已完成网格内用实际聚合和参考敏感性估计。聚合常有帮助，但效果依数据集、seed池及参考定义而变；支持按研究条件报告预算与不确定性，不支持统一最低预算或必然收敛。

## 分析运行时与追溯

本轮计算Python 3.12.14、NumPy 2.3.5、pandas 2.2.3。未使用PyTorch或GPU训练；scipy/statsmodels/matplotlib未安装且未用于本轮数值计算。sign-flip与IRLS用NumPy直接实现，代码和交叉核对实现均保存。此运行时不能倒填到历史训练run。

核对3315份来源hash、20份主计算表hash；1100个raw run全部覆盖，16层点估计检查、150单元直接预算循环、四层旧bootstrap重放通过。另有完整交付manifest收录报告及附加核查输出。

第一次执行完成数值计算后，因记录未安装可选包的版本而在metadata阶段终止；修复仅允许未安装包记为null。再次执行的20份主计算CSV全部与第一次生成的hash一致；执行记录留存，未改变统计方法或数字。

## 统计解释核查（11/11）

| 核查项 | 本轮处理 / 边界 |
|---|---|
| Simpson反转 | 不把四数据集、A/B或候选池混成一个总体均值；关联模型有dataset固定效应但仍为补充 |
| 生态推断 | run/context稳定性不外推到单患者临床性能 |
| 选择偏倚 | 固定公开数据集，非随机抽取临床人群；限制保持 |
| collider | 不按结果筛选run、split或胜出模型后只报告有利效应 |
| 基率忽略 | BA是既定benchmark指标，不报告PPV/临床筛查效益推断 |
| 均值回归 | 有限子集/极端排名与全网格比较不称为干预改善 |
| 幸存者 | 全1100个计划完成run纳入对应层，未按性能排除 |
| 多重比较 | 模型、policy及输入方案分别预列family；完整输出保留，Holm不纠正split依赖 |
| 分析自由度 | 本轮在既有结果之后冻结说明，明确修订重分析而非追称预注册；旧结果不覆盖 |
| 因果误读 | checkpoint/分差/软件记录不作不稳定性单一因果归因 |
| 反向因果 | 分差与discordance同context关联不写因果方向或普遍机制 |
