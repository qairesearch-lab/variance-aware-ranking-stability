# Table1统计证据候选表 v0.1

状态：供B2/B4整合；不是已修改的投稿Table1。原800的主设计保留，新增两数据集另行标注。

| Dataset / rule | Full-grid reference top model | Observed contexts | Agreement with full-grid reference [95% hierarchical interval] | Identity-correct LSO agreement [95% hierarchical interval] | Paired hierarchical bootstrap selection frequency of full-grid reference model | Aggregate5×3 agreement with disjoint split-ID block |
|---|---|---:|---|---|---:|---:|
| organamnist / A | resnet18 | 50 | 0.620 [0.440, 0.800] | 0.620 [0.300, 0.800] | 0.9518 | 0.890 |
| organamnist / B | resnet50 | 50 | 0.600 [0.400, 0.780] | 0.600 [0.260, 0.760] | 0.7723 | 0.587 |
| sipakmed / A | resnet18 | 50 | 0.520 [0.360, 0.740] | 0.520 [0.120, 0.740] | 0.8688 | 0.694 |
| sipakmed / B | resnet50 | 50 | 0.440 [0.380, 0.680] | 0.220 [0.100, 0.680] | 0.5346 | 0.204 |
| isic2019 / A | resnet50 | 30 | 0.900 [0.733, 1.000] | 0.900 [0.733, 1.000] | 1.0000 | 1.000 |
| mura / A | resnet50 | 30 | 0.600 [0.300, 0.800] | 0.600 [0.133, 0.800] | 0.7824 | 0.571 |

CI条件于当前固定数据集的评估随机性，10000抽样不作为实际N。最后列是252有方向5/5划分内真实聚合的有限网格统计，不附二项CI；disjoint仅指split ID，不能声称来源图像独立。原数据budget seed子集来自完整五seed池，扩展来自三seed池；共同三seed另表。Full-grid模型名称仅用于定位参考，不表示普遍模型优劣；每个LSO/disjoint块的实际参考身份另表保存。
