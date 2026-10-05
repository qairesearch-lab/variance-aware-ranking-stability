# Figure2扩展候选：2a与2b v2.0.5

作者确认Figure2合并展示；正文仍1表3图，不另列Figure4。原前二BA差为2a，新增规则/候选池敏感性为2b。当前是候选排图，正文v2.0未整合。

**English caption:** Performance separation and sensitivity of model selection to evaluation conditions. (a) Distribution of the difference between the two highest balanced-accuracy (BA) values within each observed split–seed context, in percentage points (pp). The leading pair is reselected within each context. Boxes show the median and interquartile range; whiskers extend to 1.5 interquartile ranges, and points show all observed contexts. Subplot y-ranges differ. (b) Fractions of matched split–seed contexts selecting a different model under alternative evaluation conditions. The checkpoint selection rule comparison uses 10 splits and five seeds per original dataset. The candidate-pool comparison adds shared-input Swin-T to four CNNs using five splits and three seeds per extension dataset. Horizontal bars retain existing paired nested percentile 95% intervals. The two condition comparisons are interpreted separately; their fractions describe changes in selected model identity, not changes in discordance rates.

来源：2a逐行复制原CF02数据/T52；2b逐行复制原CF04数据/T83（T57/T58重排）。不增加实验、重采样或统计检验。所有字段、原始数值和区间保持来源版本。2a的数据位置分布与2b的条件对照比例分别定义；固定模型显著性分析在ST02，A/B BA差在ST03。
