# CF04：选择规则与候选池变化下的首位模型身份变化（候选）

Historical rendering v2.0.4 from ST04/ST05; source for Figure 2b.

**English caption:** Changes in selected model identity under alternative evaluation conditions. (A) Fraction of matched split–seed contexts selecting different models under checkpoint selection Rules A and B, using 10 splits and five training seeds per original dataset. (B) Fraction selecting a different model after adding shared-input Swin-T to the four-CNN candidate pool, using five splits and three training seeds per extension dataset. Points show observed changed-context fractions; horizontal bars retain the existing paired nested percentile 95% intervals. Comparisons are paired within each panel; the panels represent different evaluation changes and are interpreted separately.

**用途：** 原RQ2的选择条件敏感性，R2.4的选择规则解释，R1.3/R2.5的候选池异质性回应。原Figure2继续提供BA差分布背景；本图显示选择条件改变时的模型身份变化。完整BA效应、输入方案对照与规格放补充材料。

**来源：** Panel A=ST04/T57（OrganAMNIST 0.460 [0.280,0.640]；SIPaKMeD 0.560 [0.360,0.740]）；Panel B=ST05/T58共享输入（ISIC2019 0.400 [0.067,0.733]；MURA 0.800 [0.533,1.000]）。本图按已生成表的显示精度作图；正式排图可直接使用同源未舍入CSV，不改变既定算法。
