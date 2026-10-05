import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv("data/cv_results_long.csv")
print("数据读取成功，共" + str(len(df)) + "条记录")
print("数据前5行:")
print(df.head())

# 拟合混合效应模型
print("\n拟合混合效应模型...")
model = MixedLM.from_formula(
    formula = "accuracy ~ 1",
    data = df,
    re_formula = "1",
    vc_formula = {
        "seed": "0 + C(seed)",
        "fold": "0 + C(fold)",
        "model_seed": "0 + C(model):C(seed)",
        "model_fold": "0 + C(model):C(fold)"
    },
    groups=df.index
)

result = model.fit(method='lbfgs')
print("模型拟合完成")
print("\n模型摘要:")
print(result.summary())

# 提取方差分量
print("\nVariance Components:")
print(result.variance_comp)

# 计算方差比例
vc = result.variance_comp
total_var = sum(vc.values()) + result.scale

components = {
    'model': 0,  # 模型间差异（固定效应）
    'seed': vc.get('C(seed)', 0),
    'fold': vc.get('C(fold)', 0),
    'model:seed': vc.get('C(model):C(seed)', 0),
    'model:fold': vc.get('C(model):C(fold)', 0),
    'residual': result.scale
}

print("\n方差分量及占比:")
for comp, var in components.items():
    print(f"{comp}: {var:.6f} ({var/total_var*100:.1f}%)")

# 可视化方差分解
print("\n生成方差分解图...")
labels = list(components.keys())
values = list(components.values())

plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.title('方差分解结果')
plt.ylabel('方差值')
plt.xlabel('方差分量')
plt.xticks(rotation=45)

# 在每个柱子上添加百分比
for i, (label, value) in enumerate(zip(labels, values)):
    percent = (value / total_var) * 100
    plt.text(i, value + 0.01, f'{percent:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('variance_decomposition.png')
print("方差分解图已保存为 variance_decomposition.png")

# 生成实验报告
report_content = f"""# 1.0数据方差分解实验报告

## 实验目的
对1.0实验的15次运行数据进行方差分解，识别模型排序不稳定性的主要来源。

## 数据描述
- 数据来源: 1.0实验记录（Supplementary Table 1）
- 数据格式: 长格式（long format）
- 样本量: 45个观测值（3个模型 × 3个种子 × 5个折）

## 统计模型
使用交叉随机效应模型：
```
accuracy ~ 1 + (1 | seed) + (1 | fold) + (1 | model:seed) + (1 | model:fold) + residual
```

## 结果分析

### 方差分量估计
{result.variance_comp}

### 方差占比
"""

for comp, var in components.items():
    percent = (var / total_var) * 100
    report_content += f"- {comp}: {var:.6f} ({percent:.1f}%)\n"

report_content += """

### 可视化结果
![方差分解图](variance_decomposition.png)

## 结论与启示
"""

# 分析主要方差来源
max_var_comp = max(components, key=components.get)
if components['fold'] > components['seed'] and components['fold'] > components['model:fold']:
   启示 = "fold引起的方差占比最大，说明数据划分方式对模型性能影响显著，印证了引入repeated outer split的必要性。"
elif components['seed'] > components['fold'] and components['seed'] > components['model:fold']:
   启示 = "seed引起的方差占比最大，说明初始化对模型性能影响显著，印证了multiple seeds的必要性。"
elif components['model:fold'] > components['fold'] and components['model:fold'] > components['seed']:
   启示 = "model:fold交互引起的方差占比最大，说明不同模型在不同数据划分下排名会剧烈变化，印证了rank stability报告的必要性。"
else:
   启示 = "残差方差占比最大，说明还有未建模的噪声（可能是seed:fold交互），提示需要更多重复。"

report_content += f"- {启示}\n"
report_content += """

## 对2.0实验的意义
方差分解结果表明，{max_var_comp}是模型排序不稳定性的主要来源，这为2.0实验设计提供了重要依据：
1. 需要引入repeated outer split来控制split-induced variance
2. 需要使用multiple seeds来控制初始化差异
3. 需要报告rank stability以评估模型在不同数据划分下的一致性

## 注意事项
- 样本量较小（45个观测点），方差分量估计的置信区间较宽
- 结果为探索性、描述性分析，不依赖p值做推断
- 1.0数据只有单次outer split，无法估计split-induced variance
"""

with open('experiment_report.md', 'w') as f:
    f.write(report_content)

print("\n实验报告已生成: experiment_report.md")
print("\n实验完成！")