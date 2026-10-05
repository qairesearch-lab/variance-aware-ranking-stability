import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
from statsmodels.regression.mixed_linear_model import MixedLM 

# ---------- 1. 读入真实数据 ---------- 
df = pd.read_csv("data/cv_results_long.csv") 
df['model'] = df['model'].astype('category') 
df['seed'] = df['seed'].astype('category') 
df['fold'] = df['fold'].astype('category') 

# 检查数据结构和基本统计 
print("数据概览：") 
print(df.head()) 
print(f"\n总观测数: {len(df)}") 
print(f"模型: {df['model'].unique().tolist()}") 
print(f"种子: {df['seed'].unique().tolist()}") 
print(f"折: {df['fold'].unique().tolist()}") 

# ---------- 2. 拟合简化混合效应模型 ---------- 
# 模型 1：固定效应 model，随机截距 fold 和 seed（无交互） 
print("\n" + "=" * 60) 
print("模型 1: accuracy ~ C(model) + (1|fold) + (1|seed)") 
model1 = MixedLM.from_formula( 
    formula="accuracy ~ C(model)", 
    data=df, 
    re_formula="1", 
    vc_formula={ 
        "fold": "0 + C(fold)", 
        "seed": "0 + C(seed)" 
    },
    groups=df.index
) 
result1 = model1.fit(method='lbfgs', maxiter=2000) 
print(result1.summary()) 

# 模型 2：增加 model:fold 交互项 
print("\n" + "=" * 60) 
print("模型 2: accuracy ~ C(model) + (1|fold) + (1|seed) + (1|model:fold)") 
model2 = MixedLM.from_formula( 
    formula="accuracy ~ C(model)", 
    data=df, 
    re_formula="1", 
    vc_formula={ 
        "fold": "0 + C(fold)", 
        "seed": "0 + C(seed)", 
        "model_fold": "0 + C(model):C(fold)" 
    },
    groups=df.index
) 
result2 = model2.fit(method='lbfgs', maxiter=2000) 
print(result2.summary()) 

# ---------- 3. 提取方差分量 ---------- 
def extract_variance_components(result, model_name): 
    vc = result.variance_comp 
    scale = result.scale 
    components = {} 
    for key, val in vc.items(): 
        if key == 'C(fold)': 
            components['fold'] = val 
        elif key == 'C(seed)': 
            components['seed'] = val 
        elif key == 'C(model):C(fold)': 
            components['model:fold'] = val 
        elif key == 'C(model):C(seed)': 
            components['model:seed'] = val 
    components['residual'] = scale 
    total = sum(components.values()) 
    df_comp = pd.DataFrame({ 
        'component': list(components.keys()), 
        'variance': list(components.values()), 
        'percent': [v / total * 100 for v in components.values()] 
    }) 
    print(f"\n{model_name} 方差分量分解：") 
    print(df_comp.round(6)) 
    return df_comp 

comp1 = extract_variance_components(result1, "模型 1") 
comp2 = extract_variance_components(result2, "模型 2") 

# ---------- 4. 模型比较（对数似然） ---------- 
print("\n" + "=" * 60) 
print("模型比较（对数似然值）：") 
print(f"模型 1 Log-Likelihood: {result1.llf:.3f}") 
print(f"模型 2 Log-Likelihood: {result2.llf:.3f}") 

# ---------- 5. 简单描述性方差估计（不依赖模型）---------- 
print("\n" + "=" * 60) 
print("描述性方差估计（组内均值方差）：") 
fold_means = df.groupby('fold')['accuracy'].mean() 
var_fold_desc = fold_means.var(ddof=0) 
seed_means = df.groupby('seed')['accuracy'].mean() 
var_seed_desc = seed_means.var(ddof=0) 
total_var_desc = df['accuracy'].var(ddof=0) 
print(f"Fold均值方差: {var_fold_desc:.6f} ({var_fold_desc/total_var_desc*100:.1f}%)") 
print(f"Seed均值方差: {var_seed_desc:.6f} ({var_seed_desc/total_var_desc*100:.1f}%)") 
print(f"总方差: {total_var_desc:.6f}")