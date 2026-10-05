import os
import argparse
import json
import pandas as pd

# 定义数据目录
parser = argparse.ArgumentParser(description="Extract legacy cross-validation metric records")
parser.add_argument("--data-dir", required=True, help="Directory containing legacy per-fold metric folders")
data_dir = parser.parse_args().data_dir

# 模型列表
models = ["baseline", "se_layer4", "se_avgpool"]

# 种子列表
seeds = [42, 52, 62]

# 折数
folds = [0, 1, 2, 3, 4]

# 存储结果
data = []

# 遍历所有模型、种子和折
for model in models:
    for seed in seeds:
        for fold in folds:
            # 构建文件路径
            if model == "baseline":
                folder_name = f"{model}_seed{seed}_fold{fold}"
            elif model == "se_layer4":
                folder_name = f"{model}_seed{seed}_fold{fold}"
            elif model == "se_avgpool":
                folder_name = f"{model}_seed{seed}_fold{fold}"
            
            file_path = os.path.join(data_dir, folder_name, "metrics.json")
            
            # 检查文件是否存在
            if os.path.exists(file_path):
                try:
                    # 读取JSON文件
                    with open(file_path, 'r') as f:
                        metrics = json.load(f)
                    
                    # 提取准确率
                    accuracy = metrics.get("val_metrics", {}).get("accuracy", None)
                    
                    if accuracy is not None:
                        # 转换模型名称以匹配要求
                        model_name = model
                        if model == "baseline":
                            model_name = "Baseline"
                        elif model == "se_layer4":
                            model_name = "layer4"
                        elif model == "se_avgpool":
                            model_name = "avgpool"
                        
                        # 保存数据
                        data.append({
                            "model": model_name,
                            "seed": seed,
                            "fold": fold + 1,  # 转换为1-5的折号
                            "accuracy": accuracy * 100  # 转换为百分比
                        })
                        print(f"成功提取: {model_name}, seed={seed}, fold={fold+1}, accuracy={accuracy*100:.2f}")
                    else:
                        print(f"警告: {file_path} 中未找到accuracy字段")
                except Exception as e:
                    print(f"错误读取 {file_path}: {e}")
            else:
                print(f"警告: 文件不存在 {file_path}")

# 转换为DataFrame
df = pd.DataFrame(data)

# 按模型、种子、折排序
df = df.sort_values(by=["model", "seed", "fold"])

# 保存为CSV文件
output_file = "cv_results_long.csv"
df.to_csv(output_file, index=False)

print(f"\n数据提取完成！")
print(f"共提取 {len(df)} 条记录")
print(f"数据已保存为: {output_file}")

# 显示数据预览
print("\n数据预览:")
print(df.head())

# 检查数据完整性
print("\n数据完整性检查:")
for model in ["Baseline", "layer4", "avgpool"]:
    model_data = df[df["model"] == model]
    print(f"{model}: {len(model_data)} 条记录 (预期15条)")
    for seed in seeds:
        seed_data = model_data[model_data["seed"] == seed]
        print(f"  seed={seed}: {len(seed_data)} 条记录 (预期5条)")
