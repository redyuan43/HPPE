#!/usr/bin/env python3
"""简单合并SFT数据集"""

import json
import random
from pathlib import Path

# 配置
INPUT_FILES = [
    "data/pii_datasets/synthetic_pii.jsonl",
    "data/pii_datasets/msra/msra_train_sft.jsonl",
    "data/pii_datasets/peoples_daily/peoples_daily_train_sft.jsonl",
]

OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 读取所有数据
print("\n📚 读取数据集...")
all_samples = []

for file_path in INPUT_FILES:
    path = Path(file_path)
    if not path.exists():
        print(f"  ⚠️  文件不存在: {file_path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
        all_samples.extend(samples)
        print(f"  ✓ {path.name}: {len(samples):,} 样本")

print(f"\n总计: {len(all_samples):,} 样本")

# 随机打乱
print("\n🔀 随机打乱数据...")
random.seed(42)
random.shuffle(all_samples)

# 分割数据
train_size = int(len(all_samples) * TRAIN_RATIO)
val_size = int(len(all_samples) * VAL_RATIO)

train_samples = all_samples[:train_size]
val_samples = all_samples[train_size:train_size + val_size]
test_samples = all_samples[train_size + val_size:]

print(f"  训练集: {len(train_samples):,} 样本")
print(f"  验证集: {len(val_samples):,} 样本")
print(f"  测试集: {len(test_samples):,} 样本")

# 保存
print("\n💾 保存数据集...")
OUTPUT_DIR.mkdir(exist_ok=True)

datasets = [
    ("merged_pii_dataset_train.jsonl", train_samples),
    ("merged_pii_dataset_validation.jsonl", val_samples),
    ("merged_pii_dataset_test.jsonl", test_samples),
]

for filename, samples in datasets:
    output_path = OUTPUT_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  ✓ {filename}: {len(samples):,} 样本")

print("\n🎉 合并完成！")
