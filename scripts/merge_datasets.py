#!/usr/bin/env python3
"""
合并多个 PII 数据集为统一的 SFT 训练格式

支持的数据集格式：
1. ai4privacy - PII脱敏数据集
2. MSRA NER - BIO标注格式
3. 合成数据 - 已经是SFT格式

输出格式：
{
  "instruction": "检测以下文本中的 PII，并以 JSON 格式输出实体列表。",
  "input": "我叫张三，电话13800138000。",
  "output": {
    "entities": [
      {"type": "PERSON_NAME", "value": "张三", "start": 2, "end": 4},
      {"type": "PHONE_NUMBER", "value": "13800138000", "start": 6, "end": 17}
    ]
  }
}

使用示例：
    # 合并所有数据集，按推荐比例
    python scripts/merge_datasets.py --all --output data/merged_pii_dataset.jsonl

    # 自定义数据集和比例
    python scripts/merge_datasets.py \\
        --datasets ai4privacy:0.3 msra:0.3 synthetic:0.4 \\
        --output data/custom_dataset.jsonl

    # 指定总样本数
    python scripts/merge_datasets.py --all --total-samples 50000 --output data/pii_50k.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

try:
    from datasets import load_from_disk, Dataset
except ImportError:
    print("❌ 缺少 datasets 库。请先安装：")
    print("   pip install datasets")
    sys.exit(1)


# PII类型映射（标准化不同数据集的实体类型）
ENTITY_TYPE_MAPPING = {
    # MSRA NER标签到标准类型
    "PER": "PERSON_NAME",
    "nr": "PERSON_NAME",
    "LOC": "ADDRESS",
    "ns": "ADDRESS",
    "ORG": "ORGANIZATION",
    "nt": "ORGANIZATION",

    # ai4privacy标签到标准类型
    "NAME": "PERSON_NAME",
    "PERSON": "PERSON_NAME",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE_NUMBER",
    "PHONE_NUM": "PHONE_NUMBER",
    "LOCATION": "ADDRESS",
    "CITY": "ADDRESS",
    "STREET": "ADDRESS",

    # BigCode标签到标准类型
    "KEY": "API_KEY",
    "IP": "IP_ADDRESS",
    "USERNAME": "USERNAME",

    # 已经是标准类型的
    "PERSON_NAME": "PERSON_NAME",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "EMAIL": "EMAIL",
    "ID_CARD": "ID_CARD",
    "BANK_CARD": "BANK_CARD",
    "ADDRESS": "ADDRESS",
    "ORGANIZATION": "ORGANIZATION",
    "IP_ADDRESS": "IP_ADDRESS",
}


def convert_ai4privacy_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    转换 ai4privacy 数据集样本为 SFT 格式

    ai4privacy 格式：
    {
      "source_text": "My name is [NAME] and I live in [CITY]",
      "target_text": "My name is John Smith and I live in New York",
      "privacy_mask": {
        "[NAME]": "John Smith",
        "[CITY]": "New York"
      }
    }
    """
    try:
        target_text = sample.get("target_text", "")
        privacy_mask = sample.get("privacy_mask", {})

        if not target_text or not privacy_mask:
            return None

        # 提取实体
        entities = []
        for placeholder, value in privacy_mask.items():
            # 从占位符推断类型
            placeholder_upper = placeholder.upper()

            if "NAME" in placeholder_upper:
                entity_type = "PERSON_NAME"
            elif "EMAIL" in placeholder_upper:
                entity_type = "EMAIL"
            elif "PHONE" in placeholder_upper:
                entity_type = "PHONE_NUMBER"
            elif "CITY" in placeholder_upper or "LOCATION" in placeholder_upper or "STREET" in placeholder_upper:
                entity_type = "ADDRESS"
            elif "ORG" in placeholder_upper or "COMPANY" in placeholder_upper:
                entity_type = "ORGANIZATION"
            else:
                # 跳过未知类型
                continue

            # 在文本中查找实体位置
            start = target_text.find(value)
            if start != -1:
                entities.append({
                    "type": entity_type,
                    "value": value,
                    "start": start,
                    "end": start + len(value)
                })

        if not entities:
            return None

        return {
            "instruction": "检测以下文本中的 PII，并以 JSON 格式输出实体列表。",
            "input": target_text,
            "output": {
                "entities": entities
            }
        }

    except Exception as e:
        return None


def convert_msra_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    转换 MSRA NER 数据集样本为 SFT 格式

    MSRA 格式（BIO标注）：
    {
      "tokens": ["我", "叫", "张", "三"],
      "ner_tags": [0, 0, 1, 2]  # 0=O, 1=B-PER, 2=I-PER, ...
    }
    """
    try:
        tokens = sample.get("tokens", [])
        ner_tags = sample.get("ner_tags", [])

        if not tokens or not ner_tags or len(tokens) != len(ner_tags):
            return None

        # 标签ID到名称的映射（MSRA使用的标签）
        # 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC
        tag_names = {
            0: "O",
            1: "B-PER", 2: "I-PER",
            3: "B-ORG", 4: "I-ORG",
            5: "B-LOC", 6: "I-LOC"
        }

        # 重建文本
        text = "".join(tokens)

        # 提取实体
        entities = []
        current_entity = None
        current_start = 0

        for i, (token, tag_id) in enumerate(zip(tokens, ner_tags)):
            tag = tag_names.get(tag_id, "O")

            if tag.startswith("B-"):
                # 保存之前的实体
                if current_entity:
                    entity_type = ENTITY_TYPE_MAPPING.get(current_entity["raw_type"], current_entity["raw_type"])
                    entities.append({
                        "type": entity_type,
                        "value": current_entity["value"],
                        "start": current_entity["start"],
                        "end": current_entity["end"]
                    })

                # 开始新实体
                current_entity = {
                    "raw_type": tag[2:],  # PER, ORG, LOC
                    "value": token,
                    "start": current_start,
                    "end": current_start + len(token)
                }

            elif tag.startswith("I-") and current_entity:
                # 继续当前实体
                current_entity["value"] += token
                current_entity["end"] = current_start + len(token)

            else:  # O
                # 保存之前的实体
                if current_entity:
                    entity_type = ENTITY_TYPE_MAPPING.get(current_entity["raw_type"], current_entity["raw_type"])
                    entities.append({
                        "type": entity_type,
                        "value": current_entity["value"],
                        "start": current_entity["start"],
                        "end": current_entity["end"]
                    })
                    current_entity = None

            current_start += len(token)

        # 保存最后的实体
        if current_entity:
            entity_type = ENTITY_TYPE_MAPPING.get(current_entity["raw_type"], current_entity["raw_type"])
            entities.append({
                "type": entity_type,
                "value": current_entity["value"],
                "start": current_entity["start"],
                "end": current_entity["end"]
            })

        if not entities:
            return None

        return {
            "instruction": "检测以下文本中的 PII，并以 JSON 格式输出实体列表。",
            "input": text,
            "output": {
                "entities": entities
            }
        }

    except Exception as e:
        return None


def load_dataset_samples(
    dataset_path: Path,
    dataset_type: str,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    加载并转换数据集

    Args:
        dataset_path: 数据集路径
        dataset_type: 数据集类型 (ai4privacy, msra, synthetic)
        max_samples: 最大样本数

    Returns:
        转换后的样本列表
    """
    print(f"\n加载数据集: {dataset_path.name} ({dataset_type})")

    samples = []

    if dataset_type == "synthetic":
        # 合成数据已经是SFT格式
        if dataset_path.suffix == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    samples.append(sample)
                    if max_samples and len(samples) >= max_samples:
                        break
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else [data]
                if max_samples:
                    samples = samples[:max_samples]

    elif dataset_type == "ai4privacy":
        # ai4privacy 数据集
        dataset = load_from_disk(str(dataset_path))

        # 获取训练集
        if hasattr(dataset, "keys"):
            split = dataset["train"] if "train" in dataset else list(dataset.values())[0]
        else:
            split = dataset

        # 转换样本
        for i, sample in enumerate(split):
            if max_samples and i >= max_samples:
                break

            converted = convert_ai4privacy_sample(sample)
            if converted:
                samples.append(converted)

    elif dataset_type == "msra":
        # MSRA NER 数据集
        dataset = load_from_disk(str(dataset_path))

        # 获取训练集
        if hasattr(dataset, "keys"):
            split = dataset["train"] if "train" in dataset else list(dataset.values())[0]
        else:
            split = dataset

        # 转换样本
        for i, sample in enumerate(split):
            if max_samples and i >= max_samples:
                break

            converted = convert_msra_sample(sample)
            if converted:
                samples.append(converted)

    print(f"  ✓ 加载了 {len(samples)} 个样本")
    return samples


def merge_datasets(
    dataset_configs: List[Dict[str, Any]],
    total_samples: Optional[int] = None,
    output_path: Optional[Path] = None,
    train_val_test_split: tuple = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    合并多个数据集

    Args:
        dataset_configs: 数据集配置列表 [{"path": Path, "type": str, "ratio": float}]
        total_samples: 总样本数（None表示使用所有样本）
        output_path: 输出文件路径
        train_val_test_split: 训练/验证/测试集划分比例
        seed: 随机种子

    Returns:
        合并后的数据集 {"train": [...], "validation": [...], "test": [...]}
    """
    random.seed(seed)

    print(f"\n{'='*70}")
    print("合并 PII 数据集")
    print(f"{'='*70}")
    print(f"数据集配置:")
    for config in dataset_configs:
        print(f"  - {config['type']}: {config['ratio']*100:.0f}%")
    if total_samples:
        print(f"总样本数: {total_samples}")
    print(f"划分比例: 训练{train_val_test_split[0]*100:.0f}% / "
          f"验证{train_val_test_split[1]*100:.0f}% / "
          f"测试{train_val_test_split[2]*100:.0f}%")
    print()

    # 加载所有数据集
    all_samples = []

    for config in dataset_configs:
        # 计算此数据集需要的样本数
        if total_samples:
            max_samples = int(total_samples * config["ratio"])
        else:
            max_samples = None

        # 加载样本
        samples = load_dataset_samples(
            config["path"],
            config["type"],
            max_samples
        )

        all_samples.extend(samples)

    print(f"\n✓ 总共加载 {len(all_samples)} 个样本")

    # 打乱数据
    print("\n打乱数据...")
    random.shuffle(all_samples)

    # 划分训练/验证/测试集
    print("划分数据集...")
    train_size = int(len(all_samples) * train_val_test_split[0])
    val_size = int(len(all_samples) * train_val_test_split[1])

    splits = {
        "train": all_samples[:train_size],
        "validation": all_samples[train_size:train_size + val_size],
        "test": all_samples[train_size + val_size:]
    }

    # 统计信息
    print(f"\n数据集统计:")
    for split_name, split_samples in splits.items():
        print(f"  {split_name}: {len(split_samples)} 样本")

    # 统计实体类型分布
    print(f"\n实体类型分布（训练集）:")
    entity_type_counts = defaultdict(int)
    total_entities = 0

    for sample in splits["train"]:
        for entity in sample["output"]["entities"]:
            entity_type_counts[entity["type"]] += 1
            total_entities += 1

    for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_entities * 100 if total_entities > 0 else 0
        print(f"  {entity_type}: {count} ({percentage:.1f}%)")

    # 保存到文件
    if output_path:
        print(f"\n保存到文件...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".jsonl":
            # JSONL格式 - 每个split分别保存
            for split_name, split_samples in splits.items():
                split_path = output_path.with_name(f"{output_path.stem}_{split_name}.jsonl")
                with open(split_path, "w", encoding="utf-8") as f:
                    for sample in split_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                print(f"  ✓ {split_name}: {split_path}")

        else:
            # JSON格式 - 保存为一个文件
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(splits, f, ensure_ascii=False, indent=2)
            print(f"  ✓ {output_path}")

    # 显示示例
    print(f"\n示例数据（训练集）:")
    example = splits["train"][0]
    print(f"  输入: {example['input'][:100]}...")
    print(f"  实体数: {len(example['output']['entities'])}")
    print(f"  实体: {[(e['type'], e['value']) for e in example['output']['entities'][:3]]}")

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="合并多个 PII 数据集为统一的 SFT 训练格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用推荐配置合并所有数据集
  python scripts/merge_datasets.py --all --output data/merged_pii_dataset.jsonl

  # 自定义数据集和比例
  python scripts/merge_datasets.py \\
      --datasets ai4privacy:0.3 msra:0.3 synthetic:0.4 \\
      --output data/custom_dataset.jsonl

  # 指定总样本数
  python scripts/merge_datasets.py --all --total-samples 50000

  # 自定义划分比例
  python scripts/merge_datasets.py --all --split 0.7 0.2 0.1
        """
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="数据集配置，格式：name:ratio (例如: ai4privacy:0.3 msra:0.3)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="使用推荐配置 (ai4privacy:0.3 + msra:0.3 + synthetic:0.4)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pii_datasets",
        help="数据集根目录 (默认: data/pii_datasets)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/merged_pii_dataset.jsonl",
        help="输出文件路径 (默认: data/merged_pii_dataset.jsonl)"
    )

    parser.add_argument(
        "--total-samples",
        type=int,
        help="总样本数（不指定则使用所有样本）"
    )

    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help="训练/验证/测试集划分比例 (默认: 0.8 0.1 0.1)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    # 解析数据集配置
    data_dir = Path(args.data_dir)

    if args.all:
        # 推荐配置
        dataset_configs = [
            {
                "path": data_dir / "ai4privacy",
                "type": "ai4privacy",
                "ratio": 0.3
            },
            {
                "path": data_dir / "msra",
                "type": "msra",
                "ratio": 0.3
            },
            {
                "path": data_dir / "synthetic_pii.jsonl",
                "type": "synthetic",
                "ratio": 0.4
            }
        ]
    elif args.datasets:
        # 自定义配置
        dataset_configs = []
        for config_str in args.datasets:
            name, ratio = config_str.split(":")
            ratio = float(ratio)

            if name == "synthetic":
                path = data_dir / "synthetic_pii.jsonl"
                dataset_type = "synthetic"
            else:
                path = data_dir / name
                dataset_type = name

            dataset_configs.append({
                "path": path,
                "type": dataset_type,
                "ratio": ratio
            })
    else:
        parser.print_help()
        return

    # 验证比例总和
    total_ratio = sum(config["ratio"] for config in dataset_configs)
    if abs(total_ratio - 1.0) > 0.01:
        print(f"⚠️  警告：比例总和为 {total_ratio:.2f}，不等于 1.0")

    # 合并数据集
    output_path = Path(args.output)
    splits = merge_datasets(
        dataset_configs=dataset_configs,
        total_samples=args.total_samples,
        output_path=output_path,
        train_val_test_split=tuple(args.split),
        seed=args.seed
    )

    print(f"\n🎉 数据集合并完成！")
    print(f"\n后续步骤:")
    print(f"  1. 查看生成的数据文件: {output_path}")
    print(f"  2. 使用训练脚本开始训练")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
