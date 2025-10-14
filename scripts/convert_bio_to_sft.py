#!/usr/bin/env python3
"""
将 BIO 格式的 NER 数据转换为 SFT 训练格式

输入格式（BIO）：
当 O
希 O
望 O
工 O
程 O
救 O
助 O

输出格式（SFT）：
{
  "instruction": "检测以下文本中的 PII，并以 JSON 格式输出实体列表。",
  "input": "当希望工程救助",
  "output": {
    "entities": [...]
  }
}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# PII类型映射
ENTITY_TYPE_MAPPING = {
    "PER": "PERSON_NAME",
    "LOC": "ADDRESS",
    "ORG": "ORGANIZATION",
    "GPE": "ADDRESS",  # Geo-Political Entity
}


def parse_bio_file(file_path: Path) -> List[List[tuple]]:
    """
    解析 BIO 格式文件

    Returns:
        句子列表，每个句子是 [(字符, 标签), ...] 的列表
    """
    sentences = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                # 空行表示句子结束
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # 解析：字符\t标签
            parts = line.split()
            if len(parts) >= 2:
                char = parts[0]
                tag = parts[1]
                current_sentence.append((char, tag))

        # 添加最后一个句子
        if current_sentence:
            sentences.append(current_sentence)

    return sentences


def convert_bio_sentence_to_sft(sentence: List[tuple]) -> Dict[str, Any]:
    """
    将 BIO 格式的句子转换为 SFT 格式

    Args:
        sentence: [(字符, 标签), ...] 列表

    Returns:
        SFT 格式的样本
    """
    # 重建文本
    text = "".join([char for char, tag in sentence])

    # 提取实体
    entities = []
    current_entity = None
    current_start = 0

    for i, (char, tag) in enumerate(sentence):
        if tag.startswith("B-"):
            # 保存之前的实体
            if current_entity:
                entity_type = ENTITY_TYPE_MAPPING.get(
                    current_entity["raw_type"],
                    current_entity["raw_type"]
                )
                entities.append({
                    "type": entity_type,
                    "value": current_entity["value"],
                    "start": current_entity["start"],
                    "end": current_entity["end"]
                })

            # 开始新实体
            raw_type = tag[2:]  # PER, ORG, LOC, etc.
            current_entity = {
                "raw_type": raw_type,
                "value": char,
                "start": i,
                "end": i + 1
            }

        elif tag.startswith("I-") and current_entity:
            # 继续当前实体
            current_entity["value"] += char
            current_entity["end"] = i + 1

        else:  # O
            # 保存之前的实体
            if current_entity:
                entity_type = ENTITY_TYPE_MAPPING.get(
                    current_entity["raw_type"],
                    current_entity["raw_type"]
                )
                entities.append({
                    "type": entity_type,
                    "value": current_entity["value"],
                    "start": current_entity["start"],
                    "end": current_entity["end"]
                })
                current_entity = None

    # 保存最后的实体
    if current_entity:
        entity_type = ENTITY_TYPE_MAPPING.get(
            current_entity["raw_type"],
            current_entity["raw_type"]
        )
        entities.append({
            "type": entity_type,
            "value": current_entity["value"],
            "start": current_entity["start"],
            "end": current_entity["end"]
        })

    # 构建 SFT 格式
    return {
        "instruction": "检测以下文本中的 PII，并以 JSON 格式输出实体列表。",
        "input": text,
        "output": {
            "entities": entities
        }
    }


def convert_bio_to_sft(
    input_file: Path,
    output_file: Path,
    max_samples: int = None,
    show_progress: bool = True
) -> int:
    """
    转换 BIO 文件为 SFT 格式

    Returns:
        转换的样本数
    """
    print(f"\n转换 BIO 文件: {input_file.name}")
    print(f"输出: {output_file}")

    # 解析 BIO 文件
    print(f"\n读取 BIO 文件...")
    sentences = parse_bio_file(input_file)
    print(f"  ✓ 读取了 {len(sentences)} 个句子")

    if max_samples:
        sentences = sentences[:max_samples]
        print(f"  限制样本数: {len(sentences)}")

    # 转换为 SFT 格式
    print(f"\n转换为 SFT 格式...")
    sft_samples = []
    samples_with_entities = 0

    for i, sentence in enumerate(sentences):
        if show_progress and (i + 1) % 10000 == 0:
            print(f"  进度: {i + 1}/{len(sentences)}", end="\r")

        sft_sample = convert_bio_sentence_to_sft(sentence)

        # 只保留包含实体的样本
        if sft_sample["output"]["entities"]:
            sft_samples.append(sft_sample)
            samples_with_entities += 1

    if show_progress:
        print(f"  进度: {len(sentences)}/{len(sentences)}")

    print(f"\n✓ 转换完成")
    print(f"  总句子数: {len(sentences)}")
    print(f"  有实体的: {samples_with_entities}")
    print(f"  转换率: {samples_with_entities / len(sentences) * 100:.1f}%")

    # 统计实体类型
    entity_type_counts = {}
    for sample in sft_samples:
        for entity in sample["output"]["entities"]:
            entity_type = entity["type"]
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

    print(f"\n实体类型分布:")
    for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / sum(entity_type_counts.values()) * 100
        print(f"  {entity_type}: {count} ({percentage:.1f}%)")

    # 保存到文件
    print(f"\n保存到文件...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  ✓ 已保存到: {output_file}")

    # 显示示例
    if sft_samples:
        print(f"\n示例数据:")
        example = sft_samples[0]
        print(f"  输入: {example['input'][:50]}...")
        print(f"  实体数: {len(example['output']['entities'])}")
        if example['output']['entities']:
            print(f"  实体: {[(e['type'], e['value']) for e in example['output']['entities'][:3]]}")

    return len(sft_samples)


def main():
    parser = argparse.ArgumentParser(
        description="将 BIO 格式的 NER 数据转换为 SFT 训练格式",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入的 BIO 格式文件"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出的 JSONL 文件"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="最大样本数（用于测试）"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)

    # 转换
    num_samples = convert_bio_to_sft(
        input_file,
        output_file,
        max_samples=args.max_samples,
        show_progress=True
    )

    print(f"\n🎉 转换完成！共 {num_samples} 个样本")


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
