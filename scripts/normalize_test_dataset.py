#!/usr/bin/env python3
"""
标准化测试数据集的PII类型名称

将测试数据中的旧类型名称转换为标准类型名称
"""

import json
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.models.pii_types import normalize_pii_type


def normalize_dataset(input_file: Path, output_file: Path) -> dict:
    """
    标准化数据集中的PII类型名称

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径

    Returns:
        统计信息字典
    """
    stats = {
        "total_cases": 0,
        "total_entities": 0,
        "normalized_count": 0,
        "type_mapping": {},
    }

    normalized_cases = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            case = json.loads(line)
            stats["total_cases"] += 1

            # 标准化实体类型
            if "entities" in case:
                for entity in case["entities"]:
                    stats["total_entities"] += 1
                    old_type = entity["type"]
                    new_type = normalize_pii_type(old_type)

                    if old_type != new_type:
                        stats["normalized_count"] += 1
                        # 记录映射
                        if old_type not in stats["type_mapping"]:
                            stats["type_mapping"][old_type] = new_type
                        print(f"  标准化: {old_type} -> {new_type}")

                    entity["type"] = new_type

            normalized_cases.append(case)

    # 写入标准化后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for case in normalized_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    return stats


def main():
    """主函数"""
    # 输入输出文件
    input_file = project_root / "data" / "test_datasets" / "17pii_test_cases.jsonl"
    output_file = project_root / "data" / "test_datasets" / "17pii_test_cases_normalized.jsonl"

    print("=" * 70)
    print("📋 标准化测试数据集PII类型")
    print("=" * 70)

    print(f"\n输入文件: {input_file}")
    print(f"输出文件: {output_file}")

    if not input_file.exists():
        print(f"\n❌ 错误：输入文件不存在: {input_file}")
        return 1

    # 执行标准化
    print("\n🔄 开始标准化...")
    stats = normalize_dataset(input_file, output_file)

    # 输出统计信息
    print("\n" + "=" * 70)
    print("📊 标准化结果统计")
    print("=" * 70)
    print(f"总测试用例数: {stats['total_cases']}")
    print(f"总实体数: {stats['total_entities']}")
    print(f"标准化实体数: {stats['normalized_count']}")

    if stats["type_mapping"]:
        print(f"\n类型映射关系:")
        for old, new in sorted(stats["type_mapping"].items()):
            print(f"  {old:30s} -> {new}")
    else:
        print("\n✅ 所有类型已是标准格式，无需修改")

    print(f"\n✅ 标准化完成！")
    print(f"   输出文件: {output_file}")

    # 备份原文件
    backup_file = input_file.with_suffix('.jsonl.backup')
    if not backup_file.exists():
        import shutil
        shutil.copy2(input_file, backup_file)
        print(f"   原文件备份: {backup_file}")

    # 替换原文件
    output_file.replace(input_file)
    print(f"   已更新原文件: {input_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
