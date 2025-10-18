#!/usr/bin/env python3
"""
验证数据一致性

检查训练数据和测试数据中的PII类型是否符合标准定义
"""

import json
import sys
from pathlib import Path
from collections import Counter

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.models.pii_types import PIIType, normalize_pii_type, ALL_17_TYPES


def validate_file(file_path: Path) -> dict:
    """
    验证单个文件

    Args:
        file_path: 数据文件路径

    Returns:
        验证结果统计
    """
    stats = {
        "total_cases": 0,
        "total_entities": 0,
        "standard_types": Counter(),
        "non_standard_types": Counter(),
        "is_valid": True
    }

    if not file_path.exists():
        print(f"⚠️  文件不存在: {file_path}")
        return stats

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
                stats["total_cases"] += 1

                if "entities" in case:
                    for entity in case["entities"]:
                        stats["total_entities"] += 1
                        pii_type = entity["type"]

                        # 检查是否为标准类型
                        if pii_type in [t.value for t in PIIType]:
                            stats["standard_types"][pii_type] += 1
                        else:
                            # 尝试标准化
                            normalized = normalize_pii_type(pii_type)
                            if normalized in [t.value for t in PIIType]:
                                stats["non_standard_types"][pii_type] += 1
                                print(f"  行 {line_num}: {pii_type} 应标准化为 {normalized}")
                                stats["is_valid"] = False
                            else:
                                stats["non_standard_types"][pii_type] += 1
                                print(f"  行 {line_num}: 未知类型 {pii_type}")
                                stats["is_valid"] = False

            except json.JSONDecodeError as e:
                print(f"  行 {line_num}: JSON解析错误 - {e}")
                stats["is_valid"] = False

    return stats


def main():
    """主函数"""
    print("=" * 70)
    print("✅ 数据一致性验证")
    print("=" * 70)

    # 要验证的文件
    files_to_check = [
        ("测试数据集", project_root / "data" / "test_datasets" / "17pii_test_cases.jsonl"),
        ("训练数据模板", project_root / "data" / "training" / "17pii_training_template.jsonl"),
    ]

    all_valid = True

    for name, file_path in files_to_check:
        print(f"\n📂 {name}: {file_path.name}")
        print("-" * 70)

        stats = validate_file(file_path)

        print(f"  测试用例数: {stats['total_cases']}")
        print(f"  实体总数: {stats['total_entities']}")

        if stats["standard_types"]:
            print(f"  标准类型数量: {len(stats['standard_types'])}")
            print(f"  标准实体数: {sum(stats['standard_types'].values())}")

        if stats["non_standard_types"]:
            print(f"\n  ⚠️  非标准类型:")
            for type_name, count in stats["non_standard_types"].most_common():
                print(f"    - {type_name}: {count} 次")
            all_valid = False
        else:
            print(f"  ✅ 所有类型均为标准格式")

        if not stats["is_valid"]:
            all_valid = False

    # 输出支持的标准类型
    print("\n" + "=" * 70)
    print("📋 标准PII类型列表 (共17种)")
    print("=" * 70)
    for i, pii_type in enumerate(ALL_17_TYPES, 1):
        print(f"  {i:2d}. {pii_type.value}")

    print("\n" + "=" * 70)
    if all_valid:
        print("✅ 所有文件验证通过！")
        return 0
    else:
        print("⚠️  发现非标准类型，请运行标准化脚本修正")
        print("   python scripts/normalize_test_dataset.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
