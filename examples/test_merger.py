#!/usr/bin/env python3
"""
实体合并器快速验证

测试EntityMerger在真实场景下的表现
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.models.entity import Entity
from hppe.refiner.merger import EntityMerger
from hppe.refiner.config import MergerConfig


def test_scenario_1_person_name_overlap():
    """场景1: 人名重叠（"张三" vs "张三先生"）"""
    print("\n" + "="*70)
    print("场景1: 人名重叠合并")
    print("="*70)

    entities = [
        Entity(
            entity_type="PERSON_NAME",
            value="张三",
            start_pos=2,
            end_pos=4,
            confidence=0.90,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        Entity(
            entity_type="PERSON_NAME",
            value="张三先生",
            start_pos=2,
            end_pos=6,
            confidence=0.85,
            detection_method="regex",
            recognizer_name="RegexPersonNameRecognizer"
        ),
    ]

    print(f"\n检测到 {len(entities)} 个重叠实体:")
    for e in entities:
        print(f"  - '{e.value}' (span={e.end_pos-e.start_pos}) conf={e.confidence:.2f}")

    merger = EntityMerger()
    merged = merger.merge(entities)

    print(f"\n合并结果: {len(merged)} 个实体")
    for e in merged:
        print(f"  ✅ '{e.value}' (span={e.end_pos-e.start_pos}) conf={e.confidence:.2f}")

    assert len(merged) == 1
    assert merged[0].value == "张三先生"  # 应保留更长的span
    print("\n✅ 场景1通过：正确保留了更长的span")


def test_scenario_2_different_type_overlap():
    """场景2: 不同类型重叠（EMAIL包含PERSON_NAME）"""
    print("\n" + "="*70)
    print("场景2: 不同类型重叠（置信度决定）")
    print("="*70)

    entities = [
        Entity(
            entity_type="EMAIL",
            value="zhangsan@example.com",
            start_pos=5,
            end_pos=26,
            confidence=0.98,
            detection_method="regex",
            recognizer_name="RegexEmailRecognizer"
        ),
        Entity(
            entity_type="PERSON_NAME",
            value="zhangsan",
            start_pos=5,
            end_pos=13,
            confidence=0.75,  # 较低置信度
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
    ]

    print(f"\n检测到 {len(entities)} 个不同类型的重叠实体:")
    for e in entities:
        print(f"  - {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    merger = EntityMerger()
    merged = merger.merge(entities)

    print(f"\n合并结果: {len(merged)} 个实体")
    for e in merged:
        print(f"  ✅ {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    assert len(merged) == 1
    assert merged[0].entity_type == "EMAIL"  # 应保留置信度更高的EMAIL
    assert merged[0].value == "zhangsan@example.com"
    print("\n✅ 场景2通过：正确保留了置信度更高的实体")


def test_scenario_3_adjacent_address():
    """场景3: 相邻地址合并"""
    print("\n" + "="*70)
    print("场景3: 相邻ADDRESS实体合并")
    print("="*70)

    entities = [
        Entity(
            entity_type="ADDRESS",
            value="北京市",
            start_pos=5,
            end_pos=8,
            confidence=0.90,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        Entity(
            entity_type="ADDRESS",
            value="海淀区",
            start_pos=8,
            end_pos=11,
            confidence=0.88,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        Entity(
            entity_type="ADDRESS",
            value="中关村大街",
            start_pos=11,
            end_pos=16,
            confidence=0.92,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
    ]

    print(f"\n检测到 {len(entities)} 个相邻ADDRESS实体:")
    for e in entities:
        print(f"  - '{e.value}' [pos={e.start_pos}-{e.end_pos}] conf={e.confidence:.2f}")

    merger = EntityMerger()
    merged = merger.merge(entities)

    print(f"\n合并结果: {len(merged)} 个实体")
    for e in merged:
        print(f"  ✅ '{e.value}' [pos={e.start_pos}-{e.end_pos}] conf={e.confidence:.2f}")
        if "merged_from" in e.metadata:
            print(f"     (由 {e.metadata['merged_from']} 个实体合并)")

    assert len(merged) == 1
    assert merged[0].value == "北京市海淀区中关村大街"
    assert merged[0].start_pos == 5
    assert merged[0].end_pos == 16
    print("\n✅ 场景3通过：成功合并相邻地址实体")


def test_scenario_4_mixed_entities():
    """场景4: 混合场景（重叠+相邻+独立）"""
    print("\n" + "="*70)
    print("场景4: 混合场景（重叠+相邻+独立）")
    print("="*70)

    entities = [
        # 人名重叠
        Entity(
            entity_type="PERSON_NAME",
            value="张三",
            start_pos=4,
            end_pos=6,
            confidence=0.95,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        Entity(
            entity_type="PERSON_NAME",
            value="张三先生",
            start_pos=4,
            end_pos=8,
            confidence=0.90,
            detection_method="regex",
            recognizer_name="RegexPersonNameRecognizer"
        ),
        # 相邻地址
        Entity(
            entity_type="ADDRESS",
            value="北京",
            start_pos=12,
            end_pos=14,
            confidence=0.92,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        Entity(
            entity_type="ADDRESS",
            value="市",
            start_pos=14,
            end_pos=15,
            confidence=0.85,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        # 独立电话
        Entity(
            entity_type="PHONE_NUMBER",
            value="13812345678",
            start_pos=20,
            end_pos=31,
            confidence=0.98,
            detection_method="regex",
            recognizer_name="RegexPhoneRecognizer"
        ),
    ]

    print(f"\n原始: {len(entities)} 个实体")
    for e in entities:
        print(f"  - {e.entity_type:15s} '{e.value}' [pos={e.start_pos}-{e.end_pos}]")

    merger = EntityMerger()
    merged = merger.merge(entities)

    print(f"\n合并后: {len(merged)} 个实体")
    for e in merged:
        print(f"  ✅ {e.entity_type:15s} '{e.value}' [pos={e.start_pos}-{e.end_pos}]")

    assert len(merged) == 3  # 人名 + 地址 + 电话
    types = {e.entity_type for e in merged}
    assert "PERSON_NAME" in types
    assert "ADDRESS" in types
    assert "PHONE_NUMBER" in types

    # 检查人名合并
    person_entities = [e for e in merged if e.entity_type == "PERSON_NAME"]
    assert len(person_entities) == 1
    assert person_entities[0].value == "张三先生"

    # 检查地址合并
    address_entities = [e for e in merged if e.entity_type == "ADDRESS"]
    assert len(address_entities) == 1
    assert address_entities[0].value == "北京市"

    print(f"\n✅ 场景4通过：正确处理了混合场景（{len(entities)-len(merged)} 个实体被合并）")


def test_scenario_5_no_merge():
    """场景5: 不应合并的情况"""
    print("\n" + "="*70)
    print("场景5: 不应合并的情况（间隔过大的ADDRESS）")
    print("="*70)

    # 配置：最大间隔为2
    config = MergerConfig(max_adjacent_gap=2)

    entities = [
        Entity(
            entity_type="ADDRESS",
            value="北京市",
            start_pos=0,
            end_pos=3,
            confidence=0.90,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        Entity(
            entity_type="ADDRESS",
            value="海淀区",
            start_pos=10,  # 间隔7个字符，超过max_adjacent_gap
            end_pos=13,
            confidence=0.88,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
    ]

    print(f"\n检测到 {len(entities)} 个ADDRESS实体（间隔={entities[1].start_pos - entities[0].end_pos}）:")
    for e in entities:
        print(f"  - '{e.value}' [pos={e.start_pos}-{e.end_pos}]")

    merger = EntityMerger(config)
    merged = merger.merge(entities)

    print(f"\n合并结果: {len(merged)} 个实体（未合并，因为间隔过大）")
    for e in merged:
        print(f"  ✅ '{e.value}' [pos={e.start_pos}-{e.end_pos}]")

    assert len(merged) == 2  # 不应合并
    print("\n✅ 场景5通过：正确拒绝了间隔过大的实体合并")


def main():
    """主函数"""
    print("="*70)
    print("🧪 实体合并器验证测试")
    print("="*70)

    try:
        test_scenario_1_person_name_overlap()
        test_scenario_2_different_type_overlap()
        test_scenario_3_adjacent_address()
        test_scenario_4_mixed_entities()
        test_scenario_5_no_merge()

        print("\n" + "="*70)
        print("🎉 所有测试通过！实体合并器工作正常")
        print("="*70)
        print("\n测试覆盖:")
        print("  ✅ 同类型重叠合并（保留最长span）")
        print("  ✅ 不同类型重叠合并（保留高置信度）")
        print("  ✅ 相邻ADDRESS实体合并")
        print("  ✅ 混合场景处理")
        print("  ✅ 拒绝不合理的合并")

        return 0

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
