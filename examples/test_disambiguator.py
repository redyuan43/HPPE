#!/usr/bin/env python3
"""
歧义消除器快速验证

测试Disambiguator在真实冲突场景下的表现
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.models.entity import Entity
from hppe.refiner.disambiguator import Disambiguator
from hppe.refiner.config import DisambiguatorConfig


def test_scenario_1_id_card_conflict():
    """场景1: 身份证号被多个识别器识别为不同类型"""
    print("\n" + "="*70)
    print("场景1: 身份证号冲突（110101199001011234）")
    print("="*70)

    # 模拟检测结果
    entities = [
        Entity(
            entity_type="ID_CARD",
            value="110101199001011234",
            start_pos=6,
            end_pos=24,
            confidence=0.95,
            detection_method="regex",
            recognizer_name="RegexIDCardRecognizer"
        ),
        Entity(
            entity_type="BANK_CARD",
            value="110101199001011234",
            start_pos=6,
            end_pos=24,
            confidence=0.85,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        Entity(
            entity_type="DRIVER_LICENSE",
            value="110101199001011234",
            start_pos=6,
            end_pos=24,
            confidence=0.80,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
    ]

    print(f"\n检测到 {len(entities)} 个冲突实体:")
    for e in entities:
        print(f"  - {e.entity_type:20s} conf={e.confidence:.2f} method={e.detection_method}")

    # 消歧
    disambiguator = Disambiguator()
    resolved = disambiguator.resolve(entities)

    print(f"\n消歧结果: {len(resolved)} 个实体")
    for e in resolved:
        print(f"  ✅ {e.entity_type:20s} conf={e.confidence:.2f} method={e.detection_method}")

    assert len(resolved) == 1
    assert resolved[0].entity_type == "ID_CARD"  # ID_CARD优先级最高
    print("\n✅ 场景1通过：正确选择ID_CARD")


def test_scenario_2_person_name_overlap():
    """场景2: 人名重叠（"张三" vs "张三先生"）"""
    print("\n" + "="*70)
    print("场景2: 人名重叠")
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
        print(f"  - '{e.value}' ({e.entity_type}) conf={e.confidence:.2f}")

    disambiguator = Disambiguator()
    resolved = disambiguator.resolve(entities)

    print(f"\n消歧结果: {len(resolved)} 个实体")
    for e in resolved:
        print(f"  ✅ '{e.value}' ({e.entity_type}) conf={e.confidence:.2f}")

    assert len(resolved) == 1
    print(f"\n✅ 场景2通过：保留了置信度更高的实体")


def test_scenario_3_multiple_conflicts():
    """场景3: 文本中有多个不同位置的冲突"""
    print("\n" + "="*70)
    print("场景3: 多个独立冲突")
    print("="*70)

    text = "联系人：张三，电话13812345678，邮箱test@example.com"

    entities = [
        # 人名（无冲突）
        Entity(
            entity_type="PERSON_NAME",
            value="张三",
            start_pos=4,
            end_pos=6,
            confidence=0.95,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        # 电话号码（有冲突）
        Entity(
            entity_type="PHONE_NUMBER",
            value="13812345678",
            start_pos=10,
            end_pos=21,
            confidence=0.90,
            detection_method="regex",
            recognizer_name="RegexPhoneRecognizer"
        ),
        Entity(
            entity_type="ID_CARD",
            value="13812345678",
            start_pos=10,
            end_pos=21,
            confidence=0.70,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
        # 邮箱（无冲突）
        Entity(
            entity_type="EMAIL",
            value="test@example.com",
            start_pos=25,
            end_pos=41,
            confidence=0.98,
            detection_method="regex",
            recognizer_name="RegexEmailRecognizer"
        ),
    ]

    print(f"\n文本: {text}")
    print(f"检测到 {len(entities)} 个实体（含冲突）")

    disambiguator = Disambiguator()
    resolved = disambiguator.resolve(entities)

    print(f"\n消歧后: {len(resolved)} 个实体")
    for e in resolved:
        print(f"  ✅ {e.entity_type:20s} '{e.value}' conf={e.confidence:.2f}")

    assert len(resolved) == 3  # 人名+电话+邮箱
    types = {e.entity_type for e in resolved}
    assert "PERSON_NAME" in types
    assert "PHONE_NUMBER" in types  # 应选择PHONE_NUMBER而非ID_CARD
    assert "EMAIL" in types

    print(f"\n✅ 场景3通过：正确处理了{len(entities)-len(resolved)}个冲突")


def main():
    """主函数"""
    print("="*70)
    print("🧪 歧义消除器验证测试")
    print("="*70)

    try:
        test_scenario_1_id_card_conflict()
        test_scenario_2_person_name_overlap()
        test_scenario_3_multiple_conflicts()

        print("\n" + "="*70)
        print("🎉 所有测试通过！歧义消除器工作正常")
        print("="*70)

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
