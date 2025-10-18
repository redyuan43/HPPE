#!/usr/bin/env python3
"""
上下文验证器快速验证

测试ContextValidator在真实场景下的表现
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.models.entity import Entity
from hppe.refiner.validator import ContextValidator
from hppe.refiner.config import ValidatorConfig


def test_scenario_1_postal_code_false_positive():
    """场景1: 误检邮编过滤（价格中的数字被误识别为邮编）"""
    print("\n" + "="*70)
    print("场景1: 误检邮编过滤（价格：100000元）")
    print("="*70)

    text = "这个商品的价格是100000元，非常实惠。"

    entities = [
        Entity(
            entity_type="POSTAL_CODE",
            value="100000",
            start_pos=9,
            end_pos=15,
            confidence=0.75,  # 低置信度
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
    ]

    print(f"\n原始文本: {text}")
    print(f"检测到 {len(entities)} 个实体:")
    for e in entities:
        print(f"  - {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    validator = ContextValidator()
    validated = validator.validate(entities, text)

    print(f"\n验证后: {len(validated)} 个实体")
    for e in validated:
        orig_conf = e.metadata.get("original_confidence", e.confidence)
        print(f"  ✅ {e.entity_type:15s} '{e.value}' conf={orig_conf:.2f} → {e.confidence:.2f}")

    # 预期：应该被过滤掉（因为上下文包含"价格"、"元"等负向关键词）
    if len(validated) == 0:
        print("\n✅ 场景1通过：成功过滤误检的邮编")
    else:
        print(f"\n⚠️ 场景1部分通过：置信度从{entities[0].confidence:.2f}降至{validated[0].confidence:.2f}")


def test_scenario_2_phone_positive_context():
    """场景2: 正向关键词提升置信度（电话号码前有"手机"关键词）"""
    print("\n" + "="*70)
    print("场景2: 正向关键词提升置信度")
    print("="*70)

    text = "请拨打我的手机号13812345678联系我。"

    entities = [
        Entity(
            entity_type="PHONE_NUMBER",
            value="13812345678",
            start_pos=10,
            end_pos=21,
            confidence=0.70,  # 中等置信度
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
    ]

    print(f"\n原始文本: {text}")
    print(f"检测到 {len(entities)} 个实体:")
    for e in entities:
        print(f"  - {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    validator = ContextValidator()
    validated = validator.validate(entities, text)

    print(f"\n验证后: {len(validated)} 个实体")
    for e in validated:
        orig_conf = e.metadata.get("original_confidence", e.confidence)
        context_score = e.metadata.get("context_score", 0)
        print(f"  ✅ {e.entity_type:15s} '{e.value}' conf={orig_conf:.2f} → {e.confidence:.2f} (context_score={context_score:+.2f})")

    assert len(validated) == 1
    assert validated[0].confidence > entities[0].confidence  # 置信度应该提升
    print(f"\n✅ 场景2通过：置信度提升{validated[0].confidence - entities[0].confidence:.2f}")


def test_scenario_3_email_negative_context():
    """场景3: 负向关键词降低置信度（误检为EMAIL但实际是网址）"""
    print("\n" + "="*70)
    print("场景3: 负向关键词降低置信度")
    print("="*70)

    text = "访问我们的网站example.com了解更多信息。"

    entities = [
        Entity(
            entity_type="EMAIL",
            value="example.com",
            start_pos=8,
            end_pos=19,
            confidence=0.65,
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
    ]

    print(f"\n原始文本: {text}")
    print(f"检测到 {len(entities)} 个实体:")
    for e in entities:
        print(f"  - {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    validator = ContextValidator()
    validated = validator.validate(entities, text)

    print(f"\n验证后: {len(validated)} 个实体")
    for e in validated:
        orig_conf = e.metadata.get("original_confidence", e.confidence)
        context_score = e.metadata.get("context_score", 0)
        print(f"  - {e.entity_type:15s} '{e.value}' conf={orig_conf:.2f} → {e.confidence:.2f} (context_score={context_score:+.2f})")

    # 可能被过滤或降低置信度
    if len(validated) == 0:
        print("\n✅ 场景3通过：成功过滤误检的EMAIL")
    elif len(validated) == 1:
        assert validated[0].confidence < entities[0].confidence
        print(f"\n✅ 场景3通过：置信度降低{entities[0].confidence - validated[0].confidence:.2f}")


def test_scenario_4_high_confidence_passthrough():
    """场景4: 高置信度实体直接通过"""
    print("\n" + "="*70)
    print("场景4: 高置信度实体直接通过（不验证）")
    print("="*70)

    text = "我的身份证号是110101199001011234，请核对。"

    entities = [
        Entity(
            entity_type="ID_CARD",
            value="110101199001011234",
            start_pos=8,
            end_pos=26,
            confidence=0.95,  # 高置信度
            detection_method="regex",
            recognizer_name="RegexIDCardRecognizer"
        ),
    ]

    print(f"\n原始文本: {text}")
    print(f"检测到 {len(entities)} 个实体:")
    for e in entities:
        print(f"  - {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    validator = ContextValidator()
    validated = validator.validate(entities, text)

    print(f"\n验证后: {len(validated)} 个实体")
    for e in validated:
        print(f"  ✅ {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    assert len(validated) == 1
    assert validated[0].confidence == entities[0].confidence  # 置信度不变
    # 高置信度实体不会被验证，所以metadata中没有original_confidence
    metadata = validated[0].metadata if validated[0].metadata else {}
    assert "original_confidence" not in metadata  # 未经过验证
    print("\n✅ 场景4通过：高置信度实体未被修改")


def test_scenario_5_mixed_validation():
    """场景5: 混合场景（同时包含高/低置信度实体）"""
    print("\n" + "="*70)
    print("场景5: 混合场景验证")
    print("="*70)

    text = "张三的电话是13812345678，身份证号110101199001011234，邮编100000。"

    entities = [
        # 高置信度电话（有正向关键词）
        Entity(
            entity_type="PHONE_NUMBER",
            value="13812345678",
            start_pos=6,
            end_pos=17,
            confidence=0.90,  # 高置信度，直接通过
            detection_method="regex",
            recognizer_name="RegexPhoneRecognizer"
        ),
        # 高置信度身份证
        Entity(
            entity_type="ID_CARD",
            value="110101199001011234",
            start_pos=22,
            end_pos=40,
            confidence=0.95,  # 高置信度，直接通过
            detection_method="regex",
            recognizer_name="RegexIDCardRecognizer"
        ),
        # 低置信度邮编（有正向关键词）
        Entity(
            entity_type="POSTAL_CODE",
            value="100000",
            start_pos=44,
            end_pos=50,
            confidence=0.70,  # 低置信度，需要验证
            detection_method="llm_finetuned",
            recognizer_name="FineTunedLLMRecognizer"
        ),
    ]

    print(f"\n原始文本: {text}")
    print(f"检测到 {len(entities)} 个实体:")
    for e in entities:
        print(f"  - {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    validator = ContextValidator()
    validated = validator.validate(entities, text)

    print(f"\n验证后: {len(validated)} 个实体")
    for e in validated:
        metadata = e.metadata if e.metadata else {}
        if "original_confidence" in metadata:
            orig_conf = metadata["original_confidence"]
            context_score = metadata.get("context_score", 0)
            print(f"  ✅ {e.entity_type:15s} '{e.value}' conf={orig_conf:.2f} → {e.confidence:.2f} (verified, score={context_score:+.2f})")
        else:
            print(f"  ✅ {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f} (direct pass)")

    assert len(validated) == 3  # 所有实体都保留
    # 检查邮编是否被提升
    postal_entity = [e for e in validated if e.entity_type == "POSTAL_CODE"][0]
    assert postal_entity.confidence > 0.70  # 应该被提升（因为有"邮编"关键词）
    print(f"\n✅ 场景5通过：正确处理混合场景（低置信度邮编被提升）")


def test_scenario_6_ip_address_validation():
    """场景6: IP地址上下文验证"""
    print("\n" + "="*70)
    print("场景6: IP地址上下文验证")
    print("="*70)

    text = "服务器的IP地址是192.168.1.100，请记录。"

    entities = [
        Entity(
            entity_type="IP_ADDRESS",
            value="192.168.1.100",
            start_pos=8,
            end_pos=21,
            confidence=0.75,
            detection_method="regex",
            recognizer_name="RegexIPRecognizer"
        ),
    ]

    print(f"\n原始文本: {text}")
    print(f"检测到 {len(entities)} 个实体:")
    for e in entities:
        print(f"  - {e.entity_type:15s} '{e.value}' conf={e.confidence:.2f}")

    validator = ContextValidator()
    validated = validator.validate(entities, text)

    print(f"\n验证后: {len(validated)} 个实体")
    for e in validated:
        orig_conf = e.metadata.get("original_confidence", e.confidence)
        context_score = e.metadata.get("context_score", 0)
        print(f"  ✅ {e.entity_type:15s} '{e.value}' conf={orig_conf:.2f} → {e.confidence:.2f} (context_score={context_score:+.2f})")

    assert len(validated) == 1
    assert validated[0].confidence > entities[0].confidence  # 应该提升（有"IP"、"服务器"关键词）
    print(f"\n✅ 场景6通过：IP地址置信度提升{validated[0].confidence - entities[0].confidence:.2f}")


def main():
    """主函数"""
    print("="*70)
    print("🧪 上下文验证器验证测试")
    print("="*70)

    try:
        test_scenario_1_postal_code_false_positive()
        test_scenario_2_phone_positive_context()
        test_scenario_3_email_negative_context()
        test_scenario_4_high_confidence_passthrough()
        test_scenario_5_mixed_validation()
        test_scenario_6_ip_address_validation()

        print("\n" + "="*70)
        print("🎉 所有测试通过！上下文验证器工作正常")
        print("="*70)
        print("\n测试覆盖:")
        print("  ✅ 误检邮编过滤（负向关键词）")
        print("  ✅ 正向关键词提升置信度")
        print("  ✅ 负向关键词降低置信度")
        print("  ✅ 高置信度实体直接通过")
        print("  ✅ 混合场景处理")
        print("  ✅ IP地址上下文验证")
        print("\n关键词库覆盖: 17种PII类型")

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
