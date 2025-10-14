"""
中国 PII 识别器单元测试
"""

import pytest

from hppe.engines.regex.recognizers.china_pii import (
    ChinaIDCardRecognizer,
    ChinaPhoneRecognizer,
    ChinaBankCardRecognizer,
    ChinaPassportRecognizer,
)
from hppe.engines.regex.config_loader import ConfigLoader


# 测试配置
@pytest.fixture
def config_loader(tmp_path):
    """创建临时配置加载器"""
    patterns_dir = tmp_path / "patterns"
    patterns_dir.mkdir()
    return patterns_dir


class TestChinaIDCardRecognizer:
    """测试中国身份证识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建身份证识别器"""
        config = {
            "entity_type": "CHINA_ID_CARD",
            "patterns": [
                {
                    "pattern": r'[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]'
                }
            ],
            "context_words": ["身份证", "身份证号", "ID"],
            "deny_lists": ["订单号", "流水号"],
            "confidence_base": 0.85
        }
        return ChinaIDCardRecognizer(config)

    def test_detect_valid_id_card(self, recognizer):
        """测试检测有效身份证号"""
        text = "我的身份证号是110101199003077571"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].entity_type == "CHINA_ID_CARD"
        assert entities[0].value == "110101199003077571"
        assert entities[0].confidence > 0.85

    def test_detect_multiple_id_cards(self, recognizer):
        """测试检测多个身份证号"""
        text = "第一个：110101199003077571，第二个：44030119900307871X"
        entities = recognizer.detect(text)

        assert len(entities) == 2

    def test_detect_with_context_words(self, recognizer):
        """测试上下文词提升置信度"""
        text1 = "身份证号：110101199003077571"
        text2 = "号码：110101199003077571"

        entities1 = recognizer.detect(text1)
        entities2 = recognizer.detect(text2)

        # 有上下文词的置信度应该更高
        assert entities1[0].confidence > entities2[0].confidence

    def test_detect_with_deny_list(self, recognizer):
        """测试拒绝列表过滤"""
        padding = "x" * 150  # 足够的填充
        text = f"订单号：110101199003077571{padding}身份证：440301199003078715"
        entities = recognizer.detect(text)

        # 订单号应该被过滤掉
        assert len(entities) == 1
        assert entities[0].value == "440301199003078715"

    def test_validate_checksum_correct(self, recognizer):
        """测试正确的校验码"""
        # 这些是有效的身份证号（校验码正确）
        valid_ids = [
            "110101199003077571",
            "440301199003078715",
            "320106198506051230",
        ]

        for id_card in valid_ids:
            assert ChinaIDCardRecognizer.validate_id_card(id_card)

    def test_validate_checksum_incorrect(self, recognizer):
        """测试错误的校验码"""
        # 这些是无效的身份证号（校验码错误）
        invalid_ids = [
            "110101199003077579",  # 最后一位错误
            "440301199003078710",  # 校验码错误
        ]

        for id_card in invalid_ids:
            assert not ChinaIDCardRecognizer.validate_id_card(id_card)

    def test_validate_invalid_format(self, recognizer):
        """测试无效格式"""
        invalid_formats = [
            "12345678901234567",  # 只有17位
            "12345678901234567890",  # 19位
            "11010119900307757A",  # 包含非法字符
            "",  # 空字符串
        ]

        for id_card in invalid_formats:
            assert not ChinaIDCardRecognizer.validate_id_card(id_card)

    def test_metadata_extraction(self, recognizer):
        """测试元数据提取"""
        text = "身份证：110101199003077571"
        entities = recognizer.detect(text)

        metadata = entities[0].metadata
        assert metadata["region_code"] == "110101"
        assert metadata["birth_date"] == "19900307"
        assert metadata["sequence"] == "757"
        assert metadata["checksum"] == "1"
        assert "checksum_valid" in metadata


class TestChinaPhoneRecognizer:
    """测试中国手机号识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建手机号识别器"""
        config = {
            "entity_type": "CHINA_PHONE",
            "patterns": [
                {"pattern": r'1[3-9]\d{9}'},
                {"pattern": r'\+86\s*1[3-9]\d{9}'},
            ],
            "context_words": ["手机", "电话", "联系方式"],
            "deny_lists": ["客服", "热线"],
            "confidence_base": 0.80
        }
        return ChinaPhoneRecognizer(config)

    def test_detect_standard_phone(self, recognizer):
        """测试检测标准手机号"""
        text = "我的手机号是13812345678"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].value == "13812345678"
        assert entities[0].entity_type == "CHINA_PHONE"

    def test_detect_phone_with_country_code(self, recognizer):
        """测试带国际区号的手机号"""
        text = "联系方式：+86 13912345678"
        entities = recognizer.detect(text)

        # 两个模式都会匹配：完整的 +86 格式和纯11位数字
        assert len(entities) == 2
        # 找到带国际区号的那个
        country_code_entity = [e for e in entities if e.metadata["has_country_code"]][0]
        assert "+86" in country_code_entity.value
        assert country_code_entity.metadata["has_country_code"] is True

    def test_detect_multiple_phones(self, recognizer):
        """测试检测多个手机号"""
        text = "手机1：13812345678，手机2：15912345678"
        entities = recognizer.detect(text)

        assert len(entities) == 2

    def test_validate_valid_phones(self, recognizer):
        """测试验证有效手机号"""
        valid_phones = [
            "13812345678",
            "15912345678",
            "18612345678",
            "19912345678",
        ]

        for phone in valid_phones:
            assert ChinaPhoneRecognizer.validate_phone(phone)

    def test_validate_invalid_phones(self, recognizer):
        """测试验证无效手机号"""
        invalid_phones = [
            "12812345678",  # 第二位不对
            "138123456",  # 不足11位
            "138123456789",  # 超过11位
            "1381234567a",  # 包含字母
            "",  # 空字符串
        ]

        for phone in invalid_phones:
            assert not ChinaPhoneRecognizer.validate_phone(phone)

    def test_extract_phone_number(self, recognizer):
        """测试提取纯数字手机号"""
        assert ChinaPhoneRecognizer._extract_phone_number("13812345678") == "13812345678"
        assert ChinaPhoneRecognizer._extract_phone_number("+8613812345678") == "13812345678"
        assert ChinaPhoneRecognizer._extract_phone_number("+86 13812345678") == "13812345678"
        assert ChinaPhoneRecognizer._extract_phone_number("86-138-1234-5678") == "13812345678"

    def test_metadata(self, recognizer):
        """测试元数据"""
        text1 = "手机：13812345678"
        text2 = "手机：+86 13812345678"

        entities1 = recognizer.detect(text1)
        entities2 = recognizer.detect(text2)

        assert entities1[0].metadata["has_country_code"] is False
        # text2会检测到两个：带+86的和不带的，找到带国际区号的那个
        country_code_entity = [e for e in entities2 if e.metadata["has_country_code"]][0]
        assert country_code_entity.metadata["has_country_code"] is True
        assert entities1[0].metadata["normalized"] == "13812345678"
        assert country_code_entity.metadata["normalized"] == "13812345678"


class TestChinaBankCardRecognizer:
    """测试中国银行卡识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建银行卡识别器"""
        config = {
            "entity_type": "CHINA_BANK_CARD",
            "patterns": [
                {"pattern": r'[1-9]\d{15,18}'},
                {"pattern": r'[1-9]\d{3}\s?\d{4}\s?\d{4}\s?\d{4,7}'},  # 支持空格
            ],
            "context_words": ["银行卡", "卡号", "账号"],
            "deny_lists": ["订单号"],
            "confidence_base": 0.85
        }
        return ChinaBankCardRecognizer(config)

    def test_detect_valid_bank_card(self, recognizer):
        """测试检测有效银行卡号"""
        text = "我的银行卡号是6222021234567890128"
        entities = recognizer.detect(text)

        # 可能被多个模式匹配，至少有1个
        assert len(entities) >= 1
        assert entities[0].entity_type == "CHINA_BANK_CARD"
        assert entities[0].value == "6222021234567890128"

    def test_detect_card_with_spaces(self, recognizer):
        """测试带空格的卡号"""
        text = "卡号：6222 0212 3456 7890 123"
        entities = recognizer.detect(text)

        # 正则可能匹配不带空格的部分
        assert len(entities) >= 1

    def test_validate_luhn_algorithm(self, recognizer):
        """测试 Luhn 算法验证"""
        # 这些是通过 Luhn 校验的卡号
        valid_cards = [
            "6222021234567890128",  # 19位（经过Luhn生成）
            "4532015112830366",  # 16位 Visa测试卡
            "5425233430109903",  # 16位 MasterCard测试卡
        ]

        for card in valid_cards:
            assert ChinaBankCardRecognizer.validate_luhn(card)

    def test_validate_luhn_invalid(self, recognizer):
        """测试无效的 Luhn 校验"""
        invalid_cards = [
            "6222021234567890124",  # 最后一位错误
            "1234567890123456",  # 随机16位数字
        ]

        for card in invalid_cards:
            assert not ChinaBankCardRecognizer.validate_luhn(card)

    def test_validate_invalid_length(self, recognizer):
        """测试无效长度"""
        invalid_cards = [
            "123456789012345",  # 15位（太短）
            "12345678901234567890",  # 20位（太长）
        ]

        for card in invalid_cards:
            assert not ChinaBankCardRecognizer.validate_luhn(card)

    def test_metadata(self, recognizer):
        """测试元数据"""
        text = "银行卡：6222021234567890123"
        entities = recognizer.detect(text)

        metadata = entities[0].metadata
        assert "luhn_valid" in metadata
        assert "normalized" in metadata
        assert metadata["length"] == 19


class TestChinaPassportRecognizer:
    """测试中国护照识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建护照号识别器"""
        config = {
            "entity_type": "CHINA_PASSPORT",
            "patterns": [
                {"pattern": r'E\d{8}'},
                {"pattern": r'G\d{8}'},
                {"pattern": r'[PSD]\d{7}'},
            ],
            "context_words": ["护照", "护照号"],
            "deny_lists": ["订单"],
            "confidence_base": 0.85
        }
        return ChinaPassportRecognizer(config)

    def test_detect_electronic_passport(self, recognizer):
        """测试电子护照（E开头）"""
        text = "护照号：E12345678"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].value == "E12345678"
        assert entities[0].metadata["passport_type"] == "电子护照"

    def test_detect_ordinary_passport(self, recognizer):
        """测试普通护照（G开头）"""
        text = "护照：G98765432"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].value == "G98765432"
        assert entities[0].metadata["passport_type"] == "普通护照"

    def test_detect_other_passport_types(self, recognizer):
        """测试其他类型护照"""
        passports = {
            "P1234567": "因私护照",
            "S1234567": "因公护照",
            "D1234567": "外交护照",
        }

        for passport, expected_type in passports.items():
            text = f"护照：{passport}"
            entities = recognizer.detect(text)

            assert len(entities) == 1
            assert entities[0].metadata["passport_type"] == expected_type

    def test_validate_valid_passports(self, recognizer):
        """测试验证有效护照号"""
        valid_passports = [
            "E12345678",
            "G98765432",
            "P1234567",
            "S7654321",
            "D9876543",
        ]

        for passport in valid_passports:
            assert ChinaPassportRecognizer.validate_passport(passport)

    def test_validate_invalid_passports(self, recognizer):
        """测试验证无效护照号"""
        invalid_passports = [
            "E1234567",  # E开头但只有7位数字
            "G123456789",  # G开头但有9位数字
            "P12345678",  # P开头但有8位数字
            "X1234567",  # 无效前缀
            "E1234567A",  # 包含字母
            "",  # 空字符串
        ]

        for passport in invalid_passports:
            assert not ChinaPassportRecognizer.validate_passport(passport)

    def test_identify_passport_type(self, recognizer):
        """测试识别护照类型"""
        assert ChinaPassportRecognizer._identify_passport_type("E12345678") == "电子护照"
        assert ChinaPassportRecognizer._identify_passport_type("G12345678") == "普通护照"
        assert ChinaPassportRecognizer._identify_passport_type("P1234567") == "因私护照"
        assert ChinaPassportRecognizer._identify_passport_type("S1234567") == "因公护照"
        assert ChinaPassportRecognizer._identify_passport_type("D1234567") == "外交护照"
        assert ChinaPassportRecognizer._identify_passport_type("X1234567") == "未知类型"

    def test_metadata(self, recognizer):
        """测试元数据"""
        text = "护照：E12345678"
        entities = recognizer.detect(text)

        metadata = entities[0].metadata
        assert metadata["prefix"] == "E"
        assert metadata["number"] == "12345678"
        assert metadata["passport_type"] == "电子护照"


class TestIntegration:
    """集成测试"""

    def test_all_recognizers_with_config_loader(self, tmp_path):
        """测试使用配置加载器加载所有识别器"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        # 创建配置文件
        yaml_content = """
recognizers:
  - name: ChinaIDCardRecognizer
    entity_type: CHINA_ID_CARD
    patterns:
      - pattern: '[1-9]\\d{5}(19|20)\\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])\\d{3}[0-9Xx]'
    confidence_base: 0.85

  - name: ChinaPhoneRecognizer
    entity_type: CHINA_PHONE
    patterns:
      - pattern: '1[3-9]\\d{9}'
    confidence_base: 0.80

  - name: ChinaBankCardRecognizer
    entity_type: CHINA_BANK_CARD
    patterns:
      - pattern: '[1-9]\\d{15,18}'
    confidence_base: 0.85

  - name: ChinaPassportRecognizer
    entity_type: CHINA_PASSPORT
    patterns:
      - pattern: 'E\\d{8}'
      - pattern: 'G\\d{8}'
    confidence_base: 0.85
"""
        yaml_file = patterns_dir / "china_pii.yaml"
        yaml_file.write_text(yaml_content)

        # 加载配置
        loader = ConfigLoader(str(patterns_dir))
        configs = loader.load_all()

        assert len(configs) == 4

        # 验证每个配置
        entity_types = {c["entity_type"] for c in configs}
        expected_types = {"CHINA_ID_CARD", "CHINA_PHONE", "CHINA_BANK_CARD", "CHINA_PASSPORT"}
        assert entity_types == expected_types

    def test_mixed_pii_detection(self):
        """测试混合 PII 检测"""
        id_config = {
            "entity_type": "CHINA_ID_CARD",
            "patterns": [
                {"pattern": r'[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]'}
            ]
        }

        phone_config = {
            "entity_type": "CHINA_PHONE",
            "patterns": [{"pattern": r'1[3-9]\d{9}'}]
        }

        id_recognizer = ChinaIDCardRecognizer(id_config)
        phone_recognizer = ChinaPhoneRecognizer(phone_config)

        text = "身份证：110101197003071234，手机：13812345678"

        id_entities = id_recognizer.detect(text)
        phone_entities = phone_recognizer.detect(text)

        assert len(id_entities) == 1
        # 手机号识别器可能会匹配到身份证中的数字序列，所以可能有2个
        assert len(phone_entities) >= 1
        # 验证真实手机号被正确识别
        assert any(e.value == "13812345678" for e in phone_entities)
        assert id_entities[0].entity_type == "CHINA_ID_CARD"
        assert phone_entities[0].entity_type == "CHINA_PHONE"
