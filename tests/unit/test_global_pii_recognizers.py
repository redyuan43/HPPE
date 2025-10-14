"""
全球 PII 识别器单元测试
"""

import pytest

from hppe.engines.regex.recognizers.global_pii import (
    EmailRecognizer,
    IPAddressRecognizer,
    URLRecognizer,
    CreditCardRecognizer,
    SSNRecognizer,
)
from hppe.engines.regex.config_loader import ConfigLoader


class TestEmailRecognizer:
    """测试电子邮件地址识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建邮箱识别器"""
        config = {
            "entity_type": "EMAIL",
            "patterns": [
                {"pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'}
            ],
            "context_words": ["邮箱", "email", "联系"],
            "deny_lists": ["example@"],
            "confidence_base": 0.80
        }
        return EmailRecognizer(config)

    def test_detect_simple_email(self, recognizer):
        """测试检测简单邮箱"""
        text = "Contact: john.doe@example.com"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].entity_type == "EMAIL"
        assert entities[0].value == "john.doe@example.com"

    def test_detect_multiple_emails(self, recognizer):
        """测试检测多个邮箱"""
        text = "Email1: user1@domain.com, Email2: user2@test.org"
        entities = recognizer.detect(text)

        assert len(entities) == 2

    def test_validate_email_format(self, recognizer):
        """测试邮箱格式验证"""
        valid_emails = [
            "john@example.com",
            "user.name@domain.co.uk",
            "test+tag@example.com",
        ]

        for email in valid_emails:
            assert EmailRecognizer.validate_email(email)

    def test_validate_invalid_emails(self, recognizer):
        """测试无效邮箱"""
        invalid_emails = [
            "invalid",
            "@example.com",
            "user@",
            "user@@example.com",
            "user@domain",  # 缺少顶级域名
        ]

        for email in invalid_emails:
            assert not EmailRecognizer.validate_email(email)

    def test_parse_email(self, recognizer):
        """测试邮箱解析"""
        local, domain = EmailRecognizer._parse_email("john@example.com")
        assert local == "john"
        assert domain == "example.com"

    def test_extract_tld(self, recognizer):
        """测试提取顶级域名"""
        assert EmailRecognizer._extract_tld("example.com") == "com"
        assert EmailRecognizer._extract_tld("example.co.uk") == "uk"

    def test_metadata(self, recognizer):
        """测试元数据提取"""
        text = "Email: john@example.com"
        entities = recognizer.detect(text)

        metadata = entities[0].metadata
        assert metadata["local_part"] == "john"
        assert metadata["domain"] == "example.com"
        assert metadata["tld"] == "com"
        assert metadata["valid"] is True

    def test_validate_entity(self, recognizer):
        """测试实体验证"""
        from hppe.models.entity import Entity

        valid_entity = Entity(
            entity_type="EMAIL",
            value="test@example.com",
            start_pos=0,
            end_pos=16,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="EmailRecognizer"
        )

        invalid_entity = Entity(
            entity_type="EMAIL",
            value="invalid-email",
            start_pos=0,
            end_pos=13,
            confidence=0.5,
            detection_method="regex",
            recognizer_name="EmailRecognizer"
        )

        assert recognizer.validate(valid_entity)
        assert not recognizer.validate(invalid_entity)

    def test_edge_cases(self, recognizer):
        """测试边界条件"""
        # 空字符串
        assert not EmailRecognizer.validate_email("")

        # 多个@符号
        assert not EmailRecognizer.validate_email("user@@example.com")

        # 域名以点开始
        assert not EmailRecognizer.validate_email("user@.example.com")

        # 域名以点结束
        assert not EmailRecognizer.validate_email("user@example.com.")

        # 只有@
        assert not EmailRecognizer.validate_email("@")


class TestIPAddressRecognizer:
    """测试IP地址识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建IP地址识别器"""
        config = {
            "entity_type": "IP_ADDRESS",
            "patterns": [
                {"pattern": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'},  # IPv4
                {"pattern": r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'},  # IPv6
            ],
            "confidence_base": 0.85
        }
        return IPAddressRecognizer(config)

    def test_detect_ipv4(self, recognizer):
        """测试检测IPv4地址"""
        text = "Server IP: 192.168.1.1"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].value == "192.168.1.1"
        assert entities[0].metadata["ip_type"] == "ipv4"

    def test_validate_ipv4_valid(self, recognizer):
        """测试有效IPv4地址"""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "8.8.8.8",
        ]

        for ip in valid_ips:
            assert IPAddressRecognizer._validate_ipv4(ip)

    def test_validate_ipv4_invalid(self, recognizer):
        """测试无效IPv4地址"""
        invalid_ips = [
            "256.1.1.1",  # 超出范围
            "192.168.1",  # 不足4段
            "192.168.1.1.1",  # 超过4段
            "abc.def.ghi.jkl",  # 非数字
        ]

        for ip in invalid_ips:
            assert not IPAddressRecognizer._validate_ipv4(ip)

    def test_identify_ip_type(self, recognizer):
        """测试IP类型识别"""
        assert IPAddressRecognizer._identify_ip_type("192.168.1.1") == "ipv4"
        assert IPAddressRecognizer._identify_ip_type("2001:db8::1") == "ipv6"

    def test_is_private_ip(self, recognizer):
        """测试私有IP判断"""
        private_ips = [
            ("10.0.0.1", "ipv4"),
            ("172.16.0.1", "ipv4"),
            ("192.168.1.1", "ipv4"),
            ("127.0.0.1", "ipv4"),
        ]

        for ip, ip_type in private_ips:
            assert IPAddressRecognizer._is_private_ip(ip, ip_type)

    def test_is_public_ip(self, recognizer):
        """测试公网IP"""
        public_ips = [
            ("8.8.8.8", "ipv4"),
            ("1.1.1.1", "ipv4"),
        ]

        for ip, ip_type in public_ips:
            assert not IPAddressRecognizer._is_private_ip(ip, ip_type)

    def test_metadata(self, recognizer):
        """测试元数据"""
        text = "IP: 192.168.1.1"
        entities = recognizer.detect(text)

        metadata = entities[0].metadata
        assert metadata["ip_type"] == "ipv4"
        assert metadata["is_private"] is True

    def test_validate_entity(self, recognizer):
        """测试实体验证"""
        from hppe.models.entity import Entity

        valid_entity = Entity(
            entity_type="IP_ADDRESS",
            value="192.168.1.1",
            start_pos=0,
            end_pos=11,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="IPAddressRecognizer"
        )

        invalid_entity = Entity(
            entity_type="IP_ADDRESS",
            value="999.999.999.999",
            start_pos=0,
            end_pos=15,
            confidence=0.5,
            detection_method="regex",
            recognizer_name="IPAddressRecognizer"
        )

        assert recognizer.validate(valid_entity)
        assert not recognizer.validate(invalid_entity)

    def test_ipv6_validation(self, recognizer):
        """测试IPv6验证"""
        # 简化格式
        assert IPAddressRecognizer._validate_ipv6("2001:db8::1")

        # 完整格式
        assert IPAddressRecognizer._validate_ipv6("2001:0db8:85a3:0000:0000:8a2e:0370:7334")

        # 无效 - 太多的::
        assert not IPAddressRecognizer._validate_ipv6("2001::db8::1")


class TestURLRecognizer:
    """测试URL识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建URL识别器"""
        config = {
            "entity_type": "URL",
            "patterns": [
                {"pattern": r'https?://[^\s]+'},
                {"pattern": r'ftp://[^\s]+'},
            ],
            "confidence_base": 0.80
        }
        return URLRecognizer(config)

    def test_detect_http_url(self, recognizer):
        """测试检测HTTP URL"""
        text = "Visit: http://example.com"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].value == "http://example.com"

    def test_detect_https_url(self, recognizer):
        """测试检测HTTPS URL"""
        text = "Secure: https://secure.example.com/path"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert "https://" in entities[0].value

    def test_parse_url(self, recognizer):
        """测试URL解析"""
        parsed = URLRecognizer._parse_url("https://example.com/path?query=value")

        assert parsed is not None
        assert parsed["scheme"] == "https"
        assert parsed["domain"] == "example.com"
        assert parsed["path"] == "/path"
        assert parsed["has_query"] is True

    def test_parse_url_invalid(self, recognizer):
        """测试无效URL解析"""
        parsed = URLRecognizer._parse_url("not-a-url")
        assert parsed is None

    def test_metadata(self, recognizer):
        """测试元数据"""
        text = "URL: https://example.com/page"
        entities = recognizer.detect(text)

        metadata = entities[0].metadata
        assert metadata["scheme"] == "https"
        assert metadata["domain"] == "example.com"
        assert metadata["valid"] is True

    def test_validate_entity(self, recognizer):
        """测试实体验证"""
        from hppe.models.entity import Entity

        valid_entity = Entity(
            entity_type="URL",
            value="https://example.com",
            start_pos=0,
            end_pos=19,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="URLRecognizer"
        )

        invalid_entity = Entity(
            entity_type="URL",
            value="not-a-url",
            start_pos=0,
            end_pos=9,
            confidence=0.5,
            detection_method="regex",
            recognizer_name="URLRecognizer"
        )

        assert recognizer.validate(valid_entity)
        assert not recognizer.validate(invalid_entity)


class TestCreditCardRecognizer:
    """测试信用卡号识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建信用卡识别器"""
        config = {
            "entity_type": "CREDIT_CARD",
            "patterns": [
                {"pattern": r'\b[0-9]{13,19}\b'},
                {"pattern": r'\b[0-9]{4}[\s\-][0-9]{4}[\s\-][0-9]{4}[\s\-][0-9]{4,7}\b'},
            ],
            "confidence_base": 0.85
        }
        return CreditCardRecognizer(config)

    def test_detect_credit_card(self, recognizer):
        """测试检测信用卡号"""
        text = "Card: 4532015112830366"
        entities = recognizer.detect(text)

        assert len(entities) >= 1
        # 找到有效的卡号
        valid_entities = [e for e in entities if e.metadata["luhn_valid"]]
        assert len(valid_entities) >= 1

    def test_validate_luhn_valid(self, recognizer):
        """测试Luhn算法验证（有效卡号）"""
        valid_cards = [
            "4532015112830366",  # Visa
            "5425233430109903",  # MasterCard
            "378282246310005",   # Amex
        ]

        for card in valid_cards:
            assert CreditCardRecognizer.validate_luhn(card)

    def test_validate_luhn_invalid(self, recognizer):
        """测试Luhn算法验证（无效卡号）"""
        invalid_cards = [
            "4532015112830367",  # 最后一位错误
            "1234567890123456",  # 随机数字
        ]

        for card in invalid_cards:
            assert not CreditCardRecognizer.validate_luhn(card)

    def test_identify_card_type(self, recognizer):
        """测试识别卡类型"""
        assert CreditCardRecognizer._identify_card_type("4532015112830366") == "Visa"
        assert CreditCardRecognizer._identify_card_type("5425233430109903") == "MasterCard"
        assert CreditCardRecognizer._identify_card_type("378282246310005") == "Amex"

    def test_mask_card_number(self, recognizer):
        """测试遮罩卡号"""
        masked = CreditCardRecognizer._mask_card_number("4532015112830366")
        assert masked == "4532********0366"

    def test_metadata(self, recognizer):
        """测试元数据"""
        text = "Card: 4532015112830366"
        entities = recognizer.detect(text)

        valid_entity = None
        for e in entities:
            if e.metadata["luhn_valid"]:
                valid_entity = e
                break

        assert valid_entity is not None
        metadata = valid_entity.metadata
        assert metadata["card_type"] == "Visa"
        assert metadata["length"] == 16
        assert metadata["masked"] == "4532********0366"

    def test_validate_entity(self, recognizer):
        """测试实体验证"""
        from hppe.models.entity import Entity

        valid_entity = Entity(
            entity_type="CREDIT_CARD",
            value="4532015112830366",
            start_pos=0,
            end_pos=16,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="CreditCardRecognizer"
        )

        invalid_entity = Entity(
            entity_type="CREDIT_CARD",
            value="1234567890123456",
            start_pos=0,
            end_pos=16,
            confidence=0.5,
            detection_method="regex",
            recognizer_name="CreditCardRecognizer"
        )

        assert recognizer.validate(valid_entity)
        assert not recognizer.validate(invalid_entity)


class TestSSNRecognizer:
    """测试美国社会安全号识别器"""

    @pytest.fixture
    def recognizer(self):
        """创建SSN识别器"""
        config = {
            "entity_type": "US_SSN",
            "patterns": [
                {"pattern": r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b'},
            ],
            "confidence_base": 0.85
        }
        return SSNRecognizer(config)

    def test_detect_ssn(self, recognizer):
        """测试检测SSN"""
        text = "SSN: 123-45-6789"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].value == "123-45-6789"

    def test_validate_ssn_valid(self, recognizer):
        """测试有效SSN"""
        valid_ssns = [
            "123-45-6789",
            "456-78-9012",
            "789-01-2345",
        ]

        for ssn in valid_ssns:
            assert SSNRecognizer.validate_ssn(ssn)

    def test_validate_ssn_invalid(self, recognizer):
        """测试无效SSN"""
        invalid_ssns = [
            "000-45-6789",  # 区域号为000
            "666-45-6789",  # 区域号为666
            "900-45-6789",  # 区域号 >= 900
            "123-00-6789",  # 组号为00
            "123-45-0000",  # 序列号为0000
        ]

        for ssn in invalid_ssns:
            assert not SSNRecognizer.validate_ssn(ssn)

    def test_parse_ssn(self, recognizer):
        """测试SSN解析"""
        area, group, serial = SSNRecognizer._parse_ssn("123-45-6789")
        assert area == "123"
        assert group == "45"
        assert serial == "6789"

    def test_mask_ssn(self, recognizer):
        """测试遮罩SSN"""
        masked = SSNRecognizer._mask_ssn("123-45-6789")
        assert masked == "***-**-6789"

    def test_metadata(self, recognizer):
        """测试元数据"""
        text = "SSN: 123-45-6789"
        entities = recognizer.detect(text)

        metadata = entities[0].metadata
        assert metadata["area"] == "123"
        assert metadata["group"] == "45"
        assert metadata["serial"] == "6789"
        assert metadata["masked"] == "***-**-6789"

    def test_validate_entity(self, recognizer):
        """测试实体验证"""
        from hppe.models.entity import Entity

        valid_entity = Entity(
            entity_type="US_SSN",
            value="123-45-6789",
            start_pos=0,
            end_pos=11,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="SSNRecognizer"
        )

        invalid_entity = Entity(
            entity_type="US_SSN",
            value="000-00-0000",
            start_pos=0,
            end_pos=11,
            confidence=0.5,
            detection_method="regex",
            recognizer_name="SSNRecognizer"
        )

        assert recognizer.validate(valid_entity)
        assert not recognizer.validate(invalid_entity)


class TestIntegration:
    """集成测试"""

    def test_all_recognizers_import(self):
        """测试所有识别器可以导入"""
        from hppe.engines.regex.recognizers.global_pii import (
            EmailRecognizer,
            IPAddressRecognizer,
            URLRecognizer,
            CreditCardRecognizer,
            SSNRecognizer,
        )

        assert EmailRecognizer is not None
        assert IPAddressRecognizer is not None
        assert URLRecognizer is not None
        assert CreditCardRecognizer is not None
        assert SSNRecognizer is not None

    def test_mixed_pii_detection(self):
        """测试混合PII检测"""
        email_config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'}]
        }

        ip_config = {
            "entity_type": "IP_ADDRESS",
            "patterns": [{"pattern": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'}]
        }

        email_recognizer = EmailRecognizer(email_config)
        ip_recognizer = IPAddressRecognizer(ip_config)

        text = "Email: admin@example.com, Server: 192.168.1.1"

        email_entities = email_recognizer.detect(text)
        ip_entities = ip_recognizer.detect(text)

        assert len(email_entities) == 1
        assert len(ip_entities) == 1
        assert email_entities[0].entity_type == "EMAIL"
        assert ip_entities[0].entity_type == "IP_ADDRESS"

    def test_all_recognizers_with_registry(self):
        """测试所有识别器与注册表集成"""
        from hppe.engines.regex.registry import RecognizerRegistry

        registry = RecognizerRegistry()

        # 创建并注册所有识别器
        email_config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'}]
        }

        ip_config = {
            "entity_type": "IP_ADDRESS",
            "patterns": [{"pattern": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'}]
        }

        registry.register(EmailRecognizer(email_config))
        registry.register(IPAddressRecognizer(ip_config))

        assert len(registry) == 2

        text = "Contact: user@test.com, IP: 10.0.0.1"
        all_entities = registry.detect(text)

        assert len(all_entities) >= 2
