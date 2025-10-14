"""
BaseRecognizer 抽象基类单元测试
"""

import pytest
import re
from typing import List

from hppe.engines.regex.base import BaseRecognizer
from hppe.models.entity import Entity


# 创建测试用的具体识别器类
class SimpleRecognizer(BaseRecognizer):
    """简单的测试识别器"""

    def detect(self, text: str) -> List[Entity]:
        """简单的检测实现"""
        entities = []

        for pattern in self.patterns:
            for match in pattern.finditer(text):
                value = match.group()
                start = match.start()
                end = match.end()

                # 检查拒绝列表
                if self._check_deny_list(text, start):
                    continue

                # 计算置信度
                confidence = self._calculate_confidence(
                    self.confidence_base,
                    text,
                    start
                )

                entity = self._create_entity(
                    value=value,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """简单验证：总是返回 True"""
        return True


class TestBaseRecognizerInitialization:
    """测试 BaseRecognizer 初始化"""

    def test_init_with_minimal_config(self):
        """测试最小配置初始化"""
        config = {
            "entity_type": "TEST_TYPE",
            "patterns": [
                {"pattern": r"\d+"}
            ]
        }

        recognizer = SimpleRecognizer(config)

        assert recognizer.entity_type == "TEST_TYPE"
        assert len(recognizer.patterns) == 1
        assert recognizer.context_words == []
        assert recognizer.deny_lists == []
        assert recognizer.confidence_base == 0.85  # 默认值

    def test_init_with_full_config(self):
        """测试完整配置初始化"""
        config = {
            "entity_type": "EMAIL",
            "name": "CustomEmailRecognizer",
            "patterns": [
                {"pattern": r"[a-z]+@[a-z]+\.com"}
            ],
            "context_words": ["email", "邮箱"],
            "deny_lists": ["test", "example"],
            "confidence_base": 0.90
        }

        recognizer = SimpleRecognizer(config)

        assert recognizer.entity_type == "EMAIL"
        assert recognizer.recognizer_name == "CustomEmailRecognizer"
        assert recognizer.context_words == ["email", "邮箱"]
        assert recognizer.deny_lists == ["test", "example"]
        assert recognizer.confidence_base == 0.90

    def test_init_missing_entity_type(self):
        """测试缺少 entity_type"""
        config = {
            "patterns": [{"pattern": r"\d+"}]
        }

        with pytest.raises(ValueError, match="缺少必需字段: entity_type"):
            SimpleRecognizer(config)

    def test_init_missing_patterns(self):
        """测试缺少 patterns"""
        config = {
            "entity_type": "TEST"
        }

        with pytest.raises(ValueError, match="缺少必需字段: patterns"):
            SimpleRecognizer(config)

    def test_init_invalid_confidence_base(self):
        """测试无效的 confidence_base"""
        config = {
            "entity_type": "TEST",
            "patterns": [{"pattern": r"\d+"}],
            "confidence_base": 1.5
        }

        with pytest.raises(ValueError, match="confidence_base 必须在"):
            SimpleRecognizer(config)


class TestPatternCompilation:
    """测试正则表达式编译"""

    def test_compile_single_pattern(self):
        """测试编译单个模式"""
        config = {
            "entity_type": "NUMBER",
            "patterns": [
                {"pattern": r"\d+"}
            ]
        }

        recognizer = SimpleRecognizer(config)

        assert len(recognizer.patterns) == 1
        assert isinstance(recognizer.patterns[0], re.Pattern)

    def test_compile_multiple_patterns(self):
        """测试编译多个模式"""
        config = {
            "entity_type": "TEST",
            "patterns": [
                {"pattern": r"\d+"},
                {"pattern": r"[a-z]+"},
                {"pattern": r"[A-Z]+"}
            ]
        }

        recognizer = SimpleRecognizer(config)

        assert len(recognizer.patterns) == 3
        for pattern in recognizer.patterns:
            assert isinstance(pattern, re.Pattern)

    def test_compile_invalid_pattern(self):
        """测试编译无效的正则表达式"""
        config = {
            "entity_type": "TEST",
            "patterns": [
                {"pattern": r"[invalid("}  # 无效的正则表达式
            ]
        }

        with pytest.raises(ValueError, match="模式 #0 编译失败"):
            SimpleRecognizer(config)

    def test_compile_pattern_without_pattern_key(self):
        """测试缺少 pattern 键的模式"""
        config = {
            "entity_type": "TEST",
            "patterns": [
                {"score": 0.9}  # 缺少 'pattern' 键
            ]
        }

        with pytest.raises(ValueError, match="缺少 'pattern' 键"):
            SimpleRecognizer(config)


class TestContextChecking:
    """测试上下文词检测"""

    def test_check_context_no_context_words(self):
        """测试没有上下文词的情况"""
        config = {
            "entity_type": "TEST",
            "patterns": [{"pattern": r"\d+"}]
        }

        recognizer = SimpleRecognizer(config)
        boost = recognizer._check_context("some text 12345", 10)

        assert boost == 0.0

    def test_check_context_found_single_word(self):
        """测试找到一个上下文词"""
        config = {
            "entity_type": "ID",
            "patterns": [{"pattern": r"\d+"}],
            "context_words": ["身份证", "ID"]
        }

        recognizer = SimpleRecognizer(config)
        text = "我的身份证号是 110101199003077578"
        match_pos = text.find("110101")

        boost = recognizer._check_context(text, match_pos)

        assert boost > 0.0
        assert boost <= 0.15

    def test_check_context_found_multiple_words(self):
        """测试找到多个上下文词"""
        config = {
            "entity_type": "ID",
            "patterns": [{"pattern": r"\d+"}],
            "context_words": ["身份证", "号码", "ID"]
        }

        recognizer = SimpleRecognizer(config)
        text = "身份证号码是 110101199003077578"
        match_pos = text.find("110101")

        boost = recognizer._check_context(text, match_pos)

        assert boost > 0.05  # 应该找到多个词
        assert boost <= 0.15

    def test_check_context_not_found(self):
        """测试未找到上下文词"""
        config = {
            "entity_type": "ID",
            "patterns": [{"pattern": r"\d+"}],
            "context_words": ["email", "邮箱"]
        }

        recognizer = SimpleRecognizer(config)
        text = "这是一个号码 12345"
        match_pos = text.find("12345")

        boost = recognizer._check_context(text, match_pos)

        assert boost == 0.0


class TestDenyListChecking:
    """测试拒绝列表检查"""

    def test_check_deny_list_no_deny_words(self):
        """测试没有拒绝词的情况"""
        config = {
            "entity_type": "TEST",
            "patterns": [{"pattern": r"\d+"}]
        }

        recognizer = SimpleRecognizer(config)
        result = recognizer._check_deny_list("some text 12345", 10)

        assert result is False

    def test_check_deny_list_found(self):
        """测试找到拒绝词"""
        config = {
            "entity_type": "ID",
            "patterns": [{"pattern": r"\d+"}],
            "deny_lists": ["订单号", "流水号"]
        }

        recognizer = SimpleRecognizer(config)
        text = "订单号：110101199003077578"
        match_pos = text.find("110101")

        result = recognizer._check_deny_list(text, match_pos)

        assert result is True

    def test_check_deny_list_not_found(self):
        """测试未找到拒绝词"""
        config = {
            "entity_type": "ID",
            "patterns": [{"pattern": r"\d+"}],
            "deny_lists": ["订单", "流水"]
        }

        recognizer = SimpleRecognizer(config)
        text = "身份证号：110101199003077578"
        match_pos = text.find("110101")

        result = recognizer._check_deny_list(text, match_pos)

        assert result is False


class TestConfidenceCalculation:
    """测试置信度计算"""

    def test_calculate_confidence_base_only(self):
        """测试仅基础分数"""
        config = {
            "entity_type": "TEST",
            "patterns": [{"pattern": r"\d+"}],
            "confidence_base": 0.80
        }

        recognizer = SimpleRecognizer(config)
        confidence = recognizer._calculate_confidence(0.80, "text", 0)

        assert confidence == 0.80

    def test_calculate_confidence_with_context_boost(self):
        """测试带上下文提升的置信度"""
        config = {
            "entity_type": "ID",
            "patterns": [{"pattern": r"\d+"}],
            "context_words": ["身份证"]
        }

        recognizer = SimpleRecognizer(config)
        text = "身份证号：110101"
        match_pos = text.find("110101")

        confidence = recognizer._calculate_confidence(0.80, text, match_pos)

        # 应该有上下文提升
        assert confidence > 0.80

    def test_calculate_confidence_validation_failed(self):
        """测试验证失败的置信度"""
        config = {
            "entity_type": "TEST",
            "patterns": [{"pattern": r"\d+"}]
        }

        recognizer = SimpleRecognizer(config)
        confidence = recognizer._calculate_confidence(
            0.90,
            "text",
            0,
            validation_passed=False
        )

        # 验证失败应该降低置信度
        assert confidence < 0.90
        assert confidence == 0.90 * 0.7

    def test_calculate_confidence_max_capped_at_one(self):
        """测试置信度上限为 1.0"""
        config = {
            "entity_type": "TEST",
            "patterns": [{"pattern": r"\d+"}]
        }

        recognizer = SimpleRecognizer(config)
        # 即使基础分数很高，也不应超过 1.0
        confidence = recognizer._calculate_confidence(0.99, "text", 0)

        assert confidence <= 1.0


class TestEntityCreation:
    """测试实体创建辅助方法"""

    def test_create_entity(self):
        """测试创建实体"""
        config = {
            "entity_type": "EMAIL",
            "name": "EmailRecognizer",
            "patterns": [{"pattern": r"[a-z]+@[a-z]+\.com"}]
        }

        recognizer = SimpleRecognizer(config)
        entity = recognizer._create_entity(
            value="test@example.com",
            start_pos=10,
            end_pos=26,
            confidence=0.95
        )

        assert entity.entity_type == "EMAIL"
        assert entity.value == "test@example.com"
        assert entity.start_pos == 10
        assert entity.end_pos == 26
        assert entity.confidence == 0.95
        assert entity.detection_method == "regex"
        assert entity.recognizer_name == "EmailRecognizer"

    def test_create_entity_with_metadata(self):
        """测试创建带元数据的实体"""
        config = {
            "entity_type": "TEST",
            "patterns": [{"pattern": r"\d+"}]
        }

        recognizer = SimpleRecognizer(config)
        metadata = {"validated": True}

        entity = recognizer._create_entity(
            value="12345",
            start_pos=0,
            end_pos=5,
            confidence=0.90,
            metadata=metadata
        )

        assert entity.metadata == metadata


class TestDetection:
    """测试检测功能"""

    def test_detect_simple_match(self):
        """测试简单匹配"""
        config = {
            "entity_type": "NUMBER",
            "patterns": [{"pattern": r"\d+"}]
        }

        recognizer = SimpleRecognizer(config)
        text = "There are 123 items"
        entities = recognizer.detect(text)

        assert len(entities) == 1
        assert entities[0].value == "123"
        assert entities[0].entity_type == "NUMBER"

    def test_detect_multiple_matches(self):
        """测试多个匹配"""
        config = {
            "entity_type": "NUMBER",
            "patterns": [{"pattern": r"\d+"}]
        }

        recognizer = SimpleRecognizer(config)
        text = "Numbers: 123, 456, 789"
        entities = recognizer.detect(text)

        assert len(entities) == 3
        assert entities[0].value == "123"
        assert entities[1].value == "456"
        assert entities[2].value == "789"

    def test_detect_with_deny_list_filtering(self):
        """测试拒绝列表过滤"""
        config = {
            "entity_type": "ID",
            "patterns": [{"pattern": r"\d{5}"}],  # 只匹配 5 位数字
            "deny_lists": ["订单"]
        }

        recognizer = SimpleRecognizer(config)
        # 创建足够长的文本，使得两个数字之间距离超过拒绝词的影响范围
        padding = "x" * 150  # 150 字符填充，超过窗口大小
        text = f"订单号 12345{padding}身份证 67890"
        entities = recognizer.detect(text)

        # 订单号应该被过滤掉，只剩身份证号
        assert len(entities) == 1
        assert entities[0].value == "67890"

    def test_detect_no_matches(self):
        """测试无匹配"""
        config = {
            "entity_type": "NUMBER",
            "patterns": [{"pattern": r"\d+"}]
        }

        recognizer = SimpleRecognizer(config)
        text = "No numbers here"
        entities = recognizer.detect(text)

        assert len(entities) == 0
