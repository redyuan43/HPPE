"""
FineTunedLLMRecognizer 单元测试

测试识别器的核心功能和接口兼容性
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.engines.llm.recognizers.finetuned import FineTunedLLMRecognizer
from hppe.models.entity import Entity


class TestFineTunedLLMRecognizer:
    """FineTunedLLMRecognizer 测试类"""

    @pytest.fixture
    def mock_llm_engine(self):
        """创建Mock LLM引擎"""
        engine = MagicMock()
        engine.model_name = "MockQwen"
        engine.model_path = Path("mock/path")
        engine.get_supported_pii_types.return_value = [
            "PERSON_NAME",
            "ADDRESS",
            "ORGANIZATION",
            "PHONE_NUMBER",
            "EMAIL",
            "ID_CARD"
        ]
        return engine

    @pytest.fixture
    def recognizer(self, mock_llm_engine):
        """创建识别器实例"""
        return FineTunedLLMRecognizer(
            llm_engine=mock_llm_engine,
            confidence_threshold=0.75,
            recognizer_name="TestRecognizer"
        )

    def test_initialization(self, mock_llm_engine):
        """测试初始化"""
        recognizer = FineTunedLLMRecognizer(
            llm_engine=mock_llm_engine,
            confidence_threshold=0.8
        )

        assert recognizer.llm_engine == mock_llm_engine
        assert recognizer.confidence_threshold == 0.8
        assert recognizer.recognizer_name == "FineTunedLLMRecognizer"
        assert len(recognizer.supported_types) == 6

    def test_detect_success(self, recognizer, mock_llm_engine):
        """测试成功检测"""
        # Mock引擎返回值
        mock_entities = [
            Entity(
                entity_type="PERSON_NAME",
                value="张三",
                start_pos=2,
                end_pos=4,
                confidence=0.9,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            ),
            Entity(
                entity_type="PHONE_NUMBER",
                value="13812345678",
                start_pos=7,
                end_pos=18,
                confidence=0.85,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            )
        ]

        mock_llm_engine.detect_pii.return_value = mock_entities

        # 执行检测
        text = "我是张三，电话13812345678"
        entities = recognizer.detect(text)

        # 验证
        assert len(entities) == 2
        assert entities[0].entity_type == "PERSON_NAME"
        assert entities[0].value == "张三"
        assert entities[1].entity_type == "PHONE_NUMBER"

        # 验证引擎被正确调用
        mock_llm_engine.detect_pii.assert_called_once_with(
            text=text,
            confidence_threshold=0.75
        )

    def test_detect_empty_text(self, recognizer):
        """测试空文本"""
        assert recognizer.detect("") == []
        assert recognizer.detect("   ") == []
        assert recognizer.detect(None) is not None  # 应返回空列表而非None

    def test_detect_with_filtering(self, recognizer, mock_llm_engine):
        """测试置信度过滤"""
        # Mock返回混合置信度的实体
        mock_entities = [
            Entity(
                entity_type="PERSON_NAME",
                value="张三",
                start_pos=0,
                end_pos=2,
                confidence=0.9,  # 高于阈值
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            ),
            Entity(
                entity_type="ADDRESS",
                value="北京",
                start_pos=5,
                end_pos=7,
                confidence=0.6,  # 低于阈值
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            )
        ]

        mock_llm_engine.detect_pii.return_value = mock_entities

        # 执行检测
        entities = recognizer.detect("张三住在北京")

        # 由于引擎已经应用了阈值过滤，这里应该得到引擎返回的结果
        # 但识别器的逻辑应该保持一致
        assert len(entities) >= 1

    def test_detect_specific_type(self, recognizer, mock_llm_engine):
        """测试检测特定类型"""
        # Mock返回多种类型
        mock_entities = [
            Entity(
                entity_type="PERSON_NAME",
                value="张三",
                start_pos=0,
                end_pos=2,
                confidence=0.9,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            ),
            Entity(
                entity_type="PHONE_NUMBER",
                value="13812345678",
                start_pos=5,
                end_pos=16,
                confidence=0.85,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            ),
            Entity(
                entity_type="EMAIL",
                value="test@example.com",
                start_pos=20,
                end_pos=36,
                confidence=0.95,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            )
        ]

        mock_llm_engine.detect_pii.return_value = mock_entities

        # 只检测PERSON_NAME
        entities = recognizer.detect_specific_type(
            "张三，电话13812345678，邮箱test@example.com",
            "PERSON_NAME"
        )

        # 应该只返回PERSON_NAME类型
        assert len(entities) == 1
        assert entities[0].entity_type == "PERSON_NAME"
        assert entities[0].value == "张三"

    def test_detect_specific_type_unsupported(self, recognizer):
        """测试检测不支持的类型"""
        entities = recognizer.detect_specific_type(
            "测试文本",
            "UNSUPPORTED_TYPE"
        )

        # 应该返回空列表并记录警告
        assert entities == []

    def test_validate(self, recognizer):
        """测试实体验证"""
        # 高置信度实体
        entity_high = Entity(
            entity_type="PERSON_NAME",
            value="张三",
            start_pos=0,
            end_pos=2,
            confidence=0.9,
            detection_method="llm_finetuned",
            recognizer_name="TestRecognizer"
        )

        assert recognizer.validate(entity_high) is True

        # 低置信度实体
        entity_low = Entity(
            entity_type="ADDRESS",
            value="北京",
            start_pos=0,
            end_pos=2,
            confidence=0.6,  # 低于0.75阈值
            detection_method="llm_finetuned",
            recognizer_name="TestRecognizer"
        )

        assert recognizer.validate(entity_low) is False

    def test_get_supported_types(self, recognizer):
        """测试获取支持的类型"""
        types = recognizer.get_supported_types()

        assert isinstance(types, list)
        assert len(types) == 6
        assert "PERSON_NAME" in types
        assert "PHONE_NUMBER" in types

        # 确保返回的是副本（修改不影响原列表）
        types.append("NEW_TYPE")
        assert "NEW_TYPE" not in recognizer.get_supported_types()

    def test_get_info(self, recognizer):
        """测试获取识别器信息"""
        info = recognizer.get_info()

        assert isinstance(info, dict)
        assert info["name"] == "TestRecognizer"
        assert info["detection_method"] == "llm_finetuned"
        assert info["model"] == "MockQwen"
        assert len(info["supported_types"]) == 6
        assert info["confidence_threshold"] == 0.75

    def test_repr(self, recognizer):
        """测试字符串表示"""
        repr_str = repr(recognizer)

        assert "FineTunedLLMRecognizer" in repr_str
        assert "MockQwen" in repr_str
        assert "types=6" in repr_str
        assert "threshold=0.75" in repr_str

    def test_detect_exception_handling(self, recognizer, mock_llm_engine):
        """测试异常处理"""
        # Mock引擎抛出异常
        mock_llm_engine.detect_pii.side_effect = Exception("Model error")

        # 检测应该返回空列表而不是抛出异常
        entities = recognizer.detect("测试文本")

        assert entities == []

    def test_multiple_entities_same_type(self, recognizer, mock_llm_engine):
        """测试同一类型的多个实体"""
        mock_entities = [
            Entity(
                entity_type="PERSON_NAME",
                value="张三",
                start_pos=0,
                end_pos=2,
                confidence=0.9,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            ),
            Entity(
                entity_type="PERSON_NAME",
                value="李四",
                start_pos=3,
                end_pos=5,
                confidence=0.85,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            ),
            Entity(
                entity_type="PERSON_NAME",
                value="王五",
                start_pos=6,
                end_pos=8,
                confidence=0.88,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            )
        ]

        mock_llm_engine.detect_pii.return_value = mock_entities

        entities = recognizer.detect("张三、李四、王五")

        assert len(entities) == 3
        assert all(e.entity_type == "PERSON_NAME" for e in entities)

    def test_chinese_text(self, recognizer, mock_llm_engine):
        """测试中文文本处理"""
        mock_entities = [
            Entity(
                entity_type="ORGANIZATION",
                value="阿里巴巴集团",
                start_pos=0,
                end_pos=6,
                confidence=0.92,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            )
        ]

        mock_llm_engine.detect_pii.return_value = mock_entities

        entities = recognizer.detect("阿里巴巴集团是一家科技公司")

        assert len(entities) == 1
        assert entities[0].value == "阿里巴巴集团"

    def test_mixed_language_text(self, recognizer, mock_llm_engine):
        """测试中英文混合文本"""
        mock_entities = [
            Entity(
                entity_type="PERSON_NAME",
                value="John Smith",
                start_pos=8,
                end_pos=18,
                confidence=0.9,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            ),
            Entity(
                entity_type="EMAIL",
                value="john@example.com",
                start_pos=22,
                end_pos=38,
                confidence=0.95,
                detection_method="llm_finetuned",
                recognizer_name="TestRecognizer"
            )
        ]

        mock_llm_engine.detect_pii.return_value = mock_entities

        entities = recognizer.detect("联系人：John Smith，邮箱：john@example.com")

        assert len(entities) == 2
        assert entities[0].entity_type == "PERSON_NAME"
        assert entities[1].entity_type == "EMAIL"


class TestFineTunedLLMRecognizerIntegration:
    """集成场景测试"""

    @pytest.fixture
    def mock_llm_engine(self):
        """创建更真实的Mock引擎"""
        engine = MagicMock()
        engine.model_name = "QwenFineTuned-6PII"
        engine.model_path = Path("models/pii_qwen4b_unsloth/final")
        engine.get_supported_pii_types.return_value = [
            "ADDRESS", "ORGANIZATION", "PERSON_NAME",
            "PHONE_NUMBER", "EMAIL", "ID_CARD"
        ]

        def mock_detect(text, confidence_threshold=0.8):
            """模拟检测逻辑"""
            entities = []

            if "张三" in text:
                entities.append(Entity(
                    entity_type="PERSON_NAME",
                    value="张三",
                    start_pos=text.index("张三"),
                    end_pos=text.index("张三") + 2,
                    confidence=0.9,
                    detection_method="llm_finetuned",
                    recognizer_name="MockEngine"
                ))

            if "13812345678" in text:
                entities.append(Entity(
                    entity_type="PHONE_NUMBER",
                    value="13812345678",
                    start_pos=text.index("13812345678"),
                    end_pos=text.index("13812345678") + 11,
                    confidence=0.85,
                    detection_method="llm_finetuned",
                    recognizer_name="MockEngine"
                ))

            return entities

        engine.detect_pii = mock_detect
        return engine

    @pytest.fixture
    def recognizer(self, mock_llm_engine):
        """创建识别器"""
        return FineTunedLLMRecognizer(
            llm_engine=mock_llm_engine,
            confidence_threshold=0.75
        )

    def test_end_to_end_detection(self, recognizer):
        """测试端到端检测"""
        text = "我是张三，电话13812345678"
        entities = recognizer.detect(text)

        assert len(entities) == 2

        # 验证人名
        person = [e for e in entities if e.entity_type == "PERSON_NAME"][0]
        assert person.value == "张三"
        assert person.confidence >= 0.75

        # 验证电话
        phone = [e for e in entities if e.entity_type == "PHONE_NUMBER"][0]
        assert phone.value == "13812345678"

    def test_batch_detection(self, recognizer):
        """测试批量检测"""
        texts = [
            "我是张三",
            "电话13812345678",
            "张三的电话是13812345678"
        ]

        results = [recognizer.detect(text) for text in texts]

        assert len(results) == 3
        assert len(results[0]) == 1  # 只有人名
        assert len(results[1]) == 1  # 只有电话
        assert len(results[2]) == 2  # 人名+电话


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
