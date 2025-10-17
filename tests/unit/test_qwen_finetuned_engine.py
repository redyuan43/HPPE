"""
QwenFineTunedEngine 单元测试

使用Mock避免GPU依赖，测试核心逻辑
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
from pathlib import Path

# 导入要测试的模块
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.engines.llm.qwen_finetuned import QwenFineTunedEngine
from hppe.models.entity import Entity


class TestQwenFineTunedEngine:
    """QwenFineTunedEngine 测试类"""

    @pytest.fixture
    def mock_model(self):
        """创建Mock模型"""
        model = MagicMock()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """创建Mock tokenizer"""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3])
        return tokenizer

    @pytest.fixture
    def engine_instance(self, mock_model, mock_tokenizer):
        """创建引擎实例（不加载真实模型）"""
        with patch('hppe.engines.llm.qwen_finetuned.FastLanguageModel') as mock_flm:
            # Mock模型加载
            mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_flm.for_inference.return_value = mock_model

            engine = QwenFineTunedEngine(
                model_path="mock/path",
                device="cpu",
                load_in_4bit=False
            )

            # 手动设置已加载状态
            engine.model = mock_model
            engine.tokenizer = mock_tokenizer
            engine._loaded = True

            yield engine

    def test_initialization(self):
        """测试引擎初始化"""
        engine = QwenFineTunedEngine(
            model_path="models/test",
            device="cpu",
            load_in_4bit=False
        )

        assert engine.model_path == Path("models/test")
        assert engine.device == "cpu"
        assert engine.load_in_4bit is False
        assert engine.model is None
        assert engine._loaded is False

    def test_supported_pii_types(self, engine_instance):
        """测试支持的PII类型列表"""
        supported_types = engine_instance.get_supported_pii_types()

        assert isinstance(supported_types, list)
        assert len(supported_types) == 6
        assert "PERSON_NAME" in supported_types
        assert "ADDRESS" in supported_types
        assert "ORGANIZATION" in supported_types
        assert "PHONE_NUMBER" in supported_types
        assert "EMAIL" in supported_types
        assert "ID_CARD" in supported_types

    def test_parse_response_valid_json(self, engine_instance):
        """测试解析有效的JSON响应"""
        response_json = json.dumps({
            "entities": [
                {
                    "type": "PERSON_NAME",
                    "value": "张三",
                    "start_pos": 2,
                    "end_pos": 4,
                    "confidence": 0.9
                },
                {
                    "type": "PHONE_NUMBER",
                    "value": "13812345678",
                    "start_pos": 7,
                    "end_pos": 18,
                    "confidence": 0.85
                }
            ]
        }, ensure_ascii=False)

        entities = engine_instance._parse_response(
            response=response_json,
            original_text="我是张三，电话13812345678"
        )

        assert len(entities) == 2

        # 验证第一个实体
        assert entities[0].entity_type == "PERSON_NAME"
        assert entities[0].value == "张三"
        assert entities[0].start_pos == 2
        assert entities[0].end_pos == 4
        assert entities[0].confidence == 0.9

        # 验证第二个实体
        assert entities[1].entity_type == "PHONE_NUMBER"
        assert entities[1].value == "13812345678"

    def test_parse_response_invalid_json(self, engine_instance):
        """测试解析无效的JSON响应"""
        invalid_response = "This is not JSON"

        entities = engine_instance._parse_response(
            response=invalid_response,
            original_text="测试文本"
        )

        # 应该返回空列表而不是抛出异常
        assert entities == []

    def test_parse_response_empty_entities(self, engine_instance):
        """测试解析空实体列表"""
        response_json = json.dumps({
            "entities": []
        })

        entities = engine_instance._parse_response(
            response=response_json,
            original_text="没有PII的文本"
        )

        assert len(entities) == 0

    def test_parse_response_missing_fields(self, engine_instance):
        """测试解析缺少字段的响应"""
        response_json = json.dumps({
            "entities": [
                {
                    "type": "PERSON_NAME",
                    "value": "张三"
                    # 缺少 start_pos, end_pos, confidence
                }
            ]
        }, ensure_ascii=False)

        entities = engine_instance._parse_response(
            response=response_json,
            original_text="我是张三"
        )

        # 应该能处理缺失字段（使用默认值或根据value计算）
        assert len(entities) == 1
        assert entities[0].entity_type == "PERSON_NAME"
        assert entities[0].value == "张三"

    def test_generate_with_mock(self, engine_instance):
        """测试generate方法（使用Mock）"""
        # Mock generate方法的返回值
        mock_output = {
            "entities": [
                {
                    "type": "EMAIL",
                    "value": "test@example.com",
                    "start_pos": 3,
                    "end_pos": 20,
                    "confidence": 0.95
                }
            ]
        }

        with patch.object(engine_instance, 'generate', return_value=json.dumps(mock_output, ensure_ascii=False)):
            entities = engine_instance.detect_pii("邮箱：test@example.com")

            assert len(entities) == 1
            assert entities[0].entity_type == "EMAIL"
            assert entities[0].value == "test@example.com"

    def test_detect_pii_with_confidence_threshold(self, engine_instance):
        """测试置信度阈值过滤"""
        mock_output = {
            "entities": [
                {
                    "type": "PERSON_NAME",
                    "value": "张三",
                    "start_pos": 0,
                    "end_pos": 2,
                    "confidence": 0.9  # 高于阈值
                },
                {
                    "type": "ADDRESS",
                    "value": "北京",
                    "start_pos": 5,
                    "end_pos": 7,
                    "confidence": 0.5  # 低于阈值
                }
            ]
        }

        with patch.object(engine_instance, 'generate', return_value=json.dumps(mock_output, ensure_ascii=False)):
            # 使用阈值0.7
            entities = engine_instance.detect_pii("张三住在北京", confidence_threshold=0.7)

            # 只应返回confidence >= 0.7的实体
            assert len(entities) == 1
            assert entities[0].entity_type == "PERSON_NAME"
            assert entities[0].confidence == 0.9

    def test_detect_pii_empty_text(self, engine_instance):
        """测试空文本输入"""
        entities = engine_instance.detect_pii("")
        assert entities == []

        entities = engine_instance.detect_pii("   ")
        assert entities == []

    def test_detect_pii_very_long_text(self, engine_instance):
        """测试超长文本（应该被截断）"""
        long_text = "测试" * 10000  # 20000字符

        mock_output = {"entities": []}

        with patch.object(engine_instance, 'generate', return_value=json.dumps(mock_output)):
            entities = engine_instance.detect_pii(long_text)

            # 验证generate被调用
            engine_instance.generate.assert_called_once()

            # 验证传入的文本长度被截断
            call_args = engine_instance.generate.call_args
            assert len(call_args[1]["prompt"]) <= engine_instance.max_seq_length

    def test_model_name(self, engine_instance):
        """测试模型名称"""
        assert engine_instance.model_name == "QwenFineTuned-6PII"

    def test_get_info(self, engine_instance):
        """测试获取引擎信息"""
        info = engine_instance.get_info()

        assert isinstance(info, dict)
        assert "model_name" in info
        assert "model_path" in info
        assert "supported_pii_types" in info
        assert "device" in info
        assert "load_in_4bit" in info

        assert info["model_name"] == "QwenFineTuned-6PII"
        assert len(info["supported_pii_types"]) == 6


class TestQwenFineTunedEngineEdgeCases:
    """边界情况和错误处理测试"""

    def test_invalid_model_path(self):
        """测试无效的模型路径"""
        with pytest.raises(Exception):
            engine = QwenFineTunedEngine(
                model_path="/nonexistent/path",
                device="cpu"
            )
            # 尝试加载模型应该失败
            engine._load_model()

    def test_parse_response_malformed_entity(self):
        """测试解析格式错误的实体"""
        engine = QwenFineTunedEngine(model_path="mock/path", device="cpu")

        # 实体缺少必需字段
        response_json = json.dumps({
            "entities": [
                {"value": "张三"},  # 缺少type
                {"type": "EMAIL"}   # 缺少value
            ]
        })

        entities = engine._parse_response(response_json, "测试")

        # 应该跳过格式错误的实体或使用默认值
        # 具体行为取决于实现
        assert isinstance(entities, list)

    def test_confidence_bounds(self):
        """测试置信度边界值"""
        engine = QwenFineTunedEngine(model_path="mock/path", device="cpu")

        # 测试置信度 = 0
        response_json = json.dumps({
            "entities": [
                {"type": "PERSON_NAME", "value": "张三", "confidence": 0.0}
            ]
        })

        entities = engine._parse_response(response_json, "张三")
        assert len(entities) >= 0  # 应该能处理

        # 测试置信度 = 1
        response_json = json.dumps({
            "entities": [
                {"type": "PERSON_NAME", "value": "张三", "confidence": 1.0}
            ]
        })

        entities = engine._parse_response(response_json, "张三")
        assert len(entities) >= 1

    def test_special_characters_in_value(self):
        """测试特殊字符处理"""
        engine = QwenFineTunedEngine(model_path="mock/path", device="cpu")

        response_json = json.dumps({
            "entities": [
                {
                    "type": "EMAIL",
                    "value": "test+special@example.com",
                    "start_pos": 0,
                    "end_pos": 24,
                    "confidence": 0.9
                }
            ]
        })

        entities = engine._parse_response(response_json, "test+special@example.com")

        assert len(entities) == 1
        assert entities[0].value == "test+special@example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
