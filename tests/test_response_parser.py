"""
测试 LLM 响应解析器
"""

import pytest
from hppe.engines.llm import LLMResponseParser, extract_pii_from_response


class TestLLMResponseParser:
    """测试 LLMResponseParser 类"""

    def test_basic_json_extraction(self):
        """测试基本 JSON 提取"""
        parser = LLMResponseParser(strict=False)

        response = '{"entities": [{"type": "NAME", "value": "张三"}]}'
        result = parser.extract_json(response)

        assert result is not None
        assert "entities" in result
        assert len(result["entities"]) == 1
        assert result["entities"][0]["value"] == "张三"

    def test_json_with_think_tags(self):
        """测试带 <think> 标签的 JSON 提取"""
        parser = LLMResponseParser(strict=False)

        response = '<think>\n\n</think>\n\n{"entities": [{"type": "PERSON_NAME", "value": "张三"}]}'
        result = parser.extract_json(response)

        assert result is not None
        assert "entities" in result
        assert result["entities"][0]["type"] == "PERSON_NAME"

    def test_json_with_long_think_content(self):
        """测试带长 <think> 内容的 JSON 提取"""
        parser = LLMResponseParser(strict=False)

        response = '''<think>
好的，我现在需要处理用户的查询，检测文本中的PII信息。
首先，用户提供的文本是："我叫张三"。
根据用户的要求，我需要识别出其中的个人信息类型...
</think>

{"entities": [{"type": "PERSON_NAME", "value": "张三", "start": 2, "end": 4}]}'''

        result = parser.extract_json(response)

        assert result is not None
        assert result["entities"][0]["value"] == "张三"
        assert result["entities"][0]["start"] == 2

    def test_malformed_json(self):
        """测试格式错误的 JSON"""
        parser = LLMResponseParser(strict=False)

        # 不完整的 JSON
        response = '{"entities": [{"type": "NAME"'
        result = parser.extract_json(response)

        assert result is None

    def test_no_json_in_response(self):
        """测试响应中没有 JSON"""
        parser = LLMResponseParser(strict=False)

        response = "这是一段纯文本，没有 JSON 数据"
        result = parser.extract_json(response)

        assert result is None

    def test_nested_json_objects(self):
        """测试嵌套的 JSON 对象"""
        parser = LLMResponseParser(strict=False)

        response = '''{"entities": [{"type": "ADDRESS", "value": "北京市", "components": {"city": "北京市", "district": "海淀区"}}]}'''
        result = parser.extract_json(response)

        assert result is not None
        assert result["entities"][0]["components"]["city"] == "北京市"

    def test_json_with_quotes_in_string(self):
        """测试 JSON 字符串中包含引号"""
        parser = LLMResponseParser(strict=False)

        response = '''{"entities": [{"type": "TEXT", "value": "他说:\\"你好\\""}]}'''
        result = parser.extract_json(response)

        assert result is not None
        assert '"' in result["entities"][0]["value"] or '\\"' in result["entities"][0]["value"]

    def test_extract_pii_entities(self):
        """测试 PII 实体提取"""
        parser = LLMResponseParser(strict=False)

        response = '''<think></think>
{"entities": [
    {"type": "PERSON_NAME", "value": "张三", "start": 0, "end": 2, "confidence": 0.95},
    {"type": "PHONE_NUMBER", "value": "13800138000", "start": 10, "end": 21, "confidence": 0.98}
]}'''

        entities = parser.extract_pii_entities(response)

        assert len(entities) == 2
        assert entities[0]["type"] == "PERSON_NAME"
        assert entities[1]["type"] == "PHONE_NUMBER"

    def test_extract_pii_with_missing_fields(self):
        """测试缺少必需字段的实体"""
        parser = LLMResponseParser(strict=False)

        response = '''{"entities": [
            {"type": "NAME", "value": "张三"},
            {"type": "PHONE"},
            {"value": "13800138000"}
        ]}'''

        entities = parser.extract_pii_entities(response, validate=True)

        # 只有第一个实体有效（有 type 和 value）
        assert len(entities) == 1
        assert entities[0]["value"] == "张三"

    def test_extract_pii_no_entities_field(self):
        """测试响应中缺少 entities 字段"""
        parser = LLMResponseParser(strict=False)

        response = '{"result": "success"}'
        entities = parser.extract_pii_entities(response)

        assert entities == []

    def test_schema_validation(self):
        """测试 schema 验证"""
        parser = LLMResponseParser(strict=False)

        response = '{"entities": [{"type": "NAME"}]}'
        expected_schema = {"entities": list}

        result = parser.extract_json(response, expected_schema=expected_schema)

        assert result is not None
        assert isinstance(result["entities"], list)

    def test_schema_validation_failure(self):
        """测试 schema 验证失败"""
        parser = LLMResponseParser(strict=False)

        response = '{"entities": "not a list"}'
        expected_schema = {"entities": list}

        result = parser.extract_json(response, expected_schema=expected_schema)

        assert result is None

    def test_strict_mode_raises_exception(self):
        """测试严格模式抛出异常"""
        parser = LLMResponseParser(strict=True)

        response = "没有 JSON"

        with pytest.raises(ValueError):
            parser.extract_json(response)

    def test_convenience_function(self):
        """测试便捷函数"""
        response = '{"entities": [{"type": "NAME", "value": "张三"}]}'
        entities = extract_pii_from_response(response)

        assert len(entities) == 1
        assert entities[0]["value"] == "张三"

    def test_remove_multiple_think_tags(self):
        """测试移除多个 <think> 标签"""
        parser = LLMResponseParser(strict=False)

        response = '<think>第一次思考</think>一些文本<think>第二次思考</think>{"key": "value"}'
        result = parser.extract_json(response)

        assert result is not None
        assert result["key"] == "value"

    def test_empty_response(self):
        """测试空响应"""
        parser = LLMResponseParser(strict=False)

        assert parser.extract_json("") is None
        assert parser.extract_json(None) is None

    def test_real_world_qwen_response(self):
        """测试真实的 Qwen 响应格式"""
        parser = LLMResponseParser(strict=False)

        # 模拟实际测试中看到的响应格式
        response = '''<think>

</think>

{"entities": [{"type": "PERSON_NAME", "value": "张三", "start": 2, "end": 4, "confidence": 0.9}, {"type": "ADDRESS", "value": "北京市海淀区中关村大街1号", "start": 5, "end": 20, "confidence": 0.9}]}'''

        entities = parser.extract_pii_entities(response)

        assert len(entities) == 2
        assert entities[0]["value"] == "张三"
        assert entities[1]["value"] == "北京市海淀区中关村大街1号"
        assert entities[0]["confidence"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
