"""
LLM 响应解析器

用于从 LLM 生成的文本中提取和验证结构化数据（主要是 JSON）
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMResponseParser:
    """
    LLM 响应解析器

    处理 LLM 输出中的各种格式问题：
    - 提取 <think> 标签后的 JSON
    - 清理多余的文本和标签
    - 验证 JSON 格式
    - 提供详细的错误信息

    Examples:
        >>> parser = LLMResponseParser()
        >>>
        >>> # 处理带 <think> 标签的响应
        >>> response = '<think>\\n\\n</think>\\n\\n{"entities": [...]}'
        >>> result = parser.extract_json(response)
        >>> print(result)
        {'entities': [...]}
        >>>
        >>> # 处理格式错误的 JSON
        >>> response = '{"entities": [{"type": "NAME"'  # 不完整
        >>> result = parser.extract_json(response, strict=False)
        >>> # 返回 None 并记录错误
    """

    def __init__(self, strict: bool = True):
        """
        初始化解析器

        Args:
            strict: 是否启用严格模式（遇到错误立即抛出异常）
        """
        self.strict = strict

    def extract_json(
        self,
        response: str,
        expected_schema: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        从 LLM 响应中提取 JSON 数据

        处理步骤：
        1. 移除 <think> 标签及其内容
        2. 查找 JSON 对象（{ ... }）
        3. 解析并验证 JSON
        4. 可选：验证是否符合预期 schema

        Args:
            response: LLM 的原始响应文本
            expected_schema: 可选的预期 JSON schema（用于验证）

        Returns:
            解析后的 JSON 对象，失败时返回 None

        Raises:
            ValueError: strict=True 时，解析失败会抛出异常

        Examples:
            >>> parser = LLMResponseParser()
            >>> response = '<think>...</think>{"key": "value"}'
            >>> result = parser.extract_json(response)
            >>> print(result)
            {'key': 'value'}
        """
        if not response or not isinstance(response, str):
            logger.warning("响应为空或类型错误")
            return None

        # 步骤 1: 移除 <think> 标签
        cleaned = self._remove_think_tags(response)

        # 步骤 2: 查找 JSON 对象
        json_str = self._find_json_object(cleaned)

        if not json_str:
            msg = "未找到有效的 JSON 对象"
            logger.warning(f"{msg}，原始响应: {response[:100]}...")
            if self.strict:
                raise ValueError(msg)
            return None

        # 步骤 3: 解析 JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            msg = f"JSON 解析失败: {e}"
            logger.error(f"{msg}，JSON 字符串: {json_str[:200]}...")
            if self.strict:
                raise ValueError(msg) from e
            return None

        # 步骤 4: 验证 schema（可选）
        if expected_schema:
            if not self._validate_schema(data, expected_schema):
                msg = "JSON 格式不符合预期 schema"
                logger.warning(f"{msg}，期望: {expected_schema}, 实际: {data}")
                if self.strict:
                    raise ValueError(msg)
                return None

        logger.debug(f"成功解析 JSON，包含 {len(data)} 个顶层键")
        return data

    def extract_pii_entities(
        self,
        response: str,
        validate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        从响应中提取 PII 实体列表

        专门用于 PII 检测响应的解析，期望格式：
        {
            "entities": [
                {
                    "type": "PERSON_NAME",
                    "value": "张三",
                    "start": 0,
                    "end": 2,
                    "confidence": 0.95
                },
                ...
            ]
        }

        Args:
            response: LLM 响应
            validate: 是否验证每个实体的必需字段

        Returns:
            PII 实体列表，失败时返回空列表

        Examples:
            >>> parser = LLMResponseParser()
            >>> response = '{"entities": [{"type": "NAME", "value": "张三", ...}]}'
            >>> entities = parser.extract_pii_entities(response)
            >>> print(len(entities))
            1
        """
        # 提取 JSON
        data = self.extract_json(response)

        if not data:
            logger.warning("无法从响应中提取 JSON")
            return []

        # 获取 entities 字段
        if "entities" not in data:
            logger.warning(f"JSON 中缺少 'entities' 字段，可用字段: {list(data.keys())}")
            return []

        entities = data["entities"]

        if not isinstance(entities, list):
            logger.warning(f"'entities' 应该是列表，实际类型: {type(entities)}")
            return []

        # 验证每个实体（可选）
        if validate:
            validated_entities = []
            required_fields = ["type", "value"]  # 最基本的必需字段

            for i, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    logger.warning(f"实体 {i} 不是字典类型: {entity}")
                    continue

                # 检查必需字段
                missing_fields = [f for f in required_fields if f not in entity]
                if missing_fields:
                    logger.warning(
                        f"实体 {i} 缺少必需字段 {missing_fields}: {entity}"
                    )
                    continue

                validated_entities.append(entity)

            logger.debug(
                f"验证完成: {len(validated_entities)}/{len(entities)} 个实体有效"
            )
            return validated_entities

        return entities

    def _remove_think_tags(self, text: str) -> str:
        """
        移除 <think> 标签及其内容

        处理各种格式：
        - <think>...</think>
        - <think>\n\n</think>
        - 多个 <think> 标签

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # 移除 <think>...</think> 及其内容
        # 使用 DOTALL 模式，使 . 匹配换行符
        cleaned = re.sub(
            r'<think>.*?</think>',
            '',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

        # 移除可能残留的单独标签
        cleaned = re.sub(r'</?think>', '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _find_json_object(self, text: str) -> Optional[str]:
        """
        从文本中查找 JSON 对象

        策略：
        1. 查找第一个 { 和最后一个 }
        2. 尝试提取中间的内容
        3. 验证括号平衡

        Args:
            text: 待搜索的文本

        Returns:
            JSON 字符串，未找到时返回 None
        """
        # 查找第一个 {
        start = text.find('{')
        if start == -1:
            return None

        # 从第一个 { 开始，找到匹配的 }
        # 使用括号计数确保正确配对
        bracket_count = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            # 处理字符串中的内容（避免字符串内的 {} 干扰）
            if char == '"' and not escape_next:
                in_string = not in_string
            elif char == '\\' and not escape_next:
                escape_next = True
                continue

            if not in_string:
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1

                    # 找到匹配的闭括号
                    if bracket_count == 0:
                        return text[start:i+1]

            escape_next = False

        # 未找到匹配的 }
        logger.warning("找到起始 { 但未找到匹配的 }")
        return None

    def _validate_schema(
        self,
        data: Dict[str, Any],
        expected_schema: Dict[str, Any]
    ) -> bool:
        """
        简单的 schema 验证

        验证：
        - 必需的顶层键是否存在
        - 值的类型是否匹配

        Args:
            data: 实际数据
            expected_schema: 预期 schema（简化格式）

        Returns:
            是否通过验证

        Examples:
            >>> parser = LLMResponseParser()
            >>> data = {"entities": []}
            >>> schema = {"entities": list}
            >>> parser._validate_schema(data, schema)
            True
        """
        for key, expected_type in expected_schema.items():
            # 检查键是否存在
            if key not in data:
                logger.warning(f"缺少必需键: {key}")
                return False

            # 检查类型（如果指定）
            if expected_type is not None:
                actual_type = type(data[key])
                if not isinstance(data[key], expected_type):
                    logger.warning(
                        f"键 '{key}' 类型不匹配: "
                        f"期望 {expected_type}, 实际 {actual_type}"
                    )
                    return False

        return True

    def format_error_context(
        self,
        response: str,
        max_length: int = 200
    ) -> str:
        """
        格式化错误上下文信息

        用于错误报告，显示响应的关键部分

        Args:
            response: 原始响应
            max_length: 最大显示长度

        Returns:
            格式化后的上下文字符串
        """
        if len(response) <= max_length:
            return response

        # 显示开头和结尾
        half = max_length // 2
        return (
            f"{response[:half]}..."
            f"[省略 {len(response) - max_length} 个字符]..."
            f"{response[-half:]}"
        )


# 便捷函数
def extract_pii_from_response(response: str) -> List[Dict[str, Any]]:
    """
    便捷函数：从 LLM 响应中提取 PII 实体

    Args:
        response: LLM 响应

    Returns:
        PII 实体列表

    Examples:
        >>> from hppe.engines.llm.response_parser import extract_pii_from_response
        >>> response = '{"entities": [{"type": "NAME", "value": "张三"}]}'
        >>> entities = extract_pii_from_response(response)
        >>> print(entities[0]["value"])
        张三
    """
    parser = LLMResponseParser(strict=False)
    return parser.extract_pii_entities(response)
