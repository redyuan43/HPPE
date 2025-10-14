"""
正则表达式识别器基类

提供 PII 识别器的基础框架和通用功能
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Pattern

from hppe.models.entity import Entity


class BaseRecognizer(ABC):
    """
    PII 识别器抽象基类

    所有正则表达式识别器的基类，提供：
    - 配置加载和验证
    - 正则表达式预编译
    - 上下文词检测
    - 拒绝列表过滤
    - 置信度计算

    子类必须实现:
    - detect(): 核心检测逻辑
    - validate(): 实体验证逻辑（可选）

    Attributes:
        entity_type: PII 类型标识符
        patterns: 预编译的正则表达式模式列表
        context_words: 上下文关键词列表，用于提升置信度
        deny_lists: 拒绝列表，用于过滤误报
        confidence_base: 基础置信度分数
        recognizer_name: 识别器名称

    Examples:
        >>> class EmailRecognizer(BaseRecognizer):
        ...     def detect(self, text: str) -> List[Entity]:
        ...         # 实现检测逻辑
        ...         pass
        ...
        ...     def validate(self, entity: Entity) -> bool:
        ...         # 实现验证逻辑
        ...         return True
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化识别器

        Args:
            config: 识别器配置字典，必须包含以下键：
                - entity_type: PII 类型
                - patterns: 正则表达式模式列表
                可选键：
                - context_words: 上下文词列表
                - deny_lists: 拒绝列表
                - confidence_base: 基础置信度（默认 0.85）
                - name: 识别器名称

        Raises:
            ValueError: 当配置缺少必需字段时
        """
        # 验证必需字段
        if "entity_type" not in config:
            raise ValueError("配置中缺少必需字段: entity_type")
        if "patterns" not in config:
            raise ValueError("配置中缺少必需字段: patterns")

        self.entity_type: str = config["entity_type"]
        self.patterns: List[Pattern] = self._compile_patterns(config["patterns"])
        self.context_words: List[str] = config.get("context_words", [])
        self.deny_lists: List[str] = config.get("deny_lists", [])
        self.confidence_base: float = config.get("confidence_base", 0.85)
        self.recognizer_name: str = config.get(
            "name", self.__class__.__name__
        )

        # 验证置信度范围
        if not 0.0 <= self.confidence_base <= 1.0:
            raise ValueError(
                f"confidence_base 必须在 [0.0, 1.0] 范围内: {self.confidence_base}"
            )

    @abstractmethod
    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的 PII 实体

        子类必须实现此方法，提供具体的检测逻辑。

        Args:
            text: 待检测的文本

        Returns:
            检测到的 Entity 实体列表

        Examples:
            >>> recognizer = MyRecognizer(config)
            >>> entities = recognizer.detect("包含 PII 的文本")
        """
        pass

    @abstractmethod
    def validate(self, entity: Entity) -> bool:
        """
        验证实体的有效性

        子类可以实现特定的验证逻辑（如校验码验证）。
        如果不需要验证，可以直接返回 True。

        Args:
            entity: 待验证的实体

        Returns:
            True 表示实体有效，False 表示无效

        Examples:
            >>> is_valid = recognizer.validate(entity)
        """
        pass

    def _compile_patterns(self, patterns: List[Dict[str, Any]]) -> List[Pattern]:
        """
        预编译正则表达式模式

        在初始化时预编译所有模式，避免运行时重复编译开销。

        Args:
            patterns: 模式配置列表，每个元素应包含 'pattern' 键

        Returns:
            预编译的正则表达式对象列表

        Raises:
            ValueError: 当模式格式错误或编译失败时
        """
        compiled_patterns = []

        for i, pattern_config in enumerate(patterns):
            if not isinstance(pattern_config, dict):
                raise ValueError(f"模式 #{i} 必须是字典类型")

            if "pattern" not in pattern_config:
                raise ValueError(f"模式 #{i} 缺少 'pattern' 键")

            pattern_str = pattern_config["pattern"]
            try:
                # 使用 re.UNICODE 支持多语言
                compiled = re.compile(pattern_str, re.UNICODE)
                compiled_patterns.append(compiled)
            except re.error as e:
                raise ValueError(
                    f"模式 #{i} 编译失败: {pattern_str}. 错误: {e}"
                )

        return compiled_patterns

    def _check_context(
        self,
        text: str,
        match_pos: int,
        window_size: int = 50
    ) -> float:
        """
        检查匹配位置附近是否有上下文关键词

        在匹配位置前后一定范围内搜索上下文词，如果找到则提升置信度。

        Args:
            text: 原始文本
            match_pos: 匹配位置（起始位置）
            window_size: 上下文窗口大小（字符数）

        Returns:
            置信度提升值，范围 [0.0, 0.15]
        """
        if not self.context_words:
            return 0.0

        # 定义上下文窗口
        start = max(0, match_pos - window_size)
        end = min(len(text), match_pos + window_size)
        context = text[start:end].lower()

        # 检查是否有上下文词出现
        found_count = 0
        for word in self.context_words:
            if word.lower() in context:
                found_count += 1

        # 根据找到的上下文词数量计算提升值
        if found_count > 0:
            # 最多提升 0.15
            boost = min(0.15, found_count * 0.05)
            return boost

        return 0.0

    def _check_deny_list(self, text: str, match_pos: int) -> bool:
        """
        检查匹配位置附近是否有拒绝列表中的词

        如果在附近发现拒绝词，则认为这是误报。

        Args:
            text: 原始文本
            match_pos: 匹配位置

        Returns:
            True 表示应该拒绝此匹配（误报），False 表示通过
        """
        if not self.deny_lists:
            return False

        # 检查更大范围的上下文（100 字符）
        window_size = 100
        start = max(0, match_pos - window_size)
        end = min(len(text), match_pos + window_size)
        context = text[start:end].lower()

        # 检查是否有拒绝词出现
        for deny_word in self.deny_lists:
            if deny_word.lower() in context:
                return True

        return False

    def _calculate_confidence(
        self,
        base_score: float,
        text: str,
        match_pos: int,
        validation_passed: bool = True
    ) -> float:
        """
        计算最终置信度

        综合考虑：
        - 模式的基础分数
        - 上下文词提升
        - 验证结果

        Args:
            base_score: 模式的基础分数
            text: 原始文本
            match_pos: 匹配位置
            validation_passed: 验证是否通过

        Returns:
            最终置信度分数，范围 [0.0, 1.0]
        """
        # 从基础分数开始
        confidence = base_score

        # 上下文词提升
        context_boost = self._check_context(text, match_pos)
        confidence = min(1.0, confidence + context_boost)

        # 验证失败降低置信度
        if not validation_passed:
            confidence *= 0.7

        return confidence

    def _create_entity(
        self,
        value: str,
        start_pos: int,
        end_pos: int,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Entity:
        """
        创建 Entity 实例的辅助方法

        Args:
            value: 实体值
            start_pos: 起始位置
            end_pos: 结束位置
            confidence: 置信度
            metadata: 可选的元数据

        Returns:
            Entity 实例
        """
        return Entity(
            entity_type=self.entity_type,
            value=value,
            start_pos=start_pos,
            end_pos=end_pos,
            confidence=confidence,
            detection_method="regex",
            recognizer_name=self.recognizer_name,
            metadata=metadata
        )
