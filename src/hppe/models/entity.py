"""
PII 实体数据模型

定义检测到的 PII 实体的数据结构
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    """
    PII 实体数据模型

    表示在文本中检测到的一个 PII 实体，包含实体类型、值、位置、
    置信度等信息。

    Attributes:
        entity_type: PII 类型标识符（如 "CHINA_ID_CARD", "EMAIL_ADDRESS"）
        value: 检测到的实际值
        start_pos: 实体在原文本中的起始字符位置（从 0 开始）
        end_pos: 实体在原文本中的结束字符位置（不包含）
        confidence: 置信度分数，范围 [0.0, 1.0]
        detection_method: 检测方法标识（"regex", "llm", "hybrid"）
        recognizer_name: 执行检测的识别器名称
        metadata: 可选的额外元数据字典（如校验结果、上下文信息等）

    Examples:
        >>> entity = Entity(
        ...     entity_type="CHINA_ID_CARD",
        ...     value="110101199003077578",
        ...     start_pos=7,
        ...     end_pos=25,
        ...     confidence=0.95,
        ...     detection_method="regex",
        ...     recognizer_name="ChinaIDCardRecognizer"
        ... )
        >>> print(entity)
        Entity(CHINA_ID_CARD: "110101199003077578", pos=[7:25], conf=0.95)
    """

    entity_type: str
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    detection_method: str
    recognizer_name: str
    metadata: Optional[dict] = field(default=None)

    def __post_init__(self) -> None:
        """
        数据验证

        在实例创建后验证数据的合法性

        Raises:
            ValueError: 当数据不符合规范时
        """
        # 验证位置
        if self.start_pos < 0:
            raise ValueError(f"start_pos 必须 >= 0，当前值: {self.start_pos}")
        if self.end_pos <= self.start_pos:
            raise ValueError(
                f"end_pos ({self.end_pos}) 必须大于 start_pos ({self.start_pos})"
            )

        # 验证置信度
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence 必须在 [0.0, 1.0] 范围内，当前值: {self.confidence}"
            )

        # 验证必填字符串字段
        if not self.entity_type:
            raise ValueError("entity_type 不能为空")
        if not self.value:
            raise ValueError("value 不能为空")
        if not self.detection_method:
            raise ValueError("detection_method 不能为空")
        if not self.recognizer_name:
            raise ValueError("recognizer_name 不能为空")

    def __str__(self) -> str:
        """
        返回实体的可读字符串表示

        Returns:
            格式化的字符串，包含关键信息
        """
        return (
            f'Entity({self.entity_type}: "{self.value}", '
            f'pos=[{self.start_pos}:{self.end_pos}], '
            f'conf={self.confidence:.2f})'
        )

    def __repr__(self) -> str:
        """
        返回实体的详细字符串表示

        Returns:
            包含所有字段的完整表示
        """
        return (
            f"Entity("
            f"entity_type={self.entity_type!r}, "
            f"value={self.value!r}, "
            f"start_pos={self.start_pos}, "
            f"end_pos={self.end_pos}, "
            f"confidence={self.confidence}, "
            f"detection_method={self.detection_method!r}, "
            f"recognizer_name={self.recognizer_name!r}, "
            f"metadata={self.metadata!r})"
        )

    def to_dict(self) -> dict:
        """
        转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "recognizer_name": self.recognizer_name,
            "metadata": self.metadata,
        }

    @property
    def length(self) -> int:
        """
        返回实体值的字符长度

        Returns:
            实体占据的字符数
        """
        return self.end_pos - self.start_pos
