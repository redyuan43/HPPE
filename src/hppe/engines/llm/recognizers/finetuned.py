"""
Fine-tuned LLM PII 识别器

使用训练好的模型直接进行 PII 检测（无需 prompt engineering）
"""

import logging
from typing import List, Optional

from hppe.models.entity import Entity
from hppe.engines.llm.qwen_finetuned import QwenFineTunedEngine

logger = logging.getLogger(__name__)


class FineTunedLLMRecognizer:
    """
    Fine-tuned LLM 识别器

    使用训练好的 Qwen 模型进行 PII 检测，支持 6 种 PII 类型：
    - ADDRESS: 地址
    - ORGANIZATION: 组织机构
    - PERSON_NAME: 人名
    - PHONE_NUMBER: 电话号码
    - EMAIL: 邮箱
    - ID_CARD: 身份证号

    与正则识别器保持相同的接口，可以直接集成到检测Pipeline

    Attributes:
        llm_engine: Fine-tuned Qwen 引擎
        confidence_threshold: 置信度阈值
        recognizer_name: 识别器名称

    Examples:
        >>> # 初始化识别器
        >>> engine = QwenFineTunedEngine()
        >>> recognizer = FineTunedLLMRecognizer(engine)
        >>>
        >>> # 检测 PII
        >>> entities = recognizer.detect("我是张三，电话13812345678")
        >>> for entity in entities:
        ...     print(f"{entity.entity_type}: {entity.value}")
    """

    def __init__(
        self,
        llm_engine: QwenFineTunedEngine,
        confidence_threshold: float = 0.75,
        recognizer_name: str = "FineTunedLLMRecognizer"
    ):
        """
        初始化 Fine-tuned LLM 识别器

        Args:
            llm_engine: QwenFineTunedEngine 实例
            confidence_threshold: 置信度阈值（0.0-1.0）
            recognizer_name: 识别器名称
        """
        self.llm_engine = llm_engine
        self.confidence_threshold = confidence_threshold
        self.recognizer_name = recognizer_name

        # 支持的实体类型列表
        self.supported_types = llm_engine.get_supported_pii_types()

        logger.info(
            f"初始化 {recognizer_name}，"
            f"支持 {len(self.supported_types)} 种 PII 类型，"
            f"置信度阈值: {confidence_threshold}"
        )

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的所有 PII 实体

        Args:
            text: 待检测的文本

        Returns:
            检测到的 Entity 实体列表

        Examples:
            >>> recognizer = FineTunedLLMRecognizer(engine)
            >>> entities = recognizer.detect("我是张三")
            >>> print(len(entities))
            1
        """
        if not text or not text.strip():
            return []

        try:
            # 调用 fine-tuned 模型的检测接口
            entities = self.llm_engine.detect_pii(
                text=text,
                confidence_threshold=self.confidence_threshold
            )

            logger.debug(
                f"检测到 {len(entities)} 个 PII 实体 "
                f"(text_len={len(text)})"
            )

            return entities

        except Exception as e:
            logger.error(f"Fine-tuned LLM 检测失败: {e}")
            return []

    def detect_specific_type(
        self,
        text: str,
        entity_type: str
    ) -> List[Entity]:
        """
        检测文本中特定类型的 PII 实体

        Args:
            text: 待检测的文本
            entity_type: 要检测的 PII 类型（如 "PERSON_NAME"）

        Returns:
            指定类型的 Entity 列表

        Examples:
            >>> recognizer = FineTunedLLMRecognizer(engine)
            >>> persons = recognizer.detect_specific_type(
            ...     "张三和李四是同事",
            ...     "PERSON_NAME"
            ... )
        """
        if entity_type not in self.supported_types:
            logger.warning(
                f"不支持的 PII 类型: {entity_type}，"
                f"支持的类型: {self.supported_types}"
            )
            return []

        # 检测所有实体，然后过滤指定类型
        all_entities = self.detect(text)

        filtered = [
            e for e in all_entities
            if e.entity_type == entity_type
        ]

        logger.debug(
            f"过滤后的 {entity_type} 实体: {len(filtered)} 个"
        )

        return filtered

    def validate(self, entity: Entity) -> bool:
        """
        验证实体的有效性

        Fine-tuned 模型已经过训练，输出通常是可靠的，
        因此默认返回 True

        Args:
            entity: 待验证的实体

        Returns:
            True（默认信任模型输出）
        """
        # 基本验证：置信度是否达标
        return entity.confidence >= self.confidence_threshold

    def get_supported_types(self) -> List[str]:
        """
        获取支持的 PII 类型列表

        Returns:
            PII 类型字符串列表
        """
        return self.supported_types.copy()

    def get_info(self) -> dict:
        """
        获取识别器信息

        Returns:
            包含识别器元信息的字典
        """
        return {
            "name": self.recognizer_name,
            "detection_method": "llm_finetuned",
            "model": self.llm_engine.model_name,
            "model_path": str(self.llm_engine.model_path),
            "supported_types": self.supported_types,
            "confidence_threshold": self.confidence_threshold,
        }

    def __repr__(self) -> str:
        """返回识别器的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"model={self.llm_engine.model_name}, "
            f"types={len(self.supported_types)}, "
            f"threshold={self.confidence_threshold})"
        )
