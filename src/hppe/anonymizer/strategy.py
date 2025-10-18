"""
脱敏策略基类 (Anonymization Strategy)

定义统一的脱敏策略接口，所有具体策略都继承此基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from hppe.models.entity import Entity


class AnonymizationStrategy(ABC):
    """
    脱敏策略抽象基类

    所有具体脱敏策略（编辑、屏蔽、哈希、加密、合成）都需要继承此类
    """

    def __init__(self, config: Any = None):
        """
        初始化策略

        Args:
            config: 策略配置对象
        """
        self.config = config

    @abstractmethod
    def anonymize(self, entity: Entity, text: str) -> str:
        """
        对实体进行脱敏处理（返回完整文本）

        Args:
            entity: 待脱敏的实体
            text: 原始完整文本

        Returns:
            脱敏后的完整文本
        """
        pass

    @abstractmethod
    def get_replacement(self, entity: Entity) -> str:
        """
        获取实体的替换文本（仅返回替换部分）

        Args:
            entity: 待脱敏的实体

        Returns:
            替换文本（如 "[ID_CARD]", "138****5678", 哈希值等）
        """
        pass

    def get_strategy_name(self) -> str:
        """
        获取策略名称

        Returns:
            策略名称（如 "redact", "mask", "hash"）
        """
        return self.__class__.__name__.replace("Strategy", "").lower()

    def get_info(self) -> Dict[str, Any]:
        """
        获取策略信息

        Returns:
            策略信息字典
        """
        return {
            "name": self.get_strategy_name(),
            "class": self.__class__.__name__,
            "config": self.config.__dict__ if self.config else {}
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
