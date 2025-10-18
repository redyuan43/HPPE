"""
脱敏引擎 (Anonymization Engine)

实现批量脱敏处理，管理策略注册和调用
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hppe.models.entity import Entity
from hppe.anonymizer.strategy import AnonymizationStrategy
from hppe.anonymizer.config import AnonymizationConfig

logger = logging.getLogger(__name__)


class AnonymizationEngine:
    """
    脱敏引擎

    负责管理脱敏策略，执行批量脱敏操作
    """

    def __init__(self, config: AnonymizationConfig = None):
        """
        初始化脱敏引擎

        Args:
            config: 脱敏配置对象
        """
        self.config = config or AnonymizationConfig()
        self.strategies: Dict[str, AnonymizationStrategy] = {}

        logger.info(
            f"脱敏引擎已初始化: default_strategy={self.config.default_strategy}, "
            f"batch_optimization={self.config.batch_optimization}"
        )

    def register_strategy(
        self,
        strategy_name: str,
        strategy: AnonymizationStrategy
    ):
        """
        注册脱敏策略

        Args:
            strategy_name: 策略名称（如 "redact", "mask", "hash"）
            strategy: 策略实例
        """
        self.strategies[strategy_name] = strategy
        logger.info(f"已注册策略: {strategy_name} -> {strategy.__class__.__name__}")

    def get_strategy(self, pii_type: str) -> Optional[AnonymizationStrategy]:
        """
        获取PII类型对应的策略

        Args:
            pii_type: PII类型（如 "ID_CARD", "PHONE_NUMBER"）

        Returns:
            策略实例，如果未找到则返回默认策略
        """
        # 1. 查找类型特定策略
        strategy_name = self.config.type_strategy_map.get(pii_type)

        # 2. 如果未配置，使用默认策略
        if not strategy_name:
            strategy_name = self.config.default_strategy

        # 3. 返回策略实例
        strategy = self.strategies.get(strategy_name)

        if not strategy:
            logger.warning(
                f"未找到策略 '{strategy_name}'（PII类型: {pii_type}），"
                f"使用默认策略 '{self.config.default_strategy}'"
            )
            strategy = self.strategies.get(self.config.default_strategy)

        return strategy

    def anonymize_text(
        self,
        entities: List[Entity],
        text: str
    ) -> str:
        """
        对文本进行批量脱敏

        Args:
            entities: 待脱敏的实体列表
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        if not entities:
            return text

        # 按位置从后向前排序（避免位置偏移问题）
        if self.config.batch_optimization:
            sorted_entities = sorted(
                entities,
                key=lambda e: e.start_pos,
                reverse=True
            )
        else:
            sorted_entities = entities

        # 逐个替换实体
        result_text = text
        for entity in sorted_entities:
            strategy = self.get_strategy(entity.entity_type)

            if not strategy:
                logger.warning(
                    f"无法获取策略，跳过实体: {entity.entity_type} '{entity.value}'"
                )
                continue

            # 获取替换文本
            replacement = strategy.get_replacement(entity)

            # 执行替换
            result_text = (
                result_text[:entity.start_pos] +
                replacement +
                result_text[entity.end_pos:]
            )

            logger.debug(
                f"已脱敏: {entity.entity_type} '{entity.value}' -> '{replacement}'"
            )

        logger.info(
            f"批量脱敏完成: 处理{len(entities)}个实体"
        )

        return result_text

    def anonymize_entities(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Entity]:
        """
        返回脱敏后的实体列表（更新实体的value为替换值）

        Args:
            entities: 待脱敏的实体列表
            text: 原始文本

        Returns:
            脱敏后的实体列表（Entity对象的value被更新）
        """
        anonymized_entities = []

        for entity in entities:
            strategy = self.get_strategy(entity.entity_type)

            if not strategy:
                logger.warning(
                    f"无法获取策略，保留原实体: {entity.entity_type}"
                )
                anonymized_entities.append(entity)
                continue

            # 获取替换文本
            replacement = strategy.get_replacement(entity)

            # 创建新实体（value为脱敏后的值）
            anonymized_entity = Entity(
                entity_type=entity.entity_type,
                value=replacement,  # 替换为脱敏值
                start_pos=entity.start_pos,
                end_pos=entity.start_pos + len(replacement),
                confidence=entity.confidence,
                detection_method=entity.detection_method,
                recognizer_name=entity.recognizer_name,
                metadata={
                    **(entity.metadata or {}),
                    "original_value": entity.value,
                    "anonymization_strategy": strategy.get_strategy_name()
                }
            )

            anonymized_entities.append(anonymized_entity)

        return anonymized_entities

    def get_info(self) -> Dict:
        """
        获取引擎信息

        Returns:
            引擎信息字典
        """
        return {
            "name": "AnonymizationEngine",
            "config": {
                "default_strategy": self.config.default_strategy,
                "batch_optimization": self.config.batch_optimization,
                "type_strategy_map": self.config.type_strategy_map
            },
            "registered_strategies": list(self.strategies.keys()),
            "num_strategies": len(self.strategies)
        }

    def __repr__(self) -> str:
        return (
            f"AnonymizationEngine(strategies={len(self.strategies)}, "
            f"default={self.config.default_strategy})"
        )
