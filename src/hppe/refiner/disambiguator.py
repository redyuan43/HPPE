"""
歧义消除器 (Disambiguator)

解决多个识别器对同一文本片段给出不同类型判断时的冲突
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hppe.models.entity import Entity
from hppe.refiner.config import DisambiguatorConfig

logger = logging.getLogger(__name__)


class Disambiguator:
    """
    歧义消除器

    当多个识别器对同一文本片段给出不同PII类型时，
    基于置信度、识别器权重和类型优先级选择最合适的实体。
    """

    def __init__(self, config: DisambiguatorConfig = None):
        """
        初始化歧义消除器

        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or DisambiguatorConfig()
        logger.info(
            f"歧义消除器已初始化: "
            f"strict_mode={self.config.strict_mode}, "
            f"min_confidence_diff={self.config.min_confidence_diff}"
        )

    def resolve(self, entities: List[Entity]) -> List[Entity]:
        """
        解决实体冲突

        Args:
            entities: 待消歧的实体列表

        Returns:
            消歧后的实体列表
        """
        if not entities:
            return []

        # 1. 按位置分组（找出重叠的实体）
        conflict_groups = self._group_overlapping_entities(entities)

        # 2. 对每组冲突实体进行解决
        resolved_entities = []
        for group in conflict_groups:
            if len(group) == 1:
                # 无冲突，直接保留
                resolved_entities.append(group[0])
            else:
                # 有冲突，进行消歧
                best_entity = self._resolve_group(group)
                resolved_entities.append(best_entity)

        logger.debug(
            f"消歧完成: 输入{len(entities)}个实体，输出{len(resolved_entities)}个实体"
        )

        return sorted(resolved_entities, key=lambda e: e.start_pos)

    def _group_overlapping_entities(
        self, entities: List[Entity]
    ) -> List[List[Entity]]:
        """
        将重叠的实体分组

        Args:
            entities: 实体列表

        Returns:
            实体组列表，每组内的实体彼此重叠
        """
        if not entities:
            return []

        # 按开始位置排序
        sorted_entities = sorted(entities, key=lambda e: (e.start_pos, e.end_pos))

        groups = []
        current_group = [sorted_entities[0]]

        for entity in sorted_entities[1:]:
            # 检查是否与当前组重叠
            if self._is_overlapping(entity, current_group):
                current_group.append(entity)
            else:
                # 不重叠，开始新组
                groups.append(current_group)
                current_group = [entity]

        # 添加最后一组
        if current_group:
            groups.append(current_group)

        return groups

    def _is_overlapping(
        self, entity: Entity, group: List[Entity]
    ) -> bool:
        """
        检查实体是否与组内任一实体重叠

        Args:
            entity: 待检查实体
            group: 实体组

        Returns:
            是否重叠
        """
        for e in group:
            # 有重叠：开始位置在另一个实体内，或结束位置在另一个实体内
            if (e.start_pos <= entity.start_pos < e.end_pos or
                e.start_pos < entity.end_pos <= e.end_pos or
                entity.start_pos <= e.start_pos < entity.end_pos or
                entity.start_pos < e.end_pos <= entity.end_pos):
                return True
        return False

    def _resolve_group(self, group: List[Entity]) -> Entity:
        """
        解决一组冲突实体，选择最佳实体

        Args:
            group: 冲突实体组

        Returns:
            最佳实体
        """
        # 计算每个实体的综合得分
        scored_entities = []
        for entity in group:
            score = self._calculate_score(entity)
            scored_entities.append((entity, score))

        # 按得分排序
        scored_entities.sort(key=lambda x: x[1], reverse=True)

        best_entity, best_score = scored_entities[0]

        logger.debug(
            f"冲突解决: 从{len(group)}个候选中选择了 "
            f"{best_entity.entity_type}(score={best_score:.3f}, conf={best_entity.confidence:.2f})"
        )

        return best_entity

    def _calculate_score(self, entity: Entity) -> float:
        """
        计算实体的综合得分

        得分 = 置信度 × 识别器权重 + 类型优先级权重

        Args:
            entity: 实体

        Returns:
            综合得分
        """
        # 1. 基础得分：置信度
        base_score = entity.confidence

        # 2. 识别器权重
        recognizer_name = entity.detection_method or "unknown"
        recognizer_weight = self.config.recognizer_weights.get(recognizer_name, 1.0)

        # 3. 类型优先级
        type_priority = self.config.type_priorities.get(entity.entity_type, 0)
        # 归一化类型优先级到 0-1 范围
        normalized_priority = type_priority / 100.0

        # 综合得分
        score = base_score * recognizer_weight + normalized_priority * 0.5

        return score

    def get_info(self) -> Dict:
        """
        获取消歧器信息

        Returns:
            信息字典
        """
        return {
            "name": "Disambiguator",
            "config": {
                "strict_mode": self.config.strict_mode,
                "min_confidence_diff": self.config.min_confidence_diff,
                "recognizer_weights": self.config.recognizer_weights,
                "num_type_priorities": len(self.config.type_priorities)
            }
        }

    def __repr__(self) -> str:
        return f"Disambiguator(strict_mode={self.config.strict_mode})"
