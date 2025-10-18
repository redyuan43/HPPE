"""
实体合并器 (Entity Merger)

处理重叠和相邻实体的合并逻辑
"""

import sys
from pathlib import Path
from typing import List, Optional
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hppe.models.entity import Entity
from hppe.refiner.config import MergerConfig

logger = logging.getLogger(__name__)


class EntityMerger:
    """
    实体合并器

    处理两种合并场景：
    1. 重叠实体合并：保留最长span或最高置信度的实体
    2. 相邻实体合并：合并特定类型的相邻实体（如ADDRESS）
    """

    def __init__(self, config: MergerConfig = None):
        """
        初始化实体合并器

        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or MergerConfig()
        logger.info(
            f"实体合并器已初始化: "
            f"merge_overlapping={self.config.merge_overlapping}, "
            f"merge_adjacent={self.config.merge_adjacent}"
        )

    def merge(self, entities: List[Entity]) -> List[Entity]:
        """
        合并重叠和相邻实体

        Args:
            entities: 待合并的实体列表

        Returns:
            合并后的实体列表
        """
        if not entities:
            return []

        result = entities.copy()

        # 1. 合并重叠实体
        if self.config.merge_overlapping:
            result = self._merge_overlapping(result)
            logger.debug(f"重叠合并后: {len(result)} 个实体")

        # 2. 合并相邻实体
        if self.config.merge_adjacent:
            result = self._merge_adjacent(result)
            logger.debug(f"相邻合并后: {len(result)} 个实体")

        return sorted(result, key=lambda e: e.start_pos)

    def _merge_overlapping(self, entities: List[Entity]) -> List[Entity]:
        """
        合并重叠实体

        策略：
        - 如果类型相同：保留span更长的，或置信度更高的
        - 如果类型不同：保留置信度更高的

        Args:
            entities: 实体列表

        Returns:
            合并后的实体列表
        """
        if not entities:
            return []

        # 按开始位置排序
        sorted_entities = sorted(entities, key=lambda e: (e.start_pos, -len(e.value)))

        merged = []
        skip_indices = set()

        for i, entity1 in enumerate(sorted_entities):
            if i in skip_indices:
                continue

            # 查找所有与entity1重叠的实体
            overlapping_group = [entity1]
            for j in range(i + 1, len(sorted_entities)):
                if j in skip_indices:
                    continue

                entity2 = sorted_entities[j]

                # 检查是否重叠
                if self._is_overlapping(entity1, entity2):
                    overlapping_group.append(entity2)
                    skip_indices.add(j)
                elif entity2.start_pos >= entity1.end_pos:
                    # 如果entity2在entity1之后且不重叠，则不再检查后续实体
                    break

            # 从重叠组中选择最佳实体
            if len(overlapping_group) > 1:
                best_entity = self._select_best_from_group(overlapping_group)
                merged.append(best_entity)
                logger.debug(
                    f"合并 {len(overlapping_group)} 个重叠实体，保留: "
                    f"{best_entity.entity_type} '{best_entity.value}' (conf={best_entity.confidence:.2f})"
                )
            else:
                merged.append(entity1)

        return merged

    def _merge_adjacent(self, entities: List[Entity]) -> List[Entity]:
        """
        合并相邻实体

        仅对特定类型（如ADDRESS）启用，将间隔较小的同类型实体合并

        Args:
            entities: 实体列表

        Returns:
            合并后的实体列表
        """
        if not entities:
            return []

        # 按开始位置排序
        sorted_entities = sorted(entities, key=lambda e: e.start_pos)

        merged = []
        i = 0

        while i < len(sorted_entities):
            current = sorted_entities[i]

            # 如果当前实体类型不在可合并列表中，直接添加
            if current.entity_type not in self.config.mergeable_types:
                merged.append(current)
                i += 1
                continue

            # 查找可以合并的相邻实体
            adjacent_group = [current]
            j = i + 1

            while j < len(sorted_entities):
                next_entity = sorted_entities[j]

                # 检查是否应该合并
                if self._should_merge_adjacent(current, next_entity):
                    adjacent_group.append(next_entity)
                    current = next_entity  # 更新current以继续查找
                    j += 1
                else:
                    break

            # 合并相邻组
            if len(adjacent_group) > 1:
                merged_entity = self._merge_adjacent_group(adjacent_group)
                merged.append(merged_entity)
                logger.debug(
                    f"合并 {len(adjacent_group)} 个相邻实体: "
                    f"'{merged_entity.value}' ({merged_entity.entity_type})"
                )
                i = j
            else:
                merged.append(current)
                i += 1

        return merged

    def _is_overlapping(self, entity1: Entity, entity2: Entity) -> bool:
        """
        检查两个实体是否重叠

        Args:
            entity1: 实体1
            entity2: 实体2

        Returns:
            是否重叠
        """
        # 计算重叠字符数
        overlap_start = max(entity1.start_pos, entity2.start_pos)
        overlap_end = min(entity1.end_pos, entity2.end_pos)
        overlap_length = max(0, overlap_end - overlap_start)

        if overlap_length == 0:
            return False

        # 计算重叠比例
        shorter_length = min(len(entity1.value), len(entity2.value))
        overlap_ratio = overlap_length / shorter_length if shorter_length > 0 else 0

        return overlap_ratio >= self.config.overlap_threshold

    def _should_merge_adjacent(self, entity1: Entity, entity2: Entity) -> bool:
        """
        判断两个实体是否应该合并（相邻且类型相同）

        Args:
            entity1: 实体1
            entity2: 实体2

        Returns:
            是否应该合并
        """
        # 必须是相同类型
        if entity1.entity_type != entity2.entity_type:
            return False

        # 必须是可合并类型
        if entity1.entity_type not in self.config.mergeable_types:
            return False

        # 间隔不能太大
        gap = entity2.start_pos - entity1.end_pos
        if gap < 0 or gap > self.config.max_adjacent_gap:
            return False

        return True

    def _select_best_from_group(self, group: List[Entity]) -> Entity:
        """
        从重叠实体组中选择最佳实体

        策略：
        1. 如果类型相同：选择span最长的，相同长度则选置信度最高的
        2. 如果类型不同：选择置信度最高的

        Args:
            group: 重叠实体组

        Returns:
            最佳实体
        """
        if not group:
            raise ValueError("实体组不能为空")

        if len(group) == 1:
            return group[0]

        # 检查是否所有实体类型相同
        types = set(e.entity_type for e in group)

        if len(types) == 1:
            # 类型相同：优先选择span最长的
            group_sorted = sorted(
                group,
                key=lambda e: (len(e.value), e.confidence),
                reverse=True
            )
        else:
            # 类型不同：选择置信度最高的
            group_sorted = sorted(group, key=lambda e: e.confidence, reverse=True)

        return group_sorted[0]

    def _merge_adjacent_group(self, group: List[Entity]) -> Entity:
        """
        合并一组相邻实体

        Args:
            group: 相邻实体组（必须已按位置排序）

        Returns:
            合并后的实体
        """
        if not group:
            raise ValueError("实体组不能为空")

        if len(group) == 1:
            return group[0]

        # 提取合并后的文本（包括中间的间隔字符）
        # 注意：这里需要原始文本来获取间隔字符，但Entity类可能没有保存原始文本
        # 简化处理：直接拼接value
        merged_value = "".join(e.value for e in group)

        # 使用第一个和最后一个实体的位置
        start_pos = group[0].start_pos
        end_pos = group[-1].end_pos

        # 置信度取平均
        avg_confidence = sum(e.confidence for e in group) / len(group)

        # 创建合并后的实体
        merged_entity = Entity(
            entity_type=group[0].entity_type,
            value=merged_value,
            start_pos=start_pos,
            end_pos=end_pos,
            confidence=avg_confidence,
            detection_method=group[0].detection_method,
            recognizer_name=group[0].recognizer_name,
            metadata={
                "merged_from": len(group),
                "source_confidences": [e.confidence for e in group]
            }
        )

        return merged_entity

    def get_info(self) -> dict:
        """
        获取合并器信息

        Returns:
            信息字典
        """
        return {
            "name": "EntityMerger",
            "config": {
                "merge_overlapping": self.config.merge_overlapping,
                "merge_adjacent": self.config.merge_adjacent,
                "mergeable_types": self.config.mergeable_types,
                "max_adjacent_gap": self.config.max_adjacent_gap,
                "overlap_threshold": self.config.overlap_threshold
            }
        }

    def __repr__(self) -> str:
        return (
            f"EntityMerger(overlap={self.config.merge_overlapping}, "
            f"adjacent={self.config.merge_adjacent})"
        )
