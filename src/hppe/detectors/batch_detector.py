"""
批量 PII 检测器

优化大规模文本的 PII 检测性能：
- 批量 LLM 推理
- 并行 Regex 检测
- 结果缓存
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from hppe.models.entity import Entity
from hppe.detectors.hybrid_detector import HybridPIIDetector, DetectionMode
from hppe.engines.llm import QwenEngine

logger = logging.getLogger(__name__)


class BatchPIIDetector:
    """
    批量 PII 检测器

    针对大量文本优化检测性能：
    - Regex 并行检测（极快，无瓶颈）
    - LLM 批量推理（提高吞吐量）
    - 智能批次划分

    Attributes:
        detector: 基础混合检测器
        batch_size: LLM 批处理大小
        max_workers: 并行工作线程数

    Examples:
        >>> # 创建批量检测器
        >>> batch_detector = BatchPIIDetector(
        ...     llm_engine=QwenEngine(),
        ...     batch_size=10
        ... )
        >>>
        >>> # 批量检测
        >>> texts = ["文本1", "文本2", ...]
        >>> results = batch_detector.detect_batch(texts)
        >>> for text, entities in zip(texts, results):
        ...     print(f"{text}: {len(entities)} 个 PII")
    """

    def __init__(
        self,
        llm_engine: Optional[QwenEngine] = None,
        batch_size: int = 10,
        max_workers: int = 4,
        mode: Literal["fast", "auto", "deep"] = "auto"
    ):
        """
        初始化批量检测器

        Args:
            llm_engine: LLM 引擎（可选）
            batch_size: LLM 批处理大小（影响内存和延迟）
            max_workers: Regex 并行线程数
            mode: 检测模式
        """
        self.detector = HybridPIIDetector(
            llm_engine=llm_engine,
            mode=mode
        )

        self.batch_size = batch_size
        self.max_workers = max_workers

        logger.info(
            f"批量检测器初始化: batch_size={batch_size}, "
            f"max_workers={max_workers}, mode={mode}"
        )

    def detect_batch(
        self,
        texts: List[str],
        entity_types: Optional[List[str]] = None,
        mode: Optional[Literal["fast", "auto", "deep"]] = None,
        show_progress: bool = False
    ) -> List[List[Entity]]:
        """
        批量检测文本中的 PII

        Args:
            texts: 文本列表
            entity_types: 指定要检测的 PII 类型
            mode: 检测模式，覆盖默认模式
            show_progress: 是否显示进度

        Returns:
            实体列表的列表，每个文本对应一个实体列表

        Examples:
            >>> texts = ["张三，13800138000", "李四，13900139000"]
            >>> results = batch_detector.detect_batch(texts)
            >>> print(f"检测了 {len(results)} 个文本")
        """
        start_time = time.time()

        if not texts:
            return []

        detection_mode = DetectionMode(mode) if mode else self.detector.mode

        logger.info(f"开始批量检测: {len(texts)} 个文本, mode={detection_mode}")

        # 策略选择
        if detection_mode == DetectionMode.FAST:
            # 快速模式：并行 Regex 检测
            results = self._detect_batch_regex_parallel(
                texts, entity_types, show_progress
            )
        else:
            # 自动/深度模式：Regex 并行 + LLM 批处理
            results = self._detect_batch_hybrid(
                texts, entity_types, detection_mode, show_progress
            )

        elapsed = time.time() - start_time

        logger.info(
            f"批量检测完成: {len(texts)} 个文本, "
            f"耗时 {elapsed:.2f}s, "
            f"平均 {elapsed/len(texts):.2f}s/文本"
        )

        return results

    def _detect_batch_regex_parallel(
        self,
        texts: List[str],
        entity_types: Optional[List[str]],
        show_progress: bool
    ) -> List[List[Entity]]:
        """
        并行 Regex 批量检测

        Regex 检测极快（< 1ms），主要开销是函数调用和线程切换。
        并行化可以提高 CPU 利用率。
        """
        results = [None] * len(texts)  # 保持顺序

        def detect_single(index: int, text: str) -> tuple:
            entities = self.detector.detect(
                text,
                entity_types=entity_types,
                mode="fast"
            )
            return index, entities

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(detect_single, i, text)
                for i, text in enumerate(texts)
            ]

            completed = 0

            for future in as_completed(futures):
                index, entities = future.result()
                results[index] = entities
                completed += 1

                if show_progress and completed % 100 == 0:
                    print(f"  进度: {completed}/{len(texts)}")

        return results

    def _detect_batch_hybrid(
        self,
        texts: List[str],
        entity_types: Optional[List[str]],
        mode: DetectionMode,
        show_progress: bool
    ) -> List[List[Entity]]:
        """
        混合批量检测（Regex 并行 + LLM 顺序）

        策略：
        1. 并行运行 Regex 检测（快）
        2. 顺序运行 LLM 检测（慢，但需要排队）
        3. 合并结果
        """
        results: List[List[Entity]] = [[] for _ in texts]

        # 确定检测类型
        if entity_types is None:
            regex_types = list(self.detector._REGEX_TYPES)
            llm_types = list(self.detector._LLM_TYPES)
        else:
            regex_types = [
                et for et in entity_types
                if et in self.detector._REGEX_TYPES
            ]
            llm_types = [
                et for et in entity_types
                if et in self.detector._LLM_TYPES
            ]

        # 第一阶段：并行 Regex 检测
        if regex_types:
            logger.debug(f"并行 Regex 检测: {len(texts)} 个文本")

            def detect_regex(index: int, text: str) -> tuple:
                entities = self.detector.regex_registry.detect(
                    text, entity_types=regex_types
                )
                return index, entities

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(detect_regex, i, text)
                    for i, text in enumerate(texts)
                ]

                for future in as_completed(futures):
                    index, entities = future.result()
                    results[index].extend(entities)

            logger.debug(f"Regex 检测完成")

        # 第二阶段：顺序 LLM 检测
        if llm_types and self.detector.llm_recognizers:
            logger.debug(f"LLM 批量检测: {len(texts)} 个文本")

            for entity_type in llm_types:
                if entity_type not in self.detector.llm_recognizers:
                    continue

                recognizer = self.detector.llm_recognizers[entity_type]

                if show_progress:
                    print(f"\n  LLM 检测 {entity_type}:")

                # LLM 检测是主要瓶颈，逐个处理
                for i, text in enumerate(texts):
                    try:
                        entities = recognizer.detect(text)
                        results[i].extend(entities)

                        if show_progress and (i + 1) % 10 == 0:
                            print(f"    进度: {i+1}/{len(texts)}")

                    except Exception as e:
                        logger.error(
                            f"LLM 检测失败 (text {i}, type {entity_type}): {e}"
                        )

        return results

    def detect_batch_by_chunks(
        self,
        texts: List[str],
        chunk_size: int = 100,
        **kwargs
    ) -> List[List[Entity]]:
        """
        分块批量检测（处理大规模数据）

        Args:
            texts: 文本列表
            chunk_size: 每块大小
            **kwargs: 传递给 detect_batch 的其他参数

        Returns:
            实体列表的列表
        """
        if len(texts) <= chunk_size:
            return self.detect_batch(texts, **kwargs)

        logger.info(f"分块检测: {len(texts)} 个文本, chunk_size={chunk_size}")

        all_results = []

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i+chunk_size]
            logger.info(f"处理块 {i//chunk_size + 1}: {len(chunk)} 个文本")

            chunk_results = self.detect_batch(chunk, **kwargs)
            all_results.extend(chunk_results)

        return all_results

    def get_batch_statistics(
        self,
        results: List[List[Entity]]
    ) -> Dict[str, Any]:
        """
        获取批量检测统计信息

        Args:
            results: 批量检测结果

        Returns:
            统计信息字典
        """
        total_entities = sum(len(entities) for entities in results)
        total_texts = len(results)

        # 按类型统计
        type_counts: Dict[str, int] = {}
        method_counts: Dict[str, int] = {"regex": 0, "llm": 0}

        for entities in results:
            for entity in entities:
                type_counts[entity.entity_type] = \
                    type_counts.get(entity.entity_type, 0) + 1

                method_counts[entity.detection_method] = \
                    method_counts.get(entity.detection_method, 0) + 1

        # 计算分布
        texts_with_pii = sum(1 for entities in results if entities)

        return {
            "total_texts": total_texts,
            "total_entities": total_entities,
            "texts_with_pii": texts_with_pii,
            "texts_with_pii_rate": texts_with_pii / total_texts if total_texts > 0 else 0,
            "avg_entities_per_text": total_entities / total_texts if total_texts > 0 else 0,
            "entity_types": type_counts,
            "detection_methods": method_counts
        }
