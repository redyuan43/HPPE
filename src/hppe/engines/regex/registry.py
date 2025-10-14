"""
识别器注册表

管理和协调所有 PII 识别器的注册、发现和执行
"""

import importlib
import inspect
import os
import pkgutil
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Set, Type

from hppe.engines.regex.base import BaseRecognizer
from hppe.models.entity import Entity


class RecognizerRegistry:
    """
    识别器注册表

    线程安全的识别器管理系统，提供：
    - 识别器注册和注销
    - 自动发现和加载配置文件
    - 批量 PII 检测
    - 按类型过滤识别器
    - 性能监控

    Attributes:
        _recognizers: 识别器字典，键为 entity_type
        _lock: 线程锁，保证并发安全
        _performance_stats: 性能统计数据

    Examples:
        >>> registry = RecognizerRegistry()
        >>> registry.register(my_recognizer)
        >>> entities = registry.detect("待检测的文本")
    """

    def __init__(self) -> None:
        """初始化注册表"""
        self._recognizers: Dict[str, BaseRecognizer] = {}
        self._lock = threading.RLock()
        self._performance_stats: Dict[str, Dict[str, float]] = {}

    def register(self, recognizer: BaseRecognizer) -> None:
        """
        注册识别器

        Args:
            recognizer: BaseRecognizer 实例

        Raises:
            ValueError: 当识别器类型已存在时（避免覆盖）
        """
        with self._lock:
            entity_type = recognizer.entity_type

            if entity_type in self._recognizers:
                raise ValueError(
                    f"识别器类型 '{entity_type}' 已注册，"
                    f"请先注销或使用不同的 entity_type"
                )

            self._recognizers[entity_type] = recognizer

            # 初始化性能统计
            self._performance_stats[entity_type] = {
                "total_calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
            }

    def unregister(self, entity_type: str) -> bool:
        """
        注销识别器

        Args:
            entity_type: 要注销的识别器类型

        Returns:
            True 表示成功注销，False 表示类型不存在
        """
        with self._lock:
            if entity_type in self._recognizers:
                del self._recognizers[entity_type]
                del self._performance_stats[entity_type]
                return True
            return False

    def get_recognizer(self, entity_type: str) -> Optional[BaseRecognizer]:
        """
        获取指定类型的识别器

        Args:
            entity_type: PII 类型

        Returns:
            识别器实例，如果不存在则返回 None
        """
        with self._lock:
            return self._recognizers.get(entity_type)

    def get_all_recognizers(self) -> List[BaseRecognizer]:
        """
        获取所有注册的识别器

        Returns:
            识别器实例列表
        """
        with self._lock:
            return list(self._recognizers.values())

    def get_entity_types(self) -> Set[str]:
        """
        获取所有已注册的 PII 类型

        Returns:
            PII 类型集合
        """
        with self._lock:
            return set(self._recognizers.keys())

    def detect(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[Entity]:
        """
        使用识别器检测文本中的 PII

        Args:
            text: 待检测的文本
            entity_types: 可选的 PII 类型过滤列表，
                        如果为 None 则使用所有识别器

        Returns:
            检测到的实体列表

        Examples:
            >>> # 使用所有识别器
            >>> entities = registry.detect("包含 PII 的文本")
            >>>
            >>> # 只使用特定识别器
            >>> entities = registry.detect(
            ...     "文本",
            ...     entity_types=["CHINA_ID_CARD", "EMAIL_ADDRESS"]
            ... )
        """
        entities: List[Entity] = []

        # 确定要使用的识别器
        with self._lock:
            if entity_types is None:
                recognizers = list(self._recognizers.values())
            else:
                recognizers = [
                    self._recognizers[et]
                    for et in entity_types
                    if et in self._recognizers
                ]

        # 使用每个识别器检测
        for recognizer in recognizers:
            start_time = time.time()

            try:
                detected = recognizer.detect(text)
                entities.extend(detected)
            except Exception as e:
                # 记录错误但继续处理其他识别器
                # TODO: 添加日志系统
                print(
                    f"识别器 {recognizer.recognizer_name} "
                    f"检测失败: {e}"
                )

            # 更新性能统计
            elapsed = time.time() - start_time
            self._update_performance_stats(
                recognizer.entity_type,
                elapsed
            )

        return entities

    def detect_with_filter(
        self,
        text: str,
        min_confidence: float = 0.0,
        entity_types: Optional[List[str]] = None
    ) -> List[Entity]:
        """
        检测并过滤实体

        Args:
            text: 待检测的文本
            min_confidence: 最低置信度阈值
            entity_types: 可选的类型过滤

        Returns:
            过滤后的实体列表
        """
        entities = self.detect(text, entity_types)

        # 按置信度过滤
        filtered = [
            e for e in entities
            if e.confidence >= min_confidence
        ]

        return filtered

    def _update_performance_stats(
        self,
        entity_type: str,
        elapsed_time: float
    ) -> None:
        """
        更新性能统计数据

        Args:
            entity_type: 识别器类型
            elapsed_time: 执行时间（秒）
        """
        with self._lock:
            if entity_type in self._performance_stats:
                stats = self._performance_stats[entity_type]
                stats["total_calls"] += 1
                stats["total_time"] += elapsed_time
                stats["avg_time"] = (
                    stats["total_time"] / stats["total_calls"]
                )

    def get_performance_stats(
        self,
        entity_type: Optional[str] = None
    ) -> Dict:
        """
        获取性能统计数据

        Args:
            entity_type: 可选的识别器类型，
                       如果为 None 则返回所有统计

        Returns:
            性能统计字典
        """
        with self._lock:
            if entity_type:
                return self._performance_stats.get(
                    entity_type,
                    {}
                ).copy()
            return {
                k: v.copy()
                for k, v in self._performance_stats.items()
            }

    def reset_performance_stats(self) -> None:
        """重置所有性能统计数据"""
        with self._lock:
            for entity_type in self._performance_stats:
                self._performance_stats[entity_type] = {
                    "total_calls": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                }

    def clear(self) -> None:
        """清空注册表（移除所有识别器）"""
        with self._lock:
            self._recognizers.clear()
            self._performance_stats.clear()

    def __len__(self) -> int:
        """返回已注册识别器的数量"""
        with self._lock:
            return len(self._recognizers)

    def __contains__(self, entity_type: str) -> bool:
        """检查是否包含指定类型的识别器"""
        with self._lock:
            return entity_type in self._recognizers

    def __repr__(self) -> str:
        """返回注册表的字符串表示"""
        with self._lock:
            entity_types = list(self._recognizers.keys())
            return (
                f"RecognizerRegistry("
                f"count={len(entity_types)}, "
                f"types={entity_types})"
            )

    def load_all(self, config: Optional[Dict] = None) -> int:
        """
        自动发现并加载所有识别器

        扫描 recognizers 包下的所有模块，自动发现并注册
        所有 BaseRecognizer 的子类实例

        Args:
            config: 可选的配置字典，包含各识别器的初始化配置
                  格式: {entity_type: config_dict}

        Returns:
            成功加载的识别器数量

        Examples:
            >>> registry = RecognizerRegistry()
            >>> # 使用默认配置加载所有识别器
            >>> count = registry.load_all()
            >>> print(f"加载了 {count} 个识别器")
            >>>
            >>> # 使用自定义配置加载
            >>> config = {
            ...     "EMAIL": {"confidence_base": 0.9},
            ...     "CHINA_ID_CARD": {"confidence_base": 0.85}
            ... }
            >>> count = registry.load_all(config)
        """
        config = config or {}
        loaded_count = 0

        # 导入所有识别器类
        try:
            # 导入中国 PII 识别器
            from hppe.engines.regex.recognizers.china_pii import (
                ChinaIDCardRecognizer,
                ChinaPhoneRecognizer,
                ChinaBankCardRecognizer,
                ChinaPassportRecognizer,
            )

            # 导入全球 PII 识别器
            from hppe.engines.regex.recognizers.global_pii import (
                EmailRecognizer,
                IPAddressRecognizer,
                URLRecognizer,
                CreditCardRecognizer,
                SSNRecognizer,
            )

            # 所有识别器类
            recognizer_classes = [
                # 中国 PII
                ChinaIDCardRecognizer,
                ChinaPhoneRecognizer,
                ChinaBankCardRecognizer,
                ChinaPassportRecognizer,
                # 全球 PII
                EmailRecognizer,
                IPAddressRecognizer,
                URLRecognizer,
                CreditCardRecognizer,
                SSNRecognizer,
            ]

            # 为每个类创建默认配置并实例化
            for recognizer_class in recognizer_classes:
                try:
                    # 获取识别器的默认配置
                    default_config = self._get_default_config(recognizer_class)

                    # 如果用户提供了自定义配置，合并它
                    entity_type = default_config.get("entity_type", "")
                    if entity_type in config:
                        default_config.update(config[entity_type])

                    # 实例化识别器
                    recognizer = recognizer_class(default_config)

                    # 尝试注册（如果已存在会跳过）
                    try:
                        self.register(recognizer)
                        loaded_count += 1
                    except ValueError:
                        # 已存在，跳过
                        pass

                except Exception as e:
                    # 记录错误但继续处理其他识别器
                    print(f"加载识别器 {recognizer_class.__name__} 失败: {e}")

        except ImportError as e:
            print(f"导入识别器模块失败: {e}")

        return loaded_count

    def _get_default_config(self, recognizer_class: Type[BaseRecognizer]) -> Dict:
        """
        获取识别器类的默认配置

        Args:
            recognizer_class: 识别器类

        Returns:
            默认配置字典
        """
        # 根据识别器类名生成默认配置
        class_name = recognizer_class.__name__

        # 中国 PII 识别器配置
        if class_name == "ChinaIDCardRecognizer":
            return {
                "entity_type": "CHINA_ID_CARD",
                "patterns": [{
                    "pattern": r'[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]'
                }],
                "confidence_base": 0.85
            }
        elif class_name == "ChinaPhoneRecognizer":
            return {
                "entity_type": "CHINA_PHONE",
                "patterns": [{"pattern": r'1[3-9]\d{9}'}],
                "confidence_base": 0.80
            }
        elif class_name == "ChinaBankCardRecognizer":
            return {
                "entity_type": "CHINA_BANK_CARD",
                "patterns": [{"pattern": r'\b[1-9]\d{15,18}\b'}],
                "confidence_base": 0.85
            }
        elif class_name == "ChinaPassportRecognizer":
            return {
                "entity_type": "CHINA_PASSPORT",
                "patterns": [
                    {"pattern": r'E\d{8}'},
                    {"pattern": r'[PG]\d{7}'},
                ],
                "confidence_base": 0.80
            }
        # 全球 PII 识别器配置
        elif class_name == "EmailRecognizer":
            return {
                "entity_type": "EMAIL",
                "patterns": [{"pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'}],
                "confidence_base": 0.80
            }
        elif class_name == "IPAddressRecognizer":
            return {
                "entity_type": "IP_ADDRESS",
                "patterns": [{"pattern": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'}],
                "confidence_base": 0.85
            }
        elif class_name == "URLRecognizer":
            return {
                "entity_type": "URL",
                "patterns": [
                    {"pattern": r'https?://[^\s]+'},
                    {"pattern": r'ftp://[^\s]+'},
                ],
                "confidence_base": 0.80
            }
        elif class_name == "CreditCardRecognizer":
            return {
                "entity_type": "CREDIT_CARD",
                "patterns": [
                    {"pattern": r'\b[0-9]{13,19}\b'},
                    {"pattern": r'\b[0-9]{4}[\s\-][0-9]{4}[\s\-][0-9]{4}[\s\-][0-9]{4,7}\b'},
                ],
                "confidence_base": 0.85
            }
        elif class_name == "SSNRecognizer":
            return {
                "entity_type": "US_SSN",
                "patterns": [{"pattern": r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b'}],
                "confidence_base": 0.85
            }
        else:
            # 未知识别器，返回空配置
            return {
                "entity_type": "UNKNOWN",
                "patterns": [],
                "confidence_base": 0.5
            }

    def get_metadata(self, entity_type: Optional[str] = None) -> Dict:
        """
        获取识别器的详细元数据

        Args:
            entity_type: 可选的识别器类型，
                       如果为 None 则返回所有识别器的元数据

        Returns:
            元数据字典，包含识别器的详细信息

        Examples:
            >>> # 获取单个识别器的元数据
            >>> metadata = registry.get_metadata("EMAIL")
            >>> print(metadata["recognizer_name"])
            >>> print(metadata["supported_patterns"])
            >>>
            >>> # 获取所有识别器的元数据
            >>> all_metadata = registry.get_metadata()
            >>> for et, meta in all_metadata.items():
            ...     print(f"{et}: {meta['recognizer_name']}")
        """
        with self._lock:
            if entity_type:
                recognizer = self._recognizers.get(entity_type)
                if not recognizer:
                    return {}

                return {
                    "entity_type": recognizer.entity_type,
                    "recognizer_name": recognizer.recognizer_name,
                    "confidence_base": recognizer.confidence_base,
                    "pattern_count": len(recognizer.patterns),
                    "supported_patterns": [p.pattern for p in recognizer.patterns],
                    "description": recognizer.__class__.__doc__ or "",
                    "performance": self._performance_stats.get(entity_type, {})
                }
            else:
                # 返回所有识别器的元数据
                return {
                    et: self.get_metadata(et)
                    for et in self._recognizers.keys()
                }

    def get_summary(self) -> Dict:
        """
        获取注册表的摘要信息

        Returns:
            包含注册表统计信息的字典

        Examples:
            >>> summary = registry.get_summary()
            >>> print(f"识别器数量: {summary['total_recognizers']}")
            >>> print(f"总检测次数: {summary['total_detections']}")
        """
        with self._lock:
            total_calls = sum(
                stats["total_calls"]
                for stats in self._performance_stats.values()
            )
            total_time = sum(
                stats["total_time"]
                for stats in self._performance_stats.values()
            )

            return {
                "total_recognizers": len(self._recognizers),
                "entity_types": list(self._recognizers.keys()),
                "total_detections": total_calls,
                "total_time": total_time,
                "avg_time_per_detection": (
                    total_time / total_calls if total_calls > 0 else 0.0
                ),
                "recognizers": {
                    et: {
                        "name": rec.recognizer_name,
                        "calls": self._performance_stats[et]["total_calls"],
                        "avg_time": self._performance_stats[et]["avg_time"]
                    }
                    for et, rec in self._recognizers.items()
                }
            }
