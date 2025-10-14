"""
混合 PII 检测器

结合 Regex 和 LLM 的优势，提供智能检测策略：
- Regex：快速检测结构化 PII（电话、身份证、邮箱等）
- LLM：深度检测非结构化 PII（姓名、地址、组织等）
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

from hppe.models.entity import Entity
from hppe.engines.regex import RecognizerRegistry
from hppe.engines.llm import QwenEngine
from hppe.engines.llm.recognizers import (
    LLMPersonNameRecognizer,
    LLMAddressRecognizer,
    LLMOrganizationRecognizer
)

logger = logging.getLogger(__name__)


class DetectionMode(str, Enum):
    """检测模式"""
    FAST = "fast"      # 仅使用 Regex（最快）
    AUTO = "auto"      # 自动选择（推荐）
    DEEP = "deep"      # 使用 Regex + LLM（最全面）


class HybridPIIDetector:
    """
    混合 PII 检测器

    根据 PII 类型自动选择最优检测方法：
    - 结构化 PII（电话、身份证等）→ Regex
    - 非结构化 PII（姓名、地址等）→ LLM

    Attributes:
        regex_registry: Regex 识别器注册表
        llm_engine: LLM 推理引擎
        llm_recognizers: LLM 识别器字典
        mode: 检测模式

    Examples:
        >>> # 创建混合检测器
        >>> detector = HybridPIIDetector(
        ...     llm_engine=QwenEngine(),
        ...     mode="auto"
        ... )
        >>>
        >>> # 检测文本
        >>> entities = detector.detect("张三的电话是13800138000")
        >>> for entity in entities:
        ...     print(f"{entity.entity_type}: {entity.value}")
        PERSON_NAME: 张三
        PHONE_NUMBER: 13800138000
    """

    # PII 类型到检测方法的映射
    _REGEX_TYPES = {
        "PHONE_NUMBER",
        "ID_CARD",
        "EMAIL",
        "BANK_CARD",
        "IP_ADDRESS"
    }

    _LLM_TYPES = {
        "PERSON_NAME",
        "ADDRESS",
        "ORGANIZATION"
    }

    def __init__(
        self,
        llm_engine: Optional[QwenEngine] = None,
        regex_registry: Optional[RecognizerRegistry] = None,
        mode: Literal["fast", "auto", "deep"] = "auto",
        enable_llm: bool = True
    ):
        """
        初始化混合检测器

        Args:
            llm_engine: LLM 引擎（可选，如果不提供则不支持 LLM 检测）
            regex_registry: Regex 注册表（可选，自动创建并注册识别器）
            mode: 检测模式（fast/auto/deep）
            enable_llm: 是否启用 LLM 检测
        """
        self.mode = DetectionMode(mode)
        self.enable_llm = enable_llm

        # 初始化 Regex 引擎
        self.regex_registry = regex_registry or RecognizerRegistry()
        self._init_regex_recognizers()

        # 初始化 LLM 引擎（如果启用）
        self.llm_engine = llm_engine
        self.llm_recognizers: Dict[str, Any] = {}

        if self.enable_llm and llm_engine is not None:
            self._init_llm_recognizers()
        elif self.mode == DetectionMode.DEEP:
            logger.warning("深度模式需要 LLM 引擎，但未提供，将使用自动模式")
            self.mode = DetectionMode.AUTO

        logger.info(
            f"混合检测器初始化完成: mode={self.mode}, "
            f"llm_enabled={self.enable_llm and llm_engine is not None}"
        )

    def _init_regex_recognizers(self):
        """初始化 Regex 识别器"""
        from hppe.engines.regex.recognizers.china_pii import (
            ChinaIDCardRecognizer,
            ChinaPhoneRecognizer,
            ChinaBankCardRecognizer
        )
        from hppe.engines.regex.recognizers.global_pii import (
            EmailRecognizer,
            IPAddressRecognizer
        )

        # 创建识别器配置并注册
        recognizers = [
            (ChinaPhoneRecognizer, {
                "entity_type": "PHONE_NUMBER",
                "recognizer_name": "ChinaPhoneRecognizer",
                "patterns": [{"pattern": r"1[3-9]\d{9}", "name": "china_mobile"}]
            }),
            (ChinaIDCardRecognizer, {
                "entity_type": "ID_CARD",
                "recognizer_name": "ChinaIDCardRecognizer",
                "patterns": [{"pattern": r"[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]", "name": "china_id_card"}]
            }),
            (ChinaBankCardRecognizer, {
                "entity_type": "BANK_CARD",
                "recognizer_name": "ChinaBankCardRecognizer",
                "patterns": [{"pattern": r"\d{16,19}", "name": "bank_card"}]
            }),
            (EmailRecognizer, {
                "entity_type": "EMAIL",
                "recognizer_name": "EmailRecognizer",
                "patterns": [{"pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "name": "email"}]
            }),
            (IPAddressRecognizer, {
                "entity_type": "IP_ADDRESS",
                "recognizer_name": "IPAddressRecognizer",
                "patterns": [{"pattern": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "name": "ipv4"}]
            })
        ]

        for recognizer_class, config in recognizers:
            try:
                recognizer = recognizer_class(config)
                self.regex_registry.register(recognizer)
                logger.debug(f"注册 Regex 识别器: {config['entity_type']}")
            except ValueError as e:
                # 识别器已存在
                logger.debug(f"识别器已存在: {config['entity_type']}")
            except Exception as e:
                logger.error(f"注册识别器失败 {config['entity_type']}: {e}")

    def _init_llm_recognizers(self):
        """初始化 LLM 识别器"""
        if self.llm_engine is None:
            return

        self.llm_recognizers = {
            "PERSON_NAME": LLMPersonNameRecognizer(llm_engine=self.llm_engine),
            "ADDRESS": LLMAddressRecognizer(llm_engine=self.llm_engine),
            "ORGANIZATION": LLMOrganizationRecognizer(llm_engine=self.llm_engine)
        }

        logger.debug("LLM 识别器初始化完成")

    def detect(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        mode: Optional[Literal["fast", "auto", "deep"]] = None
    ) -> List[Entity]:
        """
        检测文本中的 PII

        Args:
            text: 待检测的文本
            entity_types: 指定要检测的 PII 类型，None 表示检测所有类型
            mode: 检测模式，覆盖默认模式

        Returns:
            检测到的实体列表

        Examples:
            >>> # 检测所有 PII
            >>> entities = detector.detect("张三的电话是13800138000")
            >>>
            >>> # 只检测特定类型
            >>> entities = detector.detect(
            ...     "张三的电话是13800138000",
            ...     entity_types=["PHONE_NUMBER"]
            ... )
            >>>
            >>> # 使用快速模式（仅 Regex）
            >>> entities = detector.detect(
            ...     "电话：13800138000",
            ...     mode="fast"
            ... )
        """
        detection_mode = DetectionMode(mode) if mode else self.mode
        entities: List[Entity] = []

        # 确定要检测的类型
        if entity_types is None:
            entity_types = list(self._REGEX_TYPES | self._LLM_TYPES)

        # 第一阶段：Regex 检测（所有模式都执行）
        regex_types = [et for et in entity_types if et in self._REGEX_TYPES]

        if regex_types:
            logger.debug(f"Regex 检测类型: {regex_types}")
            regex_entities = self.regex_registry.detect(
                text,
                entity_types=regex_types
            )
            entities.extend(regex_entities)
            logger.debug(f"Regex 检测到 {len(regex_entities)} 个实体")

        # 第二阶段：LLM 检测（仅 auto 和 deep 模式）
        if detection_mode != DetectionMode.FAST:
            llm_types = [et for et in entity_types if et in self._LLM_TYPES]

            if llm_types and self.llm_recognizers:
                logger.debug(f"LLM 检测类型: {llm_types}")

                for entity_type in llm_types:
                    if entity_type in self.llm_recognizers:
                        try:
                            llm_entities = self.llm_recognizers[entity_type].detect(text)
                            entities.extend(llm_entities)
                            logger.debug(
                                f"LLM 检测 {entity_type} 得到 {len(llm_entities)} 个实体"
                            )
                        except Exception as e:
                            logger.error(f"LLM 检测失败 {entity_type}: {e}")

        logger.info(f"检测完成，共 {len(entities)} 个实体")
        return entities

    def detect_with_confidence(
        self,
        text: str,
        min_confidence: float = 0.0,
        entity_types: Optional[List[str]] = None,
        mode: Optional[Literal["fast", "auto", "deep"]] = None
    ) -> List[Entity]:
        """
        检测并过滤低置信度实体

        Args:
            text: 待检测的文本
            min_confidence: 最小置信度阈值
            entity_types: 指定要检测的 PII 类型
            mode: 检测模式

        Returns:
            过滤后的实体列表
        """
        entities = self.detect(text, entity_types, mode)

        # 过滤低置信度实体
        filtered_entities = [
            e for e in entities
            if e.confidence >= min_confidence
        ]

        logger.debug(
            f"置信度过滤: {len(entities)} → {len(filtered_entities)} "
            f"(threshold={min_confidence})"
        )

        return filtered_entities

    def get_supported_types(self) -> Dict[str, List[str]]:
        """
        获取支持的 PII 类型

        Returns:
            字典，键为检测方法（regex/llm），值为支持的类型列表
        """
        return {
            "regex": list(self._REGEX_TYPES),
            "llm": list(self._LLM_TYPES) if self.llm_recognizers else [],
            "all": list(self._REGEX_TYPES | (self._LLM_TYPES if self.llm_recognizers else set()))
        }

    def get_detector_info(self) -> Dict[str, Any]:
        """
        获取检测器信息

        Returns:
            包含检测器配置和状态的字典
        """
        return {
            "mode": self.mode.value,
            "llm_enabled": bool(self.llm_recognizers),
            "regex_types": list(self._REGEX_TYPES),
            "llm_types": list(self._LLM_TYPES) if self.llm_recognizers else [],
            "total_types": len(self._REGEX_TYPES) + (len(self._LLM_TYPES) if self.llm_recognizers else 0)
        }
