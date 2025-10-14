"""
基于 LLM 的地址识别器
"""

import logging
from typing import List, Optional

from hppe.models.entity import Entity
from hppe.engines.llm.qwen_engine import QwenEngine
from hppe.engines.llm.recognizers.base import BaseLLMRecognizer

logger = logging.getLogger(__name__)


class LLMAddressRecognizer(BaseLLMRecognizer):
    """
    LLM 地址识别器

    使用大语言模型识别地址信息，包括：
    - 完整地址（省市区街道门牌号）
    - 部分地址（只有城市或街道等）

    相比正则表达式，LLM 能更好地：
    - 识别地址边界（避免将"北京大学"误判为地址）
    - 处理各种地址格式变体
    - 理解自然语言描述的地址

    Examples:
        >>> engine = QwenEngine()
        >>> recognizer = LLMAddressRecognizer(llm_engine=engine)
        >>> entities = recognizer.detect("我住在北京市海淀区中关村大街1号")
        >>> print(entities[0].value)
        北京市海淀区中关村大街1号
    """

    def __init__(
        self,
        llm_engine: QwenEngine,
        prompts_file: str = "data/prompts/pii_detection_prompts.yaml",
        recognizer_name: Optional[str] = None
    ):
        """
        初始化地址识别器

        Args:
            llm_engine: LLM 推理引擎
            prompts_file: Prompt 模板文件路径
            recognizer_name: 识别器名称（可选）
        """
        super().__init__(
            entity_type="ADDRESS",
            llm_engine=llm_engine,
            prompts_file=prompts_file,
            prompt_key="address_detection",
            recognizer_name=recognizer_name or "LLMAddressRecognizer"
        )

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的地址

        Args:
            text: 待检测的文本

        Returns:
            检测到的地址实体列表

        Examples:
            >>> recognizer = LLMAddressRecognizer(engine)
            >>> entities = recognizer.detect("公司地址是上海市浦东新区陆家嘴")
            >>> len(entities)
            1
        """
        return self._detect_with_llm(
            text,
            filter_entity_type="ADDRESS"
        )
