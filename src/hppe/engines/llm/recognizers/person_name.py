"""
基于 LLM 的姓名识别器
"""

import logging
from typing import List, Optional

from hppe.models.entity import Entity
from hppe.engines.llm.qwen_engine import QwenEngine
from hppe.engines.llm.recognizers.base import BaseLLMRecognizer

logger = logging.getLogger(__name__)


class LLMPersonNameRecognizer(BaseLLMRecognizer):
    """
    LLM 姓名识别器

    使用大语言模型识别中英文人名，包括：
    - 完整姓名
    - 姓氏（如：张先生、李总）
    - 昵称（如：老张、小李）

    相比正则表达式，LLM 能更好地理解上下文，避免将地名误判为人名。

    Examples:
        >>> engine = QwenEngine()
        >>> recognizer = LLMPersonNameRecognizer(llm_engine=engine)
        >>> entities = recognizer.detect("张经理让李助理通知王总明天开会")
        >>> for entity in entities:
        ...     print(f"{entity.value} (置信度: {entity.confidence})")
        张经理 (置信度: 0.9)
        李助理 (置信度: 0.9)
        王总 (置信度: 0.9)
    """

    def __init__(
        self,
        llm_engine: QwenEngine,
        prompts_file: str = "data/prompts/pii_detection_prompts.yaml",
        recognizer_name: Optional[str] = None
    ):
        """
        初始化姓名识别器

        Args:
            llm_engine: LLM 推理引擎
            prompts_file: Prompt 模板文件路径
            recognizer_name: 识别器名称（可选）
        """
        super().__init__(
            entity_type="PERSON_NAME",
            llm_engine=llm_engine,
            prompts_file=prompts_file,
            prompt_key="person_name_detection",
            recognizer_name=recognizer_name or "LLMPersonNameRecognizer"
        )

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的人名

        Args:
            text: 待检测的文本

        Returns:
            检测到的人名实体列表

        Examples:
            >>> recognizer = LLMPersonNameRecognizer(engine)
            >>> entities = recognizer.detect("我叫张三，我的同事是李四")
            >>> len(entities)
            2
        """
        return self._detect_with_llm(
            text,
            filter_entity_type="PERSON_NAME"
        )
