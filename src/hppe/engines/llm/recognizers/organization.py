"""
基于 LLM 的组织机构识别器
"""

import logging
from typing import List, Optional

from hppe.models.entity import Entity
from hppe.engines.llm.qwen_engine import QwenEngine
from hppe.engines.llm.recognizers.base import BaseLLMRecognizer

logger = logging.getLogger(__name__)


class LLMOrganizationRecognizer(BaseLLMRecognizer):
    """
    LLM 组织机构识别器

    使用大语言模型识别组织机构名称，包括：
    - 公司/企业
    - 政府部门
    - 学校
    - 医院
    - 其他组织（协会、基金会等）

    相比正则表达式，LLM 能更好地：
    - 区分组织名称和地名
    - 处理简称和全称
    - 识别各种组织类型

    Examples:
        >>> engine = QwenEngine()
        >>> recognizer = LLMOrganizationRecognizer(llm_engine=engine)
        >>> text = "我在北京科技有限公司工作，之前在清华大学读书"
        >>> entities = recognizer.detect(text)
        >>> for entity in entities:
        ...     print(entity.value)
        北京科技有限公司
        清华大学
    """

    def __init__(
        self,
        llm_engine: QwenEngine,
        prompts_file: str = "data/prompts/pii_detection_prompts.yaml",
        recognizer_name: Optional[str] = None
    ):
        """
        初始化组织机构识别器

        Args:
            llm_engine: LLM 推理引擎
            prompts_file: Prompt 模板文件路径
            recognizer_name: 识别器名称（可选）
        """
        super().__init__(
            entity_type="ORGANIZATION",
            llm_engine=llm_engine,
            prompts_file=prompts_file,
            prompt_key="organization_detection",
            recognizer_name=recognizer_name or "LLMOrganizationRecognizer"
        )

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的组织机构

        Args:
            text: 待检测的文本

        Returns:
            检测到的组织机构实体列表

        Examples:
            >>> recognizer = LLMOrganizationRecognizer(engine)
            >>> text = "我在阿里巴巴集团工作"
            >>> entities = recognizer.detect(text)
            >>> print(entities[0].value)
            阿里巴巴集团
        """
        return self._detect_with_llm(
            text,
            filter_entity_type="ORGANIZATION"
        )
