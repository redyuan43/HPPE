"""
LLM 识别器基类

提供基于大语言模型的 PII 识别框架
"""

import logging
import yaml
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from hppe.models.entity import Entity
from hppe.engines.llm.qwen_engine import QwenEngine
from hppe.engines.llm.response_parser import LLMResponseParser

logger = logging.getLogger(__name__)


class BaseLLMRecognizer(ABC):
    """
    LLM 识别器抽象基类

    使用大语言模型进行 PII 检测，提供：
    - Prompt 模板加载
    - LLM 推理
    - 响应解析
    - Entity 创建

    与正则识别器保持相同的接口，便于互换使用

    Attributes:
        entity_type: PII 类型标识符
        llm_engine: LLM 推理引擎
        parser: 响应解析器
        prompt_template: Prompt 模板配置
        recognizer_name: 识别器名称

    Examples:
        >>> recognizer = LLMPersonNameRecognizer(
        ...     llm_engine=engine,
        ...     prompts_file="prompts.yaml"
        ... )
        >>> entities = recognizer.detect("我叫张三")
    """

    def __init__(
        self,
        entity_type: str,
        llm_engine: QwenEngine,
        prompts_file: str,
        prompt_key: str,
        recognizer_name: Optional[str] = None
    ):
        """
        初始化 LLM 识别器

        Args:
            entity_type: PII 类型（如 PERSON_NAME）
            llm_engine: LLM 推理引擎实例
            prompts_file: Prompt 模板文件路径
            prompt_key: Prompt 模板的键名
            recognizer_name: 识别器名称（可选）
        """
        self.entity_type = entity_type
        self.llm_engine = llm_engine
        self.parser = LLMResponseParser(strict=False)
        self.recognizer_name = recognizer_name or self.__class__.__name__

        # 加载 Prompt 模板
        self.prompt_template = self._load_prompt_template(
            prompts_file,
            prompt_key
        )

        logger.info(
            f"初始化 {self.recognizer_name}，"
            f"实体类型: {entity_type}，"
            f"Prompt: {prompt_key}"
        )

    def _load_prompt_template(
        self,
        prompts_file: str,
        prompt_key: str
    ) -> Dict[str, Any]:
        """
        从 YAML 文件加载 Prompt 模板

        Args:
            prompts_file: Prompt 文件路径（相对于项目根目录）
            prompt_key: 要加载的 Prompt 键名

        Returns:
            Prompt 模板配置字典

        Raises:
            FileNotFoundError: Prompt 文件不存在
            KeyError: Prompt 键不存在
        """
        # 支持相对路径和绝对路径
        prompts_path = Path(prompts_file)

        if not prompts_path.is_absolute():
            # 尝试相对于项目根目录（从当前文件往上找）
            # 当前文件: .../src/hppe/engines/llm/recognizers/base.py
            # 项目根: .../
            project_root = Path(__file__).parent.parent.parent.parent.parent
            prompts_path = project_root / prompts_file

        # 如果还是不存在，尝试相对于当前工作目录
        if not prompts_path.exists():
            prompts_path = Path.cwd() / prompts_file

        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompts_path}")

        # 加载 YAML
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)

        if prompt_key not in prompts:
            available_keys = list(prompts.keys())
            raise KeyError(
                f"Prompt 键 '{prompt_key}' 不存在。"
                f"可用键: {available_keys}"
            )

        template = prompts[prompt_key]
        logger.debug(f"加载 Prompt 模板: {prompt_key}")

        return template

    @abstractmethod
    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的 PII 实体

        子类必须实现此方法

        Args:
            text: 待检测的文本

        Returns:
            检测到的 Entity 列表
        """
        pass

    def _detect_with_llm(
        self,
        text: str,
        filter_entity_type: Optional[str] = None
    ) -> List[Entity]:
        """
        使用 LLM 检测 PII

        通用的 LLM 检测流程：
        1. 构建 Prompt
        2. 调用 LLM 生成响应
        3. 解析响应
        4. 创建 Entity 列表

        Args:
            text: 待检测文本
            filter_entity_type: 可选的实体类型过滤器

        Returns:
            Entity 列表
        """
        # 步骤 1: 构建 Prompt
        user_prompt = self.prompt_template['user_prompt_template'].format(
            text=text
        )
        system_prompt = self.prompt_template['system_prompt']
        params = self.prompt_template.get('parameters', {})

        # 步骤 2: 调用 LLM
        try:
            response = self.llm_engine.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=params.get('temperature', 0.05),
                max_tokens=params.get('max_tokens', 512),
                top_p=params.get('top_p', 0.9)
                # 注意：不使用 stop 参数，因为格式化的 JSON 会有多个 }
            )
        except Exception as e:
            logger.error(f"LLM 推理失败: {e}")
            return []

        # 步骤 3: 解析响应
        pii_entities = self.parser.extract_pii_entities(response)

        if not pii_entities:
            logger.debug("未检测到任何 PII 实体")
            return []

        # 步骤 4: 创建 Entity 列表
        entities = []
        for pii_entity in pii_entities:
            # 可选的类型过滤
            if filter_entity_type and pii_entity.get('type') != filter_entity_type:
                continue

            entity = self._create_entity_from_llm_output(
                pii_entity,
                text
            )

            if entity:
                entities.append(entity)

        logger.debug(f"检测到 {len(entities)} 个实体")
        return entities

    def _create_entity_from_llm_output(
        self,
        llm_entity: Dict[str, Any],
        original_text: str
    ) -> Optional[Entity]:
        """
        从 LLM 输出创建 Entity 实例

        Args:
            llm_entity: LLM 返回的实体字典
            original_text: 原始文本（用于验证位置）

        Returns:
            Entity 实例，创建失败时返回 None
        """
        try:
            # 提取必需字段
            value = llm_entity['value']
            entity_type = llm_entity.get('type', self.entity_type)

            # 位置信息（可能不准确，需要验证）
            start_pos = llm_entity.get('start', 0)
            end_pos = llm_entity.get('end', 0)

            # 验证位置信息
            if start_pos < 0 or end_pos > len(original_text):
                logger.warning(
                    f"位置信息超出范围: start={start_pos}, end={end_pos}, "
                    f"text_len={len(original_text)}"
                )
                # 尝试在文本中查找实体值
                start_pos = original_text.find(value)
                if start_pos >= 0:
                    end_pos = start_pos + len(value)
                else:
                    logger.warning(f"无法在文本中找到实体值: {value}")
                    return None

            # 置信度
            confidence = llm_entity.get('confidence', 0.85)

            # 创建 Entity
            entity = Entity(
                entity_type=entity_type,
                value=value,
                start_pos=start_pos,
                end_pos=end_pos,
                confidence=confidence,
                detection_method="llm",
                recognizer_name=self.recognizer_name,
                metadata={
                    "llm_output": llm_entity,
                    "model": self.llm_engine.model_name
                }
            )

            return entity

        except KeyError as e:
            logger.error(f"LLM 输出缺少必需字段: {e}")
            return None
        except Exception as e:
            logger.error(f"创建 Entity 失败: {e}")
            return None

    def validate(self, entity: Entity) -> bool:
        """
        验证实体的有效性

        LLM 识别器默认相信 LLM 的判断，因此默认返回 True
        子类可以覆盖此方法实现自定义验证逻辑

        Args:
            entity: 待验证的实体

        Returns:
            True（默认）
        """
        return True

    def get_info(self) -> Dict[str, Any]:
        """
        获取识别器信息

        Returns:
            包含识别器元信息的字典
        """
        return {
            "name": self.recognizer_name,
            "entity_type": self.entity_type,
            "detection_method": "llm",
            "model": self.llm_engine.model_name,
            "prompt_name": self.prompt_template.get('name', 'Unknown')
        }
