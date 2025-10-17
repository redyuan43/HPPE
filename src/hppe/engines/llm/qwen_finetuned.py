"""
Qwen Fine-tuned 模型引擎实现

用于加载和推理已训练的 PII 检测模型（6种PII类型）
"""

import json
import logging
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path

from hppe.engines.llm.base import BaseLLMEngine
from hppe.models.entity import Entity

logger = logging.getLogger(__name__)


class QwenFineTunedEngine(BaseLLMEngine):
    """
    Qwen Fine-tuned PII 检测引擎

    加载已训练的 LoRA 权重，直接进行 PII 检测，无需外部 vLLM 服务

    支持的 PII 类型（6种）:
    - ADDRESS: 地址
    - ORGANIZATION: 组织机构
    - PERSON_NAME: 人名
    - PHONE_NUMBER: 电话号码
    - EMAIL: 邮箱
    - ID_CARD: 身份证号

    Attributes:
        model: 加载的 Qwen 模型
        tokenizer: 分词器
        device: 计算设备 (cuda/cpu)
        max_seq_length: 最大序列长度

    Examples:
        >>> # 初始化引擎
        >>> engine = QwenFineTunedEngine(
        ...     model_path="models/pii_qwen4b_unsloth/final",
        ...     device="cuda"
        ... )
        >>>
        >>> # 检测 PII
        >>> entities = engine.detect_pii("我是张三，电话13812345678")
        >>> for entity in entities:
        ...     print(f"{entity.entity_type}: {entity.value}")
    """

    def __init__(
        self,
        model_path: str = "models/pii_qwen4b_unsloth/final",
        base_model: str = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B",
        device: str = "cuda",
        max_seq_length: int = 512,
        load_in_4bit: bool = True,
        **kwargs
    ):
        """
        初始化 Fine-tuned 引擎

        Args:
            model_path: LoRA 权重路径
            base_model: 基础 Qwen 模型路径
            device: 计算设备 ('cuda' 或 'cpu')
            max_seq_length: 最大序列长度
            load_in_4bit: 是否使用4-bit量化加载
            **kwargs: 其他配置参数
        """
        super().__init__(
            model_name=f"QwenFineTuned-6PII",
            base_url="local",  # 本地推理
            timeout=30,
            **kwargs
        )

        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = device
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit

        # 延迟加载（需要时才加载模型）
        self.model = None
        self.tokenizer = None
        self._loaded = False

        logger.info(
            f"初始化 QwenFineTunedEngine: "
            f"model_path={model_path}, device={device}"
        )

    def _load_model(self):
        """加载模型和分词器（延迟加载）"""
        if self._loaded:
            return

        try:
            from unsloth import FastLanguageModel

            logger.info("开始加载 fine-tuned 模型...")

            # 检查模型路径
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"模型路径不存在: {self.model_path}"
                )

            # 加载模型
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )

            # 设置为推理模式
            FastLanguageModel.for_inference(self.model)

            # 移动到指定设备
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info(f"模型已加载到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("模型已加载到 CPU")

            self._loaded = True
            logger.info("✅ 模型加载完成")

        except ImportError:
            logger.error("无法导入 unsloth，请安装: pip install unsloth")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        生成文本响应（底层推理方法）

        Args:
            prompt: 用户输入文本
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成token数
            **kwargs: 其他生成参数

        Returns:
            生成的JSON格式响应
        """
        # 确保模型已加载
        if not self._loaded:
            self._load_model()

        # 构建完整提示词
        if system_prompt is None:
            system_prompt = (
                "你是 PII 检测专家。检测以下文本中的 PII，"
                "并以 JSON 格式输出实体列表。"
            )

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        )

        # 移动到设备
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text

    def detect_pii(
        self,
        text: str,
        confidence_threshold: float = 0.8
    ) -> List[Entity]:
        """
        检测文本中的 PII 实体（高级接口）

        Args:
            text: 待检测的文本
            confidence_threshold: 置信度阈值（过滤低置信度实体）

        Returns:
            检测到的 PII 实体列表

        Examples:
            >>> engine = QwenFineTunedEngine()
            >>> entities = engine.detect_pii("我是张三，电话13812345678")
            >>> print(f"检测到 {len(entities)} 个 PII")
        """
        # 生成响应
        response = self.generate(
            prompt=text,
            temperature=0.1,
            max_tokens=256
        )

        # 解析 JSON 响应
        entities = self._parse_response(response, text)

        # 过滤低置信度实体
        filtered_entities = [
            e for e in entities
            if e.confidence >= confidence_threshold
        ]

        logger.debug(
            f"检测到 {len(entities)} 个实体，"
            f"过滤后剩余 {len(filtered_entities)} 个"
        )

        return filtered_entities

    def _parse_response(
        self,
        response: str,
        original_text: str
    ) -> List[Entity]:
        """
        解析模型 JSON 响应为 Entity 对象

        Args:
            response: 模型生成的 JSON 响应
            original_text: 原始输入文本

        Returns:
            Entity 对象列表
        """
        entities = []

        try:
            # 尝试解析 JSON
            data = json.loads(response)

            # 提取 entities 列表
            entity_list = data.get("entities", [])

            for item in entity_list:
                # 提取字段
                entity_type = item.get("type", "UNKNOWN")
                value = item.get("value", "")

                # 查找实体在原文中的位置
                start_pos = item.get("start_pos")
                end_pos = item.get("end_pos")

                # 如果位置未提供，尝试在原文中查找
                if start_pos is None or end_pos is None:
                    start_pos = original_text.find(value)
                    if start_pos != -1:
                        end_pos = start_pos + len(value)
                    else:
                        # 找不到位置，跳过
                        logger.warning(f"无法在原文中找到实体: {value}")
                        continue

                # 创建 Entity 对象
                entity = Entity(
                    entity_type=entity_type,
                    value=value,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=item.get("confidence", 0.85),
                    detection_method="llm_finetuned",
                    recognizer_name="QwenFineTuned-6PII",
                    metadata={
                        "model_path": str(self.model_path),
                        "model_version": "6pii_v1"
                    }
                )

                entities.append(entity)

            logger.debug(f"成功解析 {len(entities)} 个实体")

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n响应内容: {response}")
            # 返回空列表，不抛出异常
            return []

        except Exception as e:
            logger.error(f"解析响应时发生错误: {e}")
            return []

        return entities

    def health_check(self) -> bool:
        """
        健康检查（检查模型是否可用）

        Returns:
            True 表示模型可用，False 表示不可用
        """
        try:
            # 尝试加载模型
            if not self._loaded:
                self._load_model()

            # 运行简单推理测试
            test_text = "测试"
            _ = self.generate(test_text, max_tokens=10)

            logger.info("✅ 健康检查通过")
            return True

        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
            return False

    def get_supported_pii_types(self) -> List[str]:
        """
        获取支持的 PII 类型列表

        Returns:
            PII 类型列表
        """
        return [
            "ADDRESS",
            "ORGANIZATION",
            "PERSON_NAME",
            "PHONE_NUMBER",
            "EMAIL",
            "ID_CARD"
        ]

    def unload_model(self):
        """卸载模型以释放内存"""
        if self._loaded:
            del self.model
            del self.tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._loaded = False
            logger.info("模型已卸载")
