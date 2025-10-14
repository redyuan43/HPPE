"""
Qwen 模型引擎实现

基于 vLLM 部署的 Qwen 模型的推理引擎
"""

import logging
from typing import Optional
import requests

from hppe.engines.llm.base import BaseLLMEngine

logger = logging.getLogger(__name__)


class QwenEngine(BaseLLMEngine):
    """
    Qwen 模型引擎

    使用 OpenAI 兼容的 API 接口与 vLLM 服务通信

    Attributes:
        model_name: Qwen 模型名称
        base_url: vLLM 服务地址
        timeout: 请求超时时间
        api_key: API 密钥（可选，本地部署通常不需要）

    Examples:
        >>> # 初始化引擎
        >>> engine = QwenEngine(
        ...     model_name="Qwen/Qwen2.5-7B-Instruct",
        ...     base_url="http://localhost:8000/v1"
        ... )
        >>>
        >>> # 检查健康状态
        >>> if engine.health_check():
        ...     # 生成响应
        ...     response = engine.generate(
        ...         prompt="检测以下文本中的 PII: 我是张三",
        ...         temperature=0.1
        ...     )
        ...     print(response)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        timeout: int = 30,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 Qwen 引擎

        Args:
            model_name: Qwen 模型名称
            base_url: vLLM 服务地址（OpenAI 兼容）
            timeout: 请求超时时间（秒）
            api_key: API 密钥（可选）
            **kwargs: 其他配置参数
        """
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout=timeout,
            **kwargs
        )
        self.api_key = api_key or "EMPTY"  # vLLM 本地部署可以使用任意 key

        # 构建完整的 API URL
        self.chat_url = f"{self.base_url}/chat/completions"
        self.health_url = f"{self.base_url.replace('/v1', '')}/health"

        logger.info(
            f"初始化 QwenEngine: model={model_name}, base_url={base_url}"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        生成文本响应

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            temperature: 温度参数 (0.0-2.0)，越低越确定性
            max_tokens: 最大生成 token 数
            top_p: 核采样参数
            **kwargs: 其他生成参数

        Returns:
            生成的文本响应

        Raises:
            requests.RequestException: 请求失败
            ValueError: 响应格式错误

        Examples:
            >>> engine = QwenEngine()
            >>> response = engine.generate(
            ...     prompt="你好",
            ...     system_prompt="你是一个友好的助手",
            ...     temperature=0.7
            ... )
        """
        # 构建消息
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # 构建请求体
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        }

        # 构建请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            # 发送请求
            logger.debug(f"发送请求到 {self.chat_url}")
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            # 检查状态码
            response.raise_for_status()

            # 解析响应
            data = response.json()

            # 提取生成的文本
            if "choices" not in data or not data["choices"]:
                raise ValueError("响应中缺少 choices 字段")

            content = data["choices"][0]["message"]["content"]

            logger.debug(f"成功生成响应，长度: {len(content)}")

            return content

        except requests.Timeout:
            logger.error(f"请求超时（{self.timeout}秒）")
            raise

        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            raise

        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"解析响应失败: {e}")
            raise ValueError(f"无效的响应格式: {e}")

    def health_check(self) -> bool:
        """
        健康检查

        检查 vLLM 服务是否可用

        Returns:
            True 表示服务正常，False 表示服务异常

        Examples:
            >>> engine = QwenEngine()
            >>> if not engine.health_check():
            ...     print("vLLM 服务不可用")
        """
        try:
            # 尝试访问健康检查端点
            response = requests.get(
                self.health_url,
                timeout=5
            )

            # 检查状态码
            if response.status_code == 200:
                logger.info("vLLM 服务健康检查通过")
                return True
            else:
                logger.warning(
                    f"vLLM 服务健康检查失败: "
                    f"status_code={response.status_code}"
                )
                return False

        except requests.RequestException as e:
            logger.error(f"vLLM 服务健康检查异常: {e}")
            return False

    def get_model_info(self) -> dict:
        """
        获取模型信息

        Returns:
            包含模型信息的字典

        Examples:
            >>> engine = QwenEngine()
            >>> info = engine.get_model_info()
            >>> print(info["model"])
        """
        try:
            # 尝试获取模型列表
            models_url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(
                models_url,
                headers=headers,
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                return data
            else:
                logger.warning("无法获取模型信息")
                return {}

        except requests.RequestException as e:
            logger.error(f"获取模型信息失败: {e}")
            return {}
