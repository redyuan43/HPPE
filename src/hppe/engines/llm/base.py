"""
LLM 引擎抽象基类

定义所有 LLM 引擎必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLMEngine(ABC):
    """
    LLM 引擎抽象基类

    所有 LLM 引擎实现都必须继承此类并实现其抽象方法

    Attributes:
        model_name: 模型名称
        base_url: API 基础 URL
        timeout: 请求超时时间（秒）

    Examples:
        >>> class MyEngine(BaseLLMEngine):
        ...     def generate(self, prompt, **kwargs):
        ...         # 实现生成逻辑
        ...         pass
        ...
        ...     def health_check(self):
        ...         # 实现健康检查
        ...         return True
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化 LLM 引擎

        Args:
            model_name: 模型名称
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            **kwargs: 其他配置参数
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        生成文本响应

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            temperature: 温度参数，控制随机性 (0.0-2.0)
            max_tokens: 最大生成 token 数
            **kwargs: 其他生成参数

        Returns:
            生成的文本响应

        Raises:
            Exception: 生成失败时抛出异常

        Examples:
            >>> engine = MyEngine()
            >>> response = engine.generate(
            ...     prompt="检测以下文本中的 PII: 我是张三",
            ...     system_prompt="你是一个专业的隐私信息检测助手。",
            ...     temperature=0.1
            ... )
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        健康检查

        检查 LLM 服务是否可用

        Returns:
            True 表示服务正常，False 表示服务异常

        Examples:
            >>> engine = MyEngine()
            >>> if engine.health_check():
            ...     print("Service is healthy")
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        获取引擎信息

        Returns:
            包含引擎配置和状态的字典

        Examples:
            >>> engine = MyEngine()
            >>> info = engine.get_info()
            >>> print(info["model_name"])
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "config": self.config,
            "engine_type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        """返回引擎的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"base_url='{self.base_url}'"
            f")"
        )
