"""
配置文件加载器

从 YAML 文件加载和验证识别器配置
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required for config loading. "
        "Install it with: pip install PyYAML"
    )


class ConfigLoader:
    """
    识别器配置加载器

    从 YAML 配置文件加载识别器配置，并验证格式正确性。

    Attributes:
        patterns_dir: 配置文件所在目录

    Examples:
        >>> loader = ConfigLoader("data/patterns")
        >>> configs = loader.load_all()
        >>> for config in configs:
        ...     print(config['entity_type'])
    """

    def __init__(self, patterns_dir: str = "data/patterns") -> None:
        """
        初始化配置加载器

        Args:
            patterns_dir: 配置文件目录路径

        Raises:
            FileNotFoundError: 当目录不存在时
        """
        self.patterns_dir = Path(patterns_dir)

        if not self.patterns_dir.exists():
            raise FileNotFoundError(
                f"配置目录不存在: {self.patterns_dir}"
            )

        if not self.patterns_dir.is_dir():
            raise ValueError(
                f"路径不是目录: {self.patterns_dir}"
            )

    def load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从单个 YAML 文件加载配置

        Args:
            file_path: YAML 文件路径（可以是相对或绝对路径）

        Returns:
            识别器配置字典列表

        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当配置格式错误时
        """
        # 支持相对路径和绝对路径
        if not os.path.isabs(file_path):
            full_path = self.patterns_dir / file_path
        else:
            full_path = Path(file_path)

        if not full_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {full_path}")

        # 读取 YAML 文件
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"YAML 解析失败 ({full_path}): {e}"
            )
        except Exception as e:
            raise ValueError(
                f"读取文件失败 ({full_path}): {e}"
            )

        # 验证顶级结构
        if not isinstance(data, dict):
            raise ValueError(
                f"配置文件顶级必须是字典 ({full_path})"
            )

        if "recognizers" not in data:
            raise ValueError(
                f"配置文件缺少 'recognizers' 键 ({full_path})"
            )

        recognizers = data["recognizers"]

        if not isinstance(recognizers, list):
            raise ValueError(
                f"'recognizers' 必须是列表 ({full_path})"
            )

        # 验证每个识别器配置
        validated_configs = []
        for i, config in enumerate(recognizers):
            try:
                validated = self._validate_config(config)
                validated_configs.append(validated)
            except ValueError as e:
                raise ValueError(
                    f"识别器 #{i} 配置错误 ({full_path}): {e}"
                )

        return validated_configs

    def load_all(
        self,
        pattern: str = "*.yaml"
    ) -> List[Dict[str, Any]]:
        """
        加载目录下所有匹配的 YAML 文件

        Args:
            pattern: 文件名匹配模式（glob 格式）

        Returns:
            所有识别器配置的列表

        Examples:
            >>> loader = ConfigLoader("data/patterns")
            >>> configs = loader.load_all()  # 加载所有 .yaml 文件
            >>> configs = loader.load_all("china_*.yaml")  # 只加载中文配置
        """
        all_configs = []
        yaml_files = list(self.patterns_dir.glob(pattern))

        if not yaml_files:
            raise FileNotFoundError(
                f"在 {self.patterns_dir} 中未找到匹配 '{pattern}' 的文件"
            )

        for yaml_file in sorted(yaml_files):
            try:
                configs = self.load_file(str(yaml_file))
                all_configs.extend(configs)
            except Exception as e:
                # 记录错误但继续加载其他文件
                # TODO: 添加日志系统
                print(f"警告: 加载 {yaml_file} 失败: {e}")
                continue

        return all_configs

    def _validate_config(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证单个识别器配置

        Args:
            config: 识别器配置字典

        Returns:
            验证后的配置字典

        Raises:
            ValueError: 当配置格式错误时
        """
        if not isinstance(config, dict):
            raise ValueError("识别器配置必须是字典")

        # 检查必需字段
        required_fields = ["entity_type", "patterns"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"缺少必需字段: {field}")

        # 验证 entity_type
        if not isinstance(config["entity_type"], str):
            raise ValueError("entity_type 必须是字符串")

        if not config["entity_type"]:
            raise ValueError("entity_type 不能为空")

        # 验证 patterns
        patterns = config["patterns"]
        if not isinstance(patterns, list):
            raise ValueError("patterns 必须是列表")

        if not patterns:
            raise ValueError("patterns 不能为空")

        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, dict):
                raise ValueError(f"pattern #{i} 必须是字典")

            if "pattern" not in pattern:
                raise ValueError(f"pattern #{i} 缺少 'pattern' 键")

            if not isinstance(pattern["pattern"], str):
                raise ValueError(f"pattern #{i} 的 'pattern' 必须是字符串")

        # 验证可选字段
        if "context_words" in config:
            if not isinstance(config["context_words"], list):
                raise ValueError("context_words 必须是列表")

        if "deny_lists" in config:
            if not isinstance(config["deny_lists"], list):
                raise ValueError("deny_lists 必须是列表")

        if "confidence_base" in config:
            base = config["confidence_base"]
            if not isinstance(base, (int, float)):
                raise ValueError("confidence_base 必须是数字")

            if not 0.0 <= base <= 1.0:
                raise ValueError(
                    f"confidence_base 必须在 [0.0, 1.0] 范围内: {base}"
                )

        return config

    def list_config_files(self) -> List[Path]:
        """
        列出配置目录下所有 YAML 文件

        Returns:
            YAML 文件路径列表
        """
        return sorted(self.patterns_dir.glob("*.yaml"))

    def get_config_by_type(
        self,
        entity_type: str,
        pattern: str = "*.yaml"
    ) -> Optional[Dict[str, Any]]:
        """
        根据 entity_type 查找配置

        Args:
            entity_type: PII 类型
            pattern: 文件名匹配模式

        Returns:
            找到的配置字典，如果不存在则返回 None
        """
        all_configs = self.load_all(pattern)

        for config in all_configs:
            if config["entity_type"] == entity_type:
                return config

        return None
