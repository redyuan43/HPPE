"""
脱敏模块配置 (Anonymization Configuration)

定义脱敏策略的配置类
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class AnonymizationConfig:
    """
    脱敏引擎全局配置

    定义PII类型到策略的映射、默认策略等
    """

    # PII类型到策略名称的映射（如 "ID_CARD" -> "mask"）
    type_strategy_map: Dict[str, str] = field(default_factory=dict)

    # 默认策略（当类型未在type_strategy_map中指定时使用）
    default_strategy: str = "redact"

    # 是否保留格式（影响屏蔽、合成等策略）
    preserve_format: bool = True

    # 是否启用批量优化（从后向前处理，避免位置偏移问题）
    batch_optimization: bool = True


@dataclass
class RedactionConfig:
    """编辑策略配置"""

    # 占位符模板（{entity_type}会被替换为实际类型）
    placeholder_template: str = "[{entity_type}]"

    # 是否使用通用占位符（如全部用"[REDACTED]"）
    use_generic_placeholder: bool = False

    # 通用占位符文本
    generic_placeholder: str = "[REDACTED]"


@dataclass
class MaskingConfig:
    """屏蔽策略配置"""

    # 屏蔽字符
    mask_char: str = "*"

    # 按类型配置保留模式（前N后M）
    # 如 "ID_CARD": (4, 4) 表示保留前4位和后4位
    reveal_patterns: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "ID_CARD": (4, 4),
        "PHONE_NUMBER": (3, 4),
        "BANK_CARD": (6, 4),
        "EMAIL": (2, 0),
        "IMEI": (4, 4),
        "VIN": (3, 4),
    })

    # 默认保留模式（对未配置的类型）
    default_reveal_pattern: Tuple[int, int] = (2, 2)


@dataclass
class HashingConfig:
    """哈希策略配置"""

    # 哈希算法
    algorithm: str = "sha256"  # 可选：sha256, sha512, md5

    # 是否使用盐值
    use_salt: bool = True

    # 盐值（应从环境变量读取）
    salt: str = "hppe-default-salt-change-in-production"

    # 是否保留长度（哈希通常很长，可选截断）
    preserve_length: bool = False

    # 哈希输出长度（如果preserve_length=True）
    output_length: int = 16

    # 是否添加前缀标识
    add_prefix: bool = True

    # 前缀格式
    prefix_format: str = "HASH:"


@dataclass
class EncryptionConfig:
    """加密策略配置"""

    # 加密算法
    algorithm: str = "AES256"  # 可选：AES256, RSA

    # 加密密钥（Base64编码，应从环境变量读取）
    encryption_key: str = None  # 必需，不应硬编码

    # 是否添加前缀标识
    add_prefix: bool = True

    # 前缀格式
    prefix_format: str = "ENC:{algorithm}:"

    # 输出格式
    output_format: str = "hex"  # 可选：hex, base64


@dataclass
class SyntheticConfig:
    """合成替换策略配置"""

    # 是否保持一致性（同一PII生成相同合成值）
    consistent_mapping: bool = True

    # 合成数据提供者
    fake_data_providers: Dict[str, str] = field(default_factory=lambda: {
        "PERSON_NAME": "faker",
        "ID_CARD": "generator",
        "PHONE_NUMBER": "generator",
        "EMAIL": "faker",
        "ADDRESS": "faker",
        "ORGANIZATION": "faker",
    })

    # 地区偏好
    locale: str = "zh_CN"

    # 一致性映射缓存大小
    cache_size: int = 10000
