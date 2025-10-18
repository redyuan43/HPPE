"""
脱敏模块 (Anonymizer)

提供PII脱敏功能，支持多种脱敏策略
"""

from .strategy import AnonymizationStrategy
from .engine import AnonymizationEngine
from .config import (
    AnonymizationConfig,
    RedactionConfig,
    MaskingConfig,
    HashingConfig,
    EncryptionConfig,
    SyntheticConfig
)

__all__ = [
    "AnonymizationStrategy",
    "AnonymizationEngine",
    "AnonymizationConfig",
    "RedactionConfig",
    "MaskingConfig",
    "HashingConfig",
    "EncryptionConfig",
    "SyntheticConfig"
]
