"""
精炼流水线配置

定义各个精炼模块的配置参数
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DisambiguatorConfig:
    """歧义消除器配置"""

    # 识别器权重（用于解决冲突时的优先级）
    recognizer_weights: Dict[str, float] = field(default_factory=lambda: {
        "regex": 1.2,        # Regex识别器权重更高（对结构化PII更准确）
        "llm_finetuned": 1.0  # LLM识别器基准权重
    })

    # PII类型优先级（数值越高优先级越高）
    type_priorities: Dict[str, int] = field(default_factory=lambda: {
        # 身份类（最高优先级）
        "ID_CARD": 100,
        "PASSPORT": 95,
        "DRIVER_LICENSE": 90,
        "MILITARY_ID": 85,
        "SOCIAL_SECURITY": 80,

        # 金融类
        "BANK_CARD": 75,
        "TAX_ID": 70,

        # 联系方式类
        "PHONE_NUMBER": 60,
        "EMAIL": 65,

        # 地址类
        "ADDRESS": 55,
        "POSTAL_CODE": 50,

        # 人名/组织
        "PERSON_NAME": 45,
        "ORGANIZATION": 40,

        # 技术标识类
        "IP_ADDRESS": 35,
        "MAC_ADDRESS": 30,
        "IMEI": 25,
        "VIN": 20,
        "VEHICLE_PLATE": 15,
    })

    # 最小置信度差异阈值（只有差异大于此值才考虑优先级）
    min_confidence_diff: float = 0.1

    # 是否启用严格模式（严格模式下只保留最高优先级的实体）
    strict_mode: bool = True


@dataclass
class MergerConfig:
    """实体合并器配置"""

    # 是否合并重叠实体
    merge_overlapping: bool = True

    # 是否合并相邻实体（仅对特定类型启用）
    merge_adjacent: bool = True

    # 允许合并的PII类型（仅这些类型的相邻实体会被合并）
    mergeable_types: List[str] = field(default_factory=lambda: [
        "ADDRESS",
        "ORGANIZATION",
        "PERSON_NAME"
    ])

    # 相邻实体最大间隔（字符数）
    max_adjacent_gap: int = 2

    # 重叠阈值（重叠字符数占较短实体的比例）
    overlap_threshold: float = 0.3


@dataclass
class ValidatorConfig:
    """上下文验证器配置"""

    # 上下文窗口大小（前后各多少字符）
    context_window: int = 20

    # 是否启用上下文验证
    enable_context_validation: bool = True

    # 上下文置信度调整幅度
    context_confidence_boost: float = 0.1  # 正向关键词提升10%
    context_confidence_penalty: float = 0.2  # 负向关键词惩罚20%

    # 最小验证置信度（低于此值的实体会被验证）
    min_validation_confidence: float = 0.8


@dataclass
class FilterConfig:
    """误报过滤器配置"""

    # 是否启用黑名单过滤
    enable_blacklist: bool = True

    # 是否启用统计特征过滤
    enable_statistical_filter: bool = True

    # 是否启用格式验证
    enable_format_validation: bool = True

    # 最小实体长度（按类型）
    min_entity_length: Dict[str, int] = field(default_factory=lambda: {
        "PHONE_NUMBER": 7,
        "ID_CARD": 15,
        "BANK_CARD": 13,
        "EMAIL": 5,
        "IMEI": 14,
    })


@dataclass
class RefinementConfig:
    """精炼流水线总配置"""

    disambiguator: DisambiguatorConfig = field(default_factory=DisambiguatorConfig)
    merger: MergerConfig = field(default_factory=MergerConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    # 是否启用各个阶段
    enable_disambiguation: bool = True
    enable_merging: bool = True
    enable_validation: bool = True
    enable_filtering: bool = True


# 默认配置实例
DEFAULT_REFINEMENT_CONFIG = RefinementConfig()
