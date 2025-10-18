"""
PII类型标准定义

定义所有支持的PII类型，确保整个系统中类型名称一致。
"""

from enum import Enum


class PIIType(str, Enum):
    """
    标准PII类型枚举

    注意：
    - 使用str继承以便JSON序列化
    - 所有值使用大写下划线命名
    - 与训练数据、模型输出保持一致
    """

    # === 基础6种PII类型（Phase 1） ===
    PERSON_NAME = "PERSON_NAME"           # 人名
    PHONE_NUMBER = "PHONE_NUMBER"         # 电话号码
    EMAIL = "EMAIL"                       # 电子邮箱
    ADDRESS = "ADDRESS"                   # 地址
    ORGANIZATION = "ORGANIZATION"         # 组织机构名
    ID_CARD = "ID_CARD"                   # 身份证号

    # === 扩展11种PII类型（Phase 2） ===
    BANK_CARD = "BANK_CARD"               # 银行卡号
    PASSPORT = "PASSPORT"                 # 护照号码
    DRIVER_LICENSE = "DRIVER_LICENSE"     # 驾驶证号（标准名称）
    VEHICLE_PLATE = "VEHICLE_PLATE"       # 车牌号（标准名称）
    IP_ADDRESS = "IP_ADDRESS"             # IP地址（标准名称）
    MAC_ADDRESS = "MAC_ADDRESS"           # MAC地址（标准名称）
    POSTAL_CODE = "POSTAL_CODE"           # 邮政编码
    IMEI = "IMEI"                         # 手机IMEI号
    VIN = "VIN"                           # 车辆识别码
    TAX_ID = "TAX_ID"                     # 税号/统一社会信用代码
    SOCIAL_SECURITY = "SOCIAL_SECURITY"   # 社保号

    # === 商业PII类型（待扩展） ===
    MILITARY_ID = "MILITARY_ID"           # 军官证
    # CREDIT_CARD = "CREDIT_CARD"         # 信用卡号（待实现）
    # SSN = "SSN"                         # 美国社保号（待实现）


# 类型分组
PHASE_1_TYPES = [
    PIIType.PERSON_NAME,
    PIIType.PHONE_NUMBER,
    PIIType.EMAIL,
    PIIType.ADDRESS,
    PIIType.ORGANIZATION,
    PIIType.ID_CARD,
]

PHASE_2_TYPES = [
    PIIType.BANK_CARD,
    PIIType.PASSPORT,
    PIIType.DRIVER_LICENSE,
    PIIType.VEHICLE_PLATE,
    PIIType.IP_ADDRESS,
    PIIType.MAC_ADDRESS,
    PIIType.POSTAL_CODE,
    PIIType.IMEI,
    PIIType.VIN,
    PIIType.TAX_ID,
    PIIType.SOCIAL_SECURITY,
]

ALL_17_TYPES = PHASE_1_TYPES + PHASE_2_TYPES


# 类型别名映射（用于兼容旧数据）
TYPE_ALIASES = {
    # 驾驶证别名
    "DRIVERS_LICENSE": PIIType.DRIVER_LICENSE,
    "DRIVING_LICENSE": PIIType.DRIVER_LICENSE,

    # 车牌号别名
    "LICENSE_PLATE": PIIType.VEHICLE_PLATE,
    "CAR_PLATE": PIIType.VEHICLE_PLATE,

    # IP地址别名
    "IP": PIIType.IP_ADDRESS,

    # MAC地址别名
    "MAC": PIIType.MAC_ADDRESS,

    # 社保号别名
    "SOCIAL_SECURITY_CARD": PIIType.SOCIAL_SECURITY,
    "SOCIAL_SECURITY_NUMBER": PIIType.SOCIAL_SECURITY,

    # 税号别名
    "TAX_CARD": PIIType.TAX_ID,
    "UNIFIED_SOCIAL_CREDIT_CODE": PIIType.TAX_ID,

    # VIN别名
    "VEHICLE_VIN": PIIType.VIN,
    "VEHICLE_IDENTIFICATION_NUMBER": PIIType.VIN,

    # IMEI别名
    "PHONE_IMEI": PIIType.IMEI,

    # 军官证别名
    "OFFICER_CARD": PIIType.MILITARY_ID,
    "OFFICE_CARD": PIIType.MILITARY_ID,  # 常见拼写错误
}


def normalize_pii_type(type_str: str) -> str:
    """
    标准化PII类型名称

    Args:
        type_str: 原始类型字符串

    Returns:
        标准化后的类型名称

    Examples:
        >>> normalize_pii_type("DRIVERS_LICENSE")
        "DRIVER_LICENSE"
        >>> normalize_pii_type("IP")
        "IP_ADDRESS"
        >>> normalize_pii_type("PERSON_NAME")
        "PERSON_NAME"
    """
    # 转大写
    type_upper = type_str.upper()

    # 检查是否为标准类型
    if type_upper in [t.value for t in PIIType]:
        return type_upper

    # 查找别名
    if type_upper in TYPE_ALIASES:
        return TYPE_ALIASES[type_upper].value

    # 无法识别，返回原值
    return type_str


def get_type_display_name(pii_type: str) -> str:
    """
    获取PII类型的中文显示名称

    Args:
        pii_type: PII类型（英文）

    Returns:
        中文显示名称
    """
    display_names = {
        PIIType.PERSON_NAME: "人名",
        PIIType.PHONE_NUMBER: "电话号码",
        PIIType.EMAIL: "电子邮箱",
        PIIType.ADDRESS: "地址",
        PIIType.ORGANIZATION: "组织机构",
        PIIType.ID_CARD: "身份证号",
        PIIType.BANK_CARD: "银行卡号",
        PIIType.PASSPORT: "护照号码",
        PIIType.DRIVER_LICENSE: "驾驶证号",
        PIIType.VEHICLE_PLATE: "车牌号",
        PIIType.IP_ADDRESS: "IP地址",
        PIIType.MAC_ADDRESS: "MAC地址",
        PIIType.POSTAL_CODE: "邮政编码",
        PIIType.IMEI: "手机IMEI",
        PIIType.VIN: "车辆识别码",
        PIIType.TAX_ID: "税号",
        PIIType.SOCIAL_SECURITY: "社保号",
        PIIType.MILITARY_ID: "军官证",
    }

    # 标准化类型名称
    normalized = normalize_pii_type(pii_type)

    # 查找显示名称
    for pii_enum, display in display_names.items():
        if pii_enum.value == normalized:
            return display

    return pii_type  # 未知类型返回原值


if __name__ == "__main__":
    # 测试标准化功能
    test_cases = [
        "PERSON_NAME",
        "DRIVERS_LICENSE",
        "LICENSE_PLATE",
        "IP",
        "MAC",
        "PHONE_IMEI",
        "VEHICLE_VIN",
        "UNKNOWN_TYPE",
    ]

    print("=== PII类型标准化测试 ===")
    for case in test_cases:
        normalized = normalize_pii_type(case)
        display = get_type_display_name(case)
        print(f"{case:30s} -> {normalized:20s} ({display})")

    print(f"\n总计支持 {len(ALL_17_TYPES)} 种PII类型")
    print(f"  - Phase 1: {len(PHASE_1_TYPES)} 种")
    print(f"  - Phase 2: {len(PHASE_2_TYPES)} 种")
    print(f"  - 别名映射: {len(TYPE_ALIASES)} 个")
