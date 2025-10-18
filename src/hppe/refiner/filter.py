"""
误报过滤器 (False Positive Filter)

实现基于规则和统计的误报过滤，削减常见的假阳性模式
"""

import sys
from pathlib import Path
from typing import List, Set
import logging
import re

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hppe.models.entity import Entity
from hppe.refiner.config import FilterConfig

logger = logging.getLogger(__name__)


# 黑名单：常见误报模式
BLACKLIST = {
    "PERSON_NAME": {
        "admin", "user", "test", "guest", "root", "system",
        "administrator", "用户", "测试", "管理员", "访客",
        "张三", "李四", "王五", "赵六",  # 常见示例名
    },
    "EMAIL": {
        "noreply@", "no-reply@", "example.com", "test@",
        "admin@example", "user@example", "example@",
    },
    "PHONE_NUMBER": {
        "123456", "000000", "111111", "999999",
        "12345678", "00000000", "11111111",
        "1234567890", "0000000000",
    },
    "ORGANIZATION": {
        "公司", "企业", "单位", "组织",  # 单独出现时无意义
        "某公司", "某企业", "XX公司",
    },
    "ADDRESS": {
        "地址", "位置", "这里", "那里",
    },
    "ID_CARD": {
        "000000000000000000",  # 全零
        "111111111111111111",  # 全1
    },
    "BANK_CARD": {
        "0000000000000000",
        "1111111111111111",
    },
}


class FalsePositiveFilter:
    """
    误报过滤器

    削减常见的假阳性模式：
    1. 黑名单过滤
    2. 统计特征过滤
    3. 格式验证
    """

    def __init__(self, config: FilterConfig = None):
        """
        初始化误报过滤器

        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or FilterConfig()
        self.blacklist = BLACKLIST

        logger.info(
            f"误报过滤器已初始化: "
            f"blacklist={self.config.enable_blacklist}, "
            f"statistical={self.config.enable_statistical_filter}, "
            f"format={self.config.enable_format_validation}"
        )

    def filter(self, entities: List[Entity]) -> List[Entity]:
        """
        过滤假阳性实体

        Args:
            entities: 待过滤的实体列表

        Returns:
            过滤后的实体列表
        """
        if not entities:
            return []

        filtered = []

        for entity in entities:
            # 黑名单过滤
            if self.config.enable_blacklist:
                if self._is_in_blacklist(entity):
                    logger.debug(f"黑名单过滤: {entity.entity_type} '{entity.value}'")
                    continue

            # 统计特征过滤
            if self.config.enable_statistical_filter:
                if self._is_statistical_false_positive(entity):
                    logger.debug(
                        f"统计特征过滤: {entity.entity_type} '{entity.value}'"
                    )
                    continue

            # 格式验证
            if self.config.enable_format_validation:
                if not self._validate_format(entity):
                    logger.debug(f"格式验证失败: {entity.entity_type} '{entity.value}'")
                    continue

            # 最小长度检查
            if not self._check_min_length(entity):
                logger.debug(
                    f"长度不足: {entity.entity_type} '{entity.value}' "
                    f"(len={len(entity.value)})"
                )
                continue

            # 通过所有过滤器
            filtered.append(entity)

        logger.debug(
            f"误报过滤完成: 输入{len(entities)}个实体，输出{len(filtered)}个实体，"
            f"过滤{len(entities) - len(filtered)}个"
        )

        return filtered

    def _is_in_blacklist(self, entity: Entity) -> bool:
        """
        检查实体是否在黑名单中

        Args:
            entity: 实体

        Returns:
            是否在黑名单中
        """
        entity_type = entity.entity_type
        value = entity.value.lower().strip()

        # 获取该类型的黑名单
        type_blacklist = self.blacklist.get(entity_type, set())

        # 完全匹配
        if value in type_blacklist:
            return True

        # 部分匹配（用于EMAIL等）
        for pattern in type_blacklist:
            if pattern.endswith("@") or pattern.endswith(".com"):
                if pattern in value:
                    return True

        return False

    def _is_statistical_false_positive(self, entity: Entity) -> bool:
        """
        基于统计特征判断是否为假阳性

        Args:
            entity: 实体

        Returns:
            是否为假阳性
        """
        value = entity.value
        entity_type = entity.entity_type

        # 电话号码：不应全为相同数字，不应为连续数字
        if entity_type == "PHONE_NUMBER":
            # 移除分隔符
            digits = re.sub(r'[-\s()]', '', value)

            # 全为相同数字
            if len(set(digits)) <= 1:
                return True

            # 连续数字（如123456, 654321）
            if self._is_sequential_digits(digits):
                return True

        # 邮编：不应全为0或9
        if entity_type == "POSTAL_CODE":
            if value in ["000000", "999999"]:
                return True

            # 全为相同数字
            if len(set(value)) == 1:
                return True

        # 身份证：不应全为相同数字
        if entity_type == "ID_CARD":
            if len(set(value)) <= 2:
                return True

        # 银行卡：不应全为相同数字
        if entity_type == "BANK_CARD":
            if len(set(value)) <= 2:
                return True

        # IP地址：不应为127.0.0.1（本地回环）或0.0.0.0
        if entity_type == "IP_ADDRESS":
            if value in ["127.0.0.1", "0.0.0.0", "localhost"]:
                return True

        return False

    def _is_sequential_digits(self, digits: str) -> bool:
        """
        判断是否为连续数字

        Args:
            digits: 数字字符串

        Returns:
            是否为连续数字
        """
        if len(digits) < 4:
            return False

        # 转换为整数列表
        try:
            nums = [int(d) for d in digits]
        except ValueError:
            return False

        # 检查递增连续
        is_ascending = all(
            nums[i+1] - nums[i] == 1
            for i in range(len(nums) - 1)
        )

        # 检查递减连续
        is_descending = all(
            nums[i] - nums[i+1] == 1
            for i in range(len(nums) - 1)
        )

        return is_ascending or is_descending

    def _validate_format(self, entity: Entity) -> bool:
        """
        格式验证

        Args:
            entity: 实体

        Returns:
            是否通过格式验证
        """
        entity_type = entity.entity_type
        value = entity.value

        # 银行卡：Luhn算法验证
        if entity_type == "BANK_CARD":
            return self._validate_luhn(value)

        # 身份证：校验位验证（中国身份证）
        if entity_type == "ID_CARD":
            if len(value) == 18:
                return self._validate_chinese_id_card(value)
            # 15位身份证不验证校验位
            return True

        # 邮箱：基本格式验证
        if entity_type == "EMAIL":
            return self._validate_email(value)

        # 手机号：中国手机号格式
        if entity_type == "PHONE_NUMBER":
            # 简单验证：11位，1开头
            digits = re.sub(r'[-\s()]', '', value)
            if len(digits) == 11:
                return digits[0] == '1'
            return True  # 非11位的不强制验证

        # 其他类型默认通过
        return True

    def _validate_luhn(self, card_number: str) -> bool:
        """
        Luhn算法验证银行卡号

        Args:
            card_number: 银行卡号

        Returns:
            是否通过验证
        """
        # 移除非数字字符
        digits = re.sub(r'\D', '', card_number)

        if not digits or len(digits) < 13:
            return False

        # Luhn算法
        def luhn_checksum(num):
            def digits_of(n):
                return [int(d) for d in str(n)]

            digits_list = digits_of(num)
            odd_digits = digits_list[-1::-2]
            even_digits = digits_list[-2::-2]

            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))

            return checksum % 10

        return luhn_checksum(int(digits)) == 0

    def _validate_chinese_id_card(self, id_card: str) -> bool:
        """
        验证中国18位身份证校验位

        Args:
            id_card: 身份证号

        Returns:
            是否通过验证
        """
        if len(id_card) != 18:
            return True  # 非18位不验证

        # 提取前17位和校验位
        body = id_card[:17]
        check_digit = id_card[17].upper()

        # 校验位系数
        coefficients = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

        # 计算校验和
        try:
            checksum = sum(
                int(body[i]) * coefficients[i]
                for i in range(17)
            )
            expected_check = check_codes[checksum % 11]
            return check_digit == expected_check
        except (ValueError, IndexError):
            return False

    def _validate_email(self, email: str) -> bool:
        """
        基本邮箱格式验证

        Args:
            email: 邮箱地址

        Returns:
            是否通过验证
        """
        # 简单的邮箱格式检查
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None

    def _check_min_length(self, entity: Entity) -> bool:
        """
        检查最小长度

        Args:
            entity: 实体

        Returns:
            是否满足最小长度
        """
        entity_type = entity.entity_type
        value = entity.value

        # 获取最小长度要求
        min_length = self.config.min_entity_length.get(entity_type, 0)

        if min_length > 0:
            # 对于数字类型，移除分隔符后计算长度
            if entity_type in ["PHONE_NUMBER", "BANK_CARD", "ID_CARD"]:
                clean_value = re.sub(r'[-\s()]', '', value)
                return len(clean_value) >= min_length
            else:
                return len(value) >= min_length

        return True

    def get_info(self) -> dict:
        """
        获取过滤器信息

        Returns:
            信息字典
        """
        return {
            "name": "FalsePositiveFilter",
            "config": {
                "enable_blacklist": self.config.enable_blacklist,
                "enable_statistical_filter": self.config.enable_statistical_filter,
                "enable_format_validation": self.config.enable_format_validation,
                "num_blacklist_types": len(self.blacklist)
            }
        }

    def __repr__(self) -> str:
        return (
            f"FalsePositiveFilter(blacklist={self.config.enable_blacklist}, "
            f"statistical={self.config.enable_statistical_filter}, "
            f"format={self.config.enable_format_validation})"
        )
