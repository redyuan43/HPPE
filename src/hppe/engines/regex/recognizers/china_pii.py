"""
中国 PII 识别器

包含中国特定的 PII 类型识别器：
- 身份证号（含校验码验证）
- 手机号
- 银行卡号（含 Luhn 校验）
- 护照号
"""

import re
from typing import List

from hppe.engines.regex.base import BaseRecognizer
from hppe.models.entity import Entity


class ChinaIDCardRecognizer(BaseRecognizer):
    """
    中国身份证号识别器

    支持 18 位身份证号码的检测和校验码验证。

    身份证号码格式：
    - 6位地区码 + 8位出生日期 + 3位顺序码 + 1位校验码
    - 示例：110101199003077578

    Examples:
        >>> config = {
        ...     "entity_type": "CHINA_ID_CARD",
        ...     "patterns": [{"pattern": r"[1-9]\\d{5}..."}]
        ... }
        >>> recognizer = ChinaIDCardRecognizer(config)
        >>> entities = recognizer.detect("身份证：110101199003077578")
    """

    # 校验码权重
    _WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    # 校验码映射
    _CHECKSUMS = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的身份证号

        Args:
            text: 待检测的文本

        Returns:
            检测到的身份证号实体列表
        """
        entities = []

        for pattern in self.patterns:
            for match in pattern.finditer(text):
                value = match.group()
                start = match.start()
                end = match.end()

                # 检查拒绝列表
                if self._check_deny_list(text, start):
                    continue

                # 验证身份证号
                is_valid = self.validate_id_card(value)

                # 计算置信度
                base_confidence = self.confidence_base
                if is_valid:
                    base_confidence += 0.05  # 校验通过额外加分

                confidence = self._calculate_confidence(
                    base_confidence,
                    text,
                    start,
                    validation_passed=is_valid
                )

                # 创建实体
                entity = self._create_entity(
                    value=value,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence,
                    metadata={
                        "checksum_valid": is_valid,
                        "region_code": value[:6],
                        "birth_date": value[6:14],
                        "sequence": value[14:17],
                        "checksum": value[17]
                    }
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证身份证号实体

        Args:
            entity: 待验证的实体

        Returns:
            True 表示有效，False 表示无效
        """
        return self.validate_id_card(entity.value)

    @staticmethod
    def validate_id_card(id_card: str) -> bool:
        """
        验证 18 位身份证号的校验码

        使用 GB 11643-1999 标准算法：
        1. 将前 17 位数字与对应权重相乘
        2. 求和后对 11 取模
        3. 用模值作为索引获取校验码

        Args:
            id_card: 18 位身份证号字符串

        Returns:
            True 表示校验码正确，False 表示错误
        """
        if len(id_card) != 18:
            return False

        # 提取前 17 位和校验码
        id_17 = id_card[:17]
        checksum = id_card[17].upper()

        # 检查前 17 位是否都是数字
        if not id_17.isdigit():
            return False

        # 计算校验码
        try:
            total = sum(
                int(id_17[i]) * ChinaIDCardRecognizer._WEIGHTS[i]
                for i in range(17)
            )
            mod = total % 11
            expected_checksum = ChinaIDCardRecognizer._CHECKSUMS[mod]

            return checksum == expected_checksum

        except (ValueError, IndexError):
            return False


class ChinaPhoneRecognizer(BaseRecognizer):
    """
    中国手机号识别器

    支持标准 11 位手机号和带 +86 前缀的格式。

    手机号格式：
    - 1[3-9]xxxxxxxxx (11位)
    - +86 1[3-9]xxxxxxxxx
    - +86-1[3-9]xxxxxxxxx

    Examples:
        >>> recognizer = ChinaPhoneRecognizer(config)
        >>> entities = recognizer.detect("手机：13812345678")
    """

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的手机号

        Args:
            text: 待检测的文本

        Returns:
            检测到的手机号实体列表
        """
        entities = []

        for pattern in self.patterns:
            for match in pattern.finditer(text):
                value = match.group()
                start = match.start()
                end = match.end()

                # 检查拒绝列表
                if self._check_deny_list(text, start):
                    continue

                # 提取纯数字手机号（去除前缀和分隔符）
                phone_number = self._extract_phone_number(value)

                # 验证手机号
                is_valid = self.validate_phone(phone_number)

                # 计算置信度
                confidence = self._calculate_confidence(
                    self.confidence_base,
                    text,
                    start,
                    validation_passed=is_valid
                )

                # 创建实体
                entity = self._create_entity(
                    value=value,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence,
                    metadata={
                        "normalized": phone_number,
                        "has_country_code": value.startswith(('+86', '86'))
                    }
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证手机号实体

        Args:
            entity: 待验证的实体

        Returns:
            True 表示有效，False 表示无效
        """
        phone_number = self._extract_phone_number(entity.value)
        return self.validate_phone(phone_number)

    @staticmethod
    def _extract_phone_number(value: str) -> str:
        """
        提取纯数字手机号

        Args:
            value: 原始值（可能包含 +86 等前缀）

        Returns:
            11 位纯数字手机号
        """
        # 移除所有非数字字符
        digits = re.sub(r'\D', '', value)

        # 如果以 86 开头且长度为 13，移除前缀
        if digits.startswith('86') and len(digits) == 13:
            return digits[2:]

        return digits

    @staticmethod
    def validate_phone(phone_number: str) -> bool:
        """
        验证手机号格式

        Args:
            phone_number: 11 位手机号

        Returns:
            True 表示格式正确，False 表示错误
        """
        # 必须是 11 位数字
        if len(phone_number) != 11:
            return False

        # 必须以 1 开头，第二位是 3-9
        if not phone_number.startswith('1'):
            return False

        if phone_number[1] not in '3456789':
            return False

        # 全部都是数字
        return phone_number.isdigit()


class ChinaBankCardRecognizer(BaseRecognizer):
    """
    中国银行卡号识别器

    支持 16-19 位银行卡号，使用 Luhn 算法验证校验码。

    银行卡号格式：
    - 16-19 位数字
    - 常见格式：6222 0212 3456 7890 123

    Examples:
        >>> recognizer = ChinaBankCardRecognizer(config)
        >>> entities = recognizer.detect("卡号：6222021234567890123")
    """

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的银行卡号

        Args:
            text: 待检测的文本

        Returns:
            检测到的银行卡号实体列表
        """
        entities = []

        for pattern in self.patterns:
            for match in pattern.finditer(text):
                value = match.group()
                start = match.start()
                end = match.end()

                # 检查拒绝列表
                if self._check_deny_list(text, start):
                    continue

                # 移除空格和分隔符
                card_number = re.sub(r'[\s\-]', '', value)

                # 验证银行卡号
                is_valid = self.validate_luhn(card_number)

                # 计算置信度
                base_confidence = self.confidence_base
                if is_valid:
                    base_confidence += 0.05  # Luhn 校验通过额外加分

                confidence = self._calculate_confidence(
                    base_confidence,
                    text,
                    start,
                    validation_passed=is_valid
                )

                # 创建实体
                entity = self._create_entity(
                    value=value,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence,
                    metadata={
                        "luhn_valid": is_valid,
                        "normalized": card_number,
                        "length": len(card_number)
                    }
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证银行卡号实体

        Args:
            entity: 待验证的实体

        Returns:
            True 表示有效，False 表示无效
        """
        card_number = re.sub(r'[\s\-]', '', entity.value)
        return self.validate_luhn(card_number)

    @staticmethod
    def validate_luhn(card_number: str) -> bool:
        """
        使用 Luhn 算法验证银行卡号

        Luhn 算法步骤：
        1. 从右往左，偶数位乘以 2
        2. 如果乘积 >= 10，则减去 9
        3. 所有位求和
        4. 和能被 10 整除则有效

        Args:
            card_number: 银行卡号字符串

        Returns:
            True 表示校验通过，False 表示失败
        """
        # 必须全部是数字
        if not card_number.isdigit():
            return False

        # 长度必须在 16-19 之间
        if not 16 <= len(card_number) <= 19:
            return False

        # Luhn 算法
        total = 0
        reverse_digits = card_number[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)

            # 偶数位（从0开始，所以是奇数索引）
            if i % 2 == 1:
                n *= 2
                if n >= 10:
                    n -= 9

            total += n

        return total % 10 == 0


class ChinaPassportRecognizer(BaseRecognizer):
    """
    中国护照号识别器

    支持新旧版本护照号码格式。

    护照号格式：
    - 新版：E + 8位数字（电子护照）
    - 旧版：G + 8位数字（普通护照）
    - 其他：P/S/D + 7位数字

    Examples:
        >>> recognizer = ChinaPassportRecognizer(config)
        >>> entities = recognizer.detect("护照：E12345678")
    """

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的护照号

        Args:
            text: 待检测的文本

        Returns:
            检测到的护照号实体列表
        """
        entities = []

        for pattern in self.patterns:
            for match in pattern.finditer(text):
                value = match.group()
                start = match.start()
                end = match.end()

                # 检查拒绝列表
                if self._check_deny_list(text, start):
                    continue

                # 识别护照类型
                passport_type = self._identify_passport_type(value)

                # 验证护照号
                is_valid = self.validate_passport(value)

                # 计算置信度
                confidence = self._calculate_confidence(
                    self.confidence_base,
                    text,
                    start,
                    validation_passed=is_valid
                )

                # 创建实体
                entity = self._create_entity(
                    value=value,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence,
                    metadata={
                        "passport_type": passport_type,
                        "prefix": value[0],
                        "number": value[1:]
                    }
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证护照号实体

        Args:
            entity: 待验证的实体

        Returns:
            True 表示有效，False 表示无效
        """
        return self.validate_passport(entity.value)

    @staticmethod
    def _identify_passport_type(passport: str) -> str:
        """
        识别护照类型

        Args:
            passport: 护照号

        Returns:
            护照类型描述
        """
        prefix = passport[0].upper()

        type_map = {
            'E': '电子护照',
            'G': '普通护照',
            'P': '因私护照',
            'S': '因公护照',
            'D': '外交护照'
        }

        return type_map.get(prefix, '未知类型')

    @staticmethod
    def validate_passport(passport: str) -> bool:
        """
        验证护照号格式

        Args:
            passport: 护照号

        Returns:
            True 表示格式正确，False 表示错误
        """
        if not passport:
            return False

        prefix = passport[0].upper()
        number = passport[1:]

        # E/G 开头，后跟 8 位数字
        if prefix in 'EG':
            return len(number) == 8 and number.isdigit()

        # P/S/D 开头，后跟 7 位数字
        if prefix in 'PSD':
            return len(number) == 7 and number.isdigit()

        return False
