"""
全球 PII 识别器

包含全球通用的 PII 类型识别器：
- 电子邮件地址
- IP 地址（IPv4 和 IPv6）
- URL
- 信用卡号（含 Luhn 校验）
- 美国社会安全号（SSN）
"""

import re
from typing import List
from urllib.parse import urlparse

from hppe.engines.regex.base import BaseRecognizer
from hppe.models.entity import Entity


class EmailRecognizer(BaseRecognizer):
    """
    电子邮件地址识别器

    支持标准 RFC 5322 格式的电子邮件地址，包括国际化域名。

    邮箱格式：
    - 本地部分：字母、数字、特殊字符（.-_+）
    - 域名部分：字母、数字、连字符
    - 示例：john.doe@example.com、user+tag@domain.co.uk

    Examples:
        >>> config = {
        ...     "entity_type": "EMAIL",
        ...     "patterns": [{...}]
        ... }
        >>> recognizer = EmailRecognizer(config)
        >>> entities = recognizer.detect("Contact: john@example.com")
    """

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的电子邮件地址

        Args:
            text: 待检测的文本

        Returns:
            检测到的电子邮件实体列表
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

                # 验证邮箱格式
                is_valid = self.validate_email(value)

                # 计算置信度
                confidence = self._calculate_confidence(
                    self.confidence_base,
                    text,
                    start,
                    validation_passed=is_valid
                )

                # 提取邮箱组成部分
                local_part, domain = self._parse_email(value)

                # 创建实体
                entity = self._create_entity(
                    value=value,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence,
                    metadata={
                        "valid": is_valid,
                        "local_part": local_part,
                        "domain": domain,
                        "tld": self._extract_tld(domain)
                    }
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证电子邮件实体

        Args:
            entity: 待验证的实体

        Returns:
            True 表示有效，False 表示无效
        """
        return self.validate_email(entity.value)

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        验证电子邮件格式

        Args:
            email: 电子邮件地址

        Returns:
            True 表示格式正确，False 表示错误
        """
        if not email or '@' not in email:
            return False

        # 基本格式检查
        parts = email.split('@')
        if len(parts) != 2:
            return False

        local_part, domain = parts

        # 本地部分不能为空
        if not local_part or not domain:
            return False

        # 域名必须包含至少一个点
        if '.' not in domain:
            return False

        # 域名不能以点开始或结束
        if domain.startswith('.') or domain.endswith('.'):
            return False

        return True

    @staticmethod
    def _parse_email(email: str) -> tuple:
        """
        解析电子邮件地址

        Args:
            email: 电子邮件地址

        Returns:
            (local_part, domain) 元组
        """
        if '@' not in email:
            return "", ""

        parts = email.split('@')
        if len(parts) != 2:
            return "", ""

        return parts[0], parts[1]

    @staticmethod
    def _extract_tld(domain: str) -> str:
        """
        提取顶级域名

        Args:
            domain: 域名

        Returns:
            顶级域名（如 .com, .org）
        """
        if not domain or '.' not in domain:
            return ""

        return domain.split('.')[-1]


class IPAddressRecognizer(BaseRecognizer):
    """
    IP 地址识别器

    支持 IPv4 和 IPv6 地址格式。

    IPv4 格式：
    - 点分十进制：192.168.1.1
    - 每段范围：0-255

    IPv6 格式：
    - 冒号十六进制：2001:0db8:85a3:0000:0000:8a2e:0370:7334
    - 简化格式：2001:db8::1

    Examples:
        >>> recognizer = IPAddressRecognizer(config)
        >>> entities = recognizer.detect("Server IP: 192.168.1.1")
    """

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的 IP 地址

        Args:
            text: 待检测的文本

        Returns:
            检测到的 IP 地址实体列表
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

                # 识别 IP 类型和验证
                ip_type = self._identify_ip_type(value)
                is_valid = self.validate_ip(value, ip_type)

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
                        "valid": is_valid,
                        "ip_type": ip_type,
                        "is_private": self._is_private_ip(value, ip_type)
                    }
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证 IP 地址实体

        Args:
            entity: 待验证的实体

        Returns:
            True 表示有效，False 表示无效
        """
        ip_type = self._identify_ip_type(entity.value)
        return self.validate_ip(entity.value, ip_type)

    @staticmethod
    def _identify_ip_type(ip: str) -> str:
        """
        识别 IP 地址类型

        Args:
            ip: IP 地址字符串

        Returns:
            "ipv4" 或 "ipv6"
        """
        if ':' in ip:
            return "ipv6"
        elif '.' in ip:
            return "ipv4"
        return "unknown"

    @staticmethod
    def validate_ip(ip: str, ip_type: str) -> bool:
        """
        验证 IP 地址格式

        Args:
            ip: IP 地址
            ip_type: IP 类型（ipv4 或 ipv6）

        Returns:
            True 表示格式正确，False 表示错误
        """
        if ip_type == "ipv4":
            return IPAddressRecognizer._validate_ipv4(ip)
        elif ip_type == "ipv6":
            return IPAddressRecognizer._validate_ipv6(ip)
        return False

    @staticmethod
    def _validate_ipv4(ip: str) -> bool:
        """
        验证 IPv4 地址

        Args:
            ip: IPv4 地址

        Returns:
            True 表示有效，False 表示无效
        """
        parts = ip.split('.')
        if len(parts) != 4:
            return False

        try:
            for part in parts:
                num = int(part)
                if num < 0 or num > 255:
                    return False
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_ipv6(ip: str) -> bool:
        """
        验证 IPv6 地址（简化版）

        Args:
            ip: IPv6 地址

        Returns:
            True 表示有效，False 表示无效
        """
        # 简化的 IPv6 验证
        if '::' in ip:
            # 简化格式
            parts = ip.split('::')
            if len(parts) > 2:
                return False

        # 检查是否包含有效的十六进制字符
        hex_chars = set('0123456789abcdefABCDEF:')
        return all(c in hex_chars for c in ip)

    @staticmethod
    def _is_private_ip(ip: str, ip_type: str) -> bool:
        """
        判断是否为私有 IP 地址

        Args:
            ip: IP 地址
            ip_type: IP 类型

        Returns:
            True 表示私有 IP，False 表示公网 IP
        """
        if ip_type == "ipv4":
            parts = ip.split('.')
            if len(parts) != 4:
                return False

            try:
                first = int(parts[0])
                second = int(parts[1])

                # 10.0.0.0/8
                if first == 10:
                    return True

                # 172.16.0.0/12
                if first == 172 and 16 <= second <= 31:
                    return True

                # 192.168.0.0/16
                if first == 192 and second == 168:
                    return True

                # 127.0.0.0/8 (localhost)
                if first == 127:
                    return True

            except (ValueError, IndexError):
                return False

        return False


class URLRecognizer(BaseRecognizer):
    """
    URL 识别器

    支持 HTTP、HTTPS、FTP 等协议的 URL。

    URL 格式：
    - http://example.com
    - https://www.example.com/path?query=value
    - ftp://files.example.com/file.txt

    Examples:
        >>> recognizer = URLRecognizer(config)
        >>> entities = recognizer.detect("Visit: https://example.com")
    """

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的 URL

        Args:
            text: 待检测的文本

        Returns:
            检测到的 URL 实体列表
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

                # 解析 URL
                parsed = self._parse_url(value)
                is_valid = parsed is not None

                # 计算置信度
                confidence = self._calculate_confidence(
                    self.confidence_base,
                    text,
                    start,
                    validation_passed=is_valid
                )

                # 创建实体
                metadata = {
                    "valid": is_valid
                }

                if parsed:
                    metadata.update({
                        "scheme": parsed.get("scheme", ""),
                        "domain": parsed.get("domain", ""),
                        "path": parsed.get("path", ""),
                        "has_query": parsed.get("has_query", False)
                    })

                entity = self._create_entity(
                    value=value,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence,
                    metadata=metadata
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证 URL 实体

        Args:
            entity: 待验证的实体

        Returns:
            True 表示有效，False 表示无效
        """
        return self._parse_url(entity.value) is not None

    @staticmethod
    def _parse_url(url: str) -> dict:
        """
        解析 URL

        Args:
            url: URL 字符串

        Returns:
            解析后的 URL 组成部分，或 None（如果解析失败）
        """
        try:
            parsed = urlparse(url)

            if not parsed.scheme or not parsed.netloc:
                return None

            return {
                "scheme": parsed.scheme,
                "domain": parsed.netloc,
                "path": parsed.path or "/",
                "has_query": bool(parsed.query)
            }

        except Exception:
            return None


class CreditCardRecognizer(BaseRecognizer):
    """
    信用卡号识别器

    支持主流信用卡类型（Visa、MasterCard、Amex等），使用 Luhn 算法验证。

    信用卡格式：
    - Visa: 4xxx-xxxx-xxxx-xxxx (16位)
    - MasterCard: 5xxx-xxxx-xxxx-xxxx (16位)
    - Amex: 3xxx-xxxxxx-xxxxx (15位)

    Examples:
        >>> recognizer = CreditCardRecognizer(config)
        >>> entities = recognizer.detect("Card: 4532-1488-0343-6467")
    """

    # 卡类型识别前缀
    _CARD_PREFIXES = {
        'Visa': ['4'],
        'MasterCard': ['51', '52', '53', '54', '55', '22', '23', '24', '25', '26', '27'],
        'Amex': ['34', '37'],
        'Discover': ['6011', '644', '645', '646', '647', '648', '649', '65'],
        'JCB': ['35']
    }

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的信用卡号

        Args:
            text: 待检测的文本

        Returns:
            检测到的信用卡号实体列表
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

                # 移除分隔符
                card_number = re.sub(r'[\s\-]', '', value)

                # 验证信用卡号（Luhn算法）
                is_valid = self.validate_luhn(card_number)

                # 识别卡类型
                card_type = self._identify_card_type(card_number)

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
                        "length": len(card_number),
                        "card_type": card_type,
                        "masked": self._mask_card_number(card_number)
                    }
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证信用卡号实体

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
        使用 Luhn 算法验证信用卡号

        Args:
            card_number: 信用卡号字符串

        Returns:
            True 表示校验通过，False 表示失败
        """
        # 必须全部是数字
        if not card_number.isdigit():
            return False

        # 长度必须在 13-19 之间
        if not 13 <= len(card_number) <= 19:
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

    @staticmethod
    def _identify_card_type(card_number: str) -> str:
        """
        识别信用卡类型

        Args:
            card_number: 信用卡号

        Returns:
            卡类型名称
        """
        for card_type, prefixes in CreditCardRecognizer._CARD_PREFIXES.items():
            for prefix in prefixes:
                if card_number.startswith(prefix):
                    return card_type

        return "Unknown"

    @staticmethod
    def _mask_card_number(card_number: str) -> str:
        """
        遮罩信用卡号（保留前4位和后4位）

        Args:
            card_number: 信用卡号

        Returns:
            遮罩后的卡号
        """
        if len(card_number) <= 8:
            return '*' * len(card_number)

        return f"{card_number[:4]}{'*' * (len(card_number) - 8)}{card_number[-4:]}"


class SSNRecognizer(BaseRecognizer):
    """
    美国社会安全号（SSN）识别器

    SSN 格式：XXX-XX-XXXX
    - 第一部分：3位数字（区域号）
    - 第二部分：2位数字（组号）
    - 第三部分：4位数字（序列号）

    Examples:
        >>> recognizer = SSNRecognizer(config)
        >>> entities = recognizer.detect("SSN: 123-45-6789")
    """

    def detect(self, text: str) -> List[Entity]:
        """
        检测文本中的 SSN

        Args:
            text: 待检测的文本

        Returns:
            检测到的 SSN 实体列表
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

                # 验证 SSN
                is_valid = self.validate_ssn(value)

                # 计算置信度
                confidence = self._calculate_confidence(
                    self.confidence_base,
                    text,
                    start,
                    validation_passed=is_valid
                )

                # 解析 SSN 组成部分
                area, group, serial = self._parse_ssn(value)

                # 创建实体
                entity = self._create_entity(
                    value=value,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence,
                    metadata={
                        "valid": is_valid,
                        "area": area,
                        "group": group,
                        "serial": serial,
                        "masked": self._mask_ssn(value)
                    }
                )
                entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """
        验证 SSN 实体

        Args:
            entity: 待验证的实体

        Returns:
            True 表示有效，False 表示无效
        """
        return self.validate_ssn(entity.value)

    @staticmethod
    def validate_ssn(ssn: str) -> bool:
        """
        验证 SSN 格式

        Args:
            ssn: SSN 字符串

        Returns:
            True 表示格式正确，False 表示错误
        """
        # 移除连字符
        ssn_digits = ssn.replace('-', '')

        # 必须是9位数字
        if len(ssn_digits) != 9 or not ssn_digits.isdigit():
            return False

        # 解析各部分
        area = int(ssn_digits[0:3])
        group = int(ssn_digits[3:5])
        serial = int(ssn_digits[5:9])

        # 区域号不能是 000、666 或 900-999
        if area == 0 or area == 666 or area >= 900:
            return False

        # 组号不能是 00
        if group == 0:
            return False

        # 序列号不能是 0000
        if serial == 0:
            return False

        return True

    @staticmethod
    def _parse_ssn(ssn: str) -> tuple:
        """
        解析 SSN 组成部分

        Args:
            ssn: SSN 字符串

        Returns:
            (area, group, serial) 元组
        """
        ssn_digits = ssn.replace('-', '')

        if len(ssn_digits) != 9:
            return "", "", ""

        return ssn_digits[0:3], ssn_digits[3:5], ssn_digits[5:9]

    @staticmethod
    def _mask_ssn(ssn: str) -> str:
        """
        遮罩 SSN（只显示后4位）

        Args:
            ssn: SSN 字符串

        Returns:
            遮罩后的 SSN
        """
        ssn_digits = ssn.replace('-', '')

        if len(ssn_digits) != 9:
            return '*' * len(ssn)

        return f"***-**-{ssn_digits[-4:]}"
