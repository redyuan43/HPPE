"""
PII 识别器实现

包含各种类型的 PII 识别器
"""

from hppe.engines.regex.recognizers.china_pii import (
    ChinaIDCardRecognizer,
    ChinaPhoneRecognizer,
    ChinaBankCardRecognizer,
    ChinaPassportRecognizer,
)

from hppe.engines.regex.recognizers.global_pii import (
    EmailRecognizer,
    IPAddressRecognizer,
    URLRecognizer,
    CreditCardRecognizer,
    SSNRecognizer,
)

__all__ = [
    # 中国 PII
    "ChinaIDCardRecognizer",
    "ChinaPhoneRecognizer",
    "ChinaBankCardRecognizer",
    "ChinaPassportRecognizer",
    # 全球 PII
    "EmailRecognizer",
    "IPAddressRecognizer",
    "URLRecognizer",
    "CreditCardRecognizer",
    "SSNRecognizer",
]
