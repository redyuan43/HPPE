"""
上下文验证器 (Context Validator)

利用前后文信息验证实体的有效性，过滤不符合上下文的误检
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import re

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hppe.models.entity import Entity
from hppe.refiner.config import ValidatorConfig

logger = logging.getLogger(__name__)


# 上下文关键词库（覆盖17种PII类型）
CONTEXT_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    # 身份证件类
    "ID_CARD": {
        "positive": [
            "身份证", "身份证号", "身份证号码", "证件号", "证件号码",
            "ID", "id", "居民身份证", "二代身份证"
        ],
        "negative": [
            "价格", "金额", "数量", "编号", "序号", "货号", "订单号"
        ]
    },
    "PASSPORT": {
        "positive": [
            "护照", "护照号", "护照号码", "passport", "旅行证件",
            "因私护照", "因公护照"
        ],
        "negative": [
            "身份证", "驾驶证", "编号", "序号"
        ]
    },
    "DRIVER_LICENSE": {
        "positive": [
            "驾驶证", "驾照", "驾驶证号", "驾照号码", "driver", "license",
            "机动车驾驶证", "准驾证"
        ],
        "negative": [
            "身份证", "护照", "车牌", "车牌号"
        ]
    },
    "MILITARY_ID": {
        "positive": [
            "军官证", "士兵证", "军人证", "武警证", "部队证件"
        ],
        "negative": [
            "身份证", "护照", "工作证"
        ]
    },
    "SOCIAL_SECURITY": {
        "positive": [
            "社保", "社保号", "社会保障号", "社保卡号", "社会保险",
            "社保账号", "参保号"
        ],
        "negative": [
            "身份证", "银行卡", "编号"
        ]
    },

    # 金融类
    "BANK_CARD": {
        "positive": [
            "银行卡", "银行卡号", "卡号", "储蓄卡", "信用卡", "借记卡",
            "账号", "银行账号", "账户", "card", "bank"
        ],
        "negative": [
            "身份证", "电话", "手机号", "订单号", "快递单号"
        ]
    },
    "TAX_ID": {
        "positive": [
            "税号", "纳税人识别号", "税务登记号", "统一社会信用代码",
            "纳税识别号", "企业税号"
        ],
        "negative": [
            "身份证", "银行卡", "编号"
        ]
    },

    # 联系方式类
    "PHONE_NUMBER": {
        "positive": [
            "电话", "手机", "电话号码", "手机号", "联系方式", "联系电话",
            "移动电话", "座机", "tel", "phone", "mobile", "拨打", "致电"
        ],
        "negative": [
            "编号", "序号", "ID", "订单号", "快递单号", "身份证",
            "银行卡", "邮编", "价格"
        ]
    },
    "EMAIL": {
        "positive": [
            "邮箱", "电子邮件", "电子邮箱", "email", "e-mail", "mail",
            "邮件地址", "联系邮箱", "发送至", "抄送", "密送"
        ],
        "negative": [
            "网址", "网站", "链接", "域名"
        ]
    },

    # 地址类
    "ADDRESS": {
        "positive": [
            "地址", "住址", "家庭住址", "通讯地址", "联系地址",
            "收货地址", "详细地址", "所在地", "居住地", "address",
            "位于", "坐落于"
        ],
        "negative": [
            "网址", "邮箱", "email", "链接"
        ]
    },
    "POSTAL_CODE": {
        "positive": [
            "邮编", "邮政编码", "zip", "zipcode", "邮递区号", "邮区"
        ],
        "negative": [
            "价格", "金额", "数量", "元", "¥", "$", "电话", "手机",
            "编号", "序号", "身份证"
        ]
    },

    # 组织/人名
    "PERSON_NAME": {
        "positive": [
            "姓名", "名字", "联系人", "先生", "女士", "老师", "经理",
            "主任", "总监", "name", "我是", "叫", "本人"
        ],
        "negative": [
            "用户名", "账号", "昵称", "网名", "品牌", "公司", "单位"
        ]
    },
    "ORGANIZATION": {
        "positive": [
            "公司", "企业", "单位", "机构", "组织", "集团", "有限公司",
            "股份有限", "工作单位", "就职于", "供职于", "任职于"
        ],
        "negative": [
            "姓名", "人名", "先生", "女士", "个人"
        ]
    },

    # 技术标识类
    "IP_ADDRESS": {
        "positive": [
            "IP", "ip", "IP地址", "ip地址", "服务器", "主机", "网络地址",
            "内网", "外网", "公网IP", "私网IP"
        ],
        "negative": [
            "版本号", "价格", "金额", "电话"
        ]
    },
    "MAC_ADDRESS": {
        "positive": [
            "MAC", "mac", "MAC地址", "物理地址", "硬件地址", "网卡地址",
            "以太网地址"
        ],
        "negative": [
            "IP", "ip", "序列号", "编号"
        ]
    },
    "IMEI": {
        "positive": [
            "IMEI", "imei", "串号", "移动设备识别码", "手机串号",
            "设备识别码", "国际移动设备识别码"
        ],
        "negative": [
            "序列号", "SN", "产品编号", "订单号"
        ]
    },
    "VIN": {
        "positive": [
            "车架号", "VIN", "vin", "车辆识别代码", "车辆识别号",
            "底盘号", "17位码"
        ],
        "negative": [
            "车牌", "车牌号", "发动机号", "序列号"
        ]
    },
    "VEHICLE_PLATE": {
        "positive": [
            "车牌", "车牌号", "车号", "牌照", "号牌", "机动车号牌",
            "license plate", "车辆号牌"
        ],
        "negative": [
            "车架号", "VIN", "发动机号", "驾驶证"
        ]
    }
}


class ContextValidator:
    """
    上下文验证器

    利用前后文关键词验证实体的有效性，调整置信度或过滤误检
    """

    def __init__(self, config: ValidatorConfig = None):
        """
        初始化上下文验证器

        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or ValidatorConfig()
        self.patterns = CONTEXT_PATTERNS

        logger.info(
            f"上下文验证器已初始化: "
            f"window={self.config.context_window}, "
            f"enabled={self.config.enable_context_validation}"
        )

    def validate(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        验证实体列表

        Args:
            entities: 待验证的实体列表
            text: 原始文本

        Returns:
            验证后的实体列表
        """
        if not self.config.enable_context_validation:
            return entities

        if not entities or not text:
            return entities

        validated_entities = []

        for entity in entities:
            # 对低置信度实体进行上下文验证
            if entity.confidence < self.config.min_validation_confidence:
                validated_entity = self._validate_entity(entity, text)

                # 如果验证后置信度过低，则过滤掉
                if validated_entity and validated_entity.confidence > 0.3:
                    validated_entities.append(validated_entity)
                else:
                    logger.debug(
                        f"过滤低置信度实体: {entity.entity_type} "
                        f"'{entity.value}' (conf={entity.confidence:.2f})"
                    )
            else:
                # 高置信度实体直接保留
                validated_entities.append(entity)

        logger.debug(
            f"上下文验证完成: 输入{len(entities)}个实体，"
            f"输出{len(validated_entities)}个实体"
        )

        return validated_entities

    def _validate_entity(self, entity: Entity, text: str) -> Optional[Entity]:
        """
        验证单个实体

        Args:
            entity: 待验证的实体
            text: 原始文本

        Returns:
            验证后的实体（可能调整了置信度），如果验证失败则返回None
        """
        # 提取上下文
        context = self._extract_context(entity, text)

        # 获取该类型的关键词模式
        patterns = self.patterns.get(entity.entity_type)
        if not patterns:
            # 如果没有定义关键词，直接返回原实体
            return entity

        # 计算上下文得分
        context_score = self._calculate_context_score(
            context,
            patterns.get("positive", []),
            patterns.get("negative", [])
        )

        # 调整置信度
        new_confidence = self._adjust_confidence(
            entity.confidence,
            context_score
        )

        # 创建新实体（更新置信度）
        # 安全处理metadata（可能为None）
        base_metadata = entity.metadata if entity.metadata is not None else {}

        validated_entity = Entity(
            entity_type=entity.entity_type,
            value=entity.value,
            start_pos=entity.start_pos,
            end_pos=entity.end_pos,
            confidence=new_confidence,
            detection_method=entity.detection_method,
            recognizer_name=entity.recognizer_name,
            metadata={
                **base_metadata,
                "context_score": context_score,
                "original_confidence": entity.confidence
            }
        )

        logger.debug(
            f"上下文验证: {entity.entity_type} '{entity.value}' "
            f"conf={entity.confidence:.2f} → {new_confidence:.2f} "
            f"(context_score={context_score:+.2f})"
        )

        return validated_entity

    def _extract_context(self, entity: Entity, text: str) -> str:
        """
        提取实体的上下文

        Args:
            entity: 实体
            text: 原始文本

        Returns:
            上下文字符串
        """
        window = self.config.context_window

        # 计算上下文范围
        start = max(0, entity.start_pos - window)
        end = min(len(text), entity.end_pos + window)

        context = text[start:end]
        return context

    def _calculate_context_score(
        self,
        context: str,
        positive_keywords: List[str],
        negative_keywords: List[str]
    ) -> float:
        """
        计算上下文得分

        Args:
            context: 上下文字符串
            positive_keywords: 正向关键词列表
            negative_keywords: 负向关键词列表

        Returns:
            上下文得分（-1.0 到 +1.0）
        """
        context_lower = context.lower()

        # 统计正向关键词出现次数
        positive_count = 0
        for keyword in positive_keywords:
            keyword_lower = keyword.lower()
            # 简单substring匹配（适用于中文和英文）
            # 注意：中文关键词通常足够明确，不需要word boundary
            count = context_lower.count(keyword_lower)
            positive_count += count

        # 统计负向关键词出现次数
        negative_count = 0
        for keyword in negative_keywords:
            keyword_lower = keyword.lower()
            count = context_lower.count(keyword_lower)
            negative_count += count

        # 计算得分：正向 - 负向
        # 归一化到 -1.0 到 +1.0
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0

        score = (positive_count - negative_count) / total_count
        return score

    def _adjust_confidence(
        self,
        original_confidence: float,
        context_score: float
    ) -> float:
        """
        根据上下文得分调整置信度

        Args:
            original_confidence: 原始置信度
            context_score: 上下文得分 (-1.0 到 +1.0)

        Returns:
            调整后的置信度
        """
        if context_score > 0:
            # 正向上下文：提升置信度
            boost = context_score * self.config.context_confidence_boost
            new_confidence = min(1.0, original_confidence + boost)
        elif context_score < 0:
            # 负向上下文：降低置信度
            penalty = abs(context_score) * self.config.context_confidence_penalty
            new_confidence = max(0.0, original_confidence - penalty)
        else:
            # 中性上下文：保持不变
            new_confidence = original_confidence

        return new_confidence

    def get_info(self) -> Dict:
        """
        获取验证器信息

        Returns:
            信息字典
        """
        return {
            "name": "ContextValidator",
            "config": {
                "context_window": self.config.context_window,
                "enable_context_validation": self.config.enable_context_validation,
                "min_validation_confidence": self.config.min_validation_confidence,
                "context_confidence_boost": self.config.context_confidence_boost,
                "context_confidence_penalty": self.config.context_confidence_penalty
            },
            "num_pii_types_covered": len(self.patterns)
        }

    def __repr__(self) -> str:
        return (
            f"ContextValidator(enabled={self.config.enable_context_validation}, "
            f"window={self.config.context_window})"
        )
