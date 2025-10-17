#!/usr/bin/env python3
"""
规则增强PII检测器
结合模型预测 + 正则规则，提升Recall
"""
import re
import json
from typing import List, Dict, Set, Tuple

class RuleEnhancedPIIDetector:
    """规则增强的PII检测器"""

    def __init__(self):
        # 编译正则表达式（提升性能）
        self.patterns = {
            # 手机号：1开头的11位数字
            'PHONE_NUMBER': re.compile(r'1[3-9]\d{9}'),

            # 身份证：18位数字或17位数字+X
            'ID_CARD': re.compile(r'\d{17}[\dXx]'),

            # 邮箱：标准邮箱格式
            'EMAIL': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),

            # 银行卡：13-19位数字
            'BANK_CARD': re.compile(r'\b\d{13,19}\b'),

            # 中国车牌号
            'VEHICLE_PLATE': re.compile(r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{5,6}'),

            # IP地址
            'IP_ADDRESS': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),

            # 中国邮政编码
            'POSTAL_CODE': re.compile(r'\b\d{6}\b'),

            # 统一社会信用代码
            'UNIFIED_SOCIAL_CREDIT_CODE': re.compile(r'[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}'),
        }

        # 中国姓氏（前100常见姓）
        self.common_surnames = {
            '王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴',
            '徐', '孙', '马', '朱', '胡', '郭', '何', '高', '林', '罗',
            '郑', '梁', '谢', '宋', '唐', '许', '韩', '冯', '邓', '曹',
            '彭', '曾', '肖', '田', '董', '袁', '潘', '于', '蒋', '蔡',
            '余', '杜', '叶', '程', '苏', '魏', '吕', '丁', '任', '沈',
            '姚', '卢', '姜', '崔', '钟', '谭', '陆', '汪', '范', '金',
            '石', '廖', '贾', '夏', '韦', '付', '方', '白', '邹', '孟',
            '熊', '秦', '邱', '江', '尹', '薛', '闫', '段', '雷', '侯',
            '龙', '史', '陶', '黎', '贺', '顾', '毛', '郝', '龚', '邵',
            '万', '钱', '严', '覃', '武', '戴', '莫', '孔', '向', '汤'
        }

        # 地址关键词
        self.address_keywords = {
            '省', '市', '县', '区', '镇', '乡', '村', '街道', '路', '号',
            '栋', '单元', '室', '楼', '层', '小区', '花园', '广场', '大厦',
            '中心', '园区', '工业区', '开发区'
        }

        # 机构关键词
        self.org_keywords = {
            '公司', '集团', '企业', '有限', '股份', '科技', '实业', '贸易',
            '银行', '医院', '学校', '大学', '学院', '研究所', '中心', '局',
            '部', '委', '厅', '署', '队', '所', '站', '馆', '院', '社',
            '会', '协会', '基金会', '联盟', '组织', '机构'
        }

    def extract_by_rules(self, text: str) -> List[Dict]:
        """使用规则提取PII实体"""
        entities = []

        # 1. 使用正则表达式提取
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                value = match.group()
                start = match.start()
                end = match.end()

                # 额外验证
                if entity_type == 'ID_CARD' and not self._validate_id_card(value):
                    continue
                if entity_type == 'BANK_CARD' and not self._validate_bank_card(value):
                    continue
                if entity_type == 'POSTAL_CODE' and not self._validate_postal_code(text, start, end):
                    continue

                entities.append({
                    'type': entity_type,
                    'value': value,
                    'start': start,
                    'end': end,
                    'source': 'rule'
                })

        # 2. 基于启发式提取人名
        person_names = self._extract_person_names(text)
        for name, start, end in person_names:
            entities.append({
                'type': 'PERSON_NAME',
                'value': name,
                'start': start,
                'end': end,
                'source': 'rule'
            })

        # 3. 基于关键词提取地址片段
        addresses = self._extract_addresses(text)
        for addr, start, end in addresses:
            entities.append({
                'type': 'ADDRESS',
                'value': addr,
                'start': start,
                'end': end,
                'source': 'rule'
            })

        # 4. 基于关键词提取机构名称
        orgs = self._extract_organizations(text)
        for org, start, end in orgs:
            entities.append({
                'type': 'ORGANIZATION',
                'value': org,
                'start': start,
                'end': end,
                'source': 'rule'
            })

        return entities

    def _validate_id_card(self, id_card: str) -> bool:
        """验证身份证号码"""
        if len(id_card) != 18:
            return False

        # 验证前17位是否为数字
        if not id_card[:17].isdigit():
            return False

        # 验证最后一位
        if not (id_card[17].isdigit() or id_card[17].upper() == 'X'):
            return False

        # 简单校验：年份合理性
        year = int(id_card[6:10])
        if year < 1900 or year > 2024:
            return False

        return True

    def _validate_bank_card(self, card: str) -> bool:
        """验证银行卡号（简单校验）"""
        # 长度检查
        if len(card) < 13 or len(card) > 19:
            return False

        # 不能全是相同数字
        if len(set(card)) == 1:
            return False

        return True

    def _validate_postal_code(self, text: str, start: int, end: int) -> bool:
        """验证邮政编码（避免误报普通6位数字）"""
        # 检查前后是否有"邮编"、"邮政编码"等关键词
        context_start = max(0, start - 10)
        context_end = min(len(text), end + 10)
        context = text[context_start:context_end]

        keywords = ['邮编', '邮政编码', 'zip', 'postal']
        return any(kw in context.lower() for kw in keywords)

    def _extract_person_names(self, text: str) -> List[Tuple[str, int, int]]:
        """提取可能的人名"""
        names = []

        # 匹配：常见姓 + 1-3个汉字
        pattern = r'[' + ''.join(self.common_surnames) + r'][\u4e00-\u9fa5]{1,3}'

        for match in re.finditer(pattern, text):
            value = match.group()
            start = match.start()
            end = match.end()

            # 长度过滤：2-4字
            if len(value) < 2 or len(value) > 4:
                continue

            # 避免误报：检查前后字符
            if start > 0 and text[start-1] in '的之':
                continue
            if end < len(text) and text[end] in '的之':
                continue  # 修复："张三的" → 只取"张三"

            names.append((value, start, end))

        return names

    def _extract_addresses(self, text: str) -> List[Tuple[str, int, int]]:
        """提取地址片段"""
        addresses = []

        # 匹配包含地址关键词的片段
        for keyword in self.address_keywords:
            pattern = r'[\u4e00-\u9fa5\d]{2,20}' + re.escape(keyword)
            for match in re.finditer(pattern, text):
                value = match.group()
                start = match.start()
                end = match.end()

                # 过滤误报："手机号"、"身份证号"
                if keyword == '号' and end < len(text):
                    # 检查前面是否有"电话"、"手机"、"身份证"等
                    prefix_keywords = ['电话', '手机', '证件', '身份', '证号', '卡号']
                    if any(pk in value for pk in prefix_keywords):
                        continue

                addresses.append((value, start, end))

        return addresses

    def _extract_organizations(self, text: str) -> List[Tuple[str, int, int]]:
        """提取机构名称"""
        orgs = []

        # 匹配包含机构关键词的片段
        for keyword in self.org_keywords:
            pattern = r'[\u4e00-\u9fa5\w]{2,30}' + re.escape(keyword)
            for match in re.finditer(pattern, text):
                value = match.group()
                start = match.start()
                end = match.end()

                # 长度过滤
                if len(value) < 3 or len(value) > 30:
                    continue

                orgs.append((value, start, end))

        return orgs

    def merge_entities(self, model_entities: List[Dict], rule_entities: List[Dict]) -> List[Dict]:
        """合并模型预测和规则提取的实体（去重）"""
        # 转换为集合形式用于去重
        entity_set = set()
        merged = []

        # 优先添加模型预测的实体
        for entity in model_entities:
            key = (entity['type'], entity['value'])
            if key not in entity_set:
                entity_set.add(key)
                entity['source'] = 'model'
                merged.append(entity)

        # 添加规则提取的新实体
        for entity in rule_entities:
            key = (entity['type'], entity['value'])
            if key not in entity_set:
                entity_set.add(key)
                merged.append(entity)

        return merged

# 使用示例
if __name__ == "__main__":
    detector = RuleEnhancedPIIDetector()

    # 测试文本
    test_text = "张三的手机号是13812345678，身份证号是110101199001011234，住在北京市朝阳区建国路1号。"

    # 提取规则实体
    rule_entities = detector.extract_by_rules(test_text)

    print("规则提取的实体：")
    for entity in rule_entities:
        print(f"  {entity['type']}: {entity['value']}")
