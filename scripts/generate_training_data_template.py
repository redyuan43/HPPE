#!/usr/bin/env python3
"""
生成标准化的训练数据模板

使用标准PII类型生成训练数据，供模型fine-tuning使用
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.models.pii_types import PIIType, ALL_17_TYPES, get_type_display_name


def generate_sample_data(pii_type: PIIType) -> List[Dict]:
    """
    为每种PII类型生成示例训练数据

    Args:
        pii_type: PII类型

    Returns:
        训练样本列表
    """
    samples = []

    # 定义各类型的示例数据
    example_data = {
        PIIType.PERSON_NAME: [
            {"text": "我是张三，来自北京", "value": "张三", "context": "self-introduction"},
            {"text": "李四是我们部门的经理", "value": "李四", "context": "workplace"},
            {"text": "王五女士今天不在", "value": "王五", "context": "absence"},
        ],
        PIIType.PHONE_NUMBER: [
            {"text": "请拨打13812345678联系我", "value": "13812345678", "context": "contact"},
            {"text": "电话：010-12345678", "value": "010-12345678", "context": "landline"},
            {"text": "手机号码是13900139000", "value": "13900139000", "context": "mobile"},
        ],
        PIIType.EMAIL: [
            {"text": "我的邮箱是zhangsan@example.com", "value": "zhangsan@example.com", "context": "personal"},
            {"text": "请发送至contact@company.com", "value": "contact@company.com", "context": "business"},
        ],
        PIIType.ADDRESS: [
            {"text": "公司地址：北京市海淀区中关村大街1号", "value": "北京市海淀区中关村大街1号", "context": "business"},
            {"text": "家住上海市浦东新区", "value": "上海市浦东新区", "context": "residential"},
        ],
        PIIType.ORGANIZATION: [
            {"text": "阿里巴巴集团控股有限公司成立于1999年", "value": "阿里巴巴集团控股有限公司", "context": "corporation"},
            {"text": "在腾讯科技有限公司工作", "value": "腾讯科技有限公司", "context": "employment"},
        ],
        PIIType.ID_CARD: [
            {"text": "身份证号：110101199001011234", "value": "110101199001011234", "context": "official"},
            {"text": "申请人身份证330106199505051234", "value": "330106199505051234", "context": "application"},
        ],
        PIIType.BANK_CARD: [
            {"text": "银行卡号：6222021234567890123", "value": "6222021234567890123", "context": "payment"},
            {"text": "账户：6228480123456789012", "value": "6228480123456789012", "context": "account"},
        ],
        PIIType.PASSPORT: [
            {"text": "护照号码：E12345678", "value": "E12345678", "context": "travel"},
            {"text": "持有护照E23456789", "value": "E23456789", "context": "immigration"},
        ],
        PIIType.DRIVER_LICENSE: [
            {"text": "驾驶证号：110101199001011234", "value": "110101199001011234", "context": "licensing"},
            {"text": "驾照：320105199201011234，A2级", "value": "320105199201011234", "context": "qualification"},
        ],
        PIIType.VEHICLE_PLATE: [
            {"text": "车牌号：京A12345，请注意", "value": "京A12345", "context": "parking"},
            {"text": "车辆信息：沪B88888", "value": "沪B88888", "context": "registration"},
        ],
        PIIType.IP_ADDRESS: [
            {"text": "IP地址192.168.1.1访问异常", "value": "192.168.1.1", "context": "network"},
            {"text": "服务器IP: 10.0.0.1", "value": "10.0.0.1", "context": "server"},
        ],
        PIIType.MAC_ADDRESS: [
            {"text": "MAC地址：00:1A:2B:3C:4D:5E", "value": "00:1A:2B:3C:4D:5E", "context": "device"},
            {"text": "网卡MAC: F0:DE:F1:12:34:56", "value": "F0:DE:F1:12:34:56", "context": "hardware"},
        ],
        PIIType.POSTAL_CODE: [
            {"text": "邮编：100000，北京市", "value": "100000", "context": "address"},
            {"text": "邮政编码100080", "value": "100080", "context": "postal"},
        ],
        PIIType.IMEI: [
            {"text": "IMEI号：123456789012345，请记录", "value": "123456789012345", "context": "device"},
            {"text": "手机IMEI：862123456789012", "value": "862123456789012", "context": "mobile"},
        ],
        PIIType.VIN: [
            {"text": "VIN码：LGBF53E02CR123456", "value": "LGBF53E02CR123456", "context": "vehicle"},
            {"text": "车架号：LFV3A23K8D3123456", "value": "LFV3A23K8D3123456", "context": "registration"},
        ],
        PIIType.TAX_ID: [
            {"text": "税号：110101199001011234567", "value": "110101199001011234567", "context": "taxation"},
            {"text": "统一社会信用代码：91110108123456789X", "value": "91110108123456789X", "context": "business"},
        ],
        PIIType.SOCIAL_SECURITY: [
            {"text": "社保号：110108199001011234", "value": "110108199001011234", "context": "welfare"},
            {"text": "社保账户：110101198001011234", "value": "110101198001011234", "context": "insurance"},
        ],
        PIIType.MILITARY_ID: [
            {"text": "军官证号：军字第1234567号", "value": "军字第1234567号", "context": "military"},
            {"text": "军官证：军字1234567号", "value": "军字1234567号", "context": "identification"},
        ],
    }

    # 获取该类型的示例
    if pii_type in example_data:
        for example in example_data[pii_type]:
            # 查找value在text中的位置
            text = example["text"]
            value = example["value"]
            start = text.find(value)

            if start >= 0:
                samples.append({
                    "text": text,
                    "entities": [{
                        "type": pii_type.value,  # 使用标准类型名称
                        "value": value,
                        "start": start,
                        "end": start + len(value),
                        "confidence": 1.0
                    }],
                    "metadata": {
                        "context": example["context"],
                        "language": "zh"
                    }
                })

    return samples


def generate_full_dataset(output_file: Path, samples_per_type: int = 3):
    """
    生成完整的训练数据集

    Args:
        output_file: 输出文件路径
        samples_per_type: 每种类型生成的样本数
    """
    all_samples = []

    print(f"🔄 生成17种PII类型的训练数据...")

    for pii_type in ALL_17_TYPES:
        type_name = pii_type.value
        display_name = get_type_display_name(type_name)

        print(f"  - {type_name:20s} ({display_name})", end=" ")

        samples = generate_sample_data(pii_type)

        if samples:
            # 限制样本数量
            samples = samples[:samples_per_type]
            all_samples.extend(samples)
            print(f"✅ {len(samples)} 个样本")
        else:
            print(f"⚠️  无示例数据")

    # 打乱顺序
    random.shuffle(all_samples)

    # 写入文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ 生成完成！")
    print(f"   总样本数: {len(all_samples)}")
    print(f"   输出文件: {output_file}")


def main():
    """主函数"""
    output_file = project_root / "data" / "training" / "17pii_training_template.jsonl"

    print("=" * 70)
    print("📝 生成标准化训练数据模板")
    print("=" * 70)
    print(f"\n输出文件: {output_file}\n")

    generate_full_dataset(output_file, samples_per_type=3)

    # 显示示例
    print(f"\n📋 前3个样本示例:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample = json.loads(line)
            print(f"\n样本 {i+1}:")
            print(f"  文本: {sample['text']}")
            print(f"  实体: {sample['entities'][0]['type']} -> {sample['entities'][0]['value']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
