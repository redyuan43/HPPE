#!/usr/bin/env python3
"""
生成合成 PII 数据用于训练

使用 Faker 库生成中文 PII 数据，包括：
- 姓名 (PERSON_NAME)
- 电话 (PHONE_NUMBER)
- 邮箱 (EMAIL)
- 身份证 (ID_CARD)
- 地址 (ADDRESS)
- 组织 (ORGANIZATION)

生成的数据格式符合 SFT 训练要求。

使用示例：
    # 生成 10,000 条中文数据
    python scripts/generate_synthetic_pii.py --num-samples 10000 --language zh_CN

    # 生成指定类型的数据
    python scripts/generate_synthetic_pii.py --num-samples 5000 --pii-types PERSON_NAME PHONE_NUMBER

    # 指定输出文件
    python scripts/generate_synthetic_pii.py --num-samples 10000 --output data/synthetic_pii.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    from faker import Faker
except ImportError:
    print("❌ 缺少 Faker 库。请先安装：")
    print("   pip install faker")
    sys.exit(1)


# 文本模板
TEMPLATES = {
    "PERSON_NAME": [
        "我叫{name}。",
        "{name}是我的名字。",
        "联系人：{name}",
        "姓名：{name}",
        "我是{name}，很高兴认识你。",
        "{name}负责这个项目。",
        "请找{name}处理。",
        "经办人：{name}"
    ],
    "PHONE_NUMBER": [
        "我的电话是{phone}。",
        "联系电话：{phone}",
        "手机号：{phone}",
        "请拨打{phone}联系我。",
        "电话号码为{phone}。",
        "致电{phone}咨询详情。",
        "客服热线：{phone}",
        "紧急联系方式：{phone}"
    ],
    "EMAIL": [
        "我的邮箱是{email}。",
        "邮箱地址：{email}",
        "请发送至{email}。",
        "联系邮箱：{email}",
        "电子邮件：{email}",
        "回复到{email}。",
        "商务合作：{email}",
        "投递简历至：{email}"
    ],
    "ID_CARD": [
        "我的身份证号是{id_card}。",
        "身份证：{id_card}",
        "证件号码：{id_card}",
        "身份证号码为{id_card}。",
        "请提供身份证{id_card}进行验证。"
    ],
    "ADDRESS": [
        "我住在{address}。",
        "地址：{address}",
        "收货地址：{address}",
        "居住地址为{address}。",
        "公司位于{address}。",
        "办公地点：{address}",
        "送货至{address}。"
    ],
    "ORGANIZATION": [
        "我在{organization}工作。",
        "{organization}是我的公司。",
        "就职于{organization}。",
        "单位：{organization}",
        "{organization}诚聘英才。",
        "供应商：{organization}",
        "合作企业：{organization}"
    ],
    "MIXED": [
        "我叫{name}，在{organization}工作，电话{phone}。",
        "联系人{name}，邮箱{email}，手机{phone}。",
        "{name}来自{organization}，地址{address}。",
        "姓名：{name}\n单位：{organization}\n电话：{phone}\n邮箱：{email}",
        "{organization}的{name}，联系方式{phone}，邮箱{email}。",
        "我是{name}，在{address}的{organization}上班，手机{phone}。"
    ]
}


def generate_chinese_id_card() -> str:
    """生成符合规则的中文身份证号"""
    # 地区码（随机选择）
    area_codes = [
        "110101",  # 北京东城
        "310101",  # 上海黄浦
        "440106",  # 广州天河
        "440305",  # 深圳南山
        "330106",  # 杭州西湖
        "510104",  # 成都锦江
    ]
    area_code = random.choice(area_codes)

    # 出生日期（1950-2005）
    year = random.randint(1950, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    birth_date = f"{year:04d}{month:02d}{day:02d}"

    # 顺序码（随机）
    sequence = f"{random.randint(0, 999):03d}"

    # 前17位
    id_17 = area_code + birth_date + sequence

    # 计算校验码
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

    total = sum(int(id_17[i]) * weights[i] for i in range(17))
    check_code = check_codes[total % 11]

    return id_17 + check_code


def generate_sample(
    fake: Faker,
    pii_types: List[str],
    include_mixed: bool = True
) -> Dict[str, Any]:
    """
    生成单个训练样本

    Args:
        fake: Faker实例
        pii_types: 要生成的PII类型列表
        include_mixed: 是否包含混合类型样本

    Returns:
        SFT格式的训练样本
    """
    # 决定生成类型
    if include_mixed and random.random() < 0.3:
        # 30% 概率生成混合类型
        template_type = "MIXED"
        selected_pii_types = random.sample(pii_types, min(len(pii_types), random.randint(2, 4)))
    else:
        # 单一类型
        template_type = random.choice(pii_types)
        selected_pii_types = [template_type]

    # 生成PII值
    pii_values = {}
    entities = []

    for pii_type in selected_pii_types:
        if pii_type == "PERSON_NAME":
            value = fake.name()
            pii_values["name"] = value
        elif pii_type == "PHONE_NUMBER":
            # 生成中国手机号
            value = f"1{random.choice([3,4,5,6,7,8,9])}{random.randint(0,9)}{random.randint(10000000,99999999)}"
            pii_values["phone"] = value
        elif pii_type == "EMAIL":
            value = fake.email()
            pii_values["email"] = value
        elif pii_type == "ID_CARD":
            value = generate_chinese_id_card()
            pii_values["id_card"] = value
        elif pii_type == "ADDRESS":
            value = fake.address().replace("\n", "")
            pii_values["address"] = value
        elif pii_type == "ORGANIZATION":
            value = fake.company()
            pii_values["organization"] = value

    # 选择模板
    template = random.choice(TEMPLATES[template_type])

    # 生成文本
    try:
        text = template.format(**pii_values)
    except KeyError:
        # 如果模板需要的字段不存在，回退到简单模板
        text = template.format(**{k: pii_values.get(k, "") for k in ["name", "phone", "email", "id_card", "address", "organization"]})

    # 提取实体位置
    for pii_type, key in [
        ("PERSON_NAME", "name"),
        ("PHONE_NUMBER", "phone"),
        ("EMAIL", "email"),
        ("ID_CARD", "id_card"),
        ("ADDRESS", "address"),
        ("ORGANIZATION", "organization")
    ]:
        if key in pii_values:
            value = pii_values[key]
            # 查找所有出现位置
            for match in re.finditer(re.escape(value), text):
                entities.append({
                    "type": pii_type,
                    "value": value,
                    "start": match.start(),
                    "end": match.end()
                })

    # 构建SFT格式
    sample = {
        "instruction": "检测以下文本中的 PII，并以 JSON 格式输出实体列表。",
        "input": text,
        "output": {
            "entities": entities
        }
    }

    return sample


def generate_dataset(
    num_samples: int,
    language: str = "zh_CN",
    pii_types: List[str] = None,
    output_path: Path = None,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    生成完整数据集

    Args:
        num_samples: 样本数量
        language: 语言代码
        pii_types: 要生成的PII类型
        output_path: 输出文件路径
        show_progress: 是否显示进度

    Returns:
        生成的样本列表
    """
    if pii_types is None:
        pii_types = ["PERSON_NAME", "PHONE_NUMBER", "EMAIL", "ID_CARD", "ADDRESS", "ORGANIZATION"]

    print(f"\n{'='*70}")
    print("生成合成 PII 数据")
    print(f"{'='*70}")
    print(f"样本数量: {num_samples}")
    print(f"语言: {language}")
    print(f"PII类型: {', '.join(pii_types)}")
    if output_path:
        print(f"输出文件: {output_path}")
    print()

    # 初始化Faker
    fake = Faker(language)

    # 生成样本
    samples = []
    print("正在生成样本...")

    for i in range(num_samples):
        if show_progress and (i + 1) % 1000 == 0:
            print(f"  进度: {i + 1}/{num_samples}", end="\r")

        sample = generate_sample(fake, pii_types, include_mixed=True)
        samples.append(sample)

    if show_progress:
        print(f"  进度: {num_samples}/{num_samples}")

    # 统计
    print(f"\n✓ 生成完成！")
    print(f"\n数据集统计:")
    print(f"  总样本数: {len(samples)}")

    # 统计实体类型
    entity_type_counts = {}
    total_entities = 0
    for sample in samples:
        for entity in sample["output"]["entities"]:
            entity_type = entity["type"]
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            total_entities += 1

    print(f"  总实体数: {total_entities}")
    print(f"  平均实体/样本: {total_entities / len(samples):.2f}")
    print(f"\n  实体类型分布:")
    for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_entities * 100
        print(f"    {entity_type}: {count} ({percentage:.1f}%)")

    # 保存到文件
    if output_path:
        print(f"\n保存到文件...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".jsonl":
            # JSONL格式（每行一个JSON）
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            # JSON格式
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"✓ 已保存到: {output_path}")

    # 显示示例
    print(f"\n示例数据:")
    example = samples[0]
    print(f"  输入: {example['input']}")
    print(f"  实体数: {len(example['output']['entities'])}")
    print(f"  实体: {[(e['type'], e['value']) for e in example['output']['entities'][:3]]}")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="生成合成 PII 数据用于训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 10,000 条中文数据
  python scripts/generate_synthetic_pii.py --num-samples 10000

  # 生成指定类型的数据
  python scripts/generate_synthetic_pii.py --num-samples 5000 --pii-types PERSON_NAME PHONE_NUMBER

  # 指定输出格式
  python scripts/generate_synthetic_pii.py --num-samples 10000 --output data/synthetic.jsonl
        """
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="生成的样本数量 (默认: 10000)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="zh_CN",
        help="语言代码 (默认: zh_CN)"
    )

    parser.add_argument(
        "--pii-types",
        nargs="+",
        choices=["PERSON_NAME", "PHONE_NUMBER", "EMAIL", "ID_CARD", "ADDRESS", "ORGANIZATION"],
        help="要生成的PII类型（默认：全部）"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/pii_datasets/synthetic_pii.jsonl",
        help="输出文件路径 (默认: data/pii_datasets/synthetic_pii.jsonl)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="随机种子（用于可重现生成）"
    )

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        Faker.seed(args.seed)

    # 生成数据集
    output_path = Path(args.output)
    samples = generate_dataset(
        num_samples=args.num_samples,
        language=args.language,
        pii_types=args.pii_types,
        output_path=output_path,
        show_progress=True
    )

    print(f"\n🎉 完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
