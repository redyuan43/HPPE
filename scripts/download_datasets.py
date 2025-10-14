#!/usr/bin/env python3
"""
下载 PII 检测训练数据集

支持的数据集：
1. ai4privacy/pii-masking-200k - 200k PII脱敏数据集
2. levow/msra_ner - MSRA中文NER数据集
3. bigcode/bigcode-pii-dataset - 代码中的PII数据集

使用示例：
    # 下载所有推荐数据集
    python scripts/download_datasets.py --all

    # 下载特定数据集
    python scripts/download_datasets.py --datasets ai4privacy msra

    # 指定输出目录
    python scripts/download_datasets.py --all --output data/pii_datasets
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset
    from huggingface_hub import login
except ImportError:
    print("❌ 缺少依赖包。请先安装：")
    print("   pip install datasets huggingface_hub")
    sys.exit(1)


DATASET_CONFIGS = {
    "ai4privacy": {
        "hf_path": "ai4privacy/pii-masking-200k",
        "description": "PII脱敏数据集 (200k样本)",
        "size_mb": 450,
        "languages": ["en", "zh"],
        "pii_types": ["NAME", "EMAIL", "PHONE", "ADDRESS", "ID_CARD"]
    },
    "msra": {
        "hf_path": "levow/msra_ner",
        "description": "MSRA中文NER数据集 (50k+样本)",
        "size_mb": 25,
        "languages": ["zh"],
        "pii_types": ["PERSON_NAME", "LOCATION", "ORGANIZATION"]
    },
    "bigcode": {
        "hf_path": "bigcode/bigcode-pii-dataset",
        "description": "BigCode代码PII数据集 (12k样本)",
        "size_mb": 120,
        "languages": ["code"],
        "pii_types": ["EMAIL", "USERNAME", "IP_ADDRESS", "KEY", "PASSWORD"]
    }
}


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    show_progress: bool = True
) -> bool:
    """
    下载单个数据集

    Args:
        dataset_name: 数据集名称 (ai4privacy, msra, bigcode)
        output_dir: 输出目录
        show_progress: 是否显示进度

    Returns:
        是否成功
    """
    if dataset_name not in DATASET_CONFIGS:
        print(f"❌ 未知数据集: {dataset_name}")
        print(f"   支持的数据集: {', '.join(DATASET_CONFIGS.keys())}")
        return False

    config = DATASET_CONFIGS[dataset_name]
    hf_path = config["hf_path"]

    print(f"\n{'='*70}")
    print(f"下载数据集: {dataset_name}")
    print(f"{'='*70}")
    print(f"描述: {config['description']}")
    print(f"大小: ~{config['size_mb']} MB")
    print(f"语言: {', '.join(config['languages'])}")
    print(f"PII类型: {', '.join(config['pii_types'])}")
    print(f"Hugging Face路径: {hf_path}")
    print()

    try:
        # 创建输出目录
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # 下载数据集
        print(f"正在下载 {hf_path}...")
        dataset = load_dataset(hf_path)

        # 保存到本地
        print(f"保存到: {dataset_output_dir}")
        dataset.save_to_disk(str(dataset_output_dir))

        # 显示统计信息
        print(f"\n✓ 下载完成！")
        print(f"\n数据集统计:")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 样本")

        # 显示示例
        if "train" in dataset:
            example = dataset["train"][0]
            print(f"\n示例数据:")
            for key, value in list(example.items())[:3]:
                value_str = str(value)[:100]
                if len(str(value)) > 100:
                    value_str += "..."
                print(f"  {key}: {value_str}")

        return True

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_multiple_datasets(
    dataset_names: List[str],
    output_dir: Path,
    show_progress: bool = True
) -> dict:
    """
    下载多个数据集

    Returns:
        下载结果统计 {dataset_name: success}
    """
    results = {}

    print(f"\n{'='*70}")
    print(f"批量下载数据集")
    print(f"{'='*70}")
    print(f"目标数据集: {', '.join(dataset_names)}")
    print(f"输出目录: {output_dir}")
    print()

    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"\n[{i}/{len(dataset_names)}] 处理: {dataset_name}")
        success = download_dataset(dataset_name, output_dir, show_progress)
        results[dataset_name] = success

    # 总结
    print(f"\n{'='*70}")
    print("下载总结")
    print(f"{'='*70}")

    for dataset_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{dataset_name}: {status}")

    successful = sum(1 for s in results.values() if s)
    total = len(results)
    print(f"\n成功: {successful}/{total}")

    if successful == total:
        print("\n🎉 所有数据集下载完成！")
    else:
        print(f"\n⚠️  {total - successful} 个数据集下载失败")

    return results


def list_available_datasets():
    """列出可用的数据集"""
    print(f"\n{'='*70}")
    print("可用的 PII 数据集")
    print(f"{'='*70}\n")

    for name, config in DATASET_CONFIGS.items():
        print(f"📦 {name}")
        print(f"   描述: {config['description']}")
        print(f"   路径: {config['hf_path']}")
        print(f"   大小: ~{config['size_mb']} MB")
        print(f"   语言: {', '.join(config['languages'])}")
        print(f"   PII类型: {', '.join(config['pii_types'])}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="下载 PII 检测训练数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载所有推荐数据集
  python scripts/download_datasets.py --all

  # 下载特定数据集
  python scripts/download_datasets.py --datasets ai4privacy msra

  # 列出可用数据集
  python scripts/download_datasets.py --list

  # 指定输出目录
  python scripts/download_datasets.py --all --output data/pii_datasets
        """
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        help="要下载的数据集名称"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有推荐数据集 (ai4privacy + msra)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用的数据集"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/pii_datasets",
        help="输出目录 (默认: data/pii_datasets)"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token (某些数据集需要)"
    )

    args = parser.parse_args()

    # 列出数据集
    if args.list:
        list_available_datasets()
        return

    # 确定要下载的数据集
    if args.all:
        # 推荐的数据集组合
        dataset_names = ["ai4privacy", "msra"]
    elif args.datasets:
        dataset_names = args.datasets
    else:
        parser.print_help()
        return

    # 登录 Hugging Face（如果提供了token）
    if args.hf_token:
        print("登录 Hugging Face...")
        login(token=args.hf_token)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 下载数据集
    results = download_multiple_datasets(dataset_names, output_dir)

    # 退出码
    all_successful = all(results.values())
    sys.exit(0 if all_successful else 1)


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
