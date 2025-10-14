#!/usr/bin/env python3
"""
Qwen3 模型下载脚本

自动下载 Qwen/Qwen2.5-7B-Instruct 模型及量化版本
支持断点续传、进度显示、校验

使用方法:
    # 标准 FP16 模型（约 14GB）
    python scripts/download_model.py

    # 4-bit AWQ 量化模型（约 4.5GB，推荐）
    python scripts/download_model.py --quantization awq --bits 4

    # 自定义保存路径
    python scripts/download_model.py --output-dir /path/to/models
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Qwen 模型下载器"""

    # 支持的模型配置
    MODELS = {
        "qwen2.5-7b": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "size_gb": 14.0,
            "description": "Qwen2.5 7B 指令微调版（FP16）",
        },
        "qwen2.5-7b-awq": {
            "name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "size_gb": 4.5,
            "description": "Qwen2.5 7B AWQ 4-bit 量化版",
        },
        "qwen2.5-7b-gptq": {
            "name": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "size_gb": 4.5,
            "description": "Qwen2.5 7B GPTQ 4-bit 量化版",
        },
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: Optional[str] = None,
        token: Optional[str] = None,
        quantization: Optional[str] = None,
        bits: int = 4,
    ):
        """
        初始化下载器

        Args:
            model_name: HuggingFace 模型名称
            output_dir: 模型保存目录（默认: ~/.cache/huggingface/hub）
            token: HuggingFace token（可选）
            quantization: 量化方法 ('awq' 或 'gptq')
            bits: 量化位数（默认 4）
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.token = token
        self.quantization = quantization
        self.bits = bits

        # 自动选择量化模型
        if quantization:
            self.model_name = self._get_quantized_model_name(
                model_name, quantization, bits
            )

        logger.info(f"模型: {self.model_name}")
        if output_dir:
            logger.info(f"保存路径: {output_dir}")

    def _get_quantized_model_name(
        self, base_model: str, method: str, bits: int
    ) -> str:
        """获取量化模型名称"""
        if method.lower() == "awq" and bits == 4:
            return f"{base_model}-AWQ"
        elif method.lower() == "gptq" and bits == 4:
            return f"{base_model}-GPTQ-Int4"
        else:
            logger.warning(
                f"不支持的量化配置: {method} {bits}-bit，使用原始模型"
            )
            return base_model

    def download(self) -> str:
        """
        下载模型

        Returns:
            模型路径
        """
        try:
            # 尝试导入 huggingface_hub
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                logger.error("未找到 huggingface_hub，正在安装...")
                import subprocess

                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "huggingface_hub"]
                )
                from huggingface_hub import snapshot_download

            logger.info("开始下载模型...")
            logger.info(
                "这可能需要较长时间（取决于网络速度和模型大小）"
            )

            # 下载参数
            download_kwargs = {
                "repo_id": self.model_name,
                "resume_download": True,  # 断点续传
                "local_files_only": False,
            }

            if self.output_dir:
                download_kwargs["cache_dir"] = self.output_dir

            if self.token:
                download_kwargs["token"] = self.token

            # 下载模型
            model_path = snapshot_download(**download_kwargs)

            logger.info(f"✓ 模型下载完成！")
            logger.info(f"模型路径: {model_path}")

            return model_path

        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise

    def verify(self, model_path: str) -> bool:
        """
        验证模型完整性

        Args:
            model_path: 模型路径

        Returns:
            是否验证通过
        """
        logger.info("验证模型文件...")

        required_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ]

        # 检查必需文件
        model_dir = Path(model_path)
        for filename in required_files:
            filepath = model_dir / filename
            if not filepath.exists():
                logger.error(f"缺少文件: {filename}")
                return False

        # 检查模型权重文件
        weight_files = list(model_dir.glob("*.safetensors")) or list(
            model_dir.glob("*.bin")
        )
        if not weight_files:
            logger.error("未找到模型权重文件")
            return False

        logger.info(f"✓ 验证通过！找到 {len(weight_files)} 个权重文件")
        return True

    def get_model_info(self) -> dict:
        """获取模型信息"""
        for key, info in self.MODELS.items():
            if info["name"] == self.model_name:
                return info

        # 未知模型，返回基本信息
        return {
            "name": self.model_name,
            "size_gb": "未知",
            "description": "自定义模型",
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载 Qwen3 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载标准 FP16 模型（约 14GB）
  python scripts/download_model.py

  # 下载 AWQ 4-bit 量化模型（约 4.5GB，推荐）
  python scripts/download_model.py --quantization awq --bits 4

  # 下载 GPTQ 4-bit 量化模型
  python scripts/download_model.py --quantization gptq --bits 4

  # 自定义保存路径
  python scripts/download_model.py --output-dir /data/models --quantization awq

推荐配置（RTX 3060 12GB）:
  python scripts/download_model.py --quantization awq --bits 4
        """,
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace 模型名称（默认: Qwen/Qwen2.5-7B-Instruct）",
    )

    parser.add_argument(
        "--quantization",
        choices=["awq", "gptq"],
        help="量化方法 (awq 或 gptq)，推荐 awq",
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位数（默认: 4）",
    )

    parser.add_argument(
        "--output-dir",
        help="模型保存目录（默认: ~/.cache/huggingface/hub）",
    )

    parser.add_argument(
        "--token",
        help="HuggingFace token（对于私有模型）",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="下载后验证模型完整性",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出支持的模型配置",
    )

    args = parser.parse_args()

    # 列出模型
    if args.list_models:
        print("\n支持的模型配置:\n")
        for key, info in ModelDownloader.MODELS.items():
            print(f"  {key}:")
            print(f"    名称: {info['name']}")
            print(f"    大小: ~{info['size_gb']} GB")
            print(f"    描述: {info['description']}")
            print()
        return

    # 显示配置信息
    print("=" * 70)
    print("Qwen3 模型下载工具")
    print("=" * 70)
    print()

    # 创建下载器
    downloader = ModelDownloader(
        model_name=args.model,
        output_dir=args.output_dir,
        token=args.token,
        quantization=args.quantization,
        bits=args.bits,
    )

    # 显示模型信息
    model_info = downloader.get_model_info()
    print(f"模型名称: {model_info['name']}")
    print(f"预计大小: {model_info['size_gb']} GB")
    print(f"描述: {model_info['description']}")
    print()

    # 确认下载
    if args.quantization:
        print(f"✓ 使用 {args.quantization.upper()} {args.bits}-bit 量化")
        print(f"✓ 显存需求: ~{model_info['size_gb']} GB")
        print(
            f"✓ 适合 RTX 3060 12GB{'  [推荐]' if args.quantization == 'awq' else ''}"
        )
    else:
        print("⚠ 使用 FP16 原始模型")
        print(f"⚠ 显存需求: ~{model_info['size_gb']} GB")
        print("⚠ RTX 3060 12GB 可能不足，推荐使用 --quantization awq")

    print()
    print("开始下载...")
    print()

    try:
        # 下载模型
        model_path = downloader.download()

        # 验证模型
        if args.verify:
            if downloader.verify(model_path):
                print()
                print("=" * 70)
                print("✓ 下载并验证成功！")
                print("=" * 70)
            else:
                print()
                print("=" * 70)
                print("✗ 验证失败，模型可能不完整")
                print("=" * 70)
                sys.exit(1)
        else:
            print()
            print("=" * 70)
            print("✓ 下载完成！")
            print("=" * 70)

        # 显示使用说明
        print()
        print("下一步:")
        print()
        print("1. 激活 conda 环境:")
        print("   conda activate hppe")
        print()
        print("2. 启动 vLLM 服务:")
        print("   ./scripts/start_vllm.sh")
        print()
        print("3. 测试 LLM 引擎:")
        print(
            "   PYTHONPATH=/home/ivan/HPPE/src python examples/llm_engine_example.py"
        )
        print()

    except KeyboardInterrupt:
        print()
        logger.warning("下载被用户中断")
        sys.exit(130)
    except Exception as e:
        print()
        logger.error(f"下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
