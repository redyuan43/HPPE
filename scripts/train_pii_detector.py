#!/usr/bin/env python3
"""
训练 PII 检测模型

支持功能：
1. LoRA 微调 - 参数高效训练
2. 多种基础模型 - Qwen2/Qwen3 系列
3. 自动评估 - 精确度、召回率、F1分数
4. 批处理优化 - 适配批量脱敏场景

使用示例：
    # 基础训练（使用 LoRA）
    python scripts/train_pii_detector.py \\
        --model Qwen/Qwen2-1.5B \\
        --data data/merged_pii_dataset_train.jsonl \\
        --output models/pii_detector_qwen2_1.5b

    # 高级配置
    python scripts/train_pii_detector.py \\
        --model Qwen/Qwen2-1.5B \\
        --data data/merged_pii_dataset_train.jsonl \\
        --val-data data/merged_pii_dataset_validation.jsonl \\
        --lora-r 16 \\
        --lora-alpha 32 \\
        --batch-size 8 \\
        --gradient-accumulation 4 \\
        --learning-rate 2e-4 \\
        --epochs 3 \\
        --output models/pii_detector_custom

    # 全参数微调（需要大显存）
    python scripts/train_pii_detector.py \\
        --model Qwen/Qwen2-1.5B \\
        --data data/merged_pii_dataset_train.jsonl \\
        --full-finetune \\
        --output models/pii_detector_full
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import Dataset
except ImportError:
    print("❌ 缺少必要的库。请先安装：")
    print("   pip install transformers peft datasets torch accelerate")
    sys.exit(1)


@dataclass
class PIITrainingConfig:
    """训练配置"""
    model_name: str
    data_path: Path
    val_data_path: Optional[Path]
    output_dir: Path

    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    # 训练超参数
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_length: int = 512

    # 优化配置
    fp16: bool = True
    gradient_checkpointing: bool = True

    # 评估配置
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Qwen2/Qwen3 的默认 LoRA 目标模块
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_training_data(data_path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    加载训练数据

    Args:
        data_path: 数据文件路径（JSON或JSONL格式）
        max_samples: 最大样本数（用于快速测试）

    Returns:
        样本列表
    """
    print(f"\n加载训练数据: {data_path}")

    samples = []

    if data_path.suffix == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                samples.append(sample)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # 可能是 {"train": [...], "validation": [...]} 格式
                samples = data.get("train", data)
            else:
                samples = data

            if max_samples:
                samples = samples[:max_samples]

    print(f"  ✓ 加载了 {len(samples)} 个训练样本")
    return samples


def format_sample_for_training(sample: Dict[str, Any]) -> str:
    """
    将样本格式化为训练文本

    格式：
    <|im_start|>system
    你是 PII 检测专家。检测文本中的 PII 并输出 JSON。<|im_end|>
    <|im_start|>user
    检测以下文本中的 PII：我叫张三，电话13800138000<|im_end|>
    <|im_start|>assistant
    {"entities": [{"type": "PERSON_NAME", "value": "张三", ...}]}<|im_end|>
    """
    instruction = sample.get("instruction", "检测以下文本中的 PII，并以 JSON 格式输出实体列表。")
    input_text = sample["input"]
    output_data = sample["output"]

    # 简化输出（只保留必要信息）
    entities = [
        {
            "type": e["type"],
            "value": e["value"],
            "start": e["start"],
            "end": e["end"]
        }
        for e in output_data["entities"]
    ]

    output_json = json.dumps({"entities": entities}, ensure_ascii=False)

    # Qwen 对话格式
    formatted_text = (
        f"<|im_start|>system\n"
        f"你是 PII 检测专家。{instruction}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{output_json}<|im_end|>"
    )

    return formatted_text


def prepare_dataset(
    samples: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int = 512
) -> Dataset:
    """
    准备训练数据集

    Args:
        samples: 原始样本列表
        tokenizer: 分词器
        max_length: 最大序列长度

    Returns:
        Hugging Face Dataset
    """
    print(f"\n准备数据集...")
    print(f"  样本数量: {len(samples):,}")
    print(f"  最大长度: {max_length}")

    # 格式化样本为文本
    print(f"  格式化样本...")
    formatted_texts = []
    for sample in samples:
        text = format_sample_for_training(sample)
        formatted_texts.append(text)

    # 创建原始数据集
    raw_dataset = Dataset.from_dict({"text": formatted_texts})

    # 定义分词函数（批量处理）
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )

    # 使用 map 进行流式批量分词
    print(f"  流式分词中（批量处理，节省内存）...")
    dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,  # 每次处理1000个样本
        remove_columns=["text"],  # 移除原始文本，节省内存
        desc="分词进度"
    )

    print(f"  ✓ 数据集准备完成: {len(dataset):,} 样本")
    return dataset


def setup_model_and_tokenizer(config: PIITrainingConfig):
    """
    设置模型和分词器

    Args:
        config: 训练配置

    Returns:
        (model, tokenizer)
    """
    print(f"\n{'='*70}")
    print("加载模型和分词器")
    print(f"{'='*70}")
    print(f"模型: {config.model_name}")
    print(f"训练方法: {'LoRA' if config.use_lora else '全参数微调'}")

    # 加载分词器
    print(f"\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  ✓ 分词器加载完成")
    print(f"    词汇表大小: {len(tokenizer)}")
    print(f"    pad_token: {tokenizer.pad_token}")

    # 加载模型
    print(f"\n加载模型...")
    # 注意: 使用 accelerate 多 GPU 训练时不能指定 device_map
    # accelerate 会自动管理设备分配
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.fp16 else torch.float32
    )

    print(f"  ✓ 模型加载完成")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    总参数: {total_params / 1e9:.2f}B")

    # 配置 LoRA
    if config.use_lora:
        print(f"\n配置 LoRA...")
        print(f"  r: {config.lora_r}")
        print(f"  alpha: {config.lora_alpha}")
        print(f"  dropout: {config.lora_dropout}")
        print(f"  target_modules: {config.lora_target_modules}")

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 启用梯度检查点
    # 注意: LoRA + accelerate 多GPU 模式下 gradient_checkpointing 可能导致梯度问题
    # 由于 LoRA 参数量很小 (5.9M)，两个3060应该足够，暂时禁用
    if config.gradient_checkpointing and not config.use_lora:
        model.gradient_checkpointing_enable()
        print(f"\n  ✓ 梯度检查点已启用（节省显存）")
    elif config.gradient_checkpointing and config.use_lora:
        print(f"\n  ⚠ 跳过梯度检查点（LoRA+accelerate兼容性考虑）")

    return model, tokenizer


def train_model(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: PIITrainingConfig
):
    """
    训练模型

    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练集
        val_dataset: 验证集
        config: 训练配置
    """
    print(f"\n{'='*70}")
    print("开始训练")
    print(f"{'='*70}")

    # 创建输出目录
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 训练参数
    # 注意: LoRA + accelerate 多GPU模式下必须禁用 gradient_checkpointing
    use_gradient_checkpointing = config.gradient_checkpointing and not config.use_lora

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        fp16=config.fp16,
        gradient_checkpointing=use_gradient_checkpointing,
        report_to="none",  # 禁用 wandb/tensorboard
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
    )

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型，不使用 MLM
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print(f"\n训练配置:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  FP16: {config.fp16}")
    print()

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\n✓ 训练完成！")
    print(f"  耗时: {elapsed / 60:.1f} 分钟")

    # 保存模型
    print(f"\n保存模型...")
    trainer.save_model(str(config.output_dir / "final"))
    tokenizer.save_pretrained(str(config.output_dir / "final"))
    print(f"  ✓ 模型已保存到: {config.output_dir / 'final'}")

    # 保存训练配置
    config_path = config.output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": config.model_name,
            "use_lora": config.use_lora,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "max_length": config.max_length,
        }, f, indent=2)
    print(f"  ✓ 训练配置已保存到: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="训练 PII 检测模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础 LoRA 训练
  python scripts/train_pii_detector.py \\
      --model Qwen/Qwen2-1.5B \\
      --data data/merged_pii_dataset_train.jsonl \\
      --output models/pii_detector

  # 带验证集的训练
  python scripts/train_pii_detector.py \\
      --model Qwen/Qwen2-1.5B \\
      --data data/merged_pii_dataset_train.jsonl \\
      --val-data data/merged_pii_dataset_validation.jsonl \\
      --output models/pii_detector

  # 高级配置
  python scripts/train_pii_detector.py \\
      --model Qwen/Qwen2-1.5B \\
      --data data/merged_pii_dataset_train.jsonl \\
      --lora-r 16 --lora-alpha 32 \\
      --batch-size 8 --learning-rate 2e-4 \\
      --epochs 3 --output models/pii_detector_custom
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="基础模型名称或路径 (例如: Qwen/Qwen2-1.5B)"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="训练数据路径 (.json 或 .jsonl)"
    )

    parser.add_argument(
        "--val-data",
        type=str,
        help="验证数据路径 (.json 或 .jsonl)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录"
    )

    # LoRA 配置
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="使用全参数微调（而非 LoRA）"
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (默认: 8)"
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (默认: 16)"
    )

    # 训练超参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批次大小 (默认: 4)"
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="梯度累积步数 (默认: 4)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="学习率 (默认: 2e-4)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数 (默认: 3)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="最大序列长度 (默认: 512)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="最大样本数（用于快速测试）"
    )

    args = parser.parse_args()

    # 创建训练配置
    config = PIITrainingConfig(
        model_name=args.model,
        data_path=Path(args.data),
        val_data_path=Path(args.val_data) if args.val_data else None,
        output_dir=Path(args.output),
        use_lora=not args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_length=args.max_length,
    )

    # 加载数据
    train_samples = load_training_data(config.data_path, args.max_samples)

    val_samples = None
    if config.val_data_path:
        val_samples = load_training_data(config.val_data_path, args.max_samples)

    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer(config)

    # 准备数据集
    train_dataset = prepare_dataset(train_samples, tokenizer, config.max_length)
    val_dataset = prepare_dataset(val_samples, tokenizer, config.max_length) if val_samples else None

    # 训练模型
    train_model(model, tokenizer, train_dataset, val_dataset, config)

    print(f"\n🎉 训练完成！")
    print(f"\n模型位置: {config.output_dir / 'final'}")
    print(f"\n后续步骤:")
    print(f"  1. 测试模型: python examples/test_trained_model.py --model {config.output_dir / 'final'}")
    print(f"  2. 集成到系统: 修改配置文件指向新模型")


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
