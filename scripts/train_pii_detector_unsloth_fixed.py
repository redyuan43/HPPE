#!/usr/bin/env python3
"""
使用Unsloth训练PII检测模型 (修复版)
修复: 使用标准 Trainer 而非 SFTTrainer,避免 labels 格式问题
"""

import os
import json
import sys
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import torch

# 强制只使用GPU 0 (必须在导入Unsloth之前设置)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"强制GPU设置: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

print("正在导入Unsloth...")
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth导入成功!")
except ImportError as e:
    print(f"❌ Unsloth导入失败: {e}")
    sys.exit(1)

# ============================================================================
# 配置参数
# ============================================================================

# 模型配置
MODEL_NAME = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"  # 使用本地Qwen3-4B
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True  # 4bit量化,节省显存

# LoRA配置 (Unsloth优化版)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0  # 关键修复: 0 for Unsloth fast patching (5-10x speedup)

# 训练配置 (终极优化: 2 epochs + batch=12 + no dropout)
PER_DEVICE_TRAIN_BATCH_SIZE = 12  # 终极优化: 增大到12
GRADIENT_ACCUMULATION_STEPS = 3  # 调整为3,保持effective_batch_size=36
NUM_TRAIN_EPOCHS = 2  # 加速: 减少到2 epochs
LEARNING_RATE = 1.5e-4
WARMUP_STEPS = 100

# 数据路径
TRAIN_DATA = "data/merged_pii_dataset_train.jsonl"
VAL_DATA = "data/merged_pii_dataset_validation.jsonl"
OUTPUT_DIR = "models/pii_qwen4b_unsloth"

# ============================================================================
# 1. 加载模型
# ============================================================================

print(f"\n{'='*70}")
print(f"加载模型: {MODEL_NAME}")
print(f"{'='*70}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,  # 自动选择FP16/BF16
    load_in_4bit = LOAD_IN_4BIT,
)

print(f"✅ 模型加载完成 ({'4bit量化' if LOAD_IN_4BIT else 'FP16'})")

# ============================================================================
# 2. 配置LoRA
# ============================================================================

print(f"\n配置LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_R,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

print(f"✅ LoRA配置完成:")
print(f"   r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")

# ============================================================================
# 3. 加载数据
# ============================================================================

def load_and_tokenize_jsonl(path: str, desc: str = "数据") -> Dataset:
    """加载JSONL数据并tokenize"""
    print(f"\n加载{desc}: {path}")

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)

            # 提取输入和实体
            text = sample["input"]
            entities = sample.get("output", {}).get("entities", [])

            # 构造完整的训练文本
            full_text = (
                f"<|im_start|>system\n"
                f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f'{{"entities": {json.dumps(entities, ensure_ascii=False)}}}<|im_end|>'
            )

            # Tokenize
            tokenized = tokenizer(
                full_text,
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors=None,  # 返回 list
            )

            # 设置 labels = input_ids (causal LM 标准做法)
            tokenized["labels"] = tokenized["input_ids"].copy()

            all_input_ids.append(tokenized["input_ids"])
            all_attention_mask.append(tokenized["attention_mask"])
            all_labels.append(tokenized["labels"])

    print(f"  ✓ 加载了 {len(all_input_ids)} 个样本")

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })

train_dataset = load_and_tokenize_jsonl(TRAIN_DATA, "训练数据")
eval_dataset = load_and_tokenize_jsonl(VAL_DATA, "验证数据")

# ============================================================================
# 4. 训练配置
# ============================================================================

print(f"\n{'='*70}")
print(f"训练配置")
print(f"{'='*70}")

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size = 4,  # 关键修复: 验证时使用更小的batch size
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    warmup_steps = WARMUP_STEPS,
    num_train_epochs = NUM_TRAIN_EPOCHS,
    learning_rate = LEARNING_RATE,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "cosine",
    seed = 42,
    save_strategy = "epoch",
    eval_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    report_to = "none",
)

print(f"  Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
print(f"  Optimizer: adamw_8bit")
print(f"  LR scheduler: cosine")

# ============================================================================
# 5. 创建Trainer (使用标准 Trainer)
# ============================================================================

print(f"\n创建Trainer...")
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
)

# ============================================================================
# 6. 开始训练
# ============================================================================

print(f"\n{'='*70}")
print(f"开始训练")
print(f"{'='*70}\n")

trainer_stats = trainer.train()

print(f"\n{'='*70}")
print(f"训练完成!")
print(f"{'='*70}")
print(f"总耗时: {trainer_stats.metrics.get('train_runtime', 0) / 60:.1f} 分钟")

# ============================================================================
# 7. 保存模型
# ============================================================================

print(f"\n保存模型到: {OUTPUT_DIR}/final")

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print(f"\n保存合并后的16bit模型到: {OUTPUT_DIR}/final_merged_16bit")
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/final_merged_16bit",
    tokenizer,
    save_method = "merged_16bit",
)

print(f"\n✅ 所有文件已保存!")
print(f"\n推荐使用: {OUTPUT_DIR}/final_merged_16bit (推理速度最快)")
