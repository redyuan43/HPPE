#!/usr/bin/env python3
"""
GPU1训练脚本 - 5种PII
VEHICLE_PLATE, DRIVER_LICENSE, SOCIAL_SECURITY_CARD, UNIFIED_SOCIAL_CREDIT_CODE, IPV6_ADDRESS
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 强制使用GPU1

import json
import sys
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import torch
from datetime import datetime

print("="*70)
print("🚀 GPU1 训练任务启动")
print("="*70)
print(f"训练PII类型: VEHICLE_PLATE, DRIVER_LICENSE, SOCIAL_SECURITY_CARD, UNIFIED_SOCIAL_CREDIT_CODE, IPV6_ADDRESS")
print(f"GPU: 1")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")

from unsloth import FastLanguageModel

# 配置
MODEL_NAME = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"
TRAIN_DATA = "data/gpu1_train.jsonl"
OUTPUT_DIR = "models/pii_11new_gpu1"
MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 12
GRADIENT_ACCUMULATION = 3
EPOCHS = 2
LR = 1.5e-4

# 加载模型
print("📦 加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print("✅ 模型加载完成\n")

# 加载数据并tokenize
print(f"📂 加载训练数据: {TRAIN_DATA}")

def load_and_tokenize_jsonl(path: str):
    """加载JSONL数据并tokenize"""
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
                return_tensors=None,
            )

            # 设置 labels = input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()

            all_input_ids.append(tokenized["input_ids"])
            all_attention_mask.append(tokenized["attention_mask"])
            all_labels.append(tokenized["labels"])

    print(f"✅ 加载了 {len(all_input_ids)} 个训练样本\n")

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    })

train_dataset = load_and_tokenize_jsonl(TRAIN_DATA)

# 训练配置
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# 开始训练
print("🚀 开始训练...\n")
start_time = datetime.now()
trainer.train()
end_time = datetime.now()

# 保存模型
print("\n💾 保存模型...")
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

duration = (end_time - start_time).total_seconds() / 60
print(f"\n{'='*70}")
print(f"✅ GPU1 训练完成！")
print(f"{'='*70}")
print(f"训练时长: {duration:.1f} 分钟")
print(f"模型保存至: {OUTPUT_DIR}/final")
print(f"完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")
