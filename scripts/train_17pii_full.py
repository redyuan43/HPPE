#!/usr/bin/env python3
"""
Phase 2: 训练完整17种PII检测模型
使用merged_17pii_train.jsonl (69,215样本)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU0

import json
import sys
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import torch
from datetime import datetime

print("="*70)
print("🚀 Phase 2: 完整17种PII模型训练")
print("="*70)
print(f"训练PII类型: 17种")
print(f"  - 原6种: ADDRESS, ORGANIZATION, PERSON_NAME, PHONE_NUMBER, EMAIL, ID_CARD")
print(f"  - 新11种: BANK_CARD, PASSPORT, HK_MACAU_PASS, POSTAL_CODE, IP_ADDRESS,")
print(f"           MAC_ADDRESS, VEHICLE_PLATE, DRIVER_LICENSE, SOCIAL_SECURITY_CARD,")
print(f"           UNIFIED_SOCIAL_CREDIT_CODE, IPV6_ADDRESS")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")

from unsloth import FastLanguageModel

# 配置
MODEL_NAME = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"
TRAIN_DATA = "data/merged_17pii_train.jsonl"
VAL_DATA = "data/merged_pii_dataset_validation.jsonl"
OUTPUT_DIR = "models/pii_qwen4b_17types_final"
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True

# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0

# 训练配置
BATCH_SIZE = 12
GRADIENT_ACCUMULATION = 3
EPOCHS = 2
LR = 1.5e-4

print("📦 加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

print("✅ 模型加载完成")
print(f"   4bit量化: {LOAD_IN_4BIT}")
print(f"   Max seq length: {MAX_SEQ_LENGTH}\n")

print("🔧 配置LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print("✅ LoRA配置完成")
print(f"   r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}\n")

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
eval_dataset = load_and_tokenize_jsonl(VAL_DATA)

# 计算总steps
total_samples = len(train_dataset)
effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION
steps_per_epoch = total_samples // effective_batch_size
total_steps = steps_per_epoch * EPOCHS

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📊 训练配置汇总")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"训练样本数: {total_samples:,}")
print(f"验证样本数: {len(eval_dataset):,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"Effective batch size: {effective_batch_size}")
print(f"Learning rate: {LR}")
print(f"Epochs: {EPOCHS}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps: {total_steps}")
print(f"输出目录: {OUTPUT_DIR}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# 训练配置
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=False,  # 关闭fp16避免dtype冲突
    bf16=True,   # 使用bf16（与Unsloth兼容）
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("🚀 开始训练...\n")
start_time = datetime.now()

trainer.train()

end_time = datetime.now()
duration = end_time - start_time

print("\n" + "="*70)
print("🎉 训练完成！")
print("="*70)
print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总耗时: {duration}")
print("="*70 + "\n")

# 保存最终模型
print("💾 保存最终模型...")
final_output = f"{OUTPUT_DIR}/final"
trainer.model.save_pretrained(final_output)
tokenizer.save_pretrained(final_output)

print(f"✅ 模型已保存到: {final_output}")
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
