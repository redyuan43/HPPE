#!/usr/bin/env python3
"""
调试Epoch 1模型输出 - 查看实际生成内容
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_PATH = "models/pii_qwen4b_unsloth/checkpoint-781"
TEST_DATA = "data/merged_pii_dataset_test.jsonl"

print("🔍 加载模型...")
peft_config = PeftConfig.from_pretrained(MODEL_PATH)
base_model_path = peft_config.base_model_name_or_path

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("✅ 模型加载完成\n")

# 测试3个样本
print("📊 测试3个样本的实际输出:\n")

with open(TEST_DATA, 'r') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break

        sample = json.loads(line)
        input_text = sample["input"]
        expected_entities = sample.get("output", {}).get("entities", [])

        print(f"{'='*70}")
        print(f"样本 {i+1}:")
        print(f"输入: {input_text[:100]}...")
        print(f"期望实体数: {len(expected_entities)}")

        # 构造prompt
        prompt = (
            f"<|im_start|>system\n"
            f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 推理
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 解码完整输出
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 提取assistant部分
        assistant_start = full_response.rfind("<|im_start|>assistant\n")
        if assistant_start != -1:
            assistant_response = full_response[assistant_start + len("<|im_start|>assistant\n"):]
            print(f"\n模型原始输出:")
            print(f"{assistant_response[:500]}")
        else:
            print(f"\n⚠️ 未找到assistant输出标记")
            print(f"完整输出: {full_response[:500]}")

        print(f"\n期望输出样例:")
        if expected_entities:
            print(f'{{"entities": [{expected_entities[0]}]}}')
        print()

print(f"{'='*70}")
print("调试完成")
