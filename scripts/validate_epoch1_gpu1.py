#!/usr/bin/env python3
"""
GPU1隔离验证 - Epoch 1模型快速测试
严格限制只使用GPU1，不影响GPU0训练
"""

import os
import sys

# ========== 关键：必须在导入torch之前设置 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"🔒 GPU隔离设置: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path

# 配置
MODEL_PATH = "models/pii_qwen4b_unsloth/checkpoint-781"
TEST_DATA = "data/merged_pii_dataset_test.jsonl"
SAMPLE_SIZE = 50  # 快速验证：只用50个样本
MAX_SEQ_LENGTH = 512

print(f"\n{'='*70}")
print(f"Epoch 1 快速验证 (GPU1隔离)")
print(f"{'='*70}")
print(f"模型路径: {MODEL_PATH}")
print(f"测试数据: {TEST_DATA}")
print(f"样本数量: {SAMPLE_SIZE}")
print(f"GPU设置: 仅使用GPU1 (不影响GPU0训练)")
print(f"{'='*70}\n")

# 验证GPU隔离
print("🔍 验证GPU隔离...")
print(f"  PyTorch可见设备数: {torch.cuda.device_count()}")
print(f"  当前设备: {torch.cuda.current_device()}")
if torch.cuda.is_available():
    print(f"  设备名称: {torch.cuda.get_device_name(0)}")  # 注意：这里的0是相对索引
print()

# 加载模型（LoRA checkpoint 需要先加载基础模型）
print("📦 加载模型...")
try:
    from peft import PeftModel, PeftConfig

    # 1. 加载LoRA配置，获取基础模型路径
    peft_config = PeftConfig.from_pretrained(MODEL_PATH)
    base_model_path = peft_config.base_model_name_or_path
    print(f"  基础模型: {base_model_path}")
    print(f"  LoRA适配器: {MODEL_PATH}")

    # 2. 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # 自动映射到可见GPU (只有GPU1)
    )

    # 3. 加载LoRA权重
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print("✅ 模型加载成功 (基础模型 + LoRA)\n")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 加载测试数据
print(f"📊 加载测试数据...")
test_samples = []
with open(TEST_DATA, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE:
            break
        test_samples.append(json.loads(line))
print(f"✅ 加载了 {len(test_samples)} 个测试样本\n")

# 验证函数
def extract_entities(text):
    """提取模型预测的实体"""
    try:
        # 查找JSON部分
        start_idx = text.find('{"entities"')
        if start_idx == -1:
            return []

        # 修复：查找<|im_end|>标记作为结束
        end_marker = '<|im_end|>'
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            # 如果没有结束标记，尝试解析到文本末尾
            json_str = text[start_idx:].strip()
        else:
            json_str = text[start_idx:end_idx].strip()

        # 尝试解析JSON
        result = json.loads(json_str)
        return result.get("entities", [])
    except Exception as e:
        # 调试：打印失败的JSON
        # print(f"JSON解析失败: {json_str[:100]}")
        return []

# 执行验证
print("🚀 开始验证...\n")
tp = fp = fn = 0

for sample in tqdm(test_samples, desc="验证进度"):
    input_text = sample["input"]
    expected_entities = set(
        (e["type"], e["value"])
        for e in sample.get("output", {}).get("entities", [])
    )

    # 构造prompt
    prompt = (
        f"<|im_start|>system\n"
        f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # 推理
    inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    predicted_entities = extract_entities(response)
    predicted_set = set((e["type"], e["value"]) for e in predicted_entities)

    # 计算指标
    tp += len(expected_entities & predicted_set)
    fp += len(predicted_set - expected_entities)
    fn += len(expected_entities - predicted_set)

# 计算最终指标
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*70}")
print(f"📊 Epoch 1 验证结果 (基于 {SAMPLE_SIZE} 样本)")
print(f"{'='*70}")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print(f"\n详细统计:")
print(f"  TP (正确检出): {tp}")
print(f"  FP (误报):     {fp}")
print(f"  FN (漏检):     {fn}")
print(f"{'='*70}\n")

# 保存结果
result_file = f"logs/epoch1_validation_gpu1_{SAMPLE_SIZE}samples.json"
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump({
        "model": MODEL_PATH,
        "sample_size": SAMPLE_SIZE,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }, f, indent=2, ensure_ascii=False)

print(f"✅ 结果已保存至: {result_file}")
