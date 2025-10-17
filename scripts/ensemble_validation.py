#!/usr/bin/env python3
"""
模型集成验证：Epoch1 + Epoch2
使用投票机制提升Recall
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 修改为GPU0

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"
MODEL1_PATH = "models/pii_qwen4b_unsloth/checkpoint-781"   # Epoch 1
MODEL2_PATH = "models/pii_qwen4b_unsloth/checkpoint-1562"  # Epoch 2
TEST_DATA = "data/merged_pii_dataset_test.jsonl"
SAMPLE_SIZE = 100

print("="*70)
print("🔄 模型集成验证")
print("="*70)
print(f"模型1: {MODEL1_PATH} (Epoch 1)")
print(f"模型2: {MODEL2_PATH} (Epoch 2)")
print(f"策略: 并集（任一模型检出即算检出）")
print("="*70 + "\n")

# 加载两个模型
print("📦 加载模型1 (Epoch 1)...")
base_model1 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model1 = PeftModel.from_pretrained(base_model1, MODEL1_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("✅ 模型1加载完成\n")

print("📦 加载模型2 (Epoch 2)...")
base_model2 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model2 = PeftModel.from_pretrained(base_model2, MODEL2_PATH)
print("✅ 模型2加载完成\n")

# 加载测试数据
test_samples = []
with open(TEST_DATA, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE:
            break
        test_samples.append(json.loads(line))

print(f"📊 加载了 {len(test_samples)} 个测试样本\n")

def extract_entities(text):
    """提取实体"""
    try:
        start_idx = text.find('{"entities"')
        if start_idx == -1:
            return []
        end_marker = '<|im_end|>'
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            json_str = text[start_idx:].strip()
        else:
            json_str = text[start_idx:end_idx].strip()
        result = json.loads(json_str)
        return result.get("entities", [])
    except:
        return []

def predict_entities(model, tokenizer, text):
    """单个模型预测"""
    prompt = (
        f"<|im_start|>system\n"
        f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return extract_entities(response)

# 执行集成验证
print("🚀 开始集成验证...\n")

# 单模型统计
model1_tp = model1_fp = model1_fn = 0
model2_tp = model2_fp = model2_fn = 0
ensemble_tp = ensemble_fp = ensemble_fn = 0

for sample in tqdm(test_samples, desc="验证进度"):
    input_text = sample["input"]
    expected_entities = set(
        (e["type"], e["value"])
        for e in sample.get("output", {}).get("entities", [])
    )

    # 模型1预测
    pred1 = predict_entities(model1, tokenizer, input_text)
    pred1_set = set((e["type"], e["value"]) for e in pred1)

    # 模型2预测
    pred2 = predict_entities(model2, tokenizer, input_text)
    pred2_set = set((e["type"], e["value"]) for e in pred2)

    # 集成预测（并集）
    ensemble_set = pred1_set | pred2_set

    # 统计模型1
    model1_tp += len(expected_entities & pred1_set)
    model1_fp += len(pred1_set - expected_entities)
    model1_fn += len(expected_entities - pred1_set)

    # 统计模型2
    model2_tp += len(expected_entities & pred2_set)
    model2_fp += len(pred2_set - expected_entities)
    model2_fn += len(expected_entities - pred2_set)

    # 统计集成
    ensemble_tp += len(expected_entities & ensemble_set)
    ensemble_fp += len(ensemble_set - expected_entities)
    ensemble_fn += len(expected_entities - ensemble_set)

# 计算指标
def calc_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

model1_p, model1_r, model1_f1 = calc_metrics(model1_tp, model1_fp, model1_fn)
model2_p, model2_r, model2_f1 = calc_metrics(model2_tp, model2_fp, model2_fn)
ensemble_p, ensemble_r, ensemble_f1 = calc_metrics(ensemble_tp, ensemble_fp, ensemble_fn)

# 输出结果
print(f"\n{'='*70}")
print(f"📊 集成验证结果 (基于 {SAMPLE_SIZE} 样本)")
print(f"{'='*70}\n")

print(f"{'模型':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"{'-'*70}")
print(f"{'Epoch 1':<15} {model1_p*100:<11.2f}% {model1_r*100:<11.2f}% {model1_f1*100:<11.2f}%")
print(f"{'Epoch 2':<15} {model2_p*100:<11.2f}% {model2_r*100:<11.2f}% {model2_f1*100:<11.2f}%")
print(f"{'-'*70}")
print(f"{'🔄 集成模型':<15} {ensemble_p*100:<11.2f}% {ensemble_r*100:<11.2f}% {ensemble_f1*100:<11.2f}%")
print(f"{'='*70}\n")

print(f"📈 集成效果提升:")
print(f"  Precision: {model2_p*100:.2f}% → {ensemble_p*100:.2f}% ({(ensemble_p-model2_p)*100:+.2f}%)")
print(f"  Recall:    {model2_r*100:.2f}% → {ensemble_r*100:.2f}% ({(ensemble_r-model2_r)*100:+.2f}%)")
print(f"  F1-Score:  {model2_f1*100:.2f}% → {ensemble_f1*100:.2f}% ({(ensemble_f1-model2_f1)*100:+.2f}%)")

print(f"\n详细统计:")
print(f"  集成模型 - TP: {ensemble_tp}, FP: {ensemble_fp}, FN: {ensemble_fn}")

# 保存结果
result = {
    "sample_size": SAMPLE_SIZE,
    "model1": {
        "path": MODEL1_PATH,
        "precision": model1_p,
        "recall": model1_r,
        "f1": model1_f1,
        "tp": model1_tp,
        "fp": model1_fp,
        "fn": model1_fn
    },
    "model2": {
        "path": MODEL2_PATH,
        "precision": model2_p,
        "recall": model2_r,
        "f1": model2_f1,
        "tp": model2_tp,
        "fp": model2_fp,
        "fn": model2_fn
    },
    "ensemble": {
        "precision": ensemble_p,
        "recall": ensemble_r,
        "f1": ensemble_f1,
        "tp": ensemble_tp,
        "fp": ensemble_fp,
        "fn": ensemble_fn
    }
}

with open("logs/ensemble_validation_result.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✅ 结果已保存至: logs/ensemble_validation_result.json")

# 判断是否达标
if ensemble_r >= 0.90:
    print(f"\n🎉 恭喜！集成模型Recall已达标: {ensemble_r*100:.2f}% ≥ 90%")
else:
    gap = 0.90 - ensemble_r
    print(f"\n⚠️ 距离目标还差: {gap*100:.2f}%")
    print(f"建议: 继续执行数据增强策略")
