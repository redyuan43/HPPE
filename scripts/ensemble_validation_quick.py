#!/usr/bin/env python3
"""
快速集成验证：Epoch1 + Epoch2 (20样本)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU1

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"
MODEL1_PATH = "models/pii_qwen4b_unsloth/checkpoint-781"
MODEL2_PATH = "models/pii_qwen4b_unsloth/checkpoint-1562"
TEST_DATA = "data/merged_pii_dataset_test.jsonl"
SAMPLE_SIZE = 20  # 快速验证

print("="*70)
print("🔄 快速集成验证 (GPU1, 20样本)")
print("="*70)
print(f"模型1: Epoch 1")
print(f"模型2: Epoch 2")
print(f"样本数: {SAMPLE_SIZE}")
print("="*70 + "\n")

# 加载模型
print("📦 加载模型1...")
base_model1 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model1 = PeftModel.from_pretrained(base_model1, MODEL1_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("✅ 模型1完成\n")

print("📦 加载模型2...")
base_model2 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model2 = PeftModel.from_pretrained(base_model2, MODEL2_PATH)
print("✅ 模型2完成\n")

# 加载测试数据
test_samples = []
with open(TEST_DATA, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE:
            break
        test_samples.append(json.loads(line))

print(f"📊 {len(test_samples)} 样本\n")

def extract_entities(text):
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

# 统计
model1_tp = model1_fp = model1_fn = 0
model2_tp = model2_fp = model2_fn = 0
ensemble_tp = ensemble_fp = ensemble_fn = 0

print("🚀 开始验证...\n")

for sample in tqdm(test_samples, desc="进度"):
    input_text = sample["input"]
    expected_entities = set(
        (e["type"], e["value"])
        for e in sample.get("output", {}).get("entities", [])
    )

    pred1 = predict_entities(model1, tokenizer, input_text)
    pred1_set = set((e["type"], e["value"]) for e in pred1)

    pred2 = predict_entities(model2, tokenizer, input_text)
    pred2_set = set((e["type"], e["value"]) for e in pred2)

    ensemble_set = pred1_set | pred2_set

    model1_tp += len(expected_entities & pred1_set)
    model1_fp += len(pred1_set - expected_entities)
    model1_fn += len(expected_entities - pred1_set)

    model2_tp += len(expected_entities & pred2_set)
    model2_fp += len(pred2_set - expected_entities)
    model2_fn += len(expected_entities - pred2_set)

    ensemble_tp += len(expected_entities & ensemble_set)
    ensemble_fp += len(ensemble_set - expected_entities)
    ensemble_fn += len(expected_entities - ensemble_set)

def calc_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

m1_p, m1_r, m1_f1 = calc_metrics(model1_tp, model1_fp, model1_fn)
m2_p, m2_r, m2_f1 = calc_metrics(model2_tp, model2_fp, model2_fn)
ens_p, ens_r, ens_f1 = calc_metrics(ensemble_tp, ensemble_fp, ensemble_fn)

print(f"\n{'='*70}")
print(f"📊 快速验证结果 ({SAMPLE_SIZE}样本)")
print(f"{'='*70}\n")
print(f"{'模型':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print(f"{'-'*70}")
print(f"{'Epoch 1':<12} {m1_p*100:<11.2f}% {m1_r*100:<11.2f}% {m1_f1*100:<11.2f}%")
print(f"{'Epoch 2':<12} {m2_p*100:<11.2f}% {m2_r*100:<11.2f}% {m2_f1*100:<11.2f}%")
print(f"{'-'*70}")
print(f"{'集成':<12} {ens_p*100:<11.2f}% {ens_r*100:<11.2f}% {ens_f1*100:<11.2f}%")
print(f"{'='*70}\n")

print(f"提升:")
print(f"  Recall: {m2_r*100:.2f}% → {ens_r*100:.2f}% ({(ens_r-m2_r)*100:+.2f}%)")
print(f"  F1:     {m2_f1*100:.2f}% → {ens_f1*100:.2f}% ({(ens_f1-m2_f1)*100:+.2f}%)\n")

print(f"统计: TP={ensemble_tp}, FP={ensemble_fp}, FN={ensemble_fn}")

with open("logs/ensemble_quick_result.json", 'w') as f:
    json.dump({
        "sample_size": SAMPLE_SIZE,
        "ensemble": {"precision": ens_p, "recall": ens_r, "f1": ens_f1,
                     "tp": ensemble_tp, "fp": ensemble_fp, "fn": ensemble_fn}
    }, f, indent=2)

print(f"\n✅ 保存至: logs/ensemble_quick_result.json")

if ens_r >= 0.90:
    print(f"\n🎉 达标！Recall {ens_r*100:.2f}% ≥ 90%")
else:
    print(f"\n⚠️ 差距: {(0.90-ens_r)*100:.2f}%")
