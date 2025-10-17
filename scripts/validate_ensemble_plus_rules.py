#!/usr/bin/env python3
"""
集成模型 + 规则增强验证
测试：Ensemble (Epoch1+Epoch2) + Rule-based
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用双GPU

import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# 导入规则检测器
sys.path.append('scripts')
from rule_enhanced_detector import RuleEnhancedPIIDetector

BASE_MODEL = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"
MODEL1_PATH = "models/pii_qwen4b_unsloth/checkpoint-781"
MODEL2_PATH = "models/pii_qwen4b_unsloth/checkpoint-1562"
TEST_DATA = "data/merged_pii_dataset_test.jsonl"
SAMPLE_SIZE = 100

print("="*70)
print("🚀 集成模型 + 规则增强验证")
print("="*70)
print(f"策略: (Epoch1 ∪ Epoch2) ∪ Rules")
print(f"样本数: {SAMPLE_SIZE}")
print("="*70 + "\n")

# 加载模型
print("📦 加载模型1 (GPU0)...")
base_model1 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model1 = PeftModel.from_pretrained(base_model1, MODEL1_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("✅ 模型1完成\n")

print("📦 加载模型2 (GPU1)...")
base_model2 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": 1},
)
model2 = PeftModel.from_pretrained(base_model2, MODEL2_PATH)
print("✅ 模型2完成\n")

# 初始化规则检测器
print("📦 初始化规则检测器...")
rule_detector = RuleEnhancedPIIDetector()
print("✅ 规则检测器完成\n")

# 加载测试数据
test_samples = []
with open(TEST_DATA, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE:
            break
        test_samples.append(json.loads(line))

print(f"📊 加载了 {len(test_samples)} 个测试样本\n")

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

# 执行验证
print("🚀 开始验证...\n")

ensemble_tp = ensemble_fp = ensemble_fn = 0
full_tp = full_fp = full_fn = 0

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

    # 规则提取
    rule_entities = rule_detector.extract_by_rules(input_text)
    rule_set = set((e["type"], e["value"]) for e in rule_entities)

    # 完整预测（集成 + 规则）
    full_set = ensemble_set | rule_set

    # 统计集成模型
    ensemble_tp += len(expected_entities & ensemble_set)
    ensemble_fp += len(ensemble_set - expected_entities)
    ensemble_fn += len(expected_entities - ensemble_set)

    # 统计完整方案
    full_tp += len(expected_entities & full_set)
    full_fp += len(full_set - expected_entities)
    full_fn += len(expected_entities - full_set)

# 计算指标
def calc_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

ens_p, ens_r, ens_f1 = calc_metrics(ensemble_tp, ensemble_fp, ensemble_fn)
full_p, full_r, full_f1 = calc_metrics(full_tp, full_fp, full_fn)

# 输出结果
print(f"\n{'='*70}")
print(f"📊 验证结果对比 ({SAMPLE_SIZE}样本)")
print(f"{'='*70}\n")

print(f"{'方案':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"{'-'*70}")
print(f"{'集成模型':<20} {ens_p*100:<11.2f}% {ens_r*100:<11.2f}% {ens_f1*100:<11.2f}%")
print(f"{'集成+规则':<20} {full_p*100:<11.2f}% {full_r*100:<11.2f}% {full_f1*100:<11.2f}%")
print(f"{'='*70}\n")

print(f"📈 规则增强效果:")
print(f"  Recall:    {ens_r*100:.2f}% → {full_r*100:.2f}% ({(full_r-ens_r)*100:+.2f}%)")
print(f"  F1-Score:  {ens_f1*100:.2f}% → {full_f1*100:.2f}% ({(full_f1-ens_f1)*100:+.2f}%)")

print(f"\n统计:")
print(f"  集成: TP={ensemble_tp}, FP={ensemble_fp}, FN={ensemble_fn}")
print(f"  完整: TP={full_tp}, FP={full_fp}, FN={full_fn}")

# 保存结果
result = {
    "sample_size": SAMPLE_SIZE,
    "ensemble_only": {
        "precision": ens_p,
        "recall": ens_r,
        "f1": ens_f1,
        "tp": ensemble_tp,
        "fp": ensemble_fp,
        "fn": ensemble_fn
    },
    "ensemble_plus_rules": {
        "precision": full_p,
        "recall": full_r,
        "f1": full_f1,
        "tp": full_tp,
        "fp": full_fp,
        "fn": full_fn
    }
}

with open("logs/ensemble_plus_rules_result.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✅ 结果已保存至: logs/ensemble_plus_rules_result.json")

if full_r >= 0.90:
    print(f"\n🎉 达标！Recall {full_r*100:.2f}% ≥ 90%")
else:
    gap = 0.90 - full_r
    print(f"\n⚠️ 距离目标: {gap*100:.2f}%")
