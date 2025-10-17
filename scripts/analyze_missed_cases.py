#!/usr/bin/env python3
"""
分析漏检样本，找出Recall低的根本原因
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import defaultdict

MODEL_PATH = "models/pii_qwen4b_unsloth/checkpoint-1562"
BASE_MODEL = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"
TEST_DATA = "data/merged_pii_dataset_test.jsonl"
SAMPLE_SIZE = 100  # 增加样本量

print("📦 加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("✅ 模型加载完成\n")

# 加载测试数据
test_samples = []
with open(TEST_DATA, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE:
            break
        test_samples.append(json.loads(line))

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

# 统计漏检情况
missed_by_type = defaultdict(lambda: {"total": 0, "missed": 0, "examples": []})
total_fn = 0
total_entities = 0

print("🔍 分析漏检模式...")
for idx, sample in enumerate(test_samples):
    input_text = sample["input"]
    expected_entities = sample.get("output", {}).get("entities", [])
    
    # 推理
    prompt = (
        f"<|im_start|>system\n"
        f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
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
    predicted_entities = extract_entities(response)
    
    # 分析漏检
    expected_set = set((e["type"], e["value"]) for e in expected_entities)
    predicted_set = set((e["type"], e["value"]) for e in predicted_entities)
    
    missed = expected_set - predicted_set
    
    for entity_type, entity_value in missed:
        missed_by_type[entity_type]["missed"] += 1
        if len(missed_by_type[entity_type]["examples"]) < 3:
            missed_by_type[entity_type]["examples"].append({
                "text": input_text[:100],
                "missed_value": entity_value
            })
    
    for entity in expected_entities:
        missed_by_type[entity["type"]]["total"] += 1
        total_entities += 1
    
    total_fn += len(missed)
    
    if (idx + 1) % 10 == 0:
        print(f"  已处理 {idx+1}/{SAMPLE_SIZE} 样本...")

# 输出分析报告
print(f"\n{'='*70}")
print(f"📊 漏检分析报告 (基于{SAMPLE_SIZE}样本)")
print(f"{'='*70}")
print(f"总实体数: {total_entities}")
print(f"总漏检数: {total_fn}")
print(f"总体Recall: {(total_entities - total_fn) / total_entities * 100:.2f}%\n")

print(f"{'类型':<15} {'总数':>6} {'漏检':>6} {'Recall':>8} {'漏检率':>8}")
print(f"{'-'*70}")

sorted_types = sorted(missed_by_type.items(), 
                      key=lambda x: x[1]["missed"] / max(x[1]["total"], 1), 
                      reverse=True)

for entity_type, stats in sorted_types:
    total = stats["total"]
    missed = stats["missed"]
    recall = (total - missed) / total * 100 if total > 0 else 0
    miss_rate = missed / total * 100 if total > 0 else 0
    
    print(f"{entity_type:<15} {total:>6} {missed:>6} {recall:>7.1f}% {miss_rate:>7.1f}%")
    
    if stats["examples"]:
        print(f"  漏检示例:")
        for ex in stats["examples"][:2]:
            print(f"    - 文本: {ex['text']}...")
            print(f"      漏检: {ex['missed_value']}")

print(f"\n{'='*70}")
print(f"💡 关键发现")
print(f"{'='*70}")

# 找出Recall最低的3个类型
worst_types = sorted_types[:3]
print(f"\nRecall最低的PII类型:")
for entity_type, stats in worst_types:
    total = stats["total"]
    missed = stats["missed"]
    recall = (total - missed) / total * 100 if total > 0 else 0
    print(f"  {entity_type}: Recall {recall:.1f}% (漏检{missed}/{total})")

# 保存详细结果
with open("logs/missed_cases_analysis.json", 'w', encoding='utf-8') as f:
    json.dump({
        "total_entities": total_entities,
        "total_missed": total_fn,
        "overall_recall": (total_entities - total_fn) / total_entities,
        "by_type": {k: v for k, v in missed_by_type.items()}
    }, f, indent=2, ensure_ascii=False)

print(f"\n✅ 详细报告已保存: logs/missed_cases_analysis.json")
