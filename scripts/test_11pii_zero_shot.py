#!/usr/bin/env python3
"""
测试Epoch 2模型对11种新PII的零样本检测能力
验证模型在未训练过这些类型时的表现
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只用GPU0

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"
MODEL_PATH = "models/pii_qwen4b_unsloth/checkpoint-1562"  # Epoch 2

print("="*70)
print("🧪 测试Epoch 2模型的11种新PII零样本检测能力")
print("="*70)
print(f"模型: {MODEL_PATH}")
print(f"测试样本: 20个（涵盖11种新PII）")
print("="*70 + "\n")

# 加载模型
print("📦 加载模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("✅ 模型加载完成\n")

# 构造测试样本（11种新PII，每种2个样本）
test_samples = [
    # 1. BANK_CARD - 银行卡号
    {
        "input": "请将款项转至银行卡6217002430009876543，户名张三。",
        "expected": [{"type": "BANK_CARD", "value": "6217002430009876543"}, {"type": "PERSON_NAME", "value": "张三"}]
    },
    {
        "input": "我的工资卡号是6222021234567890123，开户行是工商银行。",
        "expected": [{"type": "BANK_CARD", "value": "6222021234567890123"}]
    },

    # 2. VEHICLE_PLATE - 车牌号
    {
        "input": "车牌号京A12345的车辆违章了，请尽快处理。",
        "expected": [{"type": "VEHICLE_PLATE", "value": "京A12345"}]
    },
    {
        "input": "停车场监控显示，沪B88888号车在18:30离开。",
        "expected": [{"type": "VEHICLE_PLATE", "value": "沪B88888"}]
    },

    # 3. IP_ADDRESS - IP地址
    {
        "input": "服务器IP地址192.168.1.100出现异常，请检查。",
        "expected": [{"type": "IP_ADDRESS", "value": "192.168.1.100"}]
    },
    {
        "input": "登录记录显示，来自10.0.0.25的访问被拒绝。",
        "expected": [{"type": "IP_ADDRESS", "value": "10.0.0.25"}]
    },

    # 4. POSTAL_CODE - 邮政编码
    {
        "input": "收件地址：北京市朝阳区建国路1号，邮编100020。",
        "expected": [{"type": "ADDRESS", "value": "北京市朝阳区建国路1号"}, {"type": "POSTAL_CODE", "value": "100020"}]
    },
    {
        "input": "请将发票寄至上海市浦东新区，邮政编码200120。",
        "expected": [{"type": "ADDRESS", "value": "上海市浦东新区"}, {"type": "POSTAL_CODE", "value": "200120"}]
    },

    # 5. UNIFIED_SOCIAL_CREDIT_CODE - 统一社会信用代码
    {
        "input": "公司统一社会信用代码：91110000MA01234567，请核实。",
        "expected": [{"type": "UNIFIED_SOCIAL_CREDIT_CODE", "value": "91110000MA01234567"}]
    },
    {
        "input": "企业信用代码91310000MA02345678已通过工商局验证。",
        "expected": [{"type": "UNIFIED_SOCIAL_CREDIT_CODE", "value": "91310000MA02345678"}]
    },

    # 6. PASSPORT - 护照号
    {
        "input": "我的护照号是E12345678，有效期至2030年。",
        "expected": [{"type": "PASSPORT", "value": "E12345678"}]
    },
    {
        "input": "请出示护照G98765432办理登机手续。",
        "expected": [{"type": "PASSPORT", "value": "G98765432"}]
    },

    # 7. HK_MACAU_PASS - 港澳通行证
    {
        "input": "港澳通行证C12345678已过期，需重新办理。",
        "expected": [{"type": "HK_MACAU_PASS", "value": "C12345678"}]
    },
    {
        "input": "请携带港澳通行证H98765432通关。",
        "expected": [{"type": "HK_MACAU_PASS", "value": "H98765432"}]
    },

    # 8. DRIVER_LICENSE - 驾驶证号
    {
        "input": "驾驶证号110101199001011234，准驾车型C1。",
        "expected": [{"type": "DRIVER_LICENSE", "value": "110101199001011234"}]
    },
    {
        "input": "请提供驾驶证320106198512152345进行核查。",
        "expected": [{"type": "DRIVER_LICENSE", "value": "320106198512152345"}]
    },

    # 9. SOCIAL_SECURITY_CARD - 社保卡号
    {
        "input": "社保卡号110101199001011234，单位缴费正常。",
        "expected": [{"type": "SOCIAL_SECURITY_CARD", "value": "110101199001011234"}]
    },
    {
        "input": "请提供社会保障卡号310101198001012345查询。",
        "expected": [{"type": "SOCIAL_SECURITY_CARD", "value": "310101198001012345"}]
    },

    # 10. MAC_ADDRESS - MAC地址
    {
        "input": "网卡MAC地址00:1A:2B:3C:4D:5E已绑定。",
        "expected": [{"type": "MAC_ADDRESS", "value": "00:1A:2B:3C:4D:5E"}]
    },
    {
        "input": "设备MAC地址为AA:BB:CC:DD:EE:FF，请记录。",
        "expected": [{"type": "MAC_ADDRESS", "value": "AA:BB:CC:DD:EE:FF"}]
    },

    # 11. IPV6_ADDRESS - IPv6地址
    {
        "input": "IPv6地址2001:0db8:85a3:0000:0000:8a2e:0370:7334已分配。",
        "expected": [{"type": "IPV6_ADDRESS", "value": "2001:0db8:85a3:0000:0000:8a2e:0370:7334"}]
    },
    {
        "input": "服务器IPv6为2001:db8::1，请配置路由。",
        "expected": [{"type": "IPV6_ADDRESS", "value": "2001:db8::1"}]
    },
]

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

def predict_entities(text):
    """预测实体"""
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

# 执行测试
print("🚀 开始零样本检测测试...\n")

results = []
for i, sample in enumerate(test_samples, 1):
    input_text = sample["input"]
    expected = sample["expected"]

    print(f"\n{'='*70}")
    print(f"测试样本 {i}/{len(test_samples)}")
    print(f"{'='*70}")
    print(f"输入: {input_text}")
    print(f"\n期望检测:")
    for e in expected:
        print(f"  - {e['type']}: {e['value']}")

    # 预测
    predicted = predict_entities(input_text)

    print(f"\n实际检测:")
    if predicted:
        for e in predicted:
            print(f"  - {e['type']}: {e['value']}")
    else:
        print("  (未检测到任何实体)")

    # 分析结果
    expected_set = set((e["type"], e["value"]) for e in expected)
    predicted_set = set((e["type"], e["value"]) for e in predicted)

    tp = len(expected_set & predicted_set)
    fp = len(predicted_set - expected_set)
    fn = len(expected_set - predicted_set)

    print(f"\n结果分析:")
    print(f"  TP (正确): {tp}")
    print(f"  FP (误报): {fp}")
    print(f"  FN (漏报): {fn}")

    if tp == len(expected) and fp == 0:
        print(f"  ✅ 完全正确")
    elif tp > 0:
        print(f"  ⚠️ 部分正确")
    else:
        print(f"  ❌ 完全漏检")

    results.append({
        "sample_id": i,
        "input": input_text,
        "expected": expected,
        "predicted": predicted,
        "tp": tp,
        "fp": fp,
        "fn": fn
    })

# 统计总体结果
total_tp = sum(r["tp"] for r in results)
total_fp = sum(r["fp"] for r in results)
total_fn = sum(r["fn"] for r in results)

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*70}")
print(f"📊 零样本检测总体结果")
print(f"{'='*70}")
print(f"测试样本数: {len(test_samples)}")
print(f"期望实体数: {total_tp + total_fn}")
print(f"\nTP (正确检测): {total_tp}")
print(f"FP (误报): {total_fp}")
print(f"FN (漏报): {total_fn}")
print(f"\nPrecision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")

# 按PII类型统计
pii_stats = {}
for result in results:
    for entity in result["expected"]:
        pii_type = entity["type"]
        if pii_type not in pii_stats:
            pii_stats[pii_type] = {"total": 0, "detected": 0}
        pii_stats[pii_type]["total"] += 1

    for entity in result["predicted"]:
        pii_type = entity["type"]
        if pii_type in pii_stats and (pii_type, entity["value"]) in set((e["type"], e["value"]) for e in result["expected"]):
            pii_stats[pii_type]["detected"] += 1

print(f"\n按PII类型统计:")
print(f"{'-'*70}")
print(f"{'PII类型':<30} {'检测数/总数':<15} {'检测率':<10}")
print(f"{'-'*70}")
for pii_type, stats in sorted(pii_stats.items()):
    rate = stats["detected"] / stats["total"] if stats["total"] > 0 else 0
    print(f"{pii_type:<30} {stats['detected']}/{stats['total']:<13} {rate*100:>6.1f}%")

# 保存结果
output_file = "logs/zero_shot_11pii_test_result.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "model_path": MODEL_PATH,
        "sample_size": len(test_samples),
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn
        },
        "by_pii_type": pii_stats,
        "details": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n✅ 详细结果已保存至: {output_file}")

# 结论建议
print(f"\n{'='*70}")
print(f"💡 结论与建议")
print(f"{'='*70}")

if recall >= 0.50:
    print(f"✅ 模型具备较强的零样本检测能力 (Recall {recall*100:.1f}% ≥ 50%)")
    print(f"   建议：可以直接使用规则增强来补充，无需大量训练数据")
elif recall >= 0.20:
    print(f"⚠️ 模型具备一定的零样本检测能力 (Recall {recall*100:.1f}% 在20-50%)")
    print(f"   建议：建议每种PII类型准备500-1000个训练样本重新训练")
else:
    print(f"❌ 模型几乎无零样本检测能力 (Recall {recall*100:.1f}% < 20%)")
    print(f"   建议：必须为每种PII类型准备1000-2000个训练样本")

print(f"\n检测率最低的3种PII类型需要重点优化：")
sorted_pii = sorted(pii_stats.items(), key=lambda x: x[1]["detected"]/x[1]["total"] if x[1]["total"] > 0 else 0)
for pii_type, stats in sorted_pii[:3]:
    rate = stats["detected"] / stats["total"] if stats["total"] > 0 else 0
    print(f"  - {pii_type}: {rate*100:.1f}% ({stats['detected']}/{stats['total']})")

print(f"\n{'='*70}")
print(f"✅ 测试完成！")
print(f"{'='*70}")
