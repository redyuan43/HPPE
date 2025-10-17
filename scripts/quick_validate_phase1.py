#!/usr/bin/env python3
"""
Phase 1 快速验证脚本 - 简化版
验证GPU0和GPU1模型对11种新PII的识别效果
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random

print("="*70)
print("📊 Phase 1 模型快速验证")
print("="*70)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")

# 配置
BASE_MODEL = "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-4B"
GPU0_MODEL = "models/pii_11new_gpu0/final"
GPU1_MODEL = "models/pii_11new_gpu1/final"
OUTPUT_REPORT = "logs/phase1_validation_report.json"

# GPU0负责的PII类型
GPU0_TYPES = ["BANK_CARD", "PASSPORT", "HK_MACAU_PASS", "POSTAL_CODE", "IP_ADDRESS", "MAC_ADDRESS"]
# GPU1负责的PII类型
GPU1_TYPES = ["VEHICLE_PLATE", "DRIVER_LICENSE", "SOCIAL_SECURITY_CARD", "UNIFIED_SOCIAL_CREDIT_CODE", "IPV6_ADDRESS"]

# 生成测试样本（简化版，每种10个）
def generate_test_samples():
    """为每种PII类型生成10个测试样本"""
    samples = {}

    print("📝 生成测试样本...")

    # BANK_CARD
    samples["BANK_CARD"] = []
    for _ in range(10):
        card = "6217" + "".join([str(random.randint(0, 9)) for _ in range(12)])
        samples["BANK_CARD"].append({
            "text": f"请转账到银行卡{card}",
            "expected_type": "BANK_CARD",
            "expected_value": card
        })

    # PASSPORT
    samples["PASSPORT"] = []
    for _ in range(10):
        passport = "E" + "".join([str(random.randint(0, 9)) for _ in range(8)])
        samples["PASSPORT"].append({
            "text": f"护照号{passport}",
            "expected_type": "PASSPORT",
            "expected_value": passport
        })

    # HK_MACAU_PASS
    samples["HK_MACAU_PASS"] = []
    for _ in range(10):
        pass_id = "C" + "".join([str(random.randint(0, 9)) for _ in range(8)])
        samples["HK_MACAU_PASS"].append({
            "text": f"港澳通行证{pass_id}",
            "expected_type": "HK_MACAU_PASS",
            "expected_value": pass_id
        })

    # POSTAL_CODE
    samples["POSTAL_CODE"] = []
    for _ in range(10):
        code = "".join([str(random.randint(0, 9)) for _ in range(6)])
        samples["POSTAL_CODE"].append({
            "text": f"邮编{code}",
            "expected_type": "POSTAL_CODE",
            "expected_value": code
        })

    # IP_ADDRESS
    samples["IP_ADDRESS"] = []
    for _ in range(10):
        ip = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        samples["IP_ADDRESS"].append({
            "text": f"IP地址{ip}",
            "expected_type": "IP_ADDRESS",
            "expected_value": ip
        })

    # MAC_ADDRESS
    samples["MAC_ADDRESS"] = []
    for _ in range(10):
        mac = ":".join([f"{random.randint(0,255):02x}" for _ in range(6)])
        samples["MAC_ADDRESS"].append({
            "text": f"MAC地址{mac}",
            "expected_type": "MAC_ADDRESS",
            "expected_value": mac
        })

    # VEHICLE_PLATE
    samples["VEHICLE_PLATE"] = []
    provinces = ["京", "沪", "粤"]
    for _ in range(10):
        plate = random.choice(provinces) + random.choice("ABCDEFGH") + "".join([random.choice("0123456789ABCDEFGH") for _ in range(5)])
        samples["VEHICLE_PLATE"].append({
            "text": f"车牌{plate}",
            "expected_type": "VEHICLE_PLATE",
            "expected_value": plate
        })

    # DRIVER_LICENSE
    samples["DRIVER_LICENSE"] = []
    for _ in range(10):
        dl = "".join([str(random.randint(0, 9)) for _ in range(18)])
        samples["DRIVER_LICENSE"].append({
            "text": f"驾驶证{dl}",
            "expected_type": "DRIVER_LICENSE",
            "expected_value": dl
        })

    # SOCIAL_SECURITY_CARD
    samples["SOCIAL_SECURITY_CARD"] = []
    for _ in range(10):
        ssc = "".join([str(random.randint(0, 9)) for _ in range(18)])
        samples["SOCIAL_SECURITY_CARD"].append({
            "text": f"社保卡{ssc}",
            "expected_type": "SOCIAL_SECURITY_CARD",
            "expected_value": ssc
        })

    # UNIFIED_SOCIAL_CREDIT_CODE
    samples["UNIFIED_SOCIAL_CREDIT_CODE"] = []
    for _ in range(10):
        code = "91" + "".join([str(random.randint(0, 9)) for _ in range(16)])
        samples["UNIFIED_SOCIAL_CREDIT_CODE"].append({
            "text": f"统一社会信用代码{code}",
            "expected_type": "UNIFIED_SOCIAL_CREDIT_CODE",
            "expected_value": code
        })

    # IPV6_ADDRESS
    samples["IPV6_ADDRESS"] = []
    for _ in range(10):
        ipv6 = ":".join([f"{random.randint(0,65535):04x}" for _ in range(8)])
        samples["IPV6_ADDRESS"].append({
            "text": f"IPv6地址{ipv6}",
            "expected_type": "IPV6_ADDRESS",
            "expected_value": ipv6
        })

    print(f"✅ 生成了 {sum(len(v) for v in samples.values())} 个测试样本\n")
    return samples

# 加载模型
def load_model(adapter_path):
    """加载模型"""
    print(f"📦 加载模型: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print(f"✅ 模型加载完成\n")
    return model, tokenizer

# 测试单个样本
def test_sample(model, tokenizer, text, expected_type):
    """测试单个样本"""
    prompt = (
        f"<|im_start|>system\n"
        f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # 判断是否检测到正确类型
    correct = expected_type in response

    return correct, response

# 测试模型
def test_model(model, tokenizer, samples, pii_types, model_name):
    """测试模型"""
    print(f"🔍 测试{model_name}...")
    print(f"   PII类型: {', '.join(pii_types)}\n")

    results = {}

    for pii_type in pii_types:
        print(f"   测试 {pii_type}...", end=" ")
        correct_count = 0
        total_count = len(samples[pii_type])
        details = []

        for sample in samples[pii_type]:
            correct, response = test_sample(model, tokenizer, sample["text"], sample["expected_type"])
            if correct:
                correct_count += 1
            details.append({
                "text": sample["text"],
                "expected": sample["expected_type"],
                "response": response[:100],  # 只保存前100字符
                "correct": correct
            })

        recall = 100 * correct_count / total_count
        print(f"{correct_count}/{total_count} ({recall:.0f}%)")

        results[pii_type] = {
            "correct": correct_count,
            "total": total_count,
            "recall": recall,
            "details": details
        }

    print(f"\n✅ {model_name}测试完成\n")
    return results

# 主函数
def main():
    # 生成测试样本
    samples = generate_test_samples()

    # 测试GPU0模型
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    gpu0_model, gpu0_tokenizer = load_model(GPU0_MODEL)
    gpu0_results = test_model(gpu0_model, gpu0_tokenizer, samples, GPU0_TYPES, "GPU0模型")
    del gpu0_model, gpu0_tokenizer
    torch.cuda.empty_cache()

    # 测试GPU1模型
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    gpu1_model, gpu1_tokenizer = load_model(GPU1_MODEL)
    gpu1_results = test_model(gpu1_model, gpu1_tokenizer, samples, GPU1_TYPES, "GPU1模型")
    del gpu1_model, gpu1_tokenizer
    torch.cuda.empty_cache()

    # 合并结果
    all_results = {**gpu0_results, **gpu1_results}

    # 计算整体指标
    total_correct = sum(r["correct"] for r in all_results.values())
    total_samples = sum(r["total"] for r in all_results.values())
    overall_recall = 100 * total_correct / total_samples

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 验证结果汇总")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    print(f"整体Recall: {overall_recall:.1f}% ({total_correct}/{total_samples})\n")

    for pii_type in GPU0_TYPES + GPU1_TYPES:
        result = all_results[pii_type]
        print(f"  {pii_type:30s}: {result['recall']:5.1f}% ({result['correct']}/{result['total']})")

    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "recall": round(overall_recall, 2),
            "correct": total_correct,
            "total": total_samples
        },
        "by_type": {
            pii_type: {
                "recall": round(r["recall"], 2),
                "correct": r["correct"],
                "total": r["total"]
            }
            for pii_type, r in all_results.items()
        },
        "details": all_results
    }

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 验证报告已保存: {OUTPUT_REPORT}")
    print("="*70)

    # 返回整体recall用于后续判断
    return overall_recall

if __name__ == "__main__":
    recall = main()
    print(f"\n最终Recall: {recall:.2f}%")
