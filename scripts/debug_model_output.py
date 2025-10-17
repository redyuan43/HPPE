#!/usr/bin/env python3
"""调试模型输出"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

# 加载模型
model_path = Path("models/pii_detector_qwen3_06b_aggressive/final")

adapter_config_path = model_path / "adapter_config.json"
with open(adapter_config_path) as f:
    adapter_config = json.load(f)

base_model_path = adapter_config["base_model_name_or_path"]

print(f"加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    trust_remote_code=True
)

print(f"加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)

print(f"加载 LoRA adapter...")
model = PeftModel.from_pretrained(base_model, str(model_path))

print(f"合并权重...")
model = model.merge_and_unload()
model.eval()

# 测试文本
test_text = "我的名字是张三，手机号是13812345678。"

# 测试3种提示词
prompts = [
    # 原始提示词 (完整版)
    (
        f"<|im_start|>system\n"
        f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{test_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    ),
    # 简化提示词 (当前闪电版)
    (
        f"<|im_start|>system\n"
        f"PII检测，JSON格式输出。<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{test_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    ),
]

max_new_tokens_list = [32, 64, 128]

for i, prompt in enumerate(prompts, 1):
    print(f"\n{'='*70}")
    print(f"提示词 #{i}:")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    print(f"{'='*70}")

    for max_tokens in max_new_tokens_list:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 提取assistant部分
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in output_text:
            response = output_text.split(assistant_marker)[-1]
        else:
            response = output_text

        print(f"\nmax_new_tokens={max_tokens}:")
        print(f"  输出长度: {len(response)} 字符")
        print(f"  输出内容:\n{response[:500]}")
        print()
