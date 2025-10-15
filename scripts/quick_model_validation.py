#!/usr/bin/env python3
"""
快速模型验证脚本 - 优化版本
专注于快速获取模型性能指标,不追求完美评估
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

class QuickEvaluator:
    """快速评估器 - 优化性能"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)

        print(f"\n🚀 快速模型验证")
        print(f"=" * 70)
        print(f"模型路径: {self.model_path}")
        print(f"设备: {self.device}")
        print(f"=" * 70)

        self.tokenizer, self.model = self._load_model()

    def _setup_device(self, device: str) -> str:
        """设置设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """加载模型"""
        print("\n[1/3] 加载模型...")

        # 读取 adapter_config 获取基础模型路径
        adapter_config_path = self.model_path / "adapter_config.json"
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)

        base_model_path = adapter_config["base_model_name_or_path"]
        print(f"  基础模型: {base_model_path}")

        # 加载 tokenizer
        print("  加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )

        # 加载基础模型
        print("  加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )

        # 加载 LoRA adapter
        print("  加载 LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(self.model_path))

        # 合并权重以加速推理
        print("  合并权重...")
        model = model.merge_and_unload()
        model.eval()

        print("  ✓ 模型加载完成")

        return tokenizer, model

    def detect_pii(self, text: str) -> Dict[str, Any]:
        """检测 PII"""
        # 格式化提示词
        prompt = (
            f"<|im_start|>system\n"
            f"你是 PII 检测专家。检测以下文本中的 PII，并以 JSON 格式输出实体列表。<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 分词 (截断到256以加速)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)

        # 生成 (限制到128 tokens以加速)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,  # 降低到128加速
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 提取 JSON
        try:
            assistant_marker = "<|im_start|>assistant\n"
            if assistant_marker in output_text:
                output_text = output_text.split(assistant_marker)[-1]

            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1

            if json_start != -1 and json_end > 0:
                json_str = output_text[json_start:json_end]
                result = json.loads(json_str)
                return result
        except:
            pass

        return {"entities": []}

    def evaluate_on_subset(self, test_data_path: str, sample_size: int = 500):
        """在子集上评估"""
        print(f"\n[2/3] 在测试子集上评估...")
        print(f"  测试数据: {test_data_path}")
        print(f"  子集大小: {sample_size}")

        # 加载测试数据
        samples = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                samples.append(json.loads(line))

        print(f"  ✓ 加载了 {len(samples)} 个样本")

        # 评估
        tp, fp, fn = 0, 0, 0
        success_count = 0
        error_count = 0

        print(f"\n  开始推理...")
        start_time = time.time()

        for sample in tqdm(samples, desc="  处理中"):
            try:
                # 预测 (input字段是文本)
                pred_result = self.detect_pii(sample["input"])
                pred_entities = pred_result.get("entities", [])

                # 真实标签 (output.entities)
                expected_entities = sample.get("output", {}).get("entities", [])

                # 简单匹配 (type + value)
                pred_set = {(e["type"], e["value"]) for e in pred_entities if "type" in e and "value" in e}
                expected_set = {(e["type"], e["value"]) for e in expected_entities if "type" in e and "value" in e}

                # 计算指标
                tp += len(pred_set & expected_set)
                fp += len(pred_set - expected_set)
                fn += len(expected_set - pred_set)

                success_count += 1

            except Exception as e:
                error_count += 1
                if error_count == 1:  # 只打印第一个错误
                    print(f"\n  ⚠️  第一个错误: {type(e).__name__}: {str(e)}")

        elapsed = time.time() - start_time

        print(f"\n  ✓ 评估完成")
        print(f"  耗时: {elapsed:.1f}秒 ({elapsed/len(samples):.2f}秒/样本)")
        print(f"  成功: {success_count}, 失败: {error_count}")

        return {
            "samples": len(samples),
            "success": success_count,
            "errors": error_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "elapsed": elapsed
        }

    def print_results(self, metrics: Dict[str, Any]):
        """打印结果"""
        print(f"\n[3/3] 验证结果")
        print(f"=" * 70)

        # 计算准确性指标
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

        print(f"\n混淆矩阵:")
        print(f"  TP (正确检测): {tp}")
        print(f"  FP (误报): {fp}")
        print(f"  FN (漏报): {fn}")

        print(f"\n准确性指标:")
        print(f"  Precision (精确率): {precision:.2%}")
        print(f"  Recall (召回率):    {recall:.2%}")
        print(f"  F1-Score:           {f1:.2%}")
        print(f"  F2-Score:           {f2:.2%}")

        print(f"\n性能指标:")
        print(f"  测试样本数: {metrics['samples']}")
        print(f"  总耗时: {metrics['elapsed']:.1f}秒")
        print(f"  平均速度: {metrics['elapsed']/metrics['samples']:.2f}秒/样本")

        # 验证通过/失败
        print(f"\n{'=' * 70}")

        target_f1 = 0.875
        target_recall = 0.90

        if f1 >= target_f1 and recall >= target_recall:
            print(f"✅ 验证通过!")
            print(f"   F1-Score {f1:.2%} >= 目标 {target_f1:.0%}")
            print(f"   Recall {recall:.2%} >= 目标 {target_recall:.0%}")
            result = "PASS"
        else:
            print(f"❌ 验证未通过")
            if f1 < target_f1:
                print(f"   F1-Score {f1:.2%} < 目标 {target_f1:.0%} (差距: {(target_f1-f1)*100:.1f}%)")
            if recall < target_recall:
                print(f"   Recall {recall:.2%} < 目标 {target_recall:.0%} (差距: {(target_recall-recall)*100:.1f}%)")
            result = "FAIL"

        print(f"{'=' * 70}")

        return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="快速模型验证")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--test-data", required=True, help="测试数据路径")
    parser.add_argument("--sample-size", type=int, default=500, help="测试样本数 (默认500)")

    args = parser.parse_args()

    # 创建评估器
    evaluator = QuickEvaluator(args.model)

    # 评估
    metrics = evaluator.evaluate_on_subset(args.test_data, args.sample_size)

    # 打印结果
    result = evaluator.print_results(metrics)

    # 返回退出码
    sys.exit(0 if result == "PASS" else 1)


if __name__ == "__main__":
    main()
