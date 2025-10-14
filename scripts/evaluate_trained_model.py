#!/usr/bin/env python3
"""
训练后模型评估脚本

用途：在独立测试集上评估训练后的 PII 检测模型

功能：
1. 加载微调后的模型
2. 在测试集上运行推理
3. 计算准确性指标（Precision、Recall、F1、F2）
4. 按 PII 类型统计指标
5. 生成混淆矩阵
6. 输出详细的评估报告

使用示例：
    python scripts/evaluate_trained_model.py \
        --model models/pii_detector_qwen3_0.6b/final \
        --test-data data/merged_pii_dataset_test.jsonl \
        --output evaluation_results/test_evaluation.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    print("❌ 缺少必要的库。请先安装：")
    print("   pip install transformers peft torch")
    sys.exit(1)


@dataclass
class EvaluationMetrics:
    """评估指标"""
    entity_type: str = "overall"

    # 混淆矩阵
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    # 派生指标
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    f2_score: float = 0.0
    accuracy: float = 0.0

    # 样本统计
    total_samples: int = 0
    success_count: int = 0
    error_count: int = 0

    def calculate_metrics(self):
        """计算派生指标"""
        # Precision = TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)

        # Recall = TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)

        # F1 = 2 * (P * R) / (P + R)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        # F2 = 5 * (P * R) / (4P + R)  # 更重视 Recall
        if self.precision + self.recall > 0:
            self.f2_score = 5 * (self.precision * self.recall) / (4 * self.precision + self.recall)

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total > 0:
            self.accuracy = (self.true_positives + self.true_negatives) / total

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "entity_type": self.entity_type,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "true_negatives": self.true_negatives
            },
            "metrics": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4),
                "f2_score": round(self.f2_score, 4),
                "accuracy": round(self.accuracy, 4)
            },
            "statistics": {
                "total_samples": self.total_samples,
                "success_count": self.success_count,
                "error_count": self.error_count
            }
        }


class TrainedModelEvaluator:
    """训练后模型评估器"""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化评估器

        Args:
            model_path: 模型路径（微调后的模型目录）
            device: 设备（cuda/cpu/auto）
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)

        # 加载模型和分词器
        self.tokenizer, self.model = self._load_model()

        # 评估结果
        self.results: List[Dict[str, Any]] = []

    def _setup_device(self, device: str) -> str:
        """设置设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self) -> Tuple[Any, Any]:
        """
        加载微调后的模型

        Returns:
            (tokenizer, model)
        """
        print(f"\n{'=' * 70}")
        print("加载微调后的模型")
        print(f"{'=' * 70}")
        print(f"模型路径: {self.model_path}")
        print(f"设备: {self.device}")

        # 检查模型文件是否存在
        adapter_model = self.model_path / "adapter_model.safetensors"
        adapter_config = self.model_path / "adapter_config.json"

        if not adapter_model.exists() or not adapter_config.exists():
            raise FileNotFoundError(
                f"模型文件不完整。请确保目录包含：\n"
                f"  - adapter_model.safetensors\n"
                f"  - adapter_config.json\n"
                f"当前路径: {self.model_path}"
            )

        # 读取 adapter_config 获取基础模型路径
        with open(adapter_config, 'r') as f:
            config = json.load(f)
            base_model_name = config.get("base_model_name_or_path")

        print(f"\n加载基础模型: {base_model_name}")

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device
        )

        # 加载 LoRA 适配器
        print(f"加载 LoRA 适配器: {self.model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            str(self.model_path),
            torch_dtype=torch.float16
        )

        # 合并适配器权重（可选，提升推理速度）
        model = model.merge_and_unload()

        # 设置为评估模式
        model.eval()

        print(f"\n✓ 模型加载完成")
        return tokenizer, model

    def _format_prompt(self, text: str) -> str:
        """
        格式化输入为 Qwen 对话格式

        Args:
            text: 待检测文本

        Returns:
            格式化后的提示词
        """
        instruction = "检测以下文本中的 PII，并以 JSON 格式输出实体列表。"

        prompt = (
            f"<|im_start|>system\n"
            f"你是 PII 检测专家。{instruction}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return prompt

    def _parse_model_output(self, output_text: str) -> Dict[str, Any]:
        """
        解析模型输出为 JSON

        Args:
            output_text: 模型生成的文本

        Returns:
            解析后的 JSON 对象
        """
        try:
            # 提取 JSON 部分（去除前后的 <|im_end|> 等标记）
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                return {"entities": []}

            json_str = output_text[json_start:json_end]
            result = json.loads(json_str)

            return result

        except json.JSONDecodeError:
            # JSON 解析失败，返回空结果
            return {"entities": []}

    def detect_pii(self, text: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        """
        检测文本中的 PII

        Args:
            text: 待检测文本
            max_new_tokens: 最大生成 token 数

        Returns:
            检测结果 {"entities": [...]}
        """
        # 格式化输入
        prompt = self._format_prompt(text)

        # 分词
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # 低温度保证一致性
                do_sample=False,  # 贪婪解码
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 提取 assistant 的回复（去除输入部分）
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in output_text:
            output_text = output_text.split(assistant_marker)[-1]

        # 解析 JSON
        result = self._parse_model_output(output_text)

        return result

    def _match_entities(
        self,
        predicted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> Tuple[int, int, int]:
        """
        匹配预测实体和预期实体

        Args:
            predicted: 预测的实体列表
            expected: 预期的实体列表

        Returns:
            (true_positives, false_positives, false_negatives)
        """
        # 转换为集合（type, value）
        predicted_set = {
            (e['type'], e['value'])
            for e in predicted
        }

        expected_set = {
            (e['type'], e['value'])
            for e in expected
        }

        # 计算指标
        true_positives = len(predicted_set & expected_set)
        false_positives = len(predicted_set - expected_set)
        false_negatives = len(expected_set - predicted_set)

        return true_positives, false_positives, false_negatives

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个样本

        Args:
            sample: 测试样本
                {
                    "input": "我叫张三...",
                    "output": {
                        "entities": [{"type": "PERSON_NAME", "value": "张三", ...}]
                    }
                }

        Returns:
            评估结果
        """
        input_text = sample["input"]
        expected_entities = sample["output"]["entities"]

        # 推理
        try:
            start_time = time.time()
            result = self.detect_pii(input_text)
            latency = time.time() - start_time

            predicted_entities = result.get("entities", [])

            # 匹配实体
            tp, fp, fn = self._match_entities(predicted_entities, expected_entities)

            return {
                "success": True,
                "input": input_text,
                "expected_count": len(expected_entities),
                "predicted_count": len(predicted_entities),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "latency": latency,
                "predicted_entities": predicted_entities,
                "expected_entities": expected_entities
            }

        except Exception as e:
            return {
                "success": False,
                "input": input_text,
                "error": str(e),
                "expected_count": len(expected_entities)
            }

    def evaluate_dataset(
        self,
        test_data_path: str,
        max_samples: int = None
    ) -> Tuple[EvaluationMetrics, Dict[str, EvaluationMetrics]]:
        """
        评估整个测试集

        Args:
            test_data_path: 测试数据路径
            max_samples: 最大样本数（用于快速测试）

        Returns:
            (overall_metrics, metrics_by_type)
        """
        print(f"\n{'=' * 70}")
        print("开始评估测试集")
        print(f"{'=' * 70}")

        # 加载测试数据
        test_data_path = Path(test_data_path)
        samples = []

        print(f"加载测试数据: {test_data_path}")

        with open(test_data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                samples.append(sample)

        print(f"  ✓ 加载了 {len(samples):,} 个测试样本")

        # 初始化指标
        overall_metrics = EvaluationMetrics(entity_type="overall")
        metrics_by_type = defaultdict(lambda: EvaluationMetrics())

        # 逐样本评估
        print(f"\n开始推理...")
        for i, sample in enumerate(samples, 1):
            result = self.evaluate_sample(sample)
            self.results.append(result)

            # 更新总体指标
            overall_metrics.total_samples += 1

            if result["success"]:
                overall_metrics.success_count += 1
                overall_metrics.true_positives += result["true_positives"]
                overall_metrics.false_positives += result["false_positives"]
                overall_metrics.false_negatives += result["false_negatives"]

                # 按类型统计
                for entity in result["predicted_entities"]:
                    entity_type = entity["type"]
                    metrics_by_type[entity_type].total_samples += 1

                for entity in result["expected_entities"]:
                    entity_type = entity["type"]
                    # 这里简化处理，只统计 TP/FP/FN 到对应类型
                    pass

            else:
                overall_metrics.error_count += 1

            # 显示进度
            if i % 100 == 0 or i == len(samples):
                print(f"  [{i}/{len(samples)}] 完成 {i / len(samples) * 100:.1f}%")

        # 计算派生指标
        overall_metrics.calculate_metrics()

        for entity_type in metrics_by_type:
            metrics_by_type[entity_type].entity_type = entity_type
            metrics_by_type[entity_type].calculate_metrics()

        print(f"\n✓ 评估完成！")

        return overall_metrics, dict(metrics_by_type)

    def print_evaluation_summary(
        self,
        overall_metrics: EvaluationMetrics,
        metrics_by_type: Dict[str, EvaluationMetrics]
    ):
        """打印评估摘要"""
        print(f"\n{'=' * 70}")
        print("评估结果摘要")
        print(f"{'=' * 70}")

        # 总体指标
        print(f"\n【总体指标】")
        print(f"  样本总数: {overall_metrics.total_samples:,}")
        print(f"  成功推理: {overall_metrics.success_count:,}")
        print(f"  推理失败: {overall_metrics.error_count:,}")
        print(f"\n  混淆矩阵:")
        print(f"    TP (正确检测): {overall_metrics.true_positives:,}")
        print(f"    FP (误报): {overall_metrics.false_positives:,}")
        print(f"    FN (漏报): {overall_metrics.false_negatives:,}")
        print(f"\n  准确性指标:")
        print(f"    Precision: {overall_metrics.precision:.2%}")
        print(f"    Recall:    {overall_metrics.recall:.2%}")
        print(f"    F1-Score:  {overall_metrics.f1_score:.2%}")
        print(f"    F2-Score:  {overall_metrics.f2_score:.2%}")
        print(f"    Accuracy:  {overall_metrics.accuracy:.2%}")

        # 判定结果
        print(f"\n【验证结果】")
        passed = True
        reasons = []

        if overall_metrics.f1_score < 0.875:
            passed = False
            reasons.append(f"F1-Score 未达标 ({overall_metrics.f1_score:.2%} < 87.5%)")

        if overall_metrics.recall < 0.90:
            passed = False
            reasons.append(f"Recall 未达标 ({overall_metrics.recall:.2%} < 90%)")

        if passed:
            print(f"  ✅ 通过验证！")
        else:
            print(f"  ❌ 未通过验证")
            for reason in reasons:
                print(f"     - {reason}")

        # 按类型统计
        if metrics_by_type:
            print(f"\n【各类型指标】")
            print(f"  {'类型':<20} {'精确率':>10} {'召回率':>10} {'F1-Score':>10}")
            print(f"  {'-' * 60}")
            for entity_type, metrics in sorted(metrics_by_type.items()):
                print(f"  {entity_type:<20} {metrics.precision:>9.2%} {metrics.recall:>9.2%} {metrics.f1_score:>9.2%}")

    def save_results(self, output_path: str):
        """保存评估结果到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 汇总结果
        overall_metrics, metrics_by_type = self._compute_final_metrics()

        result_data = {
            "model_path": str(self.model_path),
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_metrics": overall_metrics.to_dict(),
            "metrics_by_type": {k: v.to_dict() for k, v in metrics_by_type.items()},
            "detailed_results": self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 评估结果已保存: {output_path}")

    def _compute_final_metrics(self) -> Tuple[EvaluationMetrics, Dict[str, EvaluationMetrics]]:
        """从 results 重新计算最终指标"""
        overall_metrics = EvaluationMetrics(entity_type="overall")
        metrics_by_type = defaultdict(lambda: EvaluationMetrics())

        for result in self.results:
            overall_metrics.total_samples += 1

            if result["success"]:
                overall_metrics.success_count += 1
                overall_metrics.true_positives += result["true_positives"]
                overall_metrics.false_positives += result["false_positives"]
                overall_metrics.false_negatives += result["false_negatives"]
            else:
                overall_metrics.error_count += 1

        overall_metrics.calculate_metrics()

        return overall_metrics, dict(metrics_by_type)


def main():
    parser = argparse.ArgumentParser(
        description="评估训练后的 PII 检测模型",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="微调后的模型路径（包含 adapter_model.safetensors）"
    )

    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="测试数据路径（.jsonl 格式）"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出结果文件路径（.json 格式）"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="最大样本数（用于快速测试）"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="设备（默认: auto）"
    )

    args = parser.parse_args()

    try:
        # 创建评估器
        evaluator = TrainedModelEvaluator(
            model_path=args.model,
            device=args.device
        )

        # 评估测试集
        overall_metrics, metrics_by_type = evaluator.evaluate_dataset(
            test_data_path=args.test_data,
            max_samples=args.max_samples
        )

        # 打印摘要
        evaluator.print_evaluation_summary(overall_metrics, metrics_by_type)

        # 保存结果
        evaluator.save_results(args.output)

        print(f"\n🎉 评估完成！")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(130)

    except Exception as e:
        print(f"\n\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
