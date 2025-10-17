#!/usr/bin/env python3
"""
6种PII模型 vs 17种PII模型对比验证

功能：
1. 加载6种和17种PII模型
2. 在相同测试集上运行
3. 计算Precision/Recall/F1
4. 生成详细对比报告
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU1

import sys
from pathlib import Path
import json
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hppe.engines.llm import QwenFineTunedEngine
from hppe.engines.llm.recognizers import FineTunedLLMRecognizer
from hppe.models.entity import Entity


class ModelComparator:
    """模型对比验证器"""

    def __init__(
        self,
        model_6pii_path: str,
        model_17pii_path: str,
        test_data_path: str
    ):
        """
        初始化对比验证器

        Args:
            model_6pii_path: 6种PII模型路径
            model_17pii_path: 17种PII模型路径
            test_data_path: 测试数据路径
        """
        self.model_6pii_path = model_6pii_path
        self.model_17pii_path = model_17pii_path
        self.test_data_path = test_data_path

        self.engine_6pii = None
        self.engine_17pii = None
        self.recognizer_6pii = None
        self.recognizer_17pii = None

        self.test_cases = []

    def load_test_data(self):
        """加载测试数据"""
        print(f"📂 加载测试数据: {self.test_data_path}")

        with open(self.test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.test_cases.append(json.loads(line))

        print(f"✅ 加载 {len(self.test_cases)} 条测试用例")

    def load_6pii_model(self):
        """加载6种PII模型"""
        print(f"\n📦 加载6种PII模型: {self.model_6pii_path}")

        self.engine_6pii = QwenFineTunedEngine(
            model_path=self.model_6pii_path,
            device="cuda",
            load_in_4bit=True
        )

        self.recognizer_6pii = FineTunedLLMRecognizer(
            llm_engine=self.engine_6pii,
            confidence_threshold=0.7
        )

        # 预热
        _ = self.recognizer_6pii.detect("测试")

        print(f"✅ 6种PII模型加载完成")
        print(f"   支持类型: {self.engine_6pii.get_supported_pii_types()}")

    def load_17pii_model(self):
        """加载17种PII模型"""
        print(f"\n📦 加载17种PII模型: {self.model_17pii_path}")

        # 先卸载6种模型释放内存
        if self.engine_6pii is not None:
            del self.engine_6pii
            del self.recognizer_6pii
            import torch
            torch.cuda.empty_cache()
            print("   6种模型已卸载，内存已释放")

        self.engine_17pii = QwenFineTunedEngine(
            model_path=self.model_17pii_path,
            device="cuda",
            load_in_4bit=True
        )

        self.recognizer_17pii = FineTunedLLMRecognizer(
            llm_engine=self.engine_17pii,
            confidence_threshold=0.7
        )

        # 预热
        _ = self.recognizer_17pii.detect("测试")

        print(f"✅ 17种PII模型加载完成")
        print(f"   支持类型: {self.engine_17pii.get_supported_pii_types()}")

    def evaluate_model(
        self,
        recognizer: FineTunedLLMRecognizer,
        model_name: str
    ) -> Dict:
        """
        评估单个模型

        Args:
            recognizer: 识别器
            model_name: 模型名称

        Returns:
            评估结果
        """
        print(f"\n🧪 评估 {model_name}...")

        # 统计变量
        total_tp = 0  # True Positives
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives

        # 按类型统计
        type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        detailed_results = []

        for i, test_case in enumerate(self.test_cases, 1):
            text = test_case["text"]
            ground_truth = test_case["entities"]

            # 运行检测
            predicted = recognizer.detect(text)

            # 转换为集合（用于比较）
            gt_set = {
                (e["type"], e["value"])
                for e in ground_truth
            }

            pred_set = {
                (e.entity_type, e.value)
                for e in predicted
            }

            # 计算TP, FP, FN
            tp_set = gt_set & pred_set
            fp_set = pred_set - gt_set
            fn_set = gt_set - pred_set

            total_tp += len(tp_set)
            total_fp += len(fp_set)
            total_fn += len(fn_set)

            # 按类型统计
            for entity_type, value in tp_set:
                type_stats[entity_type]["tp"] += 1

            for entity_type, value in fp_set:
                type_stats[entity_type]["fp"] += 1

            for entity_type, value in fn_set:
                type_stats[entity_type]["fn"] += 1

            # 保存详细结果
            detailed_results.append({
                "case_id": i,
                "text": text,
                "ground_truth": list(gt_set),
                "predicted": list(pred_set),
                "tp": list(tp_set),
                "fp": list(fp_set),
                "fn": list(fn_set),
            })

            # 打印进度
            if i % 10 == 0:
                print(f"  进度: {i}/{len(self.test_cases)}")

        # 计算总体指标
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 计算每种类型的指标
        type_metrics = {}
        for entity_type, stats in type_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_type = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            type_metrics[entity_type] = {
                "precision": prec,
                "recall": rec,
                "f1": f1_type,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn  # 真实样本数
            }

        print(f"✅ {model_name} 评估完成")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1-Score: {f1:.2%}")

        return {
            "model_name": model_name,
            "overall": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn
            },
            "by_type": type_metrics,
            "detailed_results": detailed_results
        }

    def compare_models(self) -> Dict:
        """
        对比两个模型

        Returns:
            对比结果
        """
        print("=" * 70)
        print("🆚 模型对比验证")
        print("=" * 70)

        # 加载测试数据
        self.load_test_data()

        # 评估6种PII模型
        self.load_6pii_model()
        results_6pii = self.evaluate_model(self.recognizer_6pii, "6-PII Model")

        # 评估17种PII模型
        self.load_17pii_model()
        results_17pii = self.evaluate_model(self.recognizer_17pii, "17-PII Model")

        # 生成对比报告
        comparison = self._generate_comparison(results_6pii, results_17pii)

        return comparison

    def _generate_comparison(
        self,
        results_6pii: Dict,
        results_17pii: Dict
    ) -> Dict:
        """生成对比报告"""
        print("\n" + "=" * 70)
        print("📊 对比结果")
        print("=" * 70)

        # 总体对比
        print("\n总体性能:")
        print(f"{'指标':<15} {'6-PII':>12} {'17-PII':>12} {'提升':>12}")
        print("-" * 55)

        metrics = ["precision", "recall", "f1"]
        for metric in metrics:
            val_6 = results_6pii["overall"][metric]
            val_17 = results_17pii["overall"][metric]
            delta = val_17 - val_6
            delta_str = f"+{delta:.2%}" if delta >= 0 else f"{delta:.2%}"

            print(f"{metric.capitalize():<15} {val_6:>11.2%} {val_17:>11.2%} {delta_str:>12}")

        # 按类型对比
        print("\n按PII类型对比:")
        print(f"{'类型':<20} {'6-PII F1':>12} {'17-PII F1':>12} {'提升':>12}")
        print("-" * 60)

        all_types = set(results_6pii["by_type"].keys()) | set(results_17pii["by_type"].keys())

        for pii_type in sorted(all_types):
            f1_6 = results_6pii["by_type"].get(pii_type, {}).get("f1", 0)
            f1_17 = results_17pii["by_type"].get(pii_type, {}).get("f1", 0)

            delta = f1_17 - f1_6
            delta_str = f"+{delta:.2%}" if delta >= 0 else f"{delta:.2%}"

            # 标记新增类型
            type_str = pii_type
            if f1_6 == 0 and f1_17 > 0:
                type_str += " *"

            print(f"{type_str:<20} {f1_6:>11.2%} {f1_17:>11.2%} {delta_str:>12}")

        print("\n* 表示17-PII模型新增支持的类型")

        # 汇总
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "test_dataset": self.test_data_path,
            "num_test_cases": len(self.test_cases),
            "model_6pii": {
                "path": self.model_6pii_path,
                "results": results_6pii
            },
            "model_17pii": {
                "path": self.model_17pii_path,
                "results": results_17pii
            },
            "improvement": {
                "precision": results_17pii["overall"]["precision"] - results_6pii["overall"]["precision"],
                "recall": results_17pii["overall"]["recall"] - results_6pii["overall"]["recall"],
                "f1": results_17pii["overall"]["f1"] - results_6pii["overall"]["f1"]
            }
        }

        return comparison

    def save_results(self, comparison: Dict, output_path: str):
        """保存结果到文件"""
        output_file = Path(output_path)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"\n💾 对比结果已保存到: {output_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="6种 vs 17种PII模型对比验证")
    parser.add_argument(
        "--model-6pii",
        default="models/pii_qwen4b_unsloth/final",
        help="6种PII模型路径"
    )
    parser.add_argument(
        "--model-17pii",
        default="models/pii_qwen4b_17types/final",
        help="17种PII模型路径"
    )
    parser.add_argument(
        "--test-data",
        default="data/test_datasets/17pii_test_cases.jsonl",
        help="测试数据路径"
    )
    parser.add_argument(
        "--output",
        default="comparison_6vs17_report.json",
        help="输出报告路径"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.model_6pii).exists():
        print(f"❌ 错误: 6种PII模型不存在: {args.model_6pii}")
        return

    if not Path(args.test_data).exists():
        print(f"❌ 错误: 测试数据不存在: {args.test_data}")
        return

    # 检查17种模型（可能还在训练中）
    if not Path(args.model_17pii).exists():
        print(f"⚠️  警告: 17种PII模型不存在: {args.model_17pii}")
        print("    训练可能还在进行中，将只评估6种PII模型")

        # 只评估6种模型
        comparator = ModelComparator(
            model_6pii_path=args.model_6pii,
            model_17pii_path=args.model_17pii,  # 保留路径，稍后处理
            test_data_path=args.test_data
        )

        comparator.load_test_data()
        comparator.load_6pii_model()
        results_6pii = comparator.evaluate_model(comparator.recognizer_6pii, "6-PII Model")

        # 保存6种模型的结果
        output_6pii = {
            "timestamp": datetime.now().isoformat(),
            "model_path": args.model_6pii,
            "results": results_6pii
        }

        output_file = Path(args.output).parent / "6pii_baseline_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_6pii, f, indent=2, ensure_ascii=False)

        print(f"\n💾 6种PII模型结果已保存到: {output_file}")
        print("\n✅ 基线评估完成！等待17种模型训练完成后再次运行对比。")
        return

    # 两个模型都存在，进行对比
    comparator = ModelComparator(
        model_6pii_path=args.model_6pii,
        model_17pii_path=args.model_17pii,
        test_data_path=args.test_data
    )

    comparison = comparator.compare_models()
    comparator.save_results(comparison, args.output)

    print("\n✅ 对比验证完成！")


if __name__ == "__main__":
    main()
