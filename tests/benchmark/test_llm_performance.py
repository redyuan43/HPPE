#!/usr/bin/env python3
"""
LLM 引擎性能基准测试

测试内容：
1. 推理延迟（P50/P95/P99）
2. GPU内存占用
3. 吞吐量（RPS）
4. 批处理性能
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU1避免与训练冲突

import sys
from pathlib import Path
import time
import statistics
import json
from typing import List, Dict
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from hppe.engines.llm import QwenFineTunedEngine
from hppe.engines.llm.recognizers import FineTunedLLMRecognizer

# GPU监控
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("⚠️  pynvml not installed. GPU monitoring disabled.")


class GPUMonitor:
    """GPU内存和利用率监控"""

    def __init__(self):
        self.enabled = HAS_NVML
        if self.enabled:
            pynvml.nvmlInit()
            # 获取GPU1（CUDA_VISIBLE_DEVICES=1会映射为设备0）
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(1)

    def get_memory_usage(self) -> Dict:
        """获取GPU内存使用情况"""
        if not self.enabled:
            return {"used_mb": 0, "total_mb": 0, "utilization_percent": 0}

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "used_mb": mem_info.used / 1024 / 1024,
            "total_mb": mem_info.total / 1024 / 1024,
            "utilization_percent": (mem_info.used / mem_info.total) * 100
        }

    def get_gpu_utilization(self) -> int:
        """获取GPU计算利用率"""
        if not self.enabled:
            return 0

        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return util.gpu

    def __del__(self):
        if self.enabled:
            pynvml.nvmlShutdown()


class PerformanceBenchmark:
    """性能基准测试类"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化性能测试

        Args:
            model_path: 模型路径
            device: 设备（cuda/cpu）
        """
        self.model_path = model_path
        self.device = device
        self.engine = None
        self.recognizer = None
        self.gpu_monitor = GPUMonitor()

        # 测试文本集合（不同长度）
        self.test_texts = self._prepare_test_texts()

    def _prepare_test_texts(self) -> Dict[str, List[str]]:
        """准备不同长度的测试文本"""
        return {
            "short": [
                "我是张三",
                "电话13812345678",
                "邮箱test@example.com",
                "北京市海淀区",
                "阿里巴巴公司",
            ],
            "medium": [
                "我是张三，电话13812345678，邮箱zhangsan@example.com",
                "联系人：李四，手机：13900139000，工作单位：腾讯科技",
                "收件地址：上海市浦东新区陆家嘴环路1000号",
                "王五先生居住在广州市天河区珠江新城",
                "客户信息：赵六，身份证号：110101199001011234",
            ],
            "long": [
                "尊敬的张三先生，您好！我们是阿里巴巴公司，地址位于杭州市余杭区文一西路969号。"
                "您的订单已发货，快递单号：SF1234567890，收货地址为北京市朝阳区建国路88号SOHO现代城。"
                "如有问题请联系客服，电话：400-123-4567，邮箱：service@example.com。",

                "客户档案：李四，性别男，身份证号：310101198506151234，手机：13900139000，"
                "邮箱：lisi@company.com，公司：腾讯科技有限公司，地址：深圳市南山区科技园。"
                "紧急联系人：王五，关系：配偶，电话：13800138000。",

                "会议通知：定于2025年10月20日上午10:00在上海市黄浦区人民广场100号会议室召开。"
                "参会人员：张三（阿里巴巴）、李四（腾讯）、王五（字节跳动）。"
                "会议主题：隐私保护技术研讨。联系人：赵六，电话：021-12345678。",
            ]
        }

    def setup(self):
        """初始化引擎（测量加载时间）"""
        print("\n📦 正在加载模型...")
        start_time = time.time()

        self.engine = QwenFineTunedEngine(
            model_path=self.model_path,
            device=self.device,
            load_in_4bit=True
        )

        self.recognizer = FineTunedLLMRecognizer(
            llm_engine=self.engine,
            confidence_threshold=0.75
        )

        load_time = time.time() - start_time

        # 预热（首次推理较慢）
        print("🔥 预热模型...")
        warmup_start = time.time()
        _ = self.recognizer.detect("测试文本")
        warmup_time = time.time() - warmup_start

        mem_usage = self.gpu_monitor.get_memory_usage()

        return {
            "load_time_sec": load_time,
            "warmup_time_sec": warmup_time,
            "gpu_memory_mb": mem_usage["used_mb"],
            "gpu_memory_percent": mem_usage["utilization_percent"]
        }

    def benchmark_latency(self, text_category: str = "medium", num_runs: int = 20) -> Dict:
        """
        测试推理延迟

        Args:
            text_category: 文本类别（short/medium/long）
            num_runs: 运行次数

        Returns:
            延迟统计数据
        """
        texts = self.test_texts[text_category]
        latencies = []

        print(f"\n⏱️  测试延迟（{text_category}文本，{num_runs}次运行）...")

        for i in range(num_runs):
            text = texts[i % len(texts)]

            start_time = time.perf_counter()
            _ = self.recognizer.detect(text)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # 计算统计指标
        latencies_sorted = sorted(latencies)

        return {
            "text_category": text_category,
            "num_runs": num_runs,
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p50_ms": latencies_sorted[int(len(latencies) * 0.50)],
            "p95_ms": latencies_sorted[int(len(latencies) * 0.95)],
            "p99_ms": latencies_sorted[int(len(latencies) * 0.99)],
            "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "all_latencies": latencies
        }

    def benchmark_throughput(self, duration_sec: int = 30, text_category: str = "medium") -> Dict:
        """
        测试吞吐量（RPS）

        Args:
            duration_sec: 测试持续时间（秒）
            text_category: 文本类别

        Returns:
            吞吐量数据
        """
        texts = self.test_texts[text_category]

        print(f"\n📊 测试吞吐量（持续{duration_sec}秒）...")

        start_time = time.time()
        end_time = start_time + duration_sec

        request_count = 0
        total_entities = 0
        gpu_utils = []

        while time.time() < end_time:
            text = texts[request_count % len(texts)]
            entities = self.recognizer.detect(text)

            request_count += 1
            total_entities += len(entities)

            # 采样GPU利用率
            if request_count % 5 == 0:
                gpu_utils.append(self.gpu_monitor.get_gpu_utilization())

        actual_duration = time.time() - start_time
        rps = request_count / actual_duration

        return {
            "duration_sec": actual_duration,
            "total_requests": request_count,
            "total_entities_detected": total_entities,
            "requests_per_second": rps,
            "avg_gpu_utilization": statistics.mean(gpu_utils) if gpu_utils else 0
        }

    def benchmark_memory(self) -> Dict:
        """
        测试内存占用

        Returns:
            内存使用数据
        """
        print("\n💾 测试内存占用...")

        # 空闲状态
        mem_idle = self.gpu_monitor.get_memory_usage()

        # 单次推理
        _ = self.recognizer.detect(self.test_texts["medium"][0])
        mem_single = self.gpu_monitor.get_memory_usage()

        # 批量推理
        for text in self.test_texts["medium"]:
            _ = self.recognizer.detect(text)
        mem_batch = self.gpu_monitor.get_memory_usage()

        return {
            "idle": mem_idle,
            "single_inference": mem_single,
            "batch_inference": mem_batch
        }

    def run_full_benchmark(self) -> Dict:
        """
        运行完整的基准测试

        Returns:
            完整的测试结果
        """
        print("=" * 70)
        print("🚀 开始性能基准测试")
        print("=" * 70)

        # 1. 模型加载和初始化
        setup_results = self.setup()

        # 2. 延迟测试（不同文本长度）
        latency_results = {}
        for category in ["short", "medium", "long"]:
            latency_results[category] = self.benchmark_latency(
                text_category=category,
                num_runs=20
            )

        # 3. 吞吐量测试
        throughput_results = self.benchmark_throughput(duration_sec=30)

        # 4. 内存测试
        memory_results = self.benchmark_memory()

        # 汇总结果
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "device": self.device,
            "setup": setup_results,
            "latency": latency_results,
            "throughput": throughput_results,
            "memory": memory_results,
        }

        # 打印摘要
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """打印测试结果摘要"""
        print("\n" + "=" * 70)
        print("📈 性能测试结果摘要")
        print("=" * 70)

        # 初始化
        setup = results["setup"]
        print(f"\n🔧 模型加载:")
        print(f"  加载时间: {setup['load_time_sec']:.2f}秒")
        print(f"  预热时间: {setup['warmup_time_sec']:.2f}秒")
        print(f"  GPU内存: {setup['gpu_memory_mb']:.0f}MB ({setup['gpu_memory_percent']:.1f}%)")

        # 延迟
        print(f"\n⏱️  推理延迟:")
        for category, data in results["latency"].items():
            print(f"  {category.upper()} 文本:")
            print(f"    P50: {data['p50_ms']:.1f}ms")
            print(f"    P95: {data['p95_ms']:.1f}ms")
            print(f"    P99: {data['p99_ms']:.1f}ms")
            print(f"    平均: {data['mean_ms']:.1f}ms ± {data['stdev_ms']:.1f}ms")

        # 吞吐量
        throughput = results["throughput"]
        print(f"\n📊 吞吐量:")
        print(f"  RPS: {throughput['requests_per_second']:.2f}")
        print(f"  总请求数: {throughput['total_requests']}")
        print(f"  检测实体数: {throughput['total_entities_detected']}")
        print(f"  平均GPU利用率: {throughput['avg_gpu_utilization']:.1f}%")

        # 内存
        memory = results["memory"]
        print(f"\n💾 内存占用:")
        print(f"  空闲: {memory['idle']['used_mb']:.0f}MB")
        print(f"  单次推理: {memory['single_inference']['used_mb']:.0f}MB")
        print(f"  批量推理: {memory['batch_inference']['used_mb']:.0f}MB")

        print("\n" + "=" * 70)


# ==================== Pytest 测试用例 ====================

@pytest.fixture(scope="module")
def benchmark():
    """创建性能测试实例"""
    model_path = "models/pii_qwen4b_unsloth/final"
    bench = PerformanceBenchmark(model_path=model_path, device="cuda")
    return bench


@pytest.mark.benchmark
def test_model_loading_time(benchmark):
    """测试模型加载时间"""
    results = benchmark.setup()

    # 验证加载时间 < 60秒
    assert results["load_time_sec"] < 60, f"模型加载时间过长: {results['load_time_sec']:.2f}s"

    # 验证GPU内存占用 < 8GB
    assert results["gpu_memory_mb"] < 8192, f"GPU内存占用过高: {results['gpu_memory_mb']:.0f}MB"

    print(f"✅ 模型加载: {results['load_time_sec']:.2f}s, GPU: {results['gpu_memory_mb']:.0f}MB")


@pytest.mark.benchmark
def test_inference_latency_short(benchmark):
    """测试短文本推理延迟"""
    if benchmark.engine is None:
        benchmark.setup()

    results = benchmark.benchmark_latency(text_category="short", num_runs=20)

    # 验证P50延迟 < 200ms
    assert results["p50_ms"] < 200, f"P50延迟过高: {results['p50_ms']:.1f}ms"

    # 验证P99延迟 < 500ms
    assert results["p99_ms"] < 500, f"P99延迟过高: {results['p99_ms']:.1f}ms"

    print(f"✅ 短文本延迟: P50={results['p50_ms']:.1f}ms, P99={results['p99_ms']:.1f}ms")


@pytest.mark.benchmark
def test_inference_latency_medium(benchmark):
    """测试中等文本推理延迟"""
    if benchmark.engine is None:
        benchmark.setup()

    results = benchmark.benchmark_latency(text_category="medium", num_runs=20)

    # 验证P50延迟 < 300ms
    assert results["p50_ms"] < 300, f"P50延迟过高: {results['p50_ms']:.1f}ms"

    print(f"✅ 中等文本延迟: P50={results['p50_ms']:.1f}ms, P99={results['p99_ms']:.1f}ms")


@pytest.mark.benchmark
def test_throughput(benchmark):
    """测试吞吐量"""
    if benchmark.engine is None:
        benchmark.setup()

    results = benchmark.benchmark_throughput(duration_sec=15)

    # 验证吞吐量 > 5 RPS
    assert results["requests_per_second"] > 5, f"吞吐量过低: {results['requests_per_second']:.2f} RPS"

    print(f"✅ 吞吐量: {results['requests_per_second']:.2f} RPS")


# ==================== 命令行运行 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM引擎性能基准测试")
    parser.add_argument(
        "--model-path",
        default="models/pii_qwen4b_unsloth/final",
        help="模型路径"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="结果输出文件"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="吞吐量测试持续时间（秒）"
    )

    args = parser.parse_args()

    # 运行基准测试
    benchmark = PerformanceBenchmark(model_path=args.model_path)
    results = benchmark.run_full_benchmark()

    # 保存结果
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 结果已保存到: {output_path}")
    print("\n✅ 基准测试完成！")
