"""
识别器注册表集成测试

测试 RecognizerRegistry 的完整功能，包括：
- 自动加载识别器
- 批量 PII 检测
- 性能监控
- 元数据查询
- 线程安全性
"""

import threading
import time
import pytest
from typing import List

from hppe.engines.regex.registry import RecognizerRegistry
from hppe.models.entity import Entity


class TestRegistryIntegration:
    """注册表集成测试"""

    def test_load_all_recognizers(self):
        """测试 load_all 自动加载所有识别器"""
        registry = RecognizerRegistry()

        # 加载所有识别器
        count = registry.load_all()

        # 验证加载了 9 个识别器（4个中国PII + 5个全球PII）
        assert count == 9
        assert len(registry) == 9

        # 验证所有识别器类型都已注册
        entity_types = registry.get_entity_types()
        expected_types = {
            "CHINA_ID_CARD",
            "CHINA_PHONE",
            "CHINA_BANK_CARD",
            "CHINA_PASSPORT",
            "EMAIL",
            "IP_ADDRESS",
            "URL",
            "CREDIT_CARD",
            "US_SSN",
        }
        assert entity_types == expected_types

    def test_load_all_with_custom_config(self):
        """测试使用自定义配置加载识别器"""
        registry = RecognizerRegistry()

        # 自定义配置
        config = {
            "EMAIL": {"confidence_base": 0.95},
            "IP_ADDRESS": {"confidence_base": 0.90},
        }

        count = registry.load_all(config)
        assert count == 9

        # 验证自定义配置生效
        email_recognizer = registry.get_recognizer("EMAIL")
        assert email_recognizer.confidence_base == 0.95

        ip_recognizer = registry.get_recognizer("IP_ADDRESS")
        assert ip_recognizer.confidence_base == 0.90

    def test_mixed_pii_detection(self):
        """测试混合 PII 类型检测"""
        registry = RecognizerRegistry()
        registry.load_all()

        # 包含多种 PII 的文本
        text = """
        用户信息：
        - 姓名: 张三
        - 身份证: 110101199003077571
        - 手机: 13812345678
        - 邮箱: zhang.san@company.com
        - IP地址: 192.168.1.100
        - 网站: https://www.example.com
        - 银行卡: 6222021234567890123
        """

        # 批量检测
        entities = registry.detect(text)

        # 验证检测到多种PII
        assert len(entities) > 0

        # 按类型分组
        entity_types = {e.entity_type for e in entities}

        # 应该至少检测到以下类型
        assert "CHINA_ID_CARD" in entity_types
        assert "CHINA_PHONE" in entity_types
        assert "EMAIL" in entity_types
        assert "IP_ADDRESS" in entity_types
        assert "URL" in entity_types
        assert "CHINA_BANK_CARD" in entity_types

    def test_filtered_detection(self):
        """测试按类型过滤检测"""
        registry = RecognizerRegistry()
        registry.load_all()

        text = """
        Email: john@example.com
        ID: 110101199003077571
        Phone: 13812345678
        """

        # 只检测邮箱和身份证
        entities = registry.detect(
            text,
            entity_types=["EMAIL", "CHINA_ID_CARD"]
        )

        # 验证只返回指定类型
        entity_types = {e.entity_type for e in entities}
        assert entity_types == {"EMAIL", "CHINA_ID_CARD"}

    def test_detect_with_confidence_filter(self):
        """测试置信度过滤"""
        registry = RecognizerRegistry()
        registry.load_all()

        text = "Email: john@example.com, IP: 192.168.1.1"

        # 使用高置信度阈值过滤
        entities = registry.detect_with_filter(text, min_confidence=0.8)

        # 所有实体的置信度都应该 >= 0.8
        for entity in entities:
            assert entity.confidence >= 0.8

    def test_performance_monitoring(self):
        """测试性能监控功能"""
        registry = RecognizerRegistry()
        registry.load_all()

        # 重置性能统计
        registry.reset_performance_stats()

        text = "Email: test@example.com"

        # 执行多次检测
        for _ in range(10):
            registry.detect(text)

        # 获取性能统计
        stats = registry.get_performance_stats()

        # 验证统计数据
        assert "EMAIL" in stats
        email_stats = stats["EMAIL"]
        assert email_stats["total_calls"] == 10
        assert email_stats["total_time"] > 0
        assert email_stats["avg_time"] > 0

        # 验证平均时间计算正确
        expected_avg = email_stats["total_time"] / email_stats["total_calls"]
        assert abs(email_stats["avg_time"] - expected_avg) < 0.0001

    def test_metadata_query(self):
        """测试元数据查询功能"""
        registry = RecognizerRegistry()
        registry.load_all()

        # 获取单个识别器的元数据
        email_metadata = registry.get_metadata("EMAIL")

        assert email_metadata["entity_type"] == "EMAIL"
        assert email_metadata["recognizer_name"] == "EmailRecognizer"
        assert "confidence_base" in email_metadata
        assert "pattern_count" in email_metadata
        assert "supported_patterns" in email_metadata
        assert "description" in email_metadata
        assert "performance" in email_metadata

        # 获取所有识别器的元数据
        all_metadata = registry.get_metadata()
        assert len(all_metadata) == 9

        # 验证每个识别器都有完整的元数据
        for entity_type, metadata in all_metadata.items():
            assert "entity_type" in metadata
            assert "recognizer_name" in metadata
            assert "confidence_base" in metadata

    def test_summary_information(self):
        """测试摘要信息"""
        registry = RecognizerRegistry()
        registry.load_all()

        # 重置统计
        registry.reset_performance_stats()

        # 执行一些检测
        text = "Email: test@example.com, Phone: 13812345678"
        for _ in range(5):
            registry.detect(text)

        # 获取摘要
        summary = registry.get_summary()

        # 验证摘要信息
        assert summary["total_recognizers"] == 9
        assert len(summary["entity_types"]) == 9
        assert summary["total_detections"] >= 5  # 至少5次（可能更多，因为每次检测会调用多个识别器）
        assert summary["total_time"] > 0
        assert "recognizers" in summary

        # 验证每个识别器的信息
        for entity_type, info in summary["recognizers"].items():
            assert "name" in info
            assert "calls" in info
            assert "avg_time" in info

    def test_thread_safety(self):
        """测试线程安全性"""
        registry = RecognizerRegistry()
        registry.load_all()

        results = []
        errors = []

        def detect_worker():
            """工作线程函数"""
            try:
                text = "Email: test@example.com, ID: 110101199003077571"
                entities = registry.detect(text)
                results.append(len(entities))
            except Exception as e:
                errors.append(str(e))

        # 创建多个线程同时检测
        threads = []
        num_threads = 20

        for _ in range(num_threads):
            thread = threading.Thread(target=detect_worker)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误
        assert len(errors) == 0

        # 验证所有线程都成功执行
        assert len(results) == num_threads

        # 验证结果一致（所有线程应该检测到相同数量的实体）
        assert len(set(results)) == 1  # 所有结果应该相同

    def test_concurrent_register_and_detect(self):
        """测试并发注册和检测"""
        registry = RecognizerRegistry()
        errors = []

        def load_worker():
            """加载识别器的工作线程"""
            try:
                # 每个线程都尝试加载（应该只有第一个成功，其他跳过）
                registry.load_all()
            except Exception as e:
                errors.append(f"Load error: {e}")

        def detect_worker():
            """检测的工作线程"""
            try:
                # 等待一点时间，确保至少有一些识别器已加载
                time.sleep(0.1)
                text = "test@example.com"
                registry.detect(text)
            except Exception as e:
                errors.append(f"Detect error: {e}")

        # 创建混合线程
        threads = []

        # 5个加载线程
        for _ in range(5):
            thread = threading.Thread(target=load_worker)
            threads.append(thread)

        # 10个检测线程
        for _ in range(10):
            thread = threading.Thread(target=detect_worker)
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误
        assert len(errors) == 0

        # 验证注册表状态正常
        assert len(registry) == 9

    def test_performance_stats_accuracy(self):
        """测试性能统计的准确性"""
        registry = RecognizerRegistry()
        registry.load_all()
        registry.reset_performance_stats()

        text = "test@example.com"

        # 记录执行次数
        num_calls = 15

        for _ in range(num_calls):
            registry.detect(text, entity_types=["EMAIL"])

        # 获取EMAIL识别器的统计
        stats = registry.get_performance_stats("EMAIL")

        # 验证调用次数准确
        assert stats["total_calls"] == num_calls

        # 验证时间统计合理
        assert stats["total_time"] > 0
        assert stats["avg_time"] > 0
        assert stats["avg_time"] == stats["total_time"] / stats["total_calls"]

    def test_empty_registry_behavior(self):
        """测试空注册表的行为"""
        registry = RecognizerRegistry()

        # 空注册表检测应该返回空列表
        entities = registry.detect("test text")
        assert entities == []

        # 获取不存在的识别器应该返回 None
        recognizer = registry.get_recognizer("NON_EXISTENT")
        assert recognizer is None

        # 获取不存在的元数据应该返回空字典
        metadata = registry.get_metadata("NON_EXISTENT")
        assert metadata == {}

    def test_large_text_processing(self):
        """测试大文本处理能力"""
        registry = RecognizerRegistry()
        registry.load_all()

        # 生成大文本（包含多个PII）
        text_parts = []
        for i in range(100):
            text_parts.append(f"用户{i}: email{i}@example.com, 手机: 1381234567{i%10}")

        large_text = "\n".join(text_parts)

        # 检测大文本
        start_time = time.time()
        entities = registry.detect(large_text)
        elapsed_time = time.time() - start_time

        # 验证检测到多个实体
        assert len(entities) > 0

        # 验证性能（应该在合理时间内完成）
        assert elapsed_time < 5.0  # 应该在5秒内完成

    def test_duplicate_registration_prevention(self):
        """测试防止重复注册"""
        from hppe.engines.regex.recognizers.global_pii import EmailRecognizer

        registry = RecognizerRegistry()

        # 创建配置
        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'}],
            "confidence_base": 0.80
        }

        # 注册第一个识别器
        recognizer1 = EmailRecognizer(config)
        registry.register(recognizer1)

        # 尝试注册相同类型的识别器应该抛出异常
        recognizer2 = EmailRecognizer(config)
        with pytest.raises(ValueError, match="已注册"):
            registry.register(recognizer2)

    def test_unregister_functionality(self):
        """测试注销功能"""
        registry = RecognizerRegistry()
        registry.load_all()

        initial_count = len(registry)

        # 注销一个识别器
        success = registry.unregister("EMAIL")
        assert success is True
        assert len(registry) == initial_count - 1
        assert "EMAIL" not in registry

        # 尝试注销不存在的识别器
        success = registry.unregister("NON_EXISTENT")
        assert success is False

    def test_clear_functionality(self):
        """测试清空功能"""
        registry = RecognizerRegistry()
        registry.load_all()

        assert len(registry) > 0

        # 清空注册表
        registry.clear()

        assert len(registry) == 0
        assert len(registry.get_entity_types()) == 0

    def test_repr_output(self):
        """测试字符串表示"""
        registry = RecognizerRegistry()
        registry.load_all()

        repr_str = repr(registry)

        assert "RecognizerRegistry" in repr_str
        assert "count=9" in repr_str
        assert "types=" in repr_str


class TestRegistryErrorHandling:
    """注册表错误处理测试"""

    def test_recognizer_error_isolation(self):
        """测试识别器错误隔离"""
        from hppe.engines.regex.base import BaseRecognizer
        from hppe.models.entity import Entity

        class FaultyRecognizer(BaseRecognizer):
            """会抛出异常的识别器"""
            def detect(self, text: str) -> List[Entity]:
                raise RuntimeError("Intentional error")

            def validate(self, entity: Entity) -> bool:
                """简单的验证实现"""
                return True

        registry = RecognizerRegistry()

        # 注册正常识别器和有问题的识别器
        config = {
            "entity_type": "FAULTY",
            "patterns": [{"pattern": r"test"}],
            "confidence_base": 0.5
        }
        faulty = FaultyRecognizer(config)
        registry.register(faulty)

        # 加载正常识别器
        registry.load_all()

        # 检测应该继续（错误应该被捕获）
        text = "test@example.com"
        entities = registry.detect(text)

        # 应该检测到邮箱（即使有问题的识别器失败了）
        entity_types = {e.entity_type for e in entities}
        assert "EMAIL" in entity_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
