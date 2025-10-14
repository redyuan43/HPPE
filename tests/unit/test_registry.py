"""
RecognizerRegistry 注册表单元测试
"""

import pytest
from typing import List

from hppe.engines.regex.registry import RecognizerRegistry
from hppe.engines.regex.base import BaseRecognizer
from hppe.models.entity import Entity


# 创建测试用的识别器
class MockRecognizer(BaseRecognizer):
    """模拟识别器用于测试"""

    def __init__(self, config):
        super().__init__(config)
        self.detect_count = 0

    def detect(self, text: str) -> List[Entity]:
        """模拟检测"""
        self.detect_count += 1
        # 简单返回一个实体
        return [
            Entity(
                entity_type=self.entity_type,
                value="mock_value",
                start_pos=0,
                end_pos=10,
                confidence=0.9,
                detection_method="regex",
                recognizer_name=self.recognizer_name
            )
        ]

    def validate(self, entity: Entity) -> bool:
        return True


class TestRegistryBasicOperations:
    """测试注册表基本操作"""

    def test_init_empty_registry(self):
        """测试初始化空注册表"""
        registry = RecognizerRegistry()

        assert len(registry) == 0
        assert registry.get_entity_types() == set()

    def test_register_single_recognizer(self):
        """测试注册单个识别器"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"[a-z]+@[a-z]+\.com"}]
        }
        recognizer = MockRecognizer(config)

        registry.register(recognizer)

        assert len(registry) == 1
        assert "EMAIL" in registry
        assert registry.get_entity_types() == {"EMAIL"}

    def test_register_multiple_recognizers(self):
        """测试注册多个识别器"""
        registry = RecognizerRegistry()

        for entity_type in ["EMAIL", "PHONE", "ID"]:
            config = {
                "entity_type": entity_type,
                "patterns": [{"pattern": r"\d+"}]
            }
            recognizer = MockRecognizer(config)
            registry.register(recognizer)

        assert len(registry) == 3
        assert registry.get_entity_types() == {"EMAIL", "PHONE", "ID"}

    def test_register_duplicate_type_raises_error(self):
        """测试注册重复类型应该报错"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }

        registry.register(MockRecognizer(config))

        # 尝试注册相同类型
        with pytest.raises(ValueError, match="识别器类型 .* 已注册"):
            registry.register(MockRecognizer(config))

    def test_unregister_existing_recognizer(self):
        """测试注销已存在的识别器"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        registry.register(MockRecognizer(config))

        assert len(registry) == 1

        result = registry.unregister("EMAIL")

        assert result is True
        assert len(registry) == 0
        assert "EMAIL" not in registry

    def test_unregister_nonexistent_recognizer(self):
        """测试注销不存在的识别器"""
        registry = RecognizerRegistry()

        result = registry.unregister("NONEXISTENT")

        assert result is False


class TestRegistryQuery:
    """测试注册表查询功能"""

    def test_get_recognizer_exists(self):
        """测试获取已存在的识别器"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        recognizer = MockRecognizer(config)
        registry.register(recognizer)

        retrieved = registry.get_recognizer("EMAIL")

        assert retrieved is not None
        assert retrieved.entity_type == "EMAIL"

    def test_get_recognizer_not_exists(self):
        """测试获取不存在的识别器"""
        registry = RecognizerRegistry()

        retrieved = registry.get_recognizer("NONEXISTENT")

        assert retrieved is None

    def test_get_all_recognizers(self):
        """测试获取所有识别器"""
        registry = RecognizerRegistry()

        for entity_type in ["EMAIL", "PHONE", "ID"]:
            config = {
                "entity_type": entity_type,
                "patterns": [{"pattern": r"\d+"}]
            }
            registry.register(MockRecognizer(config))

        all_recognizers = registry.get_all_recognizers()

        assert len(all_recognizers) == 3
        types = {r.entity_type for r in all_recognizers}
        assert types == {"EMAIL", "PHONE", "ID"}

    def test_get_entity_types(self):
        """测试获取所有 PII 类型"""
        registry = RecognizerRegistry()

        for entity_type in ["EMAIL", "PHONE"]:
            config = {
                "entity_type": entity_type,
                "patterns": [{"pattern": r"\d+"}]
            }
            registry.register(MockRecognizer(config))

        types = registry.get_entity_types()

        assert types == {"EMAIL", "PHONE"}


class TestRegistryDetection:
    """测试注册表检测功能"""

    def test_detect_with_all_recognizers(self):
        """测试使用所有识别器检测"""
        registry = RecognizerRegistry()

        for entity_type in ["EMAIL", "PHONE"]:
            config = {
                "entity_type": entity_type,
                "patterns": [{"pattern": r"\w+"}]
            }
            registry.register(MockRecognizer(config))

        entities = registry.detect("test text")

        # 应该有 2 个实体（每个识别器一个）
        assert len(entities) == 2
        types = {e.entity_type for e in entities}
        assert types == {"EMAIL", "PHONE"}

    def test_detect_with_filtered_types(self):
        """测试使用过滤后的类型检测"""
        registry = RecognizerRegistry()

        for entity_type in ["EMAIL", "PHONE", "ID"]:
            config = {
                "entity_type": entity_type,
                "patterns": [{"pattern": r"\w+"}]
            }
            registry.register(MockRecognizer(config))

        # 只使用 EMAIL 和 PHONE
        entities = registry.detect("test", entity_types=["EMAIL", "PHONE"])

        assert len(entities) == 2
        types = {e.entity_type for e in entities}
        assert types == {"EMAIL", "PHONE"}

    def test_detect_with_nonexistent_type(self):
        """测试使用不存在的类型检测"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        registry.register(MockRecognizer(config))

        # 请求不存在的类型
        entities = registry.detect("test", entity_types=["NONEXISTENT"])

        # 应该返回空列表
        assert len(entities) == 0

    def test_detect_with_filter(self):
        """测试带置信度过滤的检测"""
        registry = RecognizerRegistry()

        # 创建返回不同置信度的识别器
        class HighConfRecognizer(MockRecognizer):
            def detect(self, text):
                return [
                    Entity(
                        entity_type=self.entity_type,
                        value="high",
                        start_pos=0,
                        end_pos=4,
                        confidence=0.95,
                        detection_method="regex",
                        recognizer_name=self.recognizer_name
                    )
                ]

        class LowConfRecognizer(MockRecognizer):
            def detect(self, text):
                return [
                    Entity(
                        entity_type=self.entity_type,
                        value="low",
                        start_pos=0,
                        end_pos=3,
                        confidence=0.50,
                        detection_method="regex",
                        recognizer_name=self.recognizer_name
                    )
                ]

        config1 = {
            "entity_type": "HIGH",
            "patterns": [{"pattern": r"\w+"}]
        }
        config2 = {
            "entity_type": "LOW",
            "patterns": [{"pattern": r"\w+"}]
        }

        registry.register(HighConfRecognizer(config1))
        registry.register(LowConfRecognizer(config2))

        # 只要置信度 >= 0.8 的
        entities = registry.detect_with_filter("test", min_confidence=0.8)

        assert len(entities) == 1
        assert entities[0].entity_type == "HIGH"


class TestRegistryPerformance:
    """测试性能统计功能"""

    def test_performance_stats_initialized(self):
        """测试性能统计初始化"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        registry.register(MockRecognizer(config))

        stats = registry.get_performance_stats("EMAIL")

        assert stats["total_calls"] == 0
        assert stats["total_time"] == 0.0
        assert stats["avg_time"] == 0.0

    def test_performance_stats_updated(self):
        """测试性能统计更新"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        registry.register(MockRecognizer(config))

        # 执行检测
        registry.detect("test text")

        stats = registry.get_performance_stats("EMAIL")

        assert stats["total_calls"] == 1
        assert stats["total_time"] > 0.0
        assert stats["avg_time"] > 0.0

    def test_performance_stats_multiple_calls(self):
        """测试多次调用的性能统计"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        registry.register(MockRecognizer(config))

        # 执行多次检测
        for _ in range(5):
            registry.detect("test")

        stats = registry.get_performance_stats("EMAIL")

        assert stats["total_calls"] == 5
        assert stats["total_time"] > 0.0
        assert stats["avg_time"] == stats["total_time"] / 5

    def test_reset_performance_stats(self):
        """测试重置性能统计"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        registry.register(MockRecognizer(config))

        # 执行检测
        registry.detect("test")

        # 重置统计
        registry.reset_performance_stats()

        stats = registry.get_performance_stats("EMAIL")

        assert stats["total_calls"] == 0
        assert stats["total_time"] == 0.0


class TestRegistryUtilityMethods:
    """测试工具方法"""

    def test_clear_registry(self):
        """测试清空注册表"""
        registry = RecognizerRegistry()

        for entity_type in ["EMAIL", "PHONE"]:
            config = {
                "entity_type": entity_type,
                "patterns": [{"pattern": r"\d+"}]
            }
            registry.register(MockRecognizer(config))

        assert len(registry) == 2

        registry.clear()

        assert len(registry) == 0
        assert registry.get_entity_types() == set()

    def test_contains_operator(self):
        """测试 in 操作符"""
        registry = RecognizerRegistry()

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        registry.register(MockRecognizer(config))

        assert "EMAIL" in registry
        assert "PHONE" not in registry

    def test_len_operator(self):
        """测试 len 操作符"""
        registry = RecognizerRegistry()

        assert len(registry) == 0

        config = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        registry.register(MockRecognizer(config))

        assert len(registry) == 1

    def test_repr(self):
        """测试字符串表示"""
        registry = RecognizerRegistry()

        config1 = {
            "entity_type": "EMAIL",
            "patterns": [{"pattern": r"\w+"}]
        }
        config2 = {
            "entity_type": "PHONE",
            "patterns": [{"pattern": r"\d+"}]
        }

        registry.register(MockRecognizer(config1))
        registry.register(MockRecognizer(config2))

        repr_str = repr(registry)

        assert "RecognizerRegistry" in repr_str
        assert "count=2" in repr_str
