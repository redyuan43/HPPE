"""
Entity 数据模型单元测试
"""

import pytest
from hppe.models.entity import Entity


class TestEntityCreation:
    """测试 Entity 创建"""

    def test_create_valid_entity(self):
        """测试创建有效的实体"""
        entity = Entity(
            entity_type="CHINA_ID_CARD",
            value="110101199003077578",
            start_pos=7,
            end_pos=25,
            confidence=0.95,
            detection_method="regex",
            recognizer_name="ChinaIDCardRecognizer"
        )

        assert entity.entity_type == "CHINA_ID_CARD"
        assert entity.value == "110101199003077578"
        assert entity.start_pos == 7
        assert entity.end_pos == 25
        assert entity.confidence == 0.95
        assert entity.detection_method == "regex"
        assert entity.recognizer_name == "ChinaIDCardRecognizer"
        assert entity.metadata is None

    def test_create_entity_with_metadata(self):
        """测试创建带元数据的实体"""
        metadata = {"validated": True, "checksum_passed": True}

        entity = Entity(
            entity_type="EMAIL_ADDRESS",
            value="test@example.com",
            start_pos=0,
            end_pos=16,
            confidence=0.90,
            detection_method="regex",
            recognizer_name="EmailRecognizer",
            metadata=metadata
        )

        assert entity.metadata == metadata
        assert entity.metadata["validated"] is True


class TestEntityValidation:
    """测试 Entity 数据验证"""

    def test_invalid_start_pos_negative(self):
        """测试负数起始位置"""
        with pytest.raises(ValueError, match="start_pos 必须 >= 0"):
            Entity(
                entity_type="TEST",
                value="test",
                start_pos=-1,
                end_pos=10,
                confidence=0.9,
                detection_method="regex",
                recognizer_name="TestRecognizer"
            )

    def test_invalid_end_pos_smaller_than_start(self):
        """测试结束位置小于起始位置"""
        with pytest.raises(ValueError, match="end_pos .* 必须大于 start_pos"):
            Entity(
                entity_type="TEST",
                value="test",
                start_pos=10,
                end_pos=5,
                confidence=0.9,
                detection_method="regex",
                recognizer_name="TestRecognizer"
            )

    def test_invalid_end_pos_equal_to_start(self):
        """测试结束位置等于起始位置"""
        with pytest.raises(ValueError, match="end_pos .* 必须大于 start_pos"):
            Entity(
                entity_type="TEST",
                value="test",
                start_pos=10,
                end_pos=10,
                confidence=0.9,
                detection_method="regex",
                recognizer_name="TestRecognizer"
            )

    def test_invalid_confidence_below_zero(self):
        """测试置信度小于 0"""
        with pytest.raises(ValueError, match="confidence 必须在"):
            Entity(
                entity_type="TEST",
                value="test",
                start_pos=0,
                end_pos=10,
                confidence=-0.1,
                detection_method="regex",
                recognizer_name="TestRecognizer"
            )

    def test_invalid_confidence_above_one(self):
        """测试置信度大于 1"""
        with pytest.raises(ValueError, match="confidence 必须在"):
            Entity(
                entity_type="TEST",
                value="test",
                start_pos=0,
                end_pos=10,
                confidence=1.5,
                detection_method="regex",
                recognizer_name="TestRecognizer"
            )

    def test_empty_entity_type(self):
        """测试空的 entity_type"""
        with pytest.raises(ValueError, match="entity_type 不能为空"):
            Entity(
                entity_type="",
                value="test",
                start_pos=0,
                end_pos=10,
                confidence=0.9,
                detection_method="regex",
                recognizer_name="TestRecognizer"
            )

    def test_empty_value(self):
        """测试空的 value"""
        with pytest.raises(ValueError, match="value 不能为空"):
            Entity(
                entity_type="TEST",
                value="",
                start_pos=0,
                end_pos=10,
                confidence=0.9,
                detection_method="regex",
                recognizer_name="TestRecognizer"
            )

    def test_empty_detection_method(self):
        """测试空的 detection_method"""
        with pytest.raises(ValueError, match="detection_method 不能为空"):
            Entity(
                entity_type="TEST",
                value="test",
                start_pos=0,
                end_pos=10,
                confidence=0.9,
                detection_method="",
                recognizer_name="TestRecognizer"
            )

    def test_empty_recognizer_name(self):
        """测试空的 recognizer_name"""
        with pytest.raises(ValueError, match="recognizer_name 不能为空"):
            Entity(
                entity_type="TEST",
                value="test",
                start_pos=0,
                end_pos=10,
                confidence=0.9,
                detection_method="regex",
                recognizer_name=""
            )


class TestEntityMethods:
    """测试 Entity 方法"""

    def test_str_representation(self):
        """测试字符串表示"""
        entity = Entity(
            entity_type="EMAIL",
            value="test@example.com",
            start_pos=10,
            end_pos=26,
            confidence=0.95,
            detection_method="regex",
            recognizer_name="EmailRecognizer"
        )

        str_repr = str(entity)
        assert "EMAIL" in str_repr
        assert "test@example.com" in str_repr
        assert "10:26" in str_repr
        assert "0.95" in str_repr

    def test_repr_representation(self):
        """测试详细字符串表示"""
        entity = Entity(
            entity_type="TEST",
            value="value",
            start_pos=0,
            end_pos=5,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="TestRecognizer"
        )

        repr_str = repr(entity)
        assert "Entity(" in repr_str
        assert "entity_type='TEST'" in repr_str
        assert "value='value'" in repr_str
        assert "start_pos=0" in repr_str
        assert "end_pos=5" in repr_str
        assert "confidence=0.9" in repr_str

    def test_to_dict(self):
        """测试转换为字典"""
        entity = Entity(
            entity_type="TEST",
            value="value",
            start_pos=0,
            end_pos=5,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="TestRecognizer",
            metadata={"key": "value"}
        )

        entity_dict = entity.to_dict()

        assert entity_dict["entity_type"] == "TEST"
        assert entity_dict["value"] == "value"
        assert entity_dict["start_pos"] == 0
        assert entity_dict["end_pos"] == 5
        assert entity_dict["confidence"] == 0.9
        assert entity_dict["detection_method"] == "regex"
        assert entity_dict["recognizer_name"] == "TestRecognizer"
        assert entity_dict["metadata"] == {"key": "value"}

    def test_length_property(self):
        """测试 length 属性"""
        entity = Entity(
            entity_type="TEST",
            value="test_value",
            start_pos=10,
            end_pos=20,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="TestRecognizer"
        )

        assert entity.length == 10

    def test_length_property_single_char(self):
        """测试单字符长度"""
        entity = Entity(
            entity_type="TEST",
            value="x",
            start_pos=5,
            end_pos=6,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="TestRecognizer"
        )

        assert entity.length == 1


class TestEntityBoundaryConditions:
    """测试边界条件"""

    def test_confidence_exactly_zero(self):
        """测试置信度正好为 0"""
        entity = Entity(
            entity_type="TEST",
            value="test",
            start_pos=0,
            end_pos=4,
            confidence=0.0,
            detection_method="regex",
            recognizer_name="TestRecognizer"
        )

        assert entity.confidence == 0.0

    def test_confidence_exactly_one(self):
        """测试置信度正好为 1"""
        entity = Entity(
            entity_type="TEST",
            value="test",
            start_pos=0,
            end_pos=4,
            confidence=1.0,
            detection_method="regex",
            recognizer_name="TestRecognizer"
        )

        assert entity.confidence == 1.0

    def test_very_long_value(self):
        """测试非常长的值"""
        long_value = "x" * 10000

        entity = Entity(
            entity_type="TEST",
            value=long_value,
            start_pos=0,
            end_pos=10000,
            confidence=0.9,
            detection_method="regex",
            recognizer_name="TestRecognizer"
        )

        assert len(entity.value) == 10000
        assert entity.length == 10000
