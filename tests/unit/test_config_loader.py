"""
ConfigLoader 配置加载器单元测试
"""

import pytest
from pathlib import Path

from hppe.engines.regex.config_loader import ConfigLoader


class TestConfigLoaderInitialization:
    """测试配置加载器初始化"""

    def test_init_with_valid_directory(self, tmp_path):
        """测试使用有效目录初始化"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        loader = ConfigLoader(str(patterns_dir))

        assert loader.patterns_dir == patterns_dir

    def test_init_with_nonexistent_directory(self):
        """测试使用不存在的目录初始化"""
        with pytest.raises(FileNotFoundError, match="配置目录不存在"):
            ConfigLoader("/nonexistent/directory")

    def test_init_with_file_instead_of_directory(self, tmp_path):
        """测试使用文件而非目录初始化"""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="路径不是目录"):
            ConfigLoader(str(file_path))


class TestLoadSingleFile:
    """测试加载单个文件"""

    def test_load_valid_yaml_file(self, tmp_path):
        """测试加载有效的 YAML 文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: EMAIL
    patterns:
      - pattern: '[a-z]+@[a-z]+\\.com'
        score: 0.9
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))
        configs = loader.load_file("test.yaml")

        assert len(configs) == 1
        assert configs[0]["entity_type"] == "EMAIL"
        assert len(configs[0]["patterns"]) == 1

    def test_load_multiple_recognizers(self, tmp_path):
        """测试加载包含多个识别器的文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: EMAIL
    patterns:
      - pattern: '[a-z]+@[a-z]+\\.com'
  - entity_type: PHONE
    patterns:
      - pattern: '\\d{11}'
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))
        configs = loader.load_file("test.yaml")

        assert len(configs) == 2
        assert configs[0]["entity_type"] == "EMAIL"
        assert configs[1]["entity_type"] == "PHONE"

    def test_load_with_optional_fields(self, tmp_path):
        """测试加载包含可选字段的配置"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: EMAIL
    name: CustomEmailRecognizer
    patterns:
      - pattern: '[a-z]+@[a-z]+\\.com'
    context_words:
      - email
      - 邮箱
    deny_lists:
      - test
      - example
    confidence_base: 0.92
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))
        configs = loader.load_file("test.yaml")

        assert len(configs) == 1
        config = configs[0]

        assert config["entity_type"] == "EMAIL"
        assert config["name"] == "CustomEmailRecognizer"
        assert config["context_words"] == ["email", "邮箱"]
        assert config["deny_lists"] == ["test", "example"]
        assert config["confidence_base"] == 0.92

    def test_load_nonexistent_file(self, tmp_path):
        """测试加载不存在的文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            loader.load_file("nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """测试加载无效的 YAML 文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_file = patterns_dir / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: syntax: [")

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(ValueError, match="YAML 解析失败"):
            loader.load_file("invalid.yaml")

    def test_load_missing_recognizers_key(self, tmp_path):
        """测试加载缺少 recognizers 键的文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
some_other_key:
  - value1
  - value2
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(ValueError, match="缺少 'recognizers' 键"):
            loader.load_file("test.yaml")


class TestConfigValidation:
    """测试配置验证"""

    def test_validate_missing_entity_type(self, tmp_path):
        """测试缺少 entity_type"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - patterns:
      - pattern: '\\d+'
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(ValueError, match="缺少必需字段: entity_type"):
            loader.load_file("test.yaml")

    def test_validate_missing_patterns(self, tmp_path):
        """测试缺少 patterns"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: TEST
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(ValueError, match="缺少必需字段: patterns"):
            loader.load_file("test.yaml")

    def test_validate_empty_patterns(self, tmp_path):
        """测试空的 patterns 列表"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: TEST
    patterns: []
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(ValueError, match="patterns 不能为空"):
            loader.load_file("test.yaml")

    def test_validate_pattern_missing_pattern_key(self, tmp_path):
        """测试模式缺少 pattern 键"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: TEST
    patterns:
      - score: 0.9
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(ValueError, match="缺少 'pattern' 键"):
            loader.load_file("test.yaml")

    def test_validate_invalid_confidence_base(self, tmp_path):
        """测试无效的 confidence_base"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: TEST
    patterns:
      - pattern: '\\d+'
    confidence_base: 1.5
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(ValueError, match="confidence_base 必须在"):
            loader.load_file("test.yaml")


class TestLoadAll:
    """测试加载所有文件"""

    def test_load_all_yaml_files(self, tmp_path):
        """测试加载所有 YAML 文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        # 创建两个 YAML 文件
        yaml1 = patterns_dir / "email.yaml"
        yaml1.write_text("""
recognizers:
  - entity_type: EMAIL
    patterns:
      - pattern: '[a-z]+@[a-z]+\\.com'
""")

        yaml2 = patterns_dir / "phone.yaml"
        yaml2.write_text("""
recognizers:
  - entity_type: PHONE
    patterns:
      - pattern: '\\d{11}'
""")

        loader = ConfigLoader(str(patterns_dir))
        configs = loader.load_all()

        assert len(configs) == 2
        types = {c["entity_type"] for c in configs}
        assert types == {"EMAIL", "PHONE"}

    def test_load_all_with_pattern(self, tmp_path):
        """测试使用模式加载文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        # 创建多个文件
        (patterns_dir / "china_id.yaml").write_text("""
recognizers:
  - entity_type: CHINA_ID
    patterns:
      - pattern: '\\d{18}'
""")

        (patterns_dir / "china_phone.yaml").write_text("""
recognizers:
  - entity_type: CHINA_PHONE
    patterns:
      - pattern: '1\\d{10}'
""")

        (patterns_dir / "global_email.yaml").write_text("""
recognizers:
  - entity_type: EMAIL
    patterns:
      - pattern: "[a-z]+@[a-z]+\\.com"
""")

        loader = ConfigLoader(str(patterns_dir))

        # 只加载 china_*.yaml
        configs = loader.load_all("china_*.yaml")

        assert len(configs) == 2
        types = {c["entity_type"] for c in configs}
        assert types == {"CHINA_ID", "CHINA_PHONE"}

    def test_load_all_no_matching_files(self, tmp_path):
        """测试没有匹配的文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        loader = ConfigLoader(str(patterns_dir))

        with pytest.raises(FileNotFoundError, match="未找到匹配"):
            loader.load_all()

    def test_load_all_skip_invalid_files(self, tmp_path, capsys):
        """测试跳过无效文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        # 创建一个有效文件
        valid_file = patterns_dir / "valid.yaml"
        valid_file.write_text("""
recognizers:
  - entity_type: VALID
    patterns:
      - pattern: '\\d+'
""")

        # 创建一个无效文件
        invalid_file = patterns_dir / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: [")

        loader = ConfigLoader(str(patterns_dir))

        # 应该加载有效文件，跳过无效文件
        configs = loader.load_all()

        # 只有一个有效配置
        assert len(configs) == 1
        assert configs[0]["entity_type"] == "VALID"

        # 检查是否有警告输出
        captured = capsys.readouterr()
        assert "警告" in captured.out


class TestUtilityMethods:
    """测试工具方法"""

    def test_list_config_files(self, tmp_path):
        """测试列出配置文件"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        # 创建几个 YAML 文件
        (patterns_dir / "a.yaml").write_text("test")
        (patterns_dir / "b.yaml").write_text("test")
        (patterns_dir / "c.txt").write_text("test")  # 非 YAML 文件

        loader = ConfigLoader(str(patterns_dir))
        yaml_files = loader.list_config_files()

        assert len(yaml_files) == 2
        names = {f.name for f in yaml_files}
        assert names == {"a.yaml", "b.yaml"}

    def test_get_config_by_type(self, tmp_path):
        """测试根据类型获取配置"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: EMAIL
    patterns:
      - pattern: '[a-z]+@[a-z]+\\.com'
  - entity_type: PHONE
    patterns:
      - pattern: '\\d{11}'
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))
        config = loader.get_config_by_type("EMAIL")

        assert config is not None
        assert config["entity_type"] == "EMAIL"

    def test_get_config_by_type_not_found(self, tmp_path):
        """测试获取不存在的类型"""
        patterns_dir = tmp_path / "patterns"
        patterns_dir.mkdir()

        yaml_content = """
recognizers:
  - entity_type: EMAIL
    patterns:
      - pattern: "[a-z]+@[a-z]+\\.com"
"""
        yaml_file = patterns_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader(str(patterns_dir))
        config = loader.get_config_by_type("NONEXISTENT")

        assert config is None
