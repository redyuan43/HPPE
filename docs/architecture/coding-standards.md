# HPPE 编码标准

**版本：** 1.0
**更新日期：** 2025年10月14日
**适用范围：** 所有 HPPE 项目代码

---

## 1. Python 编码规范

### 1.1 基础规范
- **Python 版本**：3.11+
- **编码格式**：UTF-8
- **行长度**：最大 100 字符（Black 默认）
- **缩进**：4 个空格

### 1.2 命名约定

```python
# 模块名：小写，下划线分隔
import regex_engine

# 类名：驼峰命名
class PatternRecognizer:
    pass

# 函数和方法：小写，下划线分隔
def detect_pii(text: str) -> List[Entity]:
    pass

# 常量：大写，下划线分隔
MAX_TEXT_LENGTH = 10000
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# 私有属性/方法：单下划线前缀
class PIIDetector:
    def __init__(self):
        self._cache = {}

    def _validate_input(self, text):
        pass

# 受保护的属性/方法：双下划线前缀（避免使用）
```

### 1.3 类型注解

**强制要求**：所有公共函数和方法必须包含类型注解

```python
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class PIIEntity:
    entity_type: str
    value: str
    start_pos: int
    end_pos: int
    confidence: float

def detect_entities(
    text: str,
    language: str = "auto",
    confidence_threshold: float = 0.85
) -> List[PIIEntity]:
    """
    检测文本中的 PII 实体。

    Args:
        text: 要检测的文本
        language: 语言代码 (zh|en|auto)
        confidence_threshold: 置信度阈值

    Returns:
        检测到的 PII 实体列表

    Raises:
        ValueError: 如果文本为空或超过最大长度
    """
    pass
```

---

## 2. 代码组织

### 2.1 模块结构

```python
"""
模块文档字符串，描述模块的功能和用途。
"""

# 1. Future imports (if needed)
from __future__ import annotations

# 2. 标准库导入
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

# 3. 第三方库导入
import numpy as np
import redis
from fastapi import FastAPI
from pydantic import BaseModel

# 4. 本地应用导入
from hppe.core import pipeline
from hppe.engines.regex import PatternRecognizer
from hppe.utils import text_utils

# 5. 常量定义
DEFAULT_TIMEOUT = 30
MAX_BATCH_SIZE = 100

# 6. 类定义
class MyClass:
    pass

# 7. 函数定义
def main_function():
    pass

# 8. 主程序入口
if __name__ == "__main__":
    main_function()
```

### 2.2 类设计原则

```python
from abc import ABC, abstractmethod
from typing import Protocol

# 使用抽象基类定义接口
class BaseRecognizer(ABC):
    """PII 识别器基类"""

    @abstractmethod
    def detect(self, text: str) -> List[PIIEntity]:
        """检测 PII 实体"""
        pass

    @abstractmethod
    def validate(self, entity: PIIEntity) -> bool:
        """验证实体有效性"""
        pass

# 使用 Protocol 定义接口（推荐）
class Recognizer(Protocol):
    """识别器协议"""

    def detect(self, text: str) -> List[PIIEntity]:
        ...

    def validate(self, entity: PIIEntity) -> bool:
        ...

# 实现类
class ChineseIDCardRecognizer(BaseRecognizer):
    """中国身份证识别器"""

    def __init__(self, confidence_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold
        self._pattern = self._compile_pattern()

    def detect(self, text: str) -> List[PIIEntity]:
        # 实现检测逻辑
        matches = self._pattern.findall(text)
        return [self._create_entity(m) for m in matches]

    def validate(self, entity: PIIEntity) -> bool:
        # 实现验证逻辑（如校验码验证）
        return self._check_id_card_checksum(entity.value)

    def _compile_pattern(self) -> re.Pattern:
        """编译正则表达式（私有方法）"""
        return re.compile(r'[1-9]\d{5}(19|20)\d{2}...')

    def _create_entity(self, match) -> PIIEntity:
        """创建实体对象（私有方法）"""
        pass

    def _check_id_card_checksum(self, id_card: str) -> bool:
        """验证身份证校验码（私有方法）"""
        pass
```

---

## 3. 异常处理

### 3.1 自定义异常

```python
# src/hppe/core/exceptions.py

class HPPEError(Exception):
    """HPPE 基础异常类"""
    pass

class ValidationError(HPPEError):
    """输入验证错误"""
    pass

class DetectionError(HPPEError):
    """PII 检测错误"""
    pass

class LLMInferenceError(HPPEError):
    """LLM 推理错误"""
    pass

class ConfigurationError(HPPEError):
    """配置错误"""
    pass
```

### 3.2 异常处理模式

```python
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def process_text(text: str) -> Optional[List[PIIEntity]]:
    """
    处理文本，返回 PII 实体或 None。
    """
    try:
        # 输入验证
        if not text:
            raise ValidationError("Text cannot be empty")

        if len(text) > MAX_TEXT_LENGTH:
            raise ValidationError(f"Text exceeds maximum length of {MAX_TEXT_LENGTH}")

        # 业务逻辑
        entities = detect_pii(text)
        return entities

    except ValidationError as e:
        # 记录验证错误，返回 None
        logger.warning(f"Validation failed: {e}")
        return None

    except LLMInferenceError as e:
        # 记录 LLM 错误，尝试降级处理
        logger.error(f"LLM inference failed: {e}")
        return fallback_detection(text)

    except Exception as e:
        # 记录意外错误，重新抛出
        logger.exception(f"Unexpected error processing text: {e}")
        raise

    finally:
        # 清理资源（如果需要）
        cleanup_resources()
```

---

## 4. 日志规范

### 4.1 结构化日志

```python
import structlog

# 配置 structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# 使用结构化日志
def detect_pii_with_logging(text: str, user_id: str):
    """带日志的 PII 检测"""

    # 记录请求
    logger.info(
        "pii_detection_started",
        user_id=user_id,
        text_length=len(text),
        timestamp=datetime.utcnow().isoformat()
    )

    try:
        start_time = time.time()
        entities = detect_pii(text)
        elapsed_ms = (time.time() - start_time) * 1000

        # 记录成功
        logger.info(
            "pii_detection_completed",
            user_id=user_id,
            entities_count=len(entities),
            processing_time_ms=elapsed_ms,
            entity_types=[e.entity_type for e in entities]
        )

        return entities

    except Exception as e:
        # 记录失败
        logger.error(
            "pii_detection_failed",
            user_id=user_id,
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True
        )
        raise
```

---

## 5. 性能优化

### 5.1 缓存策略

```python
from functools import lru_cache, cache
from typing import Tuple
import hashlib

class CachedDetector:
    """带缓存的 PII 检测器"""

    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        # 使用 LRU 缓存
        self._detect = lru_cache(maxsize=cache_size)(self._detect_impl)

    def detect(self, text: str) -> List[PIIEntity]:
        # 计算文本哈希作为缓存键
        text_hash = self._hash_text(text)
        return self._detect(text_hash, text)

    @staticmethod
    def _hash_text(text: str) -> str:
        """生成文本哈希"""
        return hashlib.sha256(text.encode()).hexdigest()

    def _detect_impl(self, text_hash: str, text: str) -> List[PIIEntity]:
        """实际的检测逻辑"""
        # 这里实现真正的检测
        pass

# 使用 functools.cache 缓存纯函数结果
@cache
def compile_pattern(pattern: str) -> re.Pattern:
    """编译并缓存正则表达式"""
    return re.compile(pattern)
```

### 5.2 批处理优化

```python
from typing import List, Iterator
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    """批处理器"""

    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_batch(self, texts: List[str]) -> List[List[PIIEntity]]:
        """异步批处理"""
        batches = self._create_batches(texts)
        tasks = []

        for batch in batches:
            task = asyncio.create_task(self._process_single_batch(batch))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return self._flatten_results(results)

    def _create_batches(self, texts: List[str]) -> Iterator[List[str]]:
        """创建批次"""
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    async def _process_single_batch(self, batch: List[str]) -> List[List[PIIEntity]]:
        """处理单个批次"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._batch_detect,
            batch
        )

    def _batch_detect(self, batch: List[str]) -> List[List[PIIEntity]]:
        """批量检测（在线程池中运行）"""
        return [detect_pii(text) for text in batch]
```

---

## 6. 测试规范

### 6.1 单元测试

```python
import pytest
from unittest.mock import Mock, patch
from typing import List

class TestChineseIDCardRecognizer:
    """中国身份证识别器测试"""

    @pytest.fixture
    def recognizer(self):
        """创建识别器实例"""
        return ChineseIDCardRecognizer()

    @pytest.fixture
    def valid_id_cards(self) -> List[str]:
        """有效身份证号样本"""
        return [
            "110101199003077578",
            "440308199901015216",
            "310104200001018734"
        ]

    def test_detect_valid_id_card(self, recognizer, valid_id_cards):
        """测试检测有效身份证"""
        for id_card in valid_id_cards:
            text = f"我的身份证号是{id_card}"
            entities = recognizer.detect(text)

            assert len(entities) == 1
            assert entities[0].entity_type == "CHINA_ID_CARD"
            assert entities[0].value == id_card
            assert entities[0].confidence > 0.9

    def test_validate_checksum(self, recognizer):
        """测试校验码验证"""
        valid_id = "110101199003077578"
        invalid_id = "110101199003077579"

        assert recognizer._check_id_card_checksum(valid_id) is True
        assert recognizer._check_id_card_checksum(invalid_id) is False

    @pytest.mark.parametrize("invalid_input", [
        "",  # 空字符串
        " " * 100,  # 纯空格
        "12345",  # 太短
        "abcdefghijklmnopqrs",  # 非数字
    ])
    def test_detect_invalid_input(self, recognizer, invalid_input):
        """测试无效输入"""
        entities = recognizer.detect(invalid_input)
        assert len(entities) == 0

    @patch('hppe.engines.regex.logger')
    def test_error_logging(self, mock_logger, recognizer):
        """测试错误日志"""
        with pytest.raises(ValidationError):
            recognizer.detect(None)

        mock_logger.error.assert_called_once()
```

### 6.2 集成测试

```python
import pytest
from fastapi.testclient import TestClient

class TestAPIIntegration:
    """API 集成测试"""

    @pytest.fixture
    def client(self):
        from hppe.main import app
        return TestClient(app)

    def test_detect_endpoint(self, client):
        """测试检测端点"""
        response = client.post(
            "/api/v1/detect",
            json={
                "text": "张三的身份证是110101199003077578",
                "language": "zh"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["entities"]) == 2  # 姓名 + 身份证

    @pytest.mark.asyncio
    async def test_batch_processing(self, client):
        """测试批处理"""
        documents = [
            {"text": f"文档{i}：身份证110101199003077{i:03d}"}
            for i in range(100)
        ]

        response = client.post(
            "/api/v1/batch",
            json={
                "documents": documents,
                "async": True
            }
        )

        assert response.status_code == 202
        task_id = response.json()["task_id"]

        # 检查任务状态
        status_response = client.get(f"/api/v1/status/{task_id}")
        assert status_response.status_code == 200
```

---

## 7. 安全规范

### 7.1 输入验证

```python
from pydantic import BaseModel, validator, Field
import re

class PIIDetectionRequest(BaseModel):
    """PII 检测请求模型"""

    text: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="auto", regex="^(zh|en|auto)$")
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    @validator('text')
    def sanitize_text(cls, v):
        """清理输入文本"""
        # 移除控制字符
        v = re.sub(r'[\x00-\x1F\x7F]', '', v)
        # 标准化空白字符
        v = ' '.join(v.split())
        return v

    @validator('language')
    def validate_language(cls, v):
        """验证语言代码"""
        supported_languages = {'zh', 'en', 'auto'}
        if v not in supported_languages:
            raise ValueError(f"Language must be one of {supported_languages}")
        return v
```

### 7.2 敏感数据处理

```python
import hashlib
from cryptography.fernet import Fernet

class SecureStorage:
    """安全存储"""

    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)

    def store_pii(self, pii_data: str) -> str:
        """加密存储 PII 数据"""
        encrypted = self.cipher.encrypt(pii_data.encode())
        return encrypted.decode()

    def retrieve_pii(self, encrypted_data: str) -> str:
        """解密检索 PII 数据"""
        decrypted = self.cipher.decrypt(encrypted_data.encode())
        return decrypted.decode()

    @staticmethod
    def hash_pii(pii_data: str, salt: str = "") -> str:
        """生成 PII 数据哈希（不可逆）"""
        data_with_salt = f"{pii_data}{salt}"
        return hashlib.sha256(data_with_salt.encode()).hexdigest()
```

---

## 8. 文档规范

### 8.1 函数文档

```python
def detect_pii(
    text: str,
    language: str = "auto",
    confidence_threshold: float = 0.85,
    return_positions: bool = True
) -> List[PIIEntity]:
    """
    检测文本中的个人身份信息（PII）。

    使用混合方法检测 PII，包括基于规则的模式匹配和
    基于 LLM 的上下文理解。

    Args:
        text: 要检测的文本内容。
        language: 文本语言，支持 'zh'（中文）、'en'（英文）
                 或 'auto'（自动检测）。
        confidence_threshold: 最低置信度阈值，范围 [0, 1]。
        return_positions: 是否返回实体在文本中的位置。

    Returns:
        检测到的 PII 实体列表。每个实体包含类型、值、
        位置和置信度信息。

    Raises:
        ValidationError: 当输入文本为空或超过最大长度时。
        LLMInferenceError: 当 LLM 推理失败时。

    Examples:
        >>> text = "张三的身份证号是110101199003077578"
        >>> entities = detect_pii(text, language="zh")
        >>> print(entities[0].entity_type)
        'PERSON_NAME'
        >>> print(entities[1].entity_type)
        'CHINA_ID_CARD'

    Note:
        对于长文本，建议使用批处理接口以获得更好的性能。
        检测结果会被缓存 1 小时以提高重复请求的性能。

    See Also:
        - batch_detect_pii: 批量检测接口
        - async_detect_pii: 异步检测接口
    """
    pass
```

### 8.2 类文档

```python
class PIIDetector:
    """
    个人身份信息（PII）检测器。

    这是 HPPE 系统的核心类，负责协调多个检测引擎
    来识别和处理文本中的 PII。

    Attributes:
        regex_engine: 基于正则表达式的检测引擎。
        llm_engine: 基于大语言模型的检测引擎。
        cache: Redis 缓存客户端。
        metrics: Prometheus 指标收集器。

    Examples:
        基本使用：
        >>> detector = PIIDetector()
        >>> result = detector.detect("我的手机号是13812345678")
        >>> print(result.entities)

        自定义配置：
        >>> config = DetectorConfig(
        ...     confidence_threshold=0.9,
        ...     enable_cache=True
        ... )
        >>> detector = PIIDetector(config=config)

    Note:
        该类是线程安全的，可以在多线程环境中使用。
        建议使用单例模式以避免重复初始化开销。
    """
    pass
```

---

## 9. 提交规范

### 9.1 Git 提交消息

```bash
# 格式：<type>(<scope>): <subject>
#
# type:
#   feat: 新功能
#   fix: 修复 bug
#   docs: 文档更新
#   style: 代码格式（不影响代码运行的变动）
#   refactor: 重构
#   perf: 性能优化
#   test: 测试相关
#   chore: 构建过程或辅助工具的变动

# 示例：
git commit -m "feat(engine): 添加中文姓名识别器"
git commit -m "fix(api): 修复批处理接口的内存泄漏"
git commit -m "docs(readme): 更新安装说明"
git commit -m "perf(cache): 优化 Redis 缓存策略"
```

### 9.2 分支策略

```bash
# 主分支
main          # 生产代码
develop       # 开发分支

# 功能分支
feature/add-chinese-pii     # 新功能
fix/memory-leak             # Bug 修复
perf/optimize-regex         # 性能优化
docs/update-api-doc         # 文档更新
```

---

## 10. 代码审查清单

- [ ] 代码符合 Python 编码规范
- [ ] 所有公共函数包含类型注解
- [ ] 所有公共函数/类包含文档字符串
- [ ] 新增代码有对应的单元测试
- [ ] 测试覆盖率 > 80%
- [ ] 无硬编码的敏感信息
- [ ] 日志使用结构化格式
- [ ] 错误处理恰当
- [ ] 性能影响已评估
- [ ] 安全影响已评估

---

**文档状态：** 完成
**维护者：** HPPE 开发团队
**最后审查：** 2025年10月14日