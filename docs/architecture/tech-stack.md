# HPPE 技术栈详细说明

**版本：** 1.0
**更新日期：** 2025年10月14日

---

## 1. 编程语言

### Python 3.11+
- **选择理由**：
  - ML/AI 生态系统最成熟
  - 丰富的 NLP 和数据处理库
  - 与 LLM 推理框架完美集成
  - 异步支持（asyncio）

---

## 2. LLM 基础设施

### Qwen3 8B
- **版本**：Qwen3-8B-Instruct
- **部署方式**：本地化部署
- **量化策略**：4-bit (NF4/GPTQ)
- **选择理由**：
  - 卓越的中文理解能力
  - 支持 8K+ 上下文窗口
  - 商业友好许可
  - 优秀的指令跟随能力

### vLLM 推理服务器
- **版本**：0.5.0+
- **优化技术**：
  - PagedAttention
  - Continuous Batching
  - CUDA Graph
- **配置示例**：
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B-Instruct",
    quantization="awq",  # 或 "gptq"
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    trust_remote_code=True
)
```

---

## 3. Web 框架

### FastAPI
- **版本**：0.100+
- **特性利用**：
  - 自动 OpenAPI 文档生成
  - 异步请求处理
  - Pydantic 数据验证
  - 依赖注入系统

**示例配置**：
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(
    title="HPPE API",
    version="1.0.0",
    docs_url="/api/docs"
)

class PIIRequest(BaseModel):
    text: str
    language: str = "auto"
    confidence_threshold: float = 0.85
```

---

## 4. 正则引擎框架

### Presidio-Inspired Custom Framework
基于 Microsoft Presidio 设计理念，但针对中文优化：

```python
class ChinesePatternRecognizer:
    """中文 PII 识别器基类"""

    patterns = {
        'CHINA_ID_CARD': r'[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]',
        'CHINA_PHONE': r'(?:(?:\+|00)86)?1[3-9]\d{9}',
        'CHINA_BANK_CARD': r'\d{16,19}',
        # 更多模式...
    }

    context_words = {
        'CHINA_ID_CARD': ['身份证', '身份证号', 'ID', '证件号'],
        'CHINA_PHONE': ['电话', '手机', '联系方式', 'Tel', 'Phone'],
        # 更多上下文词...
    }
```

---

## 5. 中文处理工具

### jieba 分词
- **版本**：0.42+
- **用途**：
  - 中文文本预处理
  - 实体边界识别
  - 上下文窗口提取

```python
import jieba
import jieba.analyse

# 加载自定义词典
jieba.load_userdict('pii_keywords.txt')

# 提取关键词用于上下文验证
keywords = jieba.analyse.extract_tags(text, topK=20)
```

---

## 6. 数据处理与存储

### Redis
- **版本**：7.0+
- **用途**：
  - 结果缓存（TTL: 1小时）
  - 会话管理
  - 速率限制

**配置**：
```python
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True,
    connection_pool_size=50
)
```

### Celery
- **版本**：5.3+
- **用途**：
  - 批量文档处理
  - 异步 PII 检测
  - 定时任务（模型更新）

**Worker 配置**：
```python
from celery import Celery

app = Celery('hppe',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

app.conf.update(
    task_routes={
        'hppe.tasks.detect_pii': {'queue': 'pii_detection'},
        'hppe.tasks.batch_process': {'queue': 'batch_processing'},
    },
    task_serializer='json',
    result_serializer='json'
)
```

---

## 7. 数据生成与测试

### Faker
- **版本**：Latest
- **扩展**：faker-china
- **用途**：
  - 生成测试数据
  - 合成替换策略

```python
from faker import Faker
from faker_china import ChinaProvider

fake = Faker('zh_CN')
fake.add_provider(ChinaProvider)

# 生成中文测试数据
test_data = {
    'name': fake.name(),
    'id_card': fake.ssn(),
    'phone': fake.phone_number(),
    'address': fake.address()
}
```

---

## 8. 监控与可观测性

### Prometheus + Grafana
- **指标收集**：
```python
from prometheus_client import Counter, Histogram, Gauge

pii_detected = Counter('pii_detected_total',
                       'Total PII entities detected',
                       ['entity_type', 'detection_method'])

processing_time = Histogram('processing_time_seconds',
                           'Time spent processing requests')

active_requests = Gauge('active_requests',
                       'Number of active requests')
```

### 日志系统
- **工具**：structlog
- **格式**：JSON
- **级别**：INFO (生产), DEBUG (开发)

```python
import structlog

logger = structlog.get_logger()

logger.info("pii_detection_completed",
    text_length=len(text),
    entities_found=len(entities),
    processing_time_ms=elapsed_ms
)
```

---

## 9. 开发工具

### 代码质量
- **格式化**：Black + isort
- **类型检查**：mypy
- **代码检查**：flake8 + pylint
- **安全扫描**：bandit

### 测试框架
- **单元测试**：pytest
- **性能测试**：locust
- **集成测试**：pytest + docker-compose

---

## 10. 部署工具

### 容器化
- **Docker**：多阶段构建
- **Docker Compose**：本地开发环境
- **Kubernetes**：生产部署

### CI/CD
- **GitHub Actions**：自动化测试
- **ArgoCD**：GitOps 部署
- **Harbor**：私有镜像仓库

---

## 11. 依赖管理

### requirements.txt
```text
# Core
python>=3.11
fastapi>=0.100.0
uvicorn[standard]>=0.23.0

# LLM
vllm>=0.5.0
transformers>=4.35.0
torch>=2.1.0

# Chinese NLP
jieba>=0.42.1
faker-china>=0.1.0

# Data Processing
redis>=5.0.0
celery>=5.3.0
faker>=19.0.0

# Monitoring
prometheus-client>=0.17.0
structlog>=23.1.0

# Development
pytest>=7.4.0
black>=23.0.0
mypy>=1.5.0
```

---

## 12. 硬件需求详解

### GPU 选型指南

| GPU 型号 | VRAM | 量化方式 | 批处理大小 | 用途 |
|----------|------|----------|------------|------|
| RTX 4090 | 24GB | 4-bit | 8-16 | 开发/测试 |
| A10G | 24GB | 4-bit | 8-16 | 小规模生产 |
| A100 40GB | 40GB | 4-bit/8-bit | 32-64 | 生产环境 |
| A100 80GB | 80GB | FP16 | 64-128 | 高性能生产 |

### 系统配置示例

**开发环境：**
```yaml
cpu: Intel i9 或 AMD Ryzen 9
memory: 32GB DDR5
gpu: RTX 4090 24GB
storage: 1TB NVMe SSD
```

**生产环境：**
```yaml
cpu: AMD EPYC 或 Intel Xeon (16+ cores)
memory: 128GB ECC
gpu: 2x A100 40GB (负载均衡)
storage: RAID 10 NVMe
network: 10Gbps
```

---

## 13. 性能基准

### 预期性能指标

| 操作 | 延迟 (P50) | 延迟 (P99) | 吞吐量 |
|------|------------|------------|---------|
| 短文本 (<500字) | 100ms | 300ms | 200 RPS |
| 中等文本 (500-2000字) | 300ms | 800ms | 100 RPS |
| 长文本 (>2000字) | 800ms | 2000ms | 50 RPS |
| 批处理 (100文档) | - | - | 1000 docs/min |

---

**文档状态：** 完成
**下一步：** 创建 API 规范文档和编码标准文档