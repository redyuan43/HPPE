# HPPE 源代码结构

**版本：** 1.0
**更新日期：** 2025年10月14日

---

## 项目目录结构

```
HPPE/
├── .bmad-core/                 # BMad 方法论文件
│   ├── agents/                 # BMad 代理配置
│   └── tasks/                  # BMad 任务定义
│
├── docs/                       # 项目文档
│   ├── architecture.md         # 主架构文档
│   ├── architecture/           # 架构相关文档
│   │   ├── tech-stack.md      # 技术栈详细说明
│   │   ├── source-tree.md     # 源代码结构（本文档）
│   │   ├── coding-standards.md # 编码标准
│   │   └── api-spec.md        # API 规范
│   ├── prd/                   # 产品需求文档
│   ├── qa/                    # 测试文档
│   └── stories/               # 用户故事
│
├── src/                       # 源代码
│   ├── hppe/                  # 主应用包
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI 应用入口
│   │   ├── config.py         # 配置管理
│   │   │
│   │   ├── api/              # API 层
│   │   │   ├── __init__.py
│   │   │   ├── v1/           # API v1
│   │   │   │   ├── __init__.py
│   │   │   │   ├── endpoints.py  # API 端点
│   │   │   │   ├── models.py     # Pydantic 模型
│   │   │   │   └── dependencies.py # 依赖注入
│   │   │   └── middleware.py     # 中间件
│   │   │
│   │   ├── core/             # 核心业务逻辑
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py   # 主处理流水线
│   │   │   ├── orchestrator.py # 流程协调器
│   │   │   └── exceptions.py # 自定义异常
│   │   │
│   │   ├── engines/          # 检测引擎
│   │   │   ├── __init__.py
│   │   │   ├── regex/        # 正则引擎
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py   # 基础识别器
│   │   │   │   ├── recognizers/ # 具体识别器
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── china_pii.py  # 中国 PII 识别器
│   │   │   │   │   ├── global_pii.py # 全球 PII 识别器
│   │   │   │   │   └── custom.py     # 自定义识别器
│   │   │   │   └── registry.py # 识别器注册表
│   │   │   │
│   │   │   └── llm/          # LLM 引擎
│   │   │       ├── __init__.py
│   │   │       ├── qwen_engine.py # Qwen3 引擎
│   │   │       ├── prompts.py     # 提示模板
│   │   │       └── inference.py   # 推理管理
│   │   │
│   │   ├── processors/       # 处理器
│   │   │   ├── __init__.py
│   │   │   ├── disambiguation.py  # 歧义消除
│   │   │   ├── validation.py      # 上下文验证
│   │   │   ├── merger.py          # 实体合并
│   │   │   └── chinese_nlp.py     # 中文 NLP 处理
│   │   │
│   │   ├── anonymization/    # 脱敏模块
│   │   │   ├── __init__.py
│   │   │   ├── strategies.py # 脱敏策略
│   │   │   ├── redaction.py  # 编辑策略
│   │   │   ├── masking.py    # 屏蔽策略
│   │   │   ├── hashing.py    # 哈希策略
│   │   │   └── synthetic.py  # 合成替换
│   │   │
│   │   ├── models/           # 数据模型
│   │   │   ├── __init__.py
│   │   │   ├── entity.py     # PII 实体模型
│   │   │   ├── request.py    # 请求模型
│   │   │   └── response.py   # 响应模型
│   │   │
│   │   ├── tasks/            # Celery 任务
│   │   │   ├── __init__.py
│   │   │   ├── detection.py  # 检测任务
│   │   │   ├── batch.py      # 批处理任务
│   │   │   └── maintenance.py # 维护任务
│   │   │
│   │   ├── cache/            # 缓存管理
│   │   │   ├── __init__.py
│   │   │   ├── redis_client.py
│   │   │   └── strategies.py
│   │   │
│   │   ├── monitoring/       # 监控
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py    # Prometheus 指标
│   │   │   └── logging.py    # 结构化日志
│   │   │
│   │   └── utils/            # 工具函数
│   │       ├── __init__.py
│   │       ├── text.py       # 文本处理
│   │       ├── validation.py # 验证工具
│   │       └── performance.py # 性能工具
│   │
│   └── cli/                  # 命令行工具
│       ├── __init__.py
│       └── commands.py
│
├── tests/                    # 测试代码
│   ├── __init__.py
│   ├── unit/                # 单元测试
│   │   ├── __init__.py
│   │   ├── test_regex_engine.py
│   │   ├── test_llm_engine.py
│   │   ├── test_pipeline.py
│   │   └── test_anonymization.py
│   │
│   ├── integration/         # 集成测试
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   └── test_full_pipeline.py
│   │
│   ├── performance/         # 性能测试
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   └── load_test.py
│   │
│   └── fixtures/            # 测试数据
│       ├── __init__.py
│       ├── sample_texts.py
│       └── pii_samples.json
│
├── data/                    # 数据文件
│   ├── patterns/           # 正则模式
│   │   ├── china_pii.yaml
│   │   └── global_pii.yaml
│   │
│   ├── dictionaries/      # 词典
│   │   ├── pii_keywords.txt
│   │   └── stop_words.txt
│   │
│   └── models/            # 模型文件
│       └── .gitkeep
│
├── scripts/               # 脚本
│   ├── setup.sh          # 环境设置
│   ├── download_model.py # 模型下载
│   ├── generate_test_data.py # 生成测试数据
│   └── benchmark.py      # 性能基准测试
│
├── deployments/          # 部署配置
│   ├── docker/          # Docker 配置
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── docker-compose.dev.yml
│   │
│   ├── kubernetes/      # K8s 配置
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── hpa.yaml
│   │
│   └── helm/           # Helm Charts
│       └── hppe/
│           ├── Chart.yaml
│           ├── values.yaml
│           └── templates/
│
├── configs/            # 配置文件
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
│
├── .github/           # GitHub 配置
│   └── workflows/
│       ├── ci.yml    # CI 流程
│       └── cd.yml    # CD 流程
│
├── .env.example       # 环境变量示例
├── .gitignore        # Git 忽略文件
├── requirements.txt  # Python 依赖
├── requirements-dev.txt # 开发依赖
├── pyproject.toml   # Python 项目配置
├── Makefile         # Make 命令
├── README.md        # 项目说明
└── LICENSE         # 许可证
```

---

## 核心模块说明

### 1. API 层 (`src/hppe/api/`)
负责处理 HTTP 请求，包括：
- RESTful API 端点定义
- 请求/响应模型验证
- 认证和授权
- 错误处理

### 2. 核心业务层 (`src/hppe/core/`)
实现主要业务逻辑：
- 四阶段处理流水线
- 流程协调和编排
- 异常处理

### 3. 检测引擎 (`src/hppe/engines/`)
PII 检测的核心实现：
- **正则引擎**：基于模式的结构化 PII 检测
- **LLM 引擎**：基于 Qwen3 的非结构化 PII 检测

### 4. 处理器 (`src/hppe/processors/`)
数据处理和优化：
- 歧义消除算法
- 上下文验证逻辑
- 实体合并策略
- 中文特定处理

### 5. 脱敏模块 (`src/hppe/anonymization/`)
实现各种脱敏策略：
- 编辑（完全删除）
- 屏蔽（部分隐藏）
- 哈希加密
- 合成数据替换

### 6. 任务队列 (`src/hppe/tasks/`)
Celery 异步任务：
- 批量文档处理
- 长时间运行的检测任务
- 定期维护任务

---

## 开发规范

### 模块导入顺序
```python
# 1. 标准库
import os
import sys
from typing import List, Dict

# 2. 第三方库
import redis
from fastapi import FastAPI
from pydantic import BaseModel

# 3. 本地模块
from hppe.core import pipeline
from hppe.engines.regex import ChineseRecognizer
```

### 文件命名规范
- 模块名：小写，下划线分隔 (`regex_engine.py`)
- 类名：驼峰命名 (`PatternRecognizer`)
- 函数名：小写，下划线分隔 (`detect_pii`)
- 常量：大写，下划线分隔 (`MAX_TEXT_LENGTH`)

### 目录职责
- `/api`: 只包含 API 相关代码
- `/core`: 业务逻辑，不依赖框架
- `/engines`: 可独立测试的检测引擎
- `/models`: 纯数据结构定义
- `/utils`: 无状态的工具函数

---

## 配置管理

### 环境变量 (`.env`)
```bash
# API 配置
API_HOST=0.0.0.0
API_PORT=8000

# LLM 配置
LLM_MODEL_PATH=/models/Qwen3-8B
LLM_DEVICE=cuda:0
LLM_MAX_LENGTH=8192

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Celery 配置
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# 监控配置
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```

### 配置文件 (`configs/development.yaml`)
```yaml
app:
  name: HPPE
  version: 1.0.0
  debug: true

pipeline:
  stages:
    - regex_detection
    - llm_detection
    - disambiguation
    - validation
    - anonymization

performance:
  cache_ttl: 3600
  max_batch_size: 100
  request_timeout: 30
```

---

## 测试组织

### 单元测试
- 每个模块对应一个测试文件
- 测试覆盖率目标：>80%
- 使用 pytest fixtures 管理测试数据

### 集成测试
- 测试完整的检测流水线
- API 端到端测试
- 与外部服务的集成

### 性能测试
- 使用 locust 进行负载测试
- 基准测试关键算法
- 内存和 CPU 分析

---

**文档状态：** 完成
**下一步：** 创建编码标准文档