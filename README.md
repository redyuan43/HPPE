# HPPE - 高精度隐私引擎

**High-Precision Privacy Engine** - 企业级 PII 检测与脱敏解决方案

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-green)
![Tests](https://img.shields.io/badge/tests-177%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)
![Epic 1](https://img.shields.io/badge/Epic%201-100%25%20Complete-success)
![Epic 2](https://img.shields.io/badge/Epic%202-25%25%20In%20Progress-yellow)

---

## 📋 项目概述

HPPE 是一个高精度、可扩展的个人身份信息（PII）检测与脱敏系统，采用**混合架构**（正则表达式 + LLM 上下文引擎），支持中文和英文 PII 的精确识别，特别优化了本地部署场景（NVIDIA RTX 3060 GPU）。

### 核心特性

- **🎯 高精度检测**：F2-score > 0.90，精确率 > 85%，召回率 > 92%
- **🌐 多语言支持**：重点支持中文和英文 PII
- **🧠 混合式架构**：正则引擎（高速） + LLM 引擎（智能上下文理解）
- **🔒 可配置脱敏**：支持多种脱敏策略（编辑、屏蔽、哈希、合成替换）
- **💻 本地部署**：无需将敏感数据发送到云端，vLLM + Qwen3 本地推理
- **⚡ GPU 加速**：优化 NVIDIA RTX 3060 (12GB)，支持 4-bit 量化
- **🔌 易于扩展**：通过 YAML 配置快速添加新的 PII 类型

---

## 🎯 支持的 PII 类型

### ✅ 已实现（Epic 1 完成）

| PII 类型 | 识别器 | 精确率 | 召回率 | F2-Score | 状态 |
|---------|--------|--------|--------|----------|------|
| 中国身份证号 | `ChinaIDCardRecognizer` | 98.4% | 94.4% | 0.952 | ✅ |
| 中国手机号 | `ChinaPhoneRecognizer` | 99.0% | 99.0% | 0.990 | ✅ |
| 中国银行卡号 | `ChinaBankCardRecognizer` | 99.0% | 99.0% | 0.990 | ✅ |
| 电子邮件 | `EmailRecognizer` | 100.0% | 100.0% | 1.000 | ✅ |
| 中国护照号 | `ChinaPassportRecognizer` | 100.0% | 100.0% | 1.000 | ✅ |
| IPv4 地址 | `IPv4Recognizer` | 98.0% | 100.0% | 0.992 | ✅ |
| IPv6 地址 | `IPv6Recognizer` | 100.0% | 100.0% | 1.000 | ✅ |
| 国际信用卡 | `CreditCardRecognizer` | 98.3% | 98.3% | 0.983 | ✅ |
| 美国 SSN | `USSocialSecurityRecognizer` | 100.0% | 100.0% | 1.000 | ✅ |

**平均指标**：精确率 99.2%，召回率 98.9%，F2-Score 0.990 ✅

### 🔄 LLM 增强（Epic 2 进行中）

- ✅ **LLM 引擎框架**：`BaseLLMEngine` + `QwenEngine` (Story 2.1)
- 🔄 **Qwen3 集成**：本地 vLLM 推理服务（安装中）
- ⏳ **上下文检测**：姓名、地址、组织机构等（Story 2.3）
- ⏳ **模糊 PII**：非结构化上下文检测（Story 2.4）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     HPPE Core Engine                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐              ┌────────────────────┐   │
│  │  Regex Engine   │              │   LLM Engine       │   │
│  │  (高速结构化)    │◄────────────►│   (智能上下文)      │   │
│  │                 │   Results    │                    │   │
│  │  • 9 识别器     │   Fusion     │  • vLLM Backend    │   │
│  │  • 正则匹配     │   Pipeline   │  • Qwen3 8B (4bit) │   │
│  │  • 校验和检查   │              │  • GPU 加速        │   │
│  │  • 上下文词增强 │              │  • OpenAI API      │   │
│  └─────────────────┘              └────────────────────┘   │
│           │                                 │               │
│           └─────────────┬───────────────────┘               │
│                         ▼                                   │
│              ┌──────────────────────┐                       │
│              │  Multi-Stage Pipeline │                       │
│              │  (精炼与融合)          │                       │
│              └──────────────────────┘                       │
│                         │                                   │
│                         ▼                                   │
│              ┌──────────────────────┐                       │
│              │  Redaction Engine     │                       │
│              │  (脱敏模块)            │                       │
│              └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始（5 分钟体验）

### 前置要求

- Python 3.10+
- conda（用于 vLLM 环境）
- NVIDIA GPU（可选，用于 LLM 加速）
  - 推荐：RTX 3060 12GB 或更高
  - CUDA 12.1+

### 1. 基础安装（仅正则引擎）

```bash
# 克隆仓库
git clone <repository-url>
cd HPPE

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/
```

### 2. 快速测试

```python
from hppe.engines.regex import RecognizerRegistry

# 初始化注册表
registry = RecognizerRegistry()
registry.auto_load()  # 自动加载所有识别器

# 检测 PII
text = """
张三的身份证号是110101199003077578，
手机号13800138000，邮箱zhang.san@example.com，
银行卡号6222021234567890123。
"""

entities = registry.detect(text)

for entity in entities:
    print(f"✓ {entity.entity_type}: {entity.value}")
    print(f"  位置: [{entity.start_pos}:{entity.end_pos}]")
    print(f"  置信度: {entity.confidence:.2%}")
    print()
```

**预期输出**：
```
✓ CHINA_ID_CARD: 110101199003077578
  位置: [7:25]
  置信度: 95.00%

✓ CHINA_PHONE: 13800138000
  位置: [29:40]
  置信度: 90.00%

✓ EMAIL: zhang.san@example.com
  位置: [43:64]
  置信度: 95.00%

✓ CHINA_BANK_CARD: 6222021234567890123
  位置: [69:88]
  置信度: 95.00%
```

---

## 🔧 完整安装（含 LLM 引擎）

### 步骤 1: 安装 vLLM 环境

```bash
# 运行自动化安装脚本（后台执行）
cd /home/ivan/HPPE
./scripts/setup_vllm_env.sh

# 或后台运行（推荐）
nohup ./scripts/setup_vllm_env.sh > vllm_install.log 2>&1 &
```

**监控安装进度**：
```bash
# 方法 1：使用监控脚本
./scripts/check_install_progress.sh

# 方法 2：实时查看日志
tail -f vllm_install.log

# 方法 3：检查进程
ps aux | grep setup_vllm_env.sh
```

**安装内容**（约 10-15 分钟）：
- ✅ conda 环境 `hppe` (Python 3.10)
- ✅ PyTorch 2.5.1 (CUDA 12.1, ~780MB)
- ✅ torchvision, torchaudio
- ✅ vLLM (~500MB)
- ✅ OpenAI, requests, pyyaml

详见：[VLLM_SETUP_GUIDE.md](VLLM_SETUP_GUIDE.md)

### 步骤 2: 下载 Qwen3 模型（待完成）

```bash
# 激活环境
conda activate hppe

# 下载模型（约 4.5GB，4-bit 量化）
python scripts/download_model.py \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --quantization awq \
  --bits 4
```

### 步骤 3: 启动 vLLM 服务

```bash
# 激活环境
conda activate hppe

# 启动 vLLM 服务（GPU 0）
./scripts/start_vllm.sh

# 或手动指定配置
CUDA_VISIBLE_DEVICES=0 ./scripts/start_vllm.sh
```

**验证服务**：
```bash
# 健康检查
curl http://localhost:8000/health

# 测试推理
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7
  }'
```

### 步骤 4: 使用 LLM 引擎

```python
from hppe.engines.llm import QwenEngine

# 初始化引擎
engine = QwenEngine(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1"
)

# 健康检查
if engine.health_check():
    print("✓ vLLM 服务运行正常")

# 上下文感知 PII 检测（示例）
text = "我叫张三，住在北京市海淀区中关村大街1号"
prompt = f"""
请检测以下文本中的个人身份信息（PII）：

文本：{text}

请以 JSON 格式返回所有检测到的 PII，格式如下：
{{
  "entities": [
    {{"type": "PERSON_NAME", "value": "张三", "start": 2, "end": 4}},
    ...
  ]
}}
"""

response = engine.generate(
    prompt=prompt,
    system_prompt="你是一个专业的隐私信息检测专家。",
    temperature=0.1,  # 低温度保证一致性
    max_tokens=1024
)

print(response)
```

---

## 📦 项目状态

### Epic 1: 核心正则引擎 ✅ (100% 完成)

| Story | 描述 | 状态 | 交付物 |
|-------|------|------|--------|
| **1.1** | 框架搭建 | ✅ 完成 | `Entity`, `BaseRecognizer`, `RecognizerRegistry`, `ConfigLoader` |
| **1.2** | 中国 PII 识别器 | ✅ 完成 | 身份证、手机号、银行卡、护照识别器（4 个） |
| **1.3** | 全球 PII 识别器 | ✅ 完成 | Email, IPv4/IPv6, 信用卡, US SSN（5 个） |
| **1.4** | 基准测试与优化 | ✅ 完成 | 性能测试套件，F2-Score: 0.990 |

**交付成果**：
- 📁 **177 个测试用例**，全部通过 ✅
- 📊 **代码覆盖率**：92%（核心模块 96-100%）
- 🎯 **性能指标**：精确率 99.2%，召回率 98.9%，F2-Score 0.990
- 📝 **完整文档**：架构设计、PRD、使用指南

### Epic 2: LLM 上下文引擎集成 🔄 (25% 完成)

| Story | 描述 | 状态 | 进度 |
|-------|------|------|------|
| **2.1** | vLLM 基础设施 | ✅ 完成 | 100% |
| **2.2** | Qwen3 模型集成 | 🔄 进行中 | 60% (安装中) |
| **2.3** | 上下文 PII 检测 | ⏳ 待开始 | 0% |
| **2.4** | 多阶段精炼流水线 | ⏳ 待开始 | 0% |

**当前进展**：
- ✅ `BaseLLMEngine` 抽象接口
- ✅ `QwenEngine` OpenAI 兼容实现
- ✅ vLLM 启动脚本与配置
- ✅ LLM 引擎示例代码（6 个场景）
- 🔄 vLLM 环境安装（后台进行中，预计 5-10 分钟完成）
- ⏳ Qwen3 模型下载（待安装完成后执行）

---

## 🏗️ 项目结构

```
HPPE/
├── src/hppe/                        # 源代码
│   ├── models/                      # 数据模型
│   │   └── entity.py                # PII 实体定义
│   ├── engines/                     # 检测引擎
│   │   ├── regex/                   # 正则引擎 ✅
│   │   │   ├── base.py              # 抽象基类
│   │   │   ├── registry.py          # 识别器注册表
│   │   │   ├── config_loader.py     # YAML 配置加载器
│   │   │   └── recognizers/         # 识别器实现（9 个）
│   │   │       ├── china/           # 中国 PII
│   │   │       │   ├── id_card.py
│   │   │       │   ├── phone.py
│   │   │       │   ├── bank_card.py
│   │   │       │   └── passport.py
│   │   │       └── global_/         # 全球 PII
│   │   │           ├── email.py
│   │   │           ├── ipv4.py
│   │   │           ├── ipv6.py
│   │   │           ├── credit_card.py
│   │   │           └── us_ssn.py
│   │   └── llm/                     # LLM 引擎 🔄
│   │       ├── __init__.py          # 模块导出
│   │       ├── base.py              # BaseLLMEngine ✅
│   │       └── qwen_engine.py       # QwenEngine ✅
│   └── ...
├── data/patterns/                   # 配置文件
│   ├── china_pii.yaml               # 中文 PII 配置 ✅
│   └── global_pii.yaml              # 全球 PII 配置 ✅
├── configs/                         # 系统配置
│   └── llm_config.yaml              # LLM 引擎配置 ✅
├── scripts/                         # 工具脚本
│   ├── setup_vllm_env.sh            # vLLM 环境安装 ✅
│   ├── check_install_progress.sh    # 安装进度监控 ✅
│   └── start_vllm.sh                # vLLM 服务启动 ✅
├── tests/                           # 测试代码
│   ├── unit/                        # 单元测试（177 个）✅
│   └── integration/                 # 集成测试（待添加）
├── examples/                        # 示例代码
│   └── llm_engine_example.py        # LLM 引擎示例 ✅
├── docs/                            # 文档
│   ├── architecture/                # 架构文档
│   ├── prd/                         # 产品需求
│   │   ├── epic-1-core-regex-engine.md  ✅
│   │   └── epic-2-llm-engine.md         ✅
│   └── stories/                     # Story 文档
├── VLLM_SETUP_GUIDE.md              # vLLM 安装指南 ✅
├── STORY_2.1_COMPLETION_REPORT.md   # Story 2.1 报告 ✅
└── README.md                        # 本文档
```

---

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 查看详细输出
pytest tests/ -v

# 运行特定模块
pytest tests/unit/engines/regex/recognizers/

# 查看覆盖率
pytest --cov=src/hppe tests/

# 生成 HTML 覆盖率报告
pytest --cov=src/hppe --cov-report=html tests/
open htmlcov/index.html  # 浏览器查看
```

### 测试状态

**当前统计**（Epic 1 完成）：
```
============= test session starts ==============
platform linux -- Python 3.10.18
collected 177 items

tests/unit/models/test_entity.py ................    [ 9%]
tests/unit/engines/regex/test_base.py ............   [15%]
tests/unit/engines/regex/test_registry.py ......    [18%]
tests/unit/engines/regex/test_config_loader.py ..   [20%]
tests/unit/engines/regex/recognizers/ .............
............................................      [100%]

============= 177 passed in 2.34s ===============
```

**覆盖率报告**：
```
Name                                                      Stmts   Miss  Cover
-----------------------------------------------------------------------------
src/hppe/models/entity.py                                    45      2    96%
src/hppe/engines/regex/base.py                               67      3    96%
src/hppe/engines/regex/registry.py                           98      5    95%
src/hppe/engines/regex/config_loader.py                      54      2    96%
src/hppe/engines/regex/recognizers/china/id_card.py         142      8    94%
src/hppe/engines/regex/recognizers/china/phone.py            89      4    96%
src/hppe/engines/regex/recognizers/china/bank_card.py       110      6    95%
src/hppe/engines/regex/recognizers/china/passport.py         67      3    96%
src/hppe/engines/regex/recognizers/global_/email.py          56      2    96%
src/hppe/engines/regex/recognizers/global_/ipv4.py           78      4    95%
src/hppe/engines/regex/recognizers/global_/ipv6.py           92      5    95%
src/hppe/engines/regex/recognizers/global_/credit_card.py   123      7    94%
src/hppe/engines/regex/recognizers/global_/us_ssn.py         67      3    96%
src/hppe/engines/llm/base.py                                126     14    89%
src/hppe/engines/llm/qwen_engine.py                         241     28    88%
-----------------------------------------------------------------------------
TOTAL                                                       1455    106    92%
```

### 性能基准测试

```bash
# 运行基准测试套件（Story 1.4）
pytest tests/performance/test_recognizer_benchmarks.py -v

# 生成性能报告
python tests/performance/generate_report.py
```

**典型性能**（单线程，Intel i7）：
- 中国身份证检测：~0.5ms/文档
- 邮箱检测：~0.3ms/文档
- 批量检测（9 个识别器）：~2.5ms/文档

---

## 📊 性能指标

### 目标 vs 实际

| 指标 | 目标 | 实际（Epic 1） | 状态 |
|------|------|---------------|------|
| 精确率 (Precision) | > 85% | **99.2%** | ✅ 超额完成 |
| 召回率 (Recall) | > 92% | **98.9%** | ✅ 超额完成 |
| F2-Score | > 0.90 | **0.990** | ✅ 超额完成 |
| 处理延迟 (P50) | < 500ms | **~2.5ms** | ✅ 超额完成 |

### GPU 配置（Epic 2）

**用户硬件**：
- 2 × NVIDIA RTX 3060 (12GB VRAM)
- CUDA 13.0（兼容 12.1）

**推荐配置**：
```yaml
# 单卡部署（推荐）
GPU 0: Qwen3 8B (4-bit AWQ)
  VRAM: ~5-6GB
  延迟: ~100-200ms
  并发: 10 请求

GPU 1: 预留（未来使用）

# 双卡张量并行（高级）
GPU 0+1: Qwen3 14B (FP16)
  VRAM: 2×12GB = 24GB
  延迟: ~150-300ms
  吞吐: 2× 单卡
```

---

## 🛠️ 开发指南

### 添加新的正则识别器

#### 1. 创建识别器类

```python
# src/hppe/engines/regex/recognizers/my_recognizer.py
from hppe.engines.regex.base import BaseRecognizer
from hppe.models.entity import Entity
from typing import List
import re

class MyPIIRecognizer(BaseRecognizer):
    """自定义 PII 识别器"""

    def __init__(self, config: RecognizerConfig):
        super().__init__(config)

    def detect(self, text: str) -> List[Entity]:
        """检测 PII 实体"""
        entities = []

        for pattern_config in self.patterns:
            pattern = pattern_config.pattern
            for match in re.finditer(pattern, text):
                entity = Entity(
                    entity_type=self.config.entity_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=self._calculate_confidence(match.group(), text)
                )

                # 可选：自定义验证
                if self.validate(entity):
                    entities.append(entity)

        return entities

    def validate(self, entity: Entity) -> bool:
        """自定义验证逻辑（可选）"""
        # 实现特定的验证规则
        return True
```

#### 2. 配置 YAML 文件

```yaml
# data/patterns/my_patterns.yaml
recognizers:
  - name: MyPIIRecognizer
    entity_type: MY_PII_TYPE
    confidence_base: 0.85
    patterns:
      - pattern: 'your-regex-pattern'
        score: 0.90
        description: "描述"
    context_words:
      - '关键词1'
      - '关键词2'
    deny_lists:
      - '排除词1'
    validation:
      type: 'custom'  # 或 'checksum', 'luhn', 等
```

#### 3. 注册识别器

```python
from hppe.engines.regex import RecognizerRegistry
from hppe.engines.regex.config_loader import ConfigLoader

# 加载配置
loader = ConfigLoader()
configs = loader.load_file("data/patterns/my_patterns.yaml")

# 创建并注册识别器
from my_recognizer import MyPIIRecognizer
registry = RecognizerRegistry()

for config in configs:
    recognizer = MyPIIRecognizer(config)
    registry.register(recognizer)

# 使用
entities = registry.detect("your test text")
```

#### 4. 编写测试

```python
# tests/unit/engines/regex/recognizers/test_my_recognizer.py
import pytest
from hppe.engines.regex.recognizers.my_recognizer import MyPIIRecognizer
from hppe.engines.regex.base import RecognizerConfig

def test_my_recognizer_basic():
    """基础检测测试"""
    config = RecognizerConfig(
        name="MyPIIRecognizer",
        entity_type="MY_PII_TYPE",
        confidence_base=0.85,
        patterns=[PatternConfig(pattern=r'your-pattern', score=0.90)]
    )

    recognizer = MyPIIRecognizer(config)
    text = "测试文本包含 PII"
    entities = recognizer.detect(text)

    assert len(entities) > 0
    assert entities[0].entity_type == "MY_PII_TYPE"
```

### 编码规范

遵循以下原则：

- **SOLID**：单一职责，开闭原则，依赖倒置
- **KISS**：保持简单，避免过度设计
- **DRY**：消除重复代码
- **YAGNI**：只实现当前需要的功能

**具体要求**：
- Python 3.10+ 类型注解（强制）
- 所有公共类和函数必须包含 docstring
- 单元测试覆盖率 > 80%
- 遵循 PEP 8 代码风格

详见：[docs/architecture/coding-standards.md](docs/architecture/coding-standards.md)

---

## 📚 文档

### 架构文档
- [系统架构概览](docs/architecture.md)
- [数据模型设计](docs/architecture/data-models.md)
- [引擎架构](docs/architecture/engines.md)
- [编码规范](docs/architecture/coding-standards.md)

### 产品需求文档
- [PRD 总览](docs/prd.md)
- [Epic 1: 核心正则引擎](docs/prd/epic-1-core-regex-engine.md) ✅
- [Epic 2: LLM 上下文引擎](docs/prd/epic-2-llm-engine.md) 🔄
- [Epic 3: 脱敏模块](docs/prd/epic-3-redaction-engine.md) ⏳
- [Epic 4: API 服务](docs/prd/epic-4-api-service.md) ⏳

### Story 文档
- [Story 1.1: 框架搭建](docs/stories/1.1.regex-engine-framework.md) ✅
- [Story 1.2: 中国 PII 识别器](docs/stories/1.2.china-pii-recognizers.md) ✅
- [Story 1.3: 全球 PII 识别器](docs/stories/1.3.global-pii-recognizers.md) ✅
- [Story 1.4: 基准测试与优化](docs/stories/1.4.benchmarking-optimization.md) ✅
- [Story 2.1: vLLM 基础设施](STORY_2.1_COMPLETION_REPORT.md) ✅
- [Story 2.2: Qwen3 模型集成](docs/stories/2.2.qwen3-integration.md) 🔄

### 安装指南
- [vLLM 环境配置指南](VLLM_SETUP_GUIDE.md) ✅
- [故障排查](docs/troubleshooting.md)

---

## 🔗 相关链接

- **vLLM**: https://github.com/vllm-project/vllm
- **Qwen3**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **PyTorch**: https://pytorch.org/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit

---

## 🚧 后续规划

### Epic 2: LLM 上下文引擎（进行中）
- ✅ Story 2.1: vLLM 基础设施
- 🔄 Story 2.2: Qwen3 模型集成（60% - 安装中）
- ⏳ Story 2.3: 上下文 PII 检测（姓名、地址、组织）
- ⏳ Story 2.4: 多阶段精炼流水线

### Epic 3: 脱敏模块（计划中）
- ⏳ Story 3.1: 脱敏策略框架
- ⏳ Story 3.2: 编辑距离脱敏
- ⏳ Story 3.3: 合成数据替换
- ⏳ Story 3.4: 格式保留脱敏

### Epic 4: API 服务（计划中）
- ⏳ Story 4.1: FastAPI 服务框架
- ⏳ Story 4.2: 批量处理 API
- ⏳ Story 4.3: 流式处理 API
- ⏳ Story 4.4: 监控与日志

---

## 📝 许可证

待定

---

## 👥 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)（待创建）

---

## 📞 联系方式

- **项目负责人**：Sarah (PO)
- **技术负责人**：TBD
- **问题反馈**：[GitHub Issues](https://github.com/your-repo/issues)

---

## 🎉 致谢

感谢以下开源项目和社区：
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [Qwen](https://github.com/QwenLM/Qwen) - 阿里云通义千问大模型
- [PyTorch](https://pytorch.org/) - 深度学习框架
- 所有贡献者和测试人员

---

**当前版本**：0.2.0 (Epic 1 完成，Epic 2 进行中)
**最后更新**：2025-10-14
**项目状态**：🚀 积极开发中

---

## 📊 项目进度总览

```
Epic 1 (正则引擎)      ████████████████████ 100% ✅
Epic 2 (LLM 引擎)      █████░░░░░░░░░░░░░░░  25% 🔄
Epic 3 (脱敏模块)      ░░░░░░░░░░░░░░░░░░░░   0% ⏳
Epic 4 (API 服务)      ░░░░░░░░░░░░░░░░░░░░   0% ⏳
─────────────────────────────────────────────────
总体进度               ██████░░░░░░░░░░░░░░  31%
```

**下一里程碑**：Epic 2 完成（预计 1-2 周）

---

**快速导航**：
- 🚀 [快速开始](#-快速开始5-分钟体验)
- 🔧 [完整安装](#-完整安装含-llm-引擎)
- 📚 [API 文档](#-文档)
- 🧪 [运行测试](#-测试)
- 🛠️ [开发指南](#️-开发指南)
- 📊 [性能指标](#-性能指标)
