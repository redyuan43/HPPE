# Epic 2: LLM 上下文引擎集成

**Epic ID:** EPIC-2
**优先级:** P0
**状态:** 🔄 进行中
**预计工作量:** 2-3 周
**负责人:** TBD
**创建日期:** 2025-10-14

---

## Epic 概述

### 目标
集成 Qwen3 8B 大语言模型，实现非结构化 PII（如姓名、地址等）的智能检测，补充正则表达式引擎无法覆盖的场景。

### 业务价值
- **提升召回率:** 检测正则表达式难以匹配的 PII（如中文姓名、自由格式地址）
- **上下文理解:** 利用 LLM 的语义理解能力，减少误报
- **灵活扩展:** 通过提示词工程快速支持新的 PII 类型
- **本地部署:** 基于用户的 2 × RTX 3060 GPU，无需云服务

### 硬件配置
**用户硬件:**
- 2 × NVIDIA RTX 3060 (12GB VRAM 每张)
- 总 VRAM: 24GB
- 适合运行量化模型

**部署策略:**
- **方案 1:** 单卡部署（推荐）
  - 使用 1 张 3060 运行 Qwen3 8B (4-bit 量化)
  - 另 1 张保留给未来扩展或批处理
  - VRAM 占用: ~5-6GB
  - 推理延迟: ~100-200ms

- **方案 2:** 张量并行
  - 使用 2 张 3060 做张量并行
  - 加速推理，降低单次延迟
  - VRAM 占用: 均分模型参数
  - 推理延迟: ~50-100ms

### 依赖条件
- ✅ Epic 1 完成（正则引擎已就绪）
- ✅ CUDA 和 GPU 驱动已安装
- 🔄 vLLM 安装（待完成）
- 🔄 Qwen3 8B 模型下载（待完成）

---

## Stories 列表

### Story 2.1: vLLM 推理服务器部署
**优先级:** P0
**工作量:** 3 天

**用户故事:**
> 作为开发者，我需要在本地 GPU 上部署 vLLM 推理服务，以便高效地运行 Qwen3 模型。

**验收标准:**
1. ✅ 成功安装 vLLM (支持 CUDA)
2. ✅ 启动 vLLM 服务器，监听本地端口
3. ✅ 验证 GPU 可见性和内存管理
4. ✅ 实现健康检查端点
5. ✅ 配置并发请求处理（支持多客户端）
6. ✅ 实现日志和性能监控

**技术要求:**
- 使用 vLLM 0.6+ 版本
- 支持 OpenAI 兼容 API
- 配置 GPU memory utilization (~0.9)
- 文件位置: `src/hppe/engines/llm/vllm_server.py`

**环境配置:**
```bash
# 安装 vLLM
pip install vllm

# 验证 GPU
nvidia-smi

# 启动服务
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000
```

**性能目标:**
- 首次加载时间: < 60秒
- 推理延迟 (P50): < 200ms
- 推理延迟 (P99): < 500ms
- 吞吐量: > 10 RPS (单卡)

---

### Story 2.2: Qwen3 模型集成
**优先级:** P0
**工作量:** 2 天

**用户故事:**
> 作为开发者，我需要集成 Qwen3 8B 模型，并提供统一的推理接口，以便在 HPPE 中调用。

**验收标准:**
1. ✅ 下载并验证 Qwen3 8B 模型
2. ✅ 实现模型加载和初始化
3. ✅ 创建 LLMEngine 抽象类
4. ✅ 实现 QwenEngine 具体类
5. ✅ 支持流式和非流式响应
6. ✅ 实现错误处理和重试机制

**技术要求:**
- 模型: Qwen/Qwen2.5-7B-Instruct (或更新版本)
- 量化: 4-bit (GPTQ/AWQ) 可选
- 接口: OpenAI 兼容 API
- 文件位置: `src/hppe/engines/llm/qwen_engine.py`

**API 设计:**
```python
from hppe.engines.llm import QwenEngine

# 初始化引擎
engine = QwenEngine(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    temperature=0.1,
    max_tokens=512
)

# 调用推理
response = engine.generate(
    prompt="检测以下文本中的 PII: 我是张三，电话13812345678",
    system_prompt="你是一个专业的隐私信息检测助手。"
)
```

**模型选择:**
- **Qwen2.5-7B-Instruct**: 7B 参数，指令微调版本
- **量化版本**: Qwen2.5-7B-Instruct-GPTQ-Int4 (推荐)
- **VRAM 需求**:
  - FP16: ~14GB (不适合单张 3060)
  - 4-bit: ~4-5GB (适合单张 3060) ✅

**依赖:**
- Story 2.1 完成

---

### Story 2.3: 零样本 PII 检测实现
**优先级:** P0
**工作量:** 4 天

**用户故事:**
> 作为开发者，我需要使用 LLM 实现零样本 PII 检测，以便识别非结构化的 PII（姓名、地址等）。

**验收标准:**
1. ✅ 设计零样本检测提示模板
2. ✅ 实现 LLMRecognizer 类（继承 BaseRecognizer）
3. ✅ 支持以下 PII 类型:
   - 中文姓名 (PERSON_NAME_ZH)
   - 英文姓名 (PERSON_NAME)
   - 中文地址 (ADDRESS_ZH)
   - 英文地址 (ADDRESS)
   - 组织名称 (ORGANIZATION)
4. ✅ 解析 LLM 输出为结构化实体
5. ✅ 实现置信度评分机制
6. ✅ 单元测试覆盖率 > 85%

**技术要求:**
- 使用 JSON Mode 或结构化输出
- 支持批量检测（一次调用处理多个 PII）
- 实现输出解析容错机制
- 文件位置: `src/hppe/engines/llm/recognizers.py`

**提示词设计原则:**
- 清晰的任务描述
- 具体的输出格式要求
- 少量示例（Few-shot，可选）
- 明确的约束条件

**输出格式:**
```json
{
  "entities": [
    {
      "type": "PERSON_NAME_ZH",
      "value": "张三",
      "start_pos": 5,
      "end_pos": 7,
      "confidence": 0.95
    }
  ]
}
```

**性能目标:**
- 单次检测延迟: < 300ms
- 准确率 (Precision): > 80%
- 召回率 (Recall): > 85%

**依赖:**
- Story 2.2 完成

---

### Story 2.4: 提示工程和优化
**优先级:** P0
**工作量:** 3 天

**用户故事:**
> 作为开发者，我需要优化 LLM 提示词和推理参数，以便提高检测准确性和性能。

**验收标准:**
1. ✅ 设计并测试至少 3 种提示词变体
2. ✅ 实现 A/B 测试框架
3. ✅ 优化推理参数（temperature, top_p, max_tokens）
4. ✅ 实现提示词模板管理系统
5. ✅ 建立评估数据集（中文 100 条，英文 100 条）
6. ✅ 对比测试，选择最佳配置

**技术要求:**
- 支持多语言提示模板
- 提示词版本控制
- 可配置的推理参数
- 文件位置: `src/hppe/engines/llm/prompts/`

**提示词优化策略:**

**策略 1: 直接指令式**
```
你是一个隐私信息检测专家。请检测以下文本中的个人身份信息（PII）。

文本: {text}

请识别所有的姓名、地址、电话、邮箱等敏感信息，并以 JSON 格式返回。
```

**策略 2: 角色扮演式**
```
你是一位资深的数据合规专家，负责审查文档中的隐私风险。
请仔细分析以下文本，识别所有可能导致隐私泄露的个人身份信息。

文本: {text}

输出格式: JSON
```

**策略 3: Few-shot 示例式**
```
任务: 检测文本中的 PII

示例 1:
输入: "我是张三，住在北京市海淀区"
输出: {"entities": [{"type": "姓名", "value": "张三"}, {"type": "地址", "value": "北京市海淀区"}]}

示例 2:
输入: "联系方式: john@example.com"
输出: {"entities": [{"type": "邮箱", "value": "john@example.com"}]}

现在请处理:
输入: {text}
输出:
```

**评估指标:**
- 准确率 (Precision)
- 召回率 (Recall)
- F1-Score
- 推理延迟
- Token 使用量

**依赖:**
- Story 2.3 完成

---

## 技术架构

### 1. 组件架构

```
┌─────────────────────────────────────────────────────────┐
│                    HPPE Core System                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐        ┌─────────────────┐        │
│  │  Regex Engine   │        │   LLM Engine    │        │
│  │  (Epic 1)       │        │   (Epic 2)      │        │
│  └────────┬────────┘        └────────┬────────┘        │
│           │                          │                  │
│           │    ┌────────────────────┐│                  │
│           └────┤ Detection Pipeline ├┘                  │
│                └─────────┬──────────┘                   │
│                          │                              │
│                ┌─────────▼──────────┐                   │
│                │  Result Merger     │                   │
│                │  (Epic 3)          │                   │
│                └────────────────────┘                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              vLLM Inference Server (Local)               │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────────────────────┐               │
│  │  Qwen3 8B Model (4-bit Quantized)    │               │
│  │  VRAM: ~5GB                           │               │
│  └──────────────────────────────────────┘               │
│                                                           │
│  ┌──────────────────────────────────────┐               │
│  │  OpenAI-compatible API                │               │
│  │  Endpoint: http://localhost:8000/v1   │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  GPU Hardware (Local)                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐        ┌─────────────────┐        │
│  │  RTX 3060 #1    │        │  RTX 3060 #2    │        │
│  │  12GB VRAM      │        │  12GB VRAM      │        │
│  │  (Active)       │        │  (Reserved)     │        │
│  └─────────────────┘        └─────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### 2. 数据流

```
用户输入文本
    │
    ▼
┌────────────────┐
│ Text Splitter  │  ← 分块（长文本）
└───────┬────────┘
        │
        ├─────────────────┐
        ▼                 ▼
┌────────────┐    ┌────────────┐
│Regex Engine│    │ LLM Engine │
│  (Fast)    │    │  (Smart)   │
└─────┬──────┘    └──────┬─────┘
      │                  │
      └────────┬─────────┘
               ▼
    ┌──────────────────┐
    │  Entity Merger   │  ← 去重、合并
    └────────┬─────────┘
             ▼
      ┌────────────┐
      │   Result   │
      └────────────┘
```

### 3. LLM Engine 类设计

```python
# src/hppe/engines/llm/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from hppe.models.entity import Entity

class BaseLLMEngine(ABC):
    """LLM 引擎抽象基类"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> str:
        """生成文本响应"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass


# src/hppe/engines/llm/qwen_engine.py
class QwenEngine(BaseLLMEngine):
    """Qwen3 模型引擎实现"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout: int = 30
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.client = OpenAI(base_url=base_url)

    def generate(self, prompt: str, **kwargs) -> str:
        """调用 vLLM API 生成响应"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": kwargs.get("system_prompt", "")},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 512)
        )
        return response.choices[0].message.content


# src/hppe/engines/llm/recognizers.py
class LLMRecognizer(BaseRecognizer):
    """基于 LLM 的 PII 识别器"""

    def __init__(self, config: Dict, llm_engine: BaseLLMEngine):
        super().__init__(config)
        self.llm_engine = llm_engine
        self.prompt_template = self._load_prompt_template()

    def detect(self, text: str) -> List[Entity]:
        """使用 LLM 检测 PII"""
        # 1. 构建提示词
        prompt = self.prompt_template.format(text=text)

        # 2. 调用 LLM
        response = self.llm_engine.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=512
        )

        # 3. 解析响应
        entities = self._parse_llm_response(response, text)

        # 4. 验证和过滤
        validated_entities = [
            e for e in entities
            if self.validate(e)
        ]

        return validated_entities

    def _parse_llm_response(
        self,
        response: str,
        original_text: str
    ) -> List[Entity]:
        """解析 LLM JSON 输出为 Entity 对象"""
        try:
            data = json.loads(response)
            entities = []

            for item in data.get("entities", []):
                entity = Entity(
                    entity_type=item["type"],
                    value=item["value"],
                    start_pos=item.get("start_pos", 0),
                    end_pos=item.get("end_pos", 0),
                    confidence=item.get("confidence", 0.8),
                    detection_method="llm",
                    recognizer_name=self.recognizer_name,
                    metadata={"llm_model": self.llm_engine.model_name}
                )
                entities.append(entity)

            return entities

        except json.JSONDecodeError:
            # 容错：尝试修复 JSON
            return []
```

---

## 测试策略

### 1. 单元测试
- **LLM Engine 测试**: Mock API 调用
- **Recognizer 测试**: 使用固定的 LLM 响应
- **提示词测试**: 验证模板渲染

### 2. 集成测试
- **端到端测试**: 启动真实 vLLM 服务
- **多语言测试**: 中文和英文文本
- **边界测试**: 长文本、特殊字符

### 3. 性能测试
- **延迟测试**: P50, P95, P99
- **吞吐量测试**: RPS
- **GPU 利用率**: nvidia-smi 监控

### 4. 准确性测试
- **基准数据集**: 200 条标注数据
- **指标计算**: Precision, Recall, F1
- **对比测试**: LLM vs Regex

---

## 性能目标

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| 模型加载时间 | < 60秒 | 首次启动计时 |
| 推理延迟 (P50) | < 200ms | 单次请求时间 |
| 推理延迟 (P99) | < 500ms | 99分位延迟 |
| GPU 内存占用 | < 8GB | nvidia-smi |
| 准确率 (Precision) | > 80% | 测试集评估 |
| 召回率 (Recall) | > 85% | 测试集评估 |
| F1-Score | > 82% | 调和平均数 |

---

## 风险与缓解

### 风险 1: GPU 内存不足
**风险级别:** 中
**描述:** Qwen3 8B FP16 版本需要 ~14GB，超过单张 3060 容量

**缓解措施:**
- ✅ 使用 4-bit 量化版本（降至 ~5GB）
- ✅ 配置 `gpu-memory-utilization=0.85`
- ✅ 限制 `max-model-len=4096`

### 风险 2: 推理延迟过高
**风险级别:** 中
**描述:** LLM 推理可能导致端到端延迟超过 500ms

**缓解措施:**
- 使用 vLLM 的 PagedAttention 优化
- 减少 `max_tokens` 到 512
- 实现请求批处理
- 缓存常见查询

### 风险 3: 模型输出不稳定
**风险级别:** 高
**描述:** LLM 可能输出格式不一致或包含幻觉

**缓解措施:**
- 使用低 temperature (0.1)
- 实现输出解析容错
- 添加后处理验证
- 使用 JSON Mode（如果支持）

### 风险 4: 中文姓名检测准确率
**风险级别:** 中
**描述:** 中文姓名多样性高，可能误报或漏报

**缓解措施:**
- 优化中文提示词
- 增加 Few-shot 示例
- 结合上下文词（"先生"、"女士"等）
- 建立中文姓名数据集

---

## 交付物清单

### 代码文件
1. `src/hppe/engines/llm/__init__.py` - LLM 引擎模块
2. `src/hppe/engines/llm/base.py` - 抽象基类
3. `src/hppe/engines/llm/qwen_engine.py` - Qwen 引擎实现
4. `src/hppe/engines/llm/recognizers.py` - LLM 识别器
5. `src/hppe/engines/llm/prompts/` - 提示词模板目录
6. `src/hppe/engines/llm/utils.py` - 工具函数

### 测试文件
1. `tests/unit/test_llm_engine.py` - 引擎单元测试
2. `tests/unit/test_llm_recognizers.py` - 识别器单元测试
3. `tests/integration/test_vllm_integration.py` - vLLM 集成测试
4. `tests/benchmark/test_llm_performance.py` - 性能基准测试

### 配置文件
1. `configs/llm_config.yaml` - LLM 配置
2. `configs/prompts.yaml` - 提示词配置
3. `scripts/start_vllm.sh` - vLLM 启动脚本

### 文档
1. `docs/llm-integration-guide.md` - 集成指南
2. `docs/prompt-engineering-guide.md` - 提示词工程指南
3. `examples/llm_detection_example.py` - 使用示例
4. `EPIC_2_COMPLETION_REPORT.md` - Epic 完成报告

---

## 时间规划

| Week | Story | 主要任务 | 里程碑 |
|------|-------|----------|--------|
| Week 1 | 2.1 | vLLM 部署 | vLLM 服务可用 |
| Week 2 | 2.2, 2.3 | 模型集成 + 零样本检测 | 基础检测功能 |
| Week 3 | 2.4 | 提示词优化 | 达到准确率目标 |

---

## 依赖与前置条件

### 前置条件
- ✅ Epic 1 完成
- ✅ 2 × NVIDIA RTX 3060 GPU
- ✅ CUDA 11.8+ 安装
- ✅ Python 3.10+

### 外部依赖
- vLLM 库
- Qwen3 8B 模型
- OpenAI Python SDK
- GPU 驱动

---

## 验收标准

Epic 2 完成需要满足以下所有条件:

1. ✅ 所有 4 个 Stories (2.1-2.4) 完成
2. ✅ vLLM 服务成功部署并运行
3. ✅ 支持至少 5 种非结构化 PII 检测
4. ✅ 单元测试覆盖率 > 80%
5. ✅ 集成测试全部通过
6. ✅ 准确率达到目标 (Precision > 80%, Recall > 85%)
7. ✅ 性能达到目标 (P50 < 200ms)
8. ✅ 完成文档和示例
9. ✅ 与 Epic 1 无缝集成

---

**文档状态:** ✅ 已完成
**下一步:** 开始 Story 2.1 实现
**更新日期:** 2025-10-14
