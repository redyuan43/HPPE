# HPPE 系统架构文档

**版本：** 1.0
**更新日期：** 2025年10月14日
**架构师：** Winston (BMad Architect)

---

## 1. 系统概述

### 1.1 项目背景
高精度隐私引擎（HPPE）是一个企业级的 PII 检测与脱敏系统，旨在自动识别和处理非结构化文本中的个人身份信息。

### 1.2 核心目标
- **高精度检测**：F2-score > 0.90
- **多语言支持**：重点支持中文和英文
- **实时处理**：支持流式和批量处理
- **合规性**：满足 GDPR、CCPA 和中国个人信息保护法

### 1.3 架构原则
- **模块化设计**：组件松耦合，高内聚
- **可扩展性**：易于添加新的 PII 类型和语言
- **性能优先**：优化延迟和吞吐量
- **可审计性**：完整的决策追踪

---

## 2. 系统架构

### 2.1 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                 │
├─────────────────────────────────────────────────────────┤
│                  HPPE Core Pipeline                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Stage 1: 并行候选实体生成                      │  │
│  │  ┌────────────┐        ┌────────────────────┐   │  │
│  │  │  Regex     │        │   LLM Context      │   │  │
│  │  │  Engine    │────────│   Engine (Qwen3)   │   │  │
│  │  └────────────┘        └────────────────────┘   │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Stage 2: LLM 驱动的歧义消除                    │  │
│  │         (Disambiguation & Merging)                │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Stage 3: 上下文验证与误报削减                  │  │
│  │         (Context Validation)                      │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Stage 4: 脱敏与屏蔽模块                        │  │
│  │         (Anonymization Module)                    │  │
│  └──────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│              Infrastructure Services                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │  Redis   │  │  Celery  │  │  vLLM Inference    │  │
│  │  Cache   │  │  Queue   │  │     Server         │  │
│  └──────────┘  └──────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流

1. **输入**：原始文本通过 API Gateway 进入
2. **阶段 1**：并行执行正则匹配和 LLM 检测
3. **阶段 2**：解决标签冲突和范围重叠
4. **阶段 3**：验证高风险实体，减少误报
5. **阶段 4**：应用脱敏策略
6. **输出**：返回脱敏后的文本和检测报告

---

## 3. 技术栈

### 3.1 核心技术选型

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **语言** | Python | 3.11+ | 主要开发语言 |
| **Web框架** | FastAPI | 0.100+ | REST API 服务 |
| **LLM** | Qwen3 8B | Latest | 中文 PII 检测 |
| **推理服务器** | vLLM | 0.5+ | LLM 推理优化 |
| **正则引擎** | Custom (Presidio-style) | 1.0 | 结构化 PII 检测 |
| **缓存** | Redis | 7.0+ | 结果缓存 |
| **任务队列** | Celery | 5.3+ | 异步处理 |
| **中文处理** | jieba | 0.42+ | 中文分词 |
| **监控** | Prometheus + Grafana | Latest | 性能监控 |

### 3.2 硬件要求

**最小配置：**
- CPU: 8 核心
- 内存: 32GB RAM
- GPU: NVIDIA RTX 4090 (24GB VRAM) 或 A10G

**推荐配置：**
- CPU: 16 核心
- 内存: 64GB RAM
- GPU: NVIDIA A100 (40GB VRAM)

---

## 4. 核心组件设计

### 4.1 正则引擎 (Deterministic Engine)

```python
# 组件结构
class PatternRecognizer:
    patterns: List[RegexPattern]
    context_words: List[str]
    deny_lists: List[str]
    confidence_score: float

    def validate_result(self, match) -> bool:
        """自定义验证逻辑（如 Luhn 算法）"""
        pass
```

**支持的中文 PII 类型：**
- 身份证号 (CHINA_ID_CARD)
- 手机号码 (CHINA_PHONE)
- 银行卡号 (CHINA_BANK_CARD)
- 护照号码 (CHINA_PASSPORT)
- 车牌号 (CHINA_LICENSE_PLATE)

### 4.2 LLM 上下文引擎

```python
class QwenContextEngine:
    model: "Qwen3-8B-Instruct"
    quantization: "4bit"  # NF4 量化
    max_length: 8192

    def detect_unstructured_pii(self, text: str) -> List[Entity]:
        """零样本 PII 检测"""
        pass

    def disambiguate(self, text: str, candidates: List[str]) -> str:
        """歧义消除"""
        pass

    def validate(self, context: str, entity: Entity) -> bool:
        """上下文验证"""
        pass
```

### 4.3 脱敏模块

```python
class AnonymizationModule:
    strategies = {
        "redaction": RedactionStrategy,      # 完全删除
        "masking": MaskingStrategy,          # 部分遮蔽
        "hashing": HashingStrategy,          # 加密哈希
        "synthetic": SyntheticStrategy       # 合成替换
    }

    def apply(self, text: str, entities: List[Entity],
              config: Dict) -> str:
        """应用配置的脱敏策略"""
        pass
```

---

## 5. API 设计

### 5.1 RESTful API 端点

```yaml
POST /api/v1/detect
  描述: 检测文本中的 PII
  请求体:
    text: string
    language: string (zh|en|auto)
    options:
      include_context: boolean
      confidence_threshold: float

POST /api/v1/anonymize
  描述: 检测并脱敏 PII
  请求体:
    text: string
    language: string
    strategy: object
      default: string (redaction|masking|synthetic)
      overrides: object

POST /api/v1/batch
  描述: 批量处理文档
  请求体:
    documents: array
    async: boolean

GET /api/v1/status/{task_id}
  描述: 查询异步任务状态
```

### 5.2 响应格式

```json
{
  "status": "success",
  "original_text": "...",
  "anonymized_text": "...",
  "entities": [
    {
      "type": "CHINA_ID_CARD",
      "value": "***",
      "start": 10,
      "end": 28,
      "confidence": 0.98,
      "detection_method": "regex|llm",
      "validation_status": "verified"
    }
  ],
  "metrics": {
    "processing_time_ms": 250,
    "entities_detected": 5,
    "entities_anonymized": 5
  }
}
```

---

## 6. 性能指标

### 6.1 目标性能

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| **精确率** | > 85% | TP/(TP+FP) |
| **召回率** | > 92% | TP/(TP+FN) |
| **F2-Score** | > 0.90 | 加权调和平均 |
| **延迟 (P50)** | < 500ms | 单文档处理 |
| **延迟 (P99)** | < 2000ms | 单文档处理 |
| **吞吐量** | > 100 RPS | 并发处理 |

### 6.2 优化策略

1. **缓存优化**：Redis 缓存频繁请求
2. **批处理**：vLLM 批量推理
3. **异步处理**：Celery 任务队列
4. **GPU 优化**：4-bit 量化，PagedAttention

---

## 7. 部署架构

### 7.1 容器化部署

```yaml
services:
  api:
    image: hppe-api:latest
    replicas: 3
    resources:
      cpu: 2
      memory: 4Gi

  llm-inference:
    image: hppe-vllm:latest
    replicas: 2
    resources:
      gpu: 1
      memory: 32Gi

  redis:
    image: redis:7-alpine

  celery-worker:
    image: hppe-worker:latest
    replicas: 4
```

### 7.2 扩展策略

- **水平扩展**：API 服务和 Worker 节点
- **GPU 集群**：多 GPU 服务器负载均衡
- **缓存分片**：Redis Cluster

---

## 8. 安全与合规

### 8.1 安全措施

- **数据加密**：传输中和静态数据加密
- **访问控制**：基于角色的 API 访问
- **审计日志**：完整的操作追踪
- **数据隔离**：多租户数据隔离

### 8.2 合规性

- **GDPR**：符合欧盟数据保护要求
- **CCPA**：满足加州隐私法
- **个保法**：符合中国个人信息保护法
- **ISO 27001**：信息安全管理体系

---

## 9. 监控与维护

### 9.1 监控指标

- **系统指标**：CPU、内存、GPU 使用率
- **业务指标**：请求量、错误率、检测准确率
- **性能指标**：延迟分布、吞吐量

### 9.2 维护计划

- **模型更新**：季度评估新模型
- **规则更新**：月度更新正则规则库
- **性能调优**：持续性能优化
- **安全补丁**：及时更新依赖

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LLM 推理延迟高 | 用户体验差 | 量化优化、缓存机制 |
| 误报率高 | 数据可用性降低 | 多阶段验证流程 |
| 漏报率高 | 合规风险 | 混合检测策略 |
| GPU 资源不足 | 系统瓶颈 | 弹性扩容、队列管理 |

---

## 附录 A：PII 类型定义

详见 [PII 分类表](./pii-taxonomy.md)

## 附录 B：技术决策记录

详见 [ADR 文档](./adr/)

---

**文档状态：** 初稿完成
**下一步：** 创建详细的技术栈文档和 API 规范