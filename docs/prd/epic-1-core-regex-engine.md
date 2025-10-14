# Epic 1: 核心确定性引擎（正则表达式引擎）

**Epic ID:** epic-1
**状态:** 待开发
**优先级:** P0 (必须有)
**负责人:** TBD
**预计工作量:** 4-6 周

---

## Epic 概述

### Epic 目标
构建一个高性能、可扩展的正则表达式引擎框架，用于检测文本中的结构化 PII（个人身份信息）。该引擎是 HPPE 系统的基础组件，提供快速、准确的模式匹配能力。

### Epic 价值
- **快速检测**：为结构化 PII 提供低延迟（<50ms）的检测能力
- **高精确率**：对于格式明确的 PII，精确率 > 95%
- **可扩展性**：易于添加新的 PII 类型和模式
- **基础平台**：为后续 LLM 引擎提供候选实体

### 成功标准
- ✅ 支持至少 10 种中文和英文结构化 PII 类型
- ✅ 结构化 PII 检测精确率 > 95%
- ✅ 单次检测延迟 < 50ms
- ✅ 单元测试覆盖率 > 80%
- ✅ 完整的识别器文档和示例

---

## Epic 范围

### 包含内容
✅ 正则表达式识别器基础框架
✅ 中文 PII 识别器（身份证、手机、银行卡等）
✅ 全球/英文 PII 识别器（邮箱、IP、URL 等）
✅ 识别器注册表和管理系统
✅ 上下文词和拒绝列表支持
✅ 校验和验证（如 Luhn 算法）
✅ 完整的单元测试

### 不包含内容
❌ LLM 集成
❌ 非结构化 PII 检测（姓名、地址等）
❌ 歧义消除逻辑
❌ 脱敏功能
❌ API 接口

---

## 架构设计

### 组件架构

```
RegexEngine
├── RecognizerRegistry
│   ├── load_recognizers()
│   ├── register()
│   └── detect()
│
├── BaseRecognizer (抽象基类)
│   ├── patterns: List[Pattern]
│   ├── context_words: List[str]
│   ├── deny_lists: List[str]
│   ├── detect() -> List[Entity]
│   └── validate() -> bool
│
├── ChinesePIIRecognizers
│   ├── ChinaIDCardRecognizer
│   ├── ChinaPhoneRecognizer
│   ├── ChinaBankCardRecognizer
│   └── ChinaPassportRecognizer
│
└── GlobalPIIRecognizers
    ├── EmailAddressRecognizer
    ├── IPAddressRecognizer
    ├── URLRecognizer
    ├── CreditCardRecognizer
    └── USSNRecognizer
```

### 关键设计决策

**决策 1: Presidio-Inspired 模式**
- **理由:** Microsoft Presidio 提供了经过验证的识别器模式
- **实现:** 采用类似的 recognizer + registry 架构
- **参考:** [docs/architecture/tech-stack.md#正则引擎框架]

**决策 2: YAML 配置驱动**
- **理由:** 允许无代码添加新模式
- **实现:** 每个识别器从 YAML 文件加载配置
- **位置:** `data/patterns/china_pii.yaml`, `data/patterns/global_pii.yaml`

**决策 3: 上下文词提升**
- **理由:** 提高歧义模式的置信度
- **实现:** 在匹配附近检测上下文词，提升分数
- **示例:** "身份证" 出现在 18 位数字附近

---

## 用户故事

### Story 1.1: 正则引擎框架搭建
**优先级:** P0
**工作量:** 3 天

**用户故事:**
作为开发者，
我需要一个可扩展的正则表达式引擎框架，
以便能够快速添加新的 PII 识别器。

**验收标准:**
1. 实现 `BaseRecognizer` 抽象基类
2. 实现 `RecognizerRegistry` 注册表系统
3. 支持从 YAML 配置文件加载识别器
4. 实现基本的 `Entity` 数据模型
5. 单元测试覆盖率 > 80%

**技术要求:**
- 使用 Python 3.11+ 类型注解
- 遵循 [docs/architecture/coding-standards.md]
- 文件位置：`src/hppe/engines/regex/`

**依赖:**
- 无

---

### Story 1.2: 中文 PII 识别器实现
**优先级:** P0
**工作量:** 5 天

**用户故事:**
作为系统用户，
我需要系统能够检测中文文本中的常见 PII，
以便对中文数据进行隐私保护。

**验收标准:**
1. 实现中国身份证识别器（含校验码验证）
2. 实现中国手机号识别器（支持 +86 前缀）
3. 实现中国银行卡识别器（含 Luhn 校验）
4. 实现中国护照号识别器
5. 每个识别器精确率 > 95%
6. 包含上下文词和拒绝列表
7. 完整的测试用例（正面和负面）

**技术要求:**
- 正则表达式避免灾难性回溯
- 支持 jieba 分词集成（为上下文词检测）
- 配置文件：`data/patterns/china_pii.yaml`
- 文件位置：`src/hppe/engines/regex/recognizers/china_pii.py`

**依赖:**
- Story 1.1 完成

**测试数据:**
```yaml
# 正面测试用例
- "我的身份证号是110101199003077578"
- "手机号码：13812345678"
- "银行卡号：6222021234567890123"

# 负面测试用例（应该不匹配）
- "订单号：110101199003077578"  # 虽然格式像身份证
- "验证码：1381"  # 不完整的手机号
```

---

### Story 1.3: 全球 PII 识别器实现
**优先级:** P0
**工作量:** 4 天

**用户故事:**
作为系统用户，
我需要系统能够检测英文和全球通用的 PII，
以便处理多语言混合文本。

**验收标准:**
1. 实现电子邮件地址识别器
2. 实现 IP 地址识别器（IPv4 和 IPv6）
3. 实现 URL 识别器
4. 实现信用卡号识别器（含 Luhn 校验）
5. 实现美国社会安全号识别器
6. 每个识别器精确率 > 95%
7. 完整的测试用例

**技术要求:**
- 支持国际化格式变体
- 配置文件：`data/patterns/global_pii.yaml`
- 文件位置：`src/hppe/engines/regex/recognizers/global_pii.py`

**依赖:**
- Story 1.1 完成

**测试数据:**
```yaml
# 正面测试用例
- "Email: john.doe@example.com"
- "IP: 192.168.1.1"
- "Card: 4532-1488-0343-6467"
- "SSN: 123-45-6789"

# 负面测试用例
- "Version: 1.2.3.4"  # 不是 IP
- "Code: 4532"  # 不完整的卡号
```

---

### Story 1.4: 识别器注册表和管理
**优先级:** P0
**工作量:** 2 天

**用户故事:**
作为开发者，
我需要一个统一的接口来管理和使用所有识别器，
以便简化系统集成和测试。

**验收标准:**
1. 实现自动识别器发现和注册
2. 支持按 PII 类型过滤识别器
3. 提供批量检测接口
4. 实现识别器性能监控（执行时间）
5. 提供识别器列表和元数据查询
6. 完整的集成测试

**技术要求:**
- 线程安全的注册表实现
- 支持识别器热重载（可选）
- 文件位置：`src/hppe/engines/regex/registry.py`

**依赖:**
- Story 1.1, 1.2, 1.3 完成

**API 示例:**
```python
from hppe.engines.regex import RecognizerRegistry

# 初始化注册表
registry = RecognizerRegistry()
registry.load_all()

# 检测 PII
text = "我的身份证是110101199003077578，邮箱是john@example.com"
entities = registry.detect(text)

# 结果
# [
#   Entity(type="CHINA_ID_CARD", value="110101199003077578", ...),
#   Entity(type="EMAIL_ADDRESS", value="john@example.com", ...)
# ]
```

---

## 技术规范

### 数据模型

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Entity:
    """PII 实体"""
    entity_type: str        # PII 类型（如 "CHINA_ID_CARD"）
    value: str              # 实体值
    start_pos: int          # 起始位置
    end_pos: int            # 结束位置
    confidence: float       # 置信度 [0, 1]
    detection_method: str   # 检测方法 "regex"
    recognizer_name: str    # 识别器名称
    metadata: Optional[dict] = None  # 额外元数据
```

### 配置文件格式

```yaml
# data/patterns/china_pii.yaml

recognizers:
  - name: ChinaIDCardRecognizer
    entity_type: CHINA_ID_CARD
    patterns:
      - pattern: '[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]'
        score: 0.9
    context_words:
      - '身份证'
      - '身份证号'
      - 'ID'
      - '证件号'
    deny_lists:
      - '订单号'
      - '流水号'
    validation:
      type: 'checksum'
      algorithm: 'china_id_card'
```

### 性能要求

| 指标 | 目标值 |
|------|--------|
| 单次检测延迟 | < 50ms |
| 10KB 文本处理 | < 200ms |
| 内存占用 | < 100MB |
| CPU 使用率 | < 30% (单核) |

---

## 测试策略

### 单元测试
- 每个识别器独立测试
- 正面和负面用例覆盖
- 边界条件测试
- 性能基准测试

### 集成测试
- 注册表功能测试
- 批量检测测试
- 并发安全性测试

### 测试数据集
1. **合成测试数据**：使用 Faker 生成
2. **真实数据样本**：脱敏后的真实文本
3. **边界用例**：特殊格式和罕见情况

---

## 风险与缓解

### 风险 1: 正则表达式性能
**风险级别:** 中
**描述:** 复杂的正则表达式可能导致灾难性回溯
**缓解措施:**
- 严格遵循正则表达式最佳实践
- 使用性能分析工具验证模式
- 设置超时机制

### 风险 2: 误报率过高
**风险级别:** 中
**描述:** 过于宽泛的模式可能匹配非 PII 数据
**缓解措施:**
- 使用上下文词提升
- 实现校验和验证
- 维护拒绝列表

### 风险 3: 中文分词准确性
**风险级别:** 低
**描述:** jieba 分词可能影响上下文词检测
**缓解措施:**
- 使用自定义词典
- 实现基于字符距离的备用方案

---

## 交付物

1. ✅ 完整的正则引擎框架代码
2. ✅ 10+ 种 PII 识别器实现
3. ✅ YAML 配置文件
4. ✅ 单元测试（覆盖率 > 80%）
5. ✅ 集成测试
6. ✅ 性能基准测试报告
7. ✅ API 文档和使用示例
8. ✅ 识别器开发指南

---

## 依赖项

### 技术依赖
- Python 3.11+
- pydantic (数据验证)
- PyYAML (配置解析)
- jieba (中文分词)
- pytest (测试框架)

### 文档依赖
- [docs/architecture/tech-stack.md]
- [docs/architecture/source-tree.md]
- [docs/architecture/coding-standards.md]

### 团队依赖
- 无阻塞依赖
- 可独立开发

---

## 时间线

| 里程碑 | 完成日期 | 负责人 | 状态 |
|--------|----------|--------|------|
| Story 1.1 完成 | Week 1 | TBD | ⏳ 待开始 |
| Story 1.2 完成 | Week 2 | TBD | ⏳ 待开始 |
| Story 1.3 完成 | Week 3 | TBD | ⏳ 待开始 |
| Story 1.4 完成 | Week 4 | TBD | ⏳ 待开始 |
| Epic 1 完成 | Week 4 | TBD | ⏳ 待开始 |

---

## 参考资料

1. **Microsoft Presidio 文档**
   https://microsoft.github.io/presidio/

2. **正则表达式最佳实践**
   [docs/architecture/coding-standards.md#性能优化]

3. **中国 PII 标准**
   - 身份证号码规则
   - 手机号段分配

4. **Luhn 算法实现**
   信用卡和银行卡校验码验证

---

**Epic 状态:** 📋 待开发
**下一步:** 创建 Story 1.1 详细任务