# Epic 3: 多阶段精炼流水线

**状态：** 🚀 进行中
**优先级：** P0
**开始日期：** 2025-10-18
**负责人：** AI Team
**Phase：** Phase 3

---

## 📋 目标概述

实现多阶段精炼流水线，通过歧义消除、实体合并、上下文验证和误报削减，**显著提升PII检测的准确性**。

**关键目标**：
- 解决多识别器输出冲突
- 合并重叠和相邻实体
- 利用上下文验证实体
- 削减假阳性（False Positives）

**成功指标**：
- **Precision提升**: +5% (从67% → 72%+)
- **F1-Score提升**: +3-5% (从67% → 70-72%)
- **误报率下降**: -20% (FPR从33% → 26%以下)

---

## 🎯 Story 列表

### Story 3.1: 歧义消除模块

**优先级**: P0
**工作量**: 3-5天
**依赖**: Epic 2 完成

**需求描述**:

实现智能歧义消除逻辑，当多个识别器对同一文本片段给出不同类型判断时，自动选择最可能正确的类型。

**问题场景**:
```
文本: "110101199001011234"
- RegexRecognizer检测为: ID_CARD (conf: 0.95)
- LLMRecognizer检测为: BANK_CARD (conf: 0.85)
- LLMRecognizer检测为: DRIVER_LICENSE (conf: 0.80)

❓ 应该选择哪个类型？
```

**解决方案**:

1. **基于置信度的优先级策略**
   ```python
   def resolve_conflict(entities: List[Entity]) -> Entity:
       # 1. 按置信度排序
       # 2. 考虑识别器权重（Regex > LLM for structured PII）
       # 3. 考虑PII类型特异性（ID_CARD更具体）
       return best_entity
   ```

2. **规则优先级表**
   ```
   对于数字串：
   - ID_CARD > BANK_CARD > PHONE_NUMBER
   - Regex识别器权重 = 1.2
   - LLM识别器权重 = 1.0
   ```

**验收标准**:
- AC1: 100%的冲突案例能被正确解决
- AC2: 解决逻辑可配置（规则可调）
- AC3: 包含完整的单元测试

---

### Story 3.2: 实体合并逻辑

**优先级**: P0
**工作量**: 2-3天
**依赖**: Story 3.1

**需求描述**:

实现智能实体合并，处理重叠和相邻实体，保留最完整和最准确的实体。

**问题场景**:
```
文本: "联系张三先生，电话13812345678"
检测结果:
- Entity1: "张三" (PERSON_NAME, start=2, end=4)
- Entity2: "张三先生" (PERSON_NAME, start=2, end=6)
- Entity3: "13812345678" (PHONE_NUMBER, start=11, end=22)

❓ 应该合并为哪个？
```

**解决方案**:

1. **重叠实体合并**
   ```python
   def merge_overlapping(entities: List[Entity]) -> List[Entity]:
       # 选择span更长的实体（"张三先生" vs "张三"）
       # 如果类型不同，保留置信度更高的
       return merged_entities
   ```

2. **相邻实体合并**（可选）
   ```python
   # 示例: "北京" + "市" + "海淀区" → "北京市海淀区"
   # 仅对ADDRESS类型启用
   ```

**验收标准**:
- AC1: 重叠实体正确合并（保留最长span）
- AC2: 相邻ADDRESS实体能正确合并
- AC3: 不同类型重叠时保留置信度高的

---

### Story 3.3: 上下文验证实现

**优先级**: P1
**工作量**: 4-6天
**依赖**: Story 3.2

**需求描述**:

利用前后文信息验证实体的有效性，过滤掉不符合上下文的误检。

**问题场景**:
```
文本: "价格：100000 元"
检测结果: POSTAL_CODE "100000" (conf: 0.75)

❓ 这真的是邮编吗？
```

**解决方案**:

1. **上下文关键词验证**
   ```python
   CONTEXT_PATTERNS = {
       "POSTAL_CODE": {
           "positive": ["邮编", "邮政编码", "zip"],
           "negative": ["价格", "金额", "数量", "元", "¥"]
       },
       "PHONE_NUMBER": {
           "positive": ["电话", "手机", "联系方式", "tel"],
           "negative": ["编号", "序号", "ID"]
       }
   }

   def validate_context(entity, text, window=20):
       context = text[max(0, entity.start-window):entity.end+window]
       score = calculate_context_score(context, entity.type)
       return score > threshold
   ```

2. **LLM辅助验证**（可选，高成本场景）
   ```python
   prompt = f"文本：{context}\n实体：{entity.value}\n类型：{entity.type}\n是否正确？"
   # 仅对低置信度（0.6-0.8）实体启用
   ```

**验收标准**:
- AC1: 上下文关键词库覆盖17种PII类型
- AC2: 误报削减≥15%
- AC3: 正确实体保留率≥98%

---

### Story 3.4: 误报削减优化

**优先级**: P1
**工作量**: 3-4天
**依赖**: Story 3.3

**需求描述**:

实现基于规则和统计的误报过滤，削减常见的假阳性模式。

**常见误报模式**:

| 模式 | 错误识别为 | 应过滤原因 |
|------|-----------|-----------|
| `123456` | PHONE_NUMBER | 太短且为连续数字 |
| `000000` | POSTAL_CODE | 全零模式 |
| `test` | PERSON_NAME | 常见测试词 |
| `example.com` | ORGANIZATION | 示例域名 |
| `admin` | PERSON_NAME | 系统关键词 |

**解决方案**:

1. **黑名单过滤**
   ```python
   BLACKLIST = {
       "PERSON_NAME": ["admin", "user", "test", "guest", ...],
       "EMAIL": ["noreply@", "no-reply@", "example.com"],
       "PHONE_NUMBER": ["123456", "000000", "111111"],
   }
   ```

2. **统计特征过滤**
   ```python
   def is_false_positive(entity):
       # PHONE_NUMBER: 不应全为相同数字
       if entity.type == "PHONE_NUMBER":
           if len(set(entity.value.replace('-', ''))) <= 2:
               return True

       # POSTAL_CODE: 不应全为0或9
       if entity.type == "POSTAL_CODE":
           if entity.value in ["000000", "999999"]:
               return True

       return False
   ```

3. **格式验证增强**
   ```python
   # 银行卡：必须通过Luhn算法
   # 身份证：必须通过校验位验证
   # 邮箱：必须符合RFC 5322
   ```

**验收标准**:
- AC1: 黑名单覆盖常见误报模式
- AC2: Precision提升≥5%
- AC3: Recall下降≤1%

---

## 🏗️ 架构设计

### 精炼流水线架构

```
Input: List[Entity] from RegexRegistry + LLMRegistry

↓
┌─────────────────────────────────────┐
│ Stage 1: 歧义消除 (Story 3.1)       │
│  - 解决类型冲突                      │
│  - 应用优先级规则                    │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Stage 2: 实体合并 (Story 3.2)       │
│  - 合并重叠实体                      │
│  - 合并相邻实体（可选）              │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Stage 3: 上下文验证 (Story 3.3)     │
│  - 检查上下文关键词                  │
│  - 计算上下文置信度                  │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Stage 4: 误报过滤 (Story 3.4)       │
│  - 黑名单过滤                        │
│  - 统计特征检查                      │
│  - 格式验证                          │
└─────────────────────────────────────┘
↓
Output: List[Entity] (精炼后)
```

### 核心类设计

```python
# src/hppe/refiner/pipeline.py
class RefinementPipeline:
    """精炼流水线主类"""

    def __init__(self, config: RefinementConfig):
        self.disambiguator = Disambiguator(config.disambiguator_config)
        self.merger = EntityMerger(config.merger_config)
        self.validator = ContextValidator(config.validator_config)
        self.filter = FalsePositiveFilter(config.filter_config)

    def refine(self, entities: List[Entity], text: str) -> List[Entity]:
        """执行完整的精炼流水线"""
        entities = self.disambiguator.resolve(entities)
        entities = self.merger.merge(entities)
        entities = self.validator.validate(entities, text)
        entities = self.filter.filter(entities)
        return entities


# src/hppe/refiner/disambiguator.py
class Disambiguator:
    """歧义消除器 (Story 3.1)"""

    def resolve(self, entities: List[Entity]) -> List[Entity]:
        """解决实体冲突"""
        pass


# src/hppe/refiner/merger.py
class EntityMerger:
    """实体合并器 (Story 3.2)"""

    def merge(self, entities: List[Entity]) -> List[Entity]:
        """合并重叠和相邻实体"""
        pass


# src/hppe/refiner/validator.py
class ContextValidator:
    """上下文验证器 (Story 3.3)"""

    def validate(self, entities: List[Entity], text: str) -> List[Entity]:
        """基于上下文验证实体"""
        pass


# src/hppe/refiner/filter.py
class FalsePositiveFilter:
    """误报过滤器 (Story 3.4)"""

    def filter(self, entities: List[Entity]) -> List[Entity]:
        """过滤假阳性"""
        pass
```

---

## 📊 测试策略

### 单元测试
- 每个模块独立测试（Disambiguator, Merger, Validator, Filter）
- Mock数据覆盖边界情况
- 测试覆盖率≥90%

### 集成测试
- 端到端流水线测试
- 使用真实数据（从17PII测试集）
- 验证Precision/Recall指标

### 性能测试
- 精炼流水线延迟≤100ms (P50)
- 吞吐量影响≤10%

---

## 📈 成功指标

| 指标 | 当前值 (Epic 2) | 目标值 (Epic 3) | 提升 |
|------|----------------|----------------|------|
| Precision | 67% | 72%+ | +5% |
| Recall | 68% | 68-70% | 持平或+2% |
| F1-Score | 67% | 70-72% | +3-5% |
| FPR (误报率) | 33% | 26%以下 | -7% |

---

## 🗓️ 时间规划

| Story | 工作量 | 开始日期 | 预计完成 |
|-------|--------|---------|---------|
| 3.1 歧义消除 | 3-5天 | 2025-10-18 | 2025-10-23 |
| 3.2 实体合并 | 2-3天 | 2025-10-23 | 2025-10-26 |
| 3.3 上下文验证 | 4-6天 | 2025-10-26 | 2025-11-01 |
| 3.4 误报削减 | 3-4天 | 2025-11-01 | 2025-11-05 |

**总计**: 12-18天（2.5-3.5周）

---

## 🚧 风险与依赖

### 风险
1. **上下文验证复杂度高**：可能需要额外的LLM调用（成本↑）
2. **合并逻辑边界情况多**：需要大量测试用例验证
3. **误报规则维护成本**：黑名单需要持续更新

### 依赖
- Epic 2 (LLM Engine) 必须完成
- 17种PII训练数据质量影响上下文验证效果

### 缓解措施
- 上下文验证优先使用规则，LLM作为可选增强
- 建立完善的测试数据集和回归测试
- 自动化黑名单更新流程（基于误报反馈）

---

## 📚 参考资料

- [Epic 2完成报告](../EPIC_2_COMPLETION_REPORT.md)
- [PII类型定义](../../src/hppe/models/pii_types.py)
- [PRD文档](../prd.md)

---

**文档版本**: 1.0
**创建日期**: 2025-10-18
**最后更新**: 2025-10-18
