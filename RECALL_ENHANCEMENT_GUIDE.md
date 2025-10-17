# PII检测Recall提升方案指南

## 📊 当前状态

**Epoch 2单模型性能** (50样本):
- Precision: 97.78%
- Recall: 81.48%
- F1-Score: 88.89%

**目标:**
- Recall ≥ 90%
- 差距: **8.52%**

---

## 🚀 提升策略路线图

### 方案1: 模型集成 (双GPU运行中) ⏳

**原理:** Epoch1 ∪ Epoch2 = 扩大召回范围

**预期效果:**
- Recall: 81% → **88-91%**
- Precision: 97% → 94-96% (略降)
- F1: 88% → **91-93%**

**状态:** 
- ✅ 正在运行 (GPU0: Model1, GPU1: Model2)
- ⏰ 预计完成: ~23:07 (还需~30分钟)

---

### 方案2: 集成 + 规则增强 ⭐ **推荐**

**原理:** (Epoch1 ∪ Epoch2) ∪ Rules

**规则覆盖:**
1. 正则表达式:
   - 手机号: `1[3-9]\d{9}`
   - 身份证: `\d{17}[\dXx]`
   - 邮箱: `\w+@\w+\.\w+`
   - 银行卡: `\d{13,19}`
   - 车牌号: 中国车牌格式
   - IP地址: IPv4格式

2. 启发式规则:
   - 人名: 常见姓氏 + 1-3汉字
   - 地址: 包含"省市县区路号"等关键词
   - 机构: 包含"公司/银行/医院/学校"等

**预期效果:**
- Recall: 88-91% → **92-95%**
- 可能达到甚至超过90%目标！

**准备情况:**
- ✅ 规则检测器已创建: `scripts/rule_enhanced_detector.py`
- ✅ 验证脚本已准备: `scripts/validate_ensemble_plus_rules.py`
- ⏳ 等待集成验证完成后立即执行

---

### 方案3: 数据增强 + 重训练 (备选)

**触发条件:** 如果方案2仍未达90%

**步骤:**
1. 分析漏检类型 (已完成):
   - ORGANIZATION: Recall 82% (最弱)
   - PERSON_NAME: Recall 84.5%
   - ADDRESS: Recall 86.9%

2. 过采样策略:
   - ORGANIZATION数据 × 3倍
   - 添加困难样本 (复合机构名、简短机构名)

3. 使用Focal Loss重新训练2 epoch

**预期效果:**
- Recall: → **90-93%**
- 时间成本: **36-48小时**

---

## 📁 关键文件

### 已完成的脚本
- `scripts/ensemble_validation_dual_gpu.py` - 双GPU集成验证 (运行中)
- `scripts/rule_enhanced_detector.py` - 规则检测器
- `scripts/validate_ensemble_plus_rules.py` - 集成+规则验证
- `scripts/analyze_missed_cases.py` - 漏检分析

### 验证结果
- `logs/validation_checkpoint-1562_50samples.json` - Epoch 2 (50样本)
- `logs/missed_cases_analysis.json` - 漏检分析 (100样本)
- `logs/ensemble_dual_gpu_result.json` - 集成结果 (待生成)
- `logs/ensemble_plus_rules_result.json` - 规则增强结果 (待生成)

### 模型文件
- `models/pii_qwen4b_unsloth/checkpoint-781/` - Epoch 1
- `models/pii_qwen4b_unsloth/checkpoint-1562/` - Epoch 2 (最佳)

---

## 🎯 执行计划

### 当前阶段 (Phase 1): 集成验证
- [x] 启动双GPU集成验证
- [ ] 等待完成 (~30分钟)
- [ ] 分析集成效果

### 下一步 (Phase 2): 规则增强
**如果集成Recall ≥ 88%:**
```bash
cd /home/ivan/HPPE
python scripts/validate_ensemble_plus_rules.py
```
- 预计额外提升: **+3-5% Recall**
- 耗时: 约50分钟

**如果达到90%:** ✅ 完成，可以部署！

### 备选方案 (Phase 3): 数据增强
**如果仍未达标:**
1. 准备增强数据集 (2-4小时)
2. 重新训练 (36小时)
3. 目标Recall: 90-93%

---

## 💡 关键洞察

### 为什么集成有效？
- Epoch 1 (step 781): 学习率较高，更"激进"
- Epoch 2 (step 1562): 学习率接近0，更"保守"
- 两者盲区不同 → 并集提升Recall

### 为什么规则有用?
- EMAIL/PHONE/ID_CARD: 模型已100% Recall ✅
- 但其他类型可能遗漏边缘case
- 规则补充模型盲区 → 进一步提升

### 权衡考虑
| 方案 | Recall提升 | 时间 | Precision影响 |
|------|-----------|------|---------------|
| 集成 | +6-10% | 立即 | -1~3% |
| +规则 | +3-5% | +50分钟 | -2~4% |
| 重训练 | +8-12% | +36小时 | 可能提升 |

---

## 📞 下一步行动

**立即:**
- ⏳ 等待双GPU集成验证完成

**完成后:**
1. 查看 `logs/ensemble_dual_gpu_result.json`
2. 如果Recall ≥ 88% → 执行规则增强
3. 如果Recall < 88% → 考虑重训练

**最终目标:**
- Recall ≥ 90% ✅
- Precision ≥ 85% (当前97%，有足够buffer)
- F1 ≥ 87.5% (当前88.89%，已达标)

---

*更新时间: 2025-10-16 22:27*
*作者: Claude (AI助手)*
