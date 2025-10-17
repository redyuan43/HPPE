# GPU资源冲突严重问题报告

**生成时间**: 2025-10-15 11:23

## 🔥 严重性能异常

### GPU0训练进程速度断崖式下降

| 步数 | 累计用时 | 速度 | 状态 |
|------|---------|------|------|
| 1-167 | ~30分钟 | **10-11s/it** | ✅ 正常 |
| 168 | 35分钟 | **84.55s/it** | ⚠️ 慢8倍 |
| 169 | 39分钟 | **126.77s/it** | ❌ **慢12倍！** |

**影响**：
- 原计划5.5小时完成 → 现需要**60+小时**
- GPU资源严重浪费
- 无法按时完成1 epoch验证

## 🔍 根本原因分析

### CUDA设备隔离失败

```bash
# GPU0进程 (PID 3778)
CUDA_VISIBLE_DEVICES=未设置  # ❌ 可见所有GPU (0,1)

# GPU1进程 (PID 14382)
CUDA_VISIBLE_DEVICES=1       # ✅ 只可见GPU1
```

### nvidia-smi证据

```
|  GPU   Process                         GPU Memory |
|    0   3778 (GPU0训练)                  11959MiB  |
|    0   14382 (GPU1训练)                 11959MiB  |  ← GPU1进程也在GPU0上!
|    1   3778 (GPU0训练)                  11986MiB  |  ← GPU0进程也在GPU1上!
|    1   14382 (GPU1训练)                 11986MiB  |
```

**GPU利用率**：
- GPU0: **1%** ❌ (严重不足，因为在等待GPU1资源)
- GPU1: **99%** ✅ (正常使用)

### 技术细节

1. **PID 3778启动命令未指定GPU**：
   ```bash
   python scripts/train_pii_detector.py \
     --model ... \
     --output models/pii_detector_qwen3_06b_epoch4-5 \
     # 缺失: CUDA_VISIBLE_DEVICES=0
   ```

2. **device_map="auto"行为**：
   - 当没有CUDA限制时，transformers会尝试使用所有可见GPU
   - 即使模型能放进单GPU，也可能将部分层分布到其他GPU

3. **资源竞争点**：
   - PID 3778占用GPU1的部分显存/算力
   - PID 14382在GPU1上全力训练
   - 两者争抢GPU1资源 → 互相阻塞

## ✅ 解决方案

### 立即行动

**步骤1**: 终止问题进程
```bash
kill 3778  # 终止GPU0保守配置训练
```

**步骤2**: 使用正确的GPU隔离重启
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_pii_detector.py \
  --model "/home/ivan/.cache/modelscope/hub/Qwen/Qwen3-0___6B" \
  --data "data/merged_pii_dataset_train.jsonl" \
  --val-data "data/merged_pii_dataset_validation.jsonl" \
  --output "models/pii_detector_qwen3_06b_epoch4-5" \
  --lora-r 8 \
  --lora-alpha 16 \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --learning-rate 0.0001 \
  --epochs 2 \
  2>&1 | tee "logs/continue_training_20251015_fixed.log" &
```

### 验证修复

```bash
# 查看进程GPU绑定
cat /proc/$(pgrep -f pii_detector_qwen3_06b_epoch4-5)/environ | tr '\0' '\n' | grep CUDA
# 预期输出: CUDA_VISIBLE_DEVICES=0

# 查看GPU利用率
watch -n 5 nvidia-smi
# 预期: GPU0 99%, GPU1 99%
```

## 📊 预期改进

修复后:
- GPU0速度: **126s/it → 11s/it** (恢复正常)
- GPU0利用率: **1% → 99%**
- 完成时间: **60小时 → 5.5小时**

## 📝 经验教训

**SOLID原则应用 - 单一职责 (Single Responsibility)**:
- ❌ 一个进程不应试图管理多个GPU的资源分配
- ✅ 每个训练进程应明确绑定到单一GPU

**最佳实践**:
1. **启动多GPU训练时必须明确设置CUDA_VISIBLE_DEVICES**
2. **使用nvidia-smi验证进程GPU绑定**
3. **监控GPU利用率，低于80%即为异常**

---

**状态**: 🔴 等待用户确认修复操作
**优先级**: P0 - 阻塞训练进度
