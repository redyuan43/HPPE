# vLLM 环境安装指南

## 📊 当前状态

✅ **安装正在后台进行中**

- **环境名称**: `hppe`
- **Python 版本**: 3.10
- **进程 ID**: 261116, 261117
- **日志文件**: `/home/ivan/HPPE/vllm_install.log`
- **预计时间**: 5-15 分钟

---

## 🔍 监控安装进度

### 方法 1: 使用监控脚本（推荐）
```bash
cd /home/ivan/HPPE
./scripts/check_install_progress.sh
```

### 方法 2: 实时查看日志
```bash
tail -f /home/ivan/HPPE/vllm_install.log
```

### 方法 3: 检查进程
```bash
ps aux | grep setup_vllm_env.sh
```

---

## 📦 安装内容

安装脚本将自动完成以下步骤：

1. ✅ **创建 conda 环境** (hppe, Python 3.10)
2. 🔄 **安装 PyTorch** (780MB, CUDA 12.1)
3. ⏳ **安装 vLLM** (约 500MB)
4. ⏳ **安装依赖** (openai, requests, pyyaml)
5. ⏳ **验证安装**

---

## ✅ 安装完成后

### 1. 激活环境
```bash
conda activate hppe
```

或使用快速激活脚本：
```bash
./activate_hppe.sh
```

### 2. 验证安装
```bash
# 检查 vLLM 版本
python -c "import vllm; print(vllm.__version__)"

# 检查 CUDA 可用性
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 检查 GPU
nvidia-smi
```

### 3. 启动 vLLM 服务
```bash
conda activate hppe
./scripts/start_vllm.sh
```

### 4. 测试服务
```bash
# 在另一个终端
conda activate hppe
PYTHONPATH=/home/ivan/HPPE/src python examples/llm_engine_example.py
```

---

## 🛠️ 故障排查

### 问题 1: 安装时间过长

**正常情况**: 下载和安装可能需要 10-15 分钟，取决于网络速度

**检查方法**:
```bash
# 查看下载进度
tail -f /home/ivan/HPPE/vllm_install.log | grep -E "(Downloading|Installing)"
```

### 问题 2: 安装失败

**检查日志**:
```bash
cat /home/ivan/HPPE/vllm_install.log | grep -i error
```

**常见原因**:
- 网络问题（PyTorch 下载很大）
- 磁盘空间不足（需要约 5GB）
- CUDA 版本不兼容

**解决方法**:
```bash
# 重新运行安装
./scripts/setup_vllm_env.sh
```

### 问题 3: conda 环境问题

**检查环境**:
```bash
conda env list
```

**手动创建环境**:
```bash
conda create -n hppe python=3.10 -y
conda activate hppe
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm
```

---

## 📈 预期结果

安装完成后，你将拥有：

✅ **完整的 vLLM 环境**
- conda 环境: `hppe`
- Python: 3.10
- PyTorch: 2.5.1 (CUDA 12.1)
- vLLM: 最新稳定版
- CUDA 支持: 已启用

✅ **可用的 GPU**
- GPU 0: RTX 3060 (12GB) - 主显示
- GPU 1: RTX 3060 (12GB) - 可用于 vLLM

✅ **工具和脚本**
- 启动脚本: `./scripts/start_vllm.sh`
- 监控脚本: `./scripts/check_install_progress.sh`
- 激活脚本: `./activate_hppe.sh`

---

## 🎯 下一步

安装完成后：

1. **激活环境并测试**
   ```bash
   conda activate hppe
   python -c "import vllm; print('vLLM OK')"
   ```

2. **启动 vLLM 服务**
   ```bash
   ./scripts/start_vllm.sh
   ```

3. **运行示例**
   ```bash
   PYTHONPATH=/home/ivan/HPPE/src python examples/llm_engine_example.py
   ```

4. **继续 Story 2.2**: Qwen3 模型集成

---

## 📞 监控命令速查

```bash
# 快速检查进度
./scripts/check_install_progress.sh

# 实时日志
tail -f vllm_install.log

# 检查进程
ps aux | grep setup_vllm_env

# 查看环境
conda env list

# 激活环境
conda activate hppe

# 检查 GPU
nvidia-smi
```

---

**创建时间**: 2025-10-14
**状态**: 🔄 安装中
**预计完成**: 约 10-15 分钟
