#!/usr/bin/env python3
"""手动下载Qwen3-4B模型"""

from huggingface_hub import snapshot_download
import os

print("开始下载 Qwen3-4B 模型...")
print("预计大小: ~8GB")
print("预计时间: 10-15分钟\n")

try:
    model_path = snapshot_download(
        repo_id="Qwen/Qwen3-4B",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        resume_download=True,
        local_files_only=False,
    )
    print(f"\n✓ 模型下载完成！")
    print(f"路径: {model_path}")

    # 验证关键文件
    import glob
    safetensors = glob.glob(f"{model_path}/*.safetensors") + glob.glob(f"{model_path}/**/*.safetensors", recursive=True)
    config_json = glob.glob(f"{model_path}/config.json")

    print(f"\n文件验证:")
    print(f"  权重文件: {len(safetensors)} 个")
    print(f"  配置文件: {'✓' if config_json else '✗'}")

    if safetensors and config_json:
        print("\n🎉 模型文件完整，可以开始训练！")
    else:
        print("\n⚠️  模型文件不完整，请重新下载")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
