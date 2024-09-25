import torch

print(torch.__version__)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print("CUDA 可用:", cuda_available)

# 如果 CUDA 可用，输出 CUDA 版本
if cuda_available:
    print("CUDA 版本:", torch.version.cuda)

# 检查 PyTorch 是否编译了 CUDA 支持
print("编译的 CUDA 版本:", torch.version.cuda)

# 检查当前使用的 GPU
if cuda_available:
    print("GPU 设备:", torch.cuda.get_device_name(0))
