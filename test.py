import torch

# 1. 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 2. 检查 CUDA (显卡驱动) 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {cuda_available}")

if cuda_available:
    # 3. 获取显卡名称
    device_name = torch.cuda.get_device_name(0)
    print(f"检测到显卡: {device_name}")
    
    # 4. 做一个简单的数学测试：在 GPU 上做矩阵乘法
    # 创建两个 1000x1000 的随机矩阵，直接放到 GPU 上
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    
    # 执行矩阵乘法 C = A * B
    c = torch.matmul(a, b)
    
    print("GPU 矩阵乘法测试成功！")
else:
    print("警告：未检测到可用 GPU，目前只能用 CPU 运行。")