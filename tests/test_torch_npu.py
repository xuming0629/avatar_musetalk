import torch
import torch_npu

# 手动设置设备
device = torch.device("npu" if torch.npu.is_available() else "cpu")

# 创建张量
x = torch.randn(2, 3).to(device)
w = torch.randn(3, 4).to(device)

# 计算
y = x @ w
print("结果:", y)

