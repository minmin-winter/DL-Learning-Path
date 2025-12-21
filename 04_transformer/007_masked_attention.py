import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数设定
batch_size = 4
block_size = 8 # 序列长度(Time)
n_embd = 32    # embedding维度(Channel)
head_size = 16 

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # 1. Q, K, V的映射矩阵
        # n_embd 输入的维度， head_size 是投射后的维度(因为是单头)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 2. 定义面具(Mask)
        # register_buffer : 此参数不u需要训练，但是模型状态的一部分
        # torch.tril 创建一个下三角矩阵
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # 1.计算k, q, v
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # 2.算attention
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        
        # 3.重点：Mask !
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # 4.Sofatmax 归一化
        wei = F.softmax(wei, dim=-1)

        # 5.加权求和
        out = wei@v
        
        return out, wei
    
# 测试代码

# 1.模拟经过Embedding 后的x
x = torch.randn(batch_size, block_size, n_embd)
print(f"输入形状 : {x.shape}")

# 2.实例化一个Head
head = Head(head_size)

# 3.前向传播
out, wei = head(x)
print(f"输出形状:{out.shape}")

# 4.看看Mask
print("\n--- 查看第一句话的 Attention 矩阵 ---")
print("行看作'当前字'，列看作'关注的字'")
print(wei[0].tolist())