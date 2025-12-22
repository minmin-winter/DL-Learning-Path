import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 1.创建个足够大的矩阵[max_len, d_model]
        pe = torch.zeros(max_len,d_model)

        # 2.生成位置索引[0, 1, 2, 3,... max_len - 1]
        position = torch.arange(0, max_len, dtype=float).unsqueeze(1)

        # 3.计算分母的 div_term(这部分是数学公式 10000^(2i/d_model))
        # 这一步是为了让不同维度的正弦波频率不一样
        div_teerm = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 4.填充 Sin 和 Cos
        # 偶数位用sin
        pe[:, 0::2] = torch.sin(position * div_teerm)
        # 奇数位用cos
        pe[:, 1::2] = torch.cos(position * div_teerm)

        # 5.增加一个batch维度
        self.pe = pe.unsqueeze(0)

        # 注册为buffer
        self.register_buffer('pe', self.pe)

    def forward(self, x):
        # x : [Batch, Seq_Len, D_Model]

        # 核心操作 : 直接把位置编码加到Embedding上
        x += self.pe[:, :x.size(1)]
        return x