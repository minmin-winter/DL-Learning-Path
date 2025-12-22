import torch
import torch.nn as nn
import math

# 1.MultiHeadAttention
class MultiHeadattention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert not d_model%num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2) / math.sqrt(self.d_k))
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.w_o(context)
        return output
        
# 2.Feed Forward Network前馈神经网络
# 两层全连接
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # d_ff 通常比d_model大很多
        # 目的：把特征投影到高维空间处理，在压回来
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x : [Batch, Seq, d_model]
        # Linear -> ReLU -> Linear
        return self.fc2(self.relu(self.fc1(x)))
            
# 3.核心： Transformer Block(Encoder Layer)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 子层1 : 多头注意力
        self.attention = MultiHeadattention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model) #层归一化
        self.dropout1 = nn.Dropout(dropout)

        # 子层2 : 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x : [Batch, Seq, d_model](已经h加上了 Positional Encoding)

        # 第一步: attention + 残差 + Norm
        # 1. 备份x(做残差)
        residual = x
        
        # 2.跑attention
        attn_output = self.attention(x)

        # 3.Dropout(防止过拟合)
        attn_output = self.dropout1(attn_output)

        # 残差连接 + 归一化(Add & Norm)
        x = self.norm1(residual + attn_output)

        # 第二步 : Feedforward + 残差 + Norm
        # 1.备份一下现在的x
        residual = x

        # 2.跑FeedForward
        ff_output = self.feed_forward(x)

        # 3.Dropout
        ff_output = self.dropout2(ff_output)

        # 4.残差连接 + 归一化
        x = self.norm2(residual + ff_output)

        return x
    
# ====测试=====
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    d_ff = 2048     #习惯是d_model的4倍

    # 模拟输入(Batch=2, Seq=10, Dim=512)
    x = torch.randn(2, 10, 512)

    # 创建一个Block
    block = TransformerBlock(d_model, num_heads, d_ff)

    output = block(x)
    print(f"输入形状 : {x.shape}")
    print(f"输出形状 : {output.shape}")
    