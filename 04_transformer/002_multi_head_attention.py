import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # d_model : 输入的维度(比如512)
        # num_heads : 要把脑袋切成几分(比如8)

        # 检查维度是否能被整除
        assert not d_model % num_heads, "d_model必须能被二num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 1.定义线性层
        # 哪怕是多头，我们也是先用一个大矩阵算，算完再切，这样效率最高。
        # W_q, W_k, W_v: 负责把输入 "投影" 成 Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 把最后8个头的结果拼起来，在融合一次
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x.shape : [Batch, Seq_Len, D_Model]
        batch_size, seq_len, _ = x.shape

        # 2. 关键： 线性变换 + 切分头(Split Heads)
        # 变换 ： [Batch, Seq, 8, 64] -> [Batch, 8, Seq, 64](把“头放在前面)
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

        # 3. 计算注意力分数
        # Q @ K.T /sqrt(d_k)
        # Q: [B, 8, Seq, 64]
        # K.transpose(-2, -1): [B, 8, 64, Seq] (把最后两维转置)
        # scores: [B, 8, Seq, Seq] (8 张注意力热力图！)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 归一化(变成概率)
        att_weights = F.softmax(scores, dim=1)

        # 4.加权求和
        # context ： [B, 8, Seq, 64] (每个头算出来的新特征)
        context = torch.matmul(att_weights, V)

        # 5.拼接
        # 把 [B, 8, Seq, 64] 变回 [B, Seq, 512]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 6.最后的线性融合
        output = self.w_o(context)

        return output, att_weights

# 测试：
if __name__ == "__main__":
    inpur_tensor = torch.randn(1, 10, 512)

    # 实例化： 8个头
    mha = MultiHeadAttention(d_model=512, num_heads=8)

    output, weights = mha(inpur_tensor)

    print(f"输入形状: {inpur_tensor.shape}")
    print(f"输出形状: {output.shape}") 
    print(f"权重形状: {weights.shape}")
