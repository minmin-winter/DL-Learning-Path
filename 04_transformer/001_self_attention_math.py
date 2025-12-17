import torch
import torch.nn.functional as F
import math

# 1.准备数据
# 假设我们要处理一句话: "I love AI" (3个词)
# 假设每个词已经变成了 Embedding 向量 (维度=4)
sentences = torch.tensor([
    [1.0, 0.0, 1.0, 0.0], # I
    [0.0, 1.0, 0.0, 1.0], # love
    [1.0, 1.0, 1.0, 1.0]  # AI
])
# 形状: [3, 4] (Seq_Len=3, Embed_Dim=4)
print(f"\n1.原始输入:\n{sentences}")

# 2. 核心概念: Q, K, V
# Transformer 把每个词拆成了三个分身：
# Query(Q) : 我在找什么？ (查询)
# Key(K) : 我有什么特征？ (标签)
# Value(V) : 我的实质内容是什么？(内容)

# 这三个是通过Linear层算出来的
# 简化：假设就是输入本身
Q = sentences
K = sentences
V = sentences

# --- 3.计算Attention! ----
# 数学公式 : Q @ K的转置
# 向量越相似，点积越大

scores = torch.matmul(Q, K.transpose(-2, -1))

print(f"\n原始分数(scores) : {scores}")

# 4.缩放(Scaling)
d_k = 4
scores /= math.sqrt(d_k)

# 5.归一化(Softmax)
# 变为概率，即为Attention Weights(注意力权重)
attention_weights = F.softmax(scores, dim=1)

print(f"\n2.注意力权重:\n{attention_weights}")
# 这张表表明： 多少注意力在"love", 多少在"AI”

# 6.加权求和(Weight Sum)
output = torch.matmul(attention_weights, V)

print(f"\n3. 最终输出(Context Vectors): \n{output}")
print(f"输出形状:{output.shape}")