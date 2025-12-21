import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 32     
block_size = 8    # GPT中最大的上下文长度
max_iters = 3000  # 训练步数
learning_rate = 1e-3
device = 'cpu'    # 如果有N卡就改为'cuda'
eval_iters = 200  # 评估时跑多少步取平均
n_embd = 32       # 嵌入维度(Channel)
n_head = 4        
n_layer = 3       # Transformer Block的层数
dropout = 0.2     # 防止过拟合的丢弃率

torch.manual_seed(1337)

# 1.单个注意力头(Single Head With Mask)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # 映射K, Q， V
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 定义Mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : [Batch, Time, Channel]
        B,T,C = x.shape
        
        # 1.计算q, k, v
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 2.计算attention
        wei = q @ k.transpose(-1, -2) * (C ** -0.5)

        # 3.Mask
        wei = wei.mask_fill(self.tril[:T, :T] == 0, float('-inf'))

        # 4.Softmax归一化
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 5.加权求和
        v = self.value(x)
        out = wei @ v
        
        return out 
    
# 2 .多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # 创建num_heads个独立的Head
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        # 最后的线性投影层，融合所有num_heads个头
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 1. 每个Head独立计算
        head_outs = [h(x) for h in self.heads]

        # 2. 在最后一个维度投影拼接
        out = torch.cat(head_outs, dim=-1)

        # 3. 投影 + Dropout
        out = self.dropout(out)
        out = self.dropout(out)

        return out
    
# 3.前馈神经网络(Feed Forward)
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 4.Transformer Block
# Attention + FFN + Residual + Norm
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        
        # attention
        self.sa = MultiHeadAttention(n_head, head_size)
        # FFN
        self.ffwd = FeedForward(n_embd)
        # 两个归一化层
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 5.GPT模型本体
class GPTLanguagerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 词嵌入表(Content)
        # vocab_size还未定义， 在main里定义
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 位置嵌入表(Position)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 堆叠Blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        
        # 最后的LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)
        # 最后的输出层
        self.lm_heads = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1.Embedding
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # 2.Transfomer Blocks
        x = self.blocks(x)

        # 3.最后的Norm和映射
        x = self.ln_f(x)
        logits = self.lm_heads(x)

        if targets is None :
            loss = None
        else:
            # 计算损失
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 裁剪上下文
            idx_cond = idx[:, -block_size:]
            # 预测
            logits, loss = self(idx_cond)
            # 只取最后一个时间步
            logits = logits[:, -1, :]
            # 概率转换
            probs = F.softmax(logits, dim=-1)
            # 采样
            ind_next = torch.multinomial(probs, num_samples=1)
            # 拼接
            idx = torch.cat((idx, ind_next), dim=1)
        return idx
    
# 6.数据准备与训练(Main Execution)
if __name__ == "__main__":
    with open("input.txt", encoding='utf-8') as f:
        text = f.read()

    # 构建词典
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars)}
    itos = { i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 转换数据
    data = torch.tensor(encode(text), dtype=torch.long)
    # 划分训练集验证集
    n = int(0.9 * len(data))
    train_data = data[:n]
    validata_data = data[n:]

    #数据加载函数
    def get_batch(split):
        data = train_data if split == "train" else validata_data
        ix = torch.randint(len(data) - block_size, (batch_size, ))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    # 实例化模型
    model = GPTLanguagerModel()
    m = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # training loop
    for iter in range(max_iters):
        xb1, yb = get_batch('train')

        logits, loss = model(xb1, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not iter%100 :
            print(f"Step : {iter}\tLoss : {loss.item():.4f}")

    # 生成测试
    print("\n--- generating Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))