import torch
import torch.nn as nn
from torch.nn import functional as F

# 1.单个注意力头(Single Head With Mask)
class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 计算attention
        wei = q @ k.transpose(-1, -2) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        
        return out 
    
# 2 .多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        
        self.heads = nn.ModuleList([Head(config, self.head_size) for _ in range (config.n_head)])

        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
# 3.前馈神经网络(Feed Forward)
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

# 4.Transformer Block
# Attention + FFN + Residual + Norm
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 5.GPT模型本体
class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_heads = nn.Linear(config.n_embd, config.vocab_size)

        # 权重初始化优化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 自动获取 device，不需要从 init 传入
        # 这样模型在 CPU 还是 GPU 上都能自动适应
        device = idx.device
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        if targets is None :
            logits = self.lm_heads(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_heads(x)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        generate 的 Docstring
        
        :param idx: (B, T)数组， 当前的context
        :param max_new_tokens:生成多少个新的etokens 
        :param temperature: >1.0 更疯狂，更开放; <1.0 更保守
        :param top_k: i只保留概率最高的k个选项(截断尾部低概率单词)
        """
        for _ in range(max_new_tokens):
            # 裁剪上下文
            idx_cond = idx[:, -self.config.block_size:]
            # 预测
            logits, _ = self(idx_cond)
            # 只取最后一个时间步
            logits = logits[:, -1, :]
            # *:应用温度
            logits = logits / temperature
            # Top-K采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 小于第k个概率的，概率全部变为-inf
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 概率转换
            probs = F.softmax(logits, dim=-1)
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)
        return idx