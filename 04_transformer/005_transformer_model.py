import torch
import torch.nn as nn
import torch.optim as optim
import math

# 1.之前的零件: multi-head attention, feed forward, positional encoding, transformer block

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert not d_model % num_heads
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        atten_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(atten_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.w_o(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # add & norm 1
        x = self.norm1(x + self.dropout1(self.attention(x)))
        # add & norm 2
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x

# 2.终极组装: Transformer 文本分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, output_dim, max_len=100):
        super().__init__()

        # 1.Embedding层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2.Positional Encoding 
        self.pos_encoder = PositionalEncoding(d_model, max_len) # 3.堆叠Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # 4.最后的分类头
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x: [Batch, Seq_Len]
        # 词向量 + 位置编码
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # 通过N层
        for layer in self.layers:
            x = layer(x)

        # Pooling(把10个词的特征合成1个)
        # 平均值池化(Mean Pooling)
        x = x.mean(dim=1) # [Batch, Seq, Dim] -> [Batch, Dim]

        # 分类
        return self.fc(x)

# 3.极简训练测试
if __name__ == "__main__":
    # 参数配置:
    VOCAB_SIZE = 100
    D_MODEL = 64
    NUM_HEADS = 4
    D_FF = 256
    NUM_LAYERS = 3 # 堆叠3层
    OUTPUT_DIM = 2   # 输出的结果：分的两类
    EPOCHS = 100

    # 1.实例化模型
    model = TransformerClassifier(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,OUTPUT_DIM)

    # 2.准备一点假数据
    # Batch = 2, Seq_len = 10
    inputs = torch.randint(0, VOCAB_SIZE, (2, 10))
    labels = torch.tensor([1, 0])

    # 3.定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Trasformer开始运行...")

    # 4.跑一步
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print(f"输入尺寸: {inputs.shape}")
        #print(f"输出尺寸: {outputs.shape}") # 应该是 [2, 2]
        if not (epoch+1)%10:
            print(f"Epoch : {epoch + 1}\tLoss: {loss.item():.4f}")