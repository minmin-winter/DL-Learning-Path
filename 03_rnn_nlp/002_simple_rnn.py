import torch
import torch.nn as nn

# 1. ----定义一个简易的RNN模型------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()

        # [Layer 1]: Embedding层(把ID变为向量)
        self.embedding = nn.Embedding(vocab_size,embed_dim)

        # [Layer 2]:RNN层（核心）
        self.rnn = nn.RNN(input_size=embed_dim,hidden_size=hidden_dim,batch_first=True)

        # [Layer 3]:全连接层(分类器)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,text_ids):
        # text_ids 形状：[Batch, Seq_Len]

        # 1.查字典，变向量
        # out形状: [Batch, Seq_len, Embed_len]
        out = self.embedding(text_ids)
        
        # 2.过RNN， 读句子
        output, hidden = self.rnn(out)

        # hidden形状: [1, Batch, Hidden_Dim]
        final_memory = hidden.squeeze(0)

        return self.fc(final_memory)

if __name__ == "__main__":
    
    VOCAB_SIZE = 10   # 词表里只有 10 个词
    EMBED_DIM = 5     # 每个词用 5 维向量表示
    HIDDEN_DIM = 8    # RNN 的记忆容量是 8
    OUTPUT_DIM = 2    # 二分类 (比如：积极/消极)
    
    model = SimpleRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    print(model)

    input_ids = torch.tensor([[2,3,4,5]],dtype=torch.long)

    prediction = model(input_ids)

    print(input_ids.shape)
    print(prediction.shape)
    print(prediction)