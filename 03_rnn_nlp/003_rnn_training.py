import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1. 准备伪数据
# (句子，标签)
raw_data = [
    ("i love this movie", 1),
    ("this film is great", 1),
    ("awesome acting and plot", 1),
    ("i really like it", 1),
    ("best movie ever", 1),
    ("i hate this movie", 0),
    ("terrible plot and acting", 0),
    ("this is garbage", 0),
    ("boring and waste of time", 0),
    ("i do not like it", 0)
]

# 2. 构建词表
word_list = set()
for text, label in raw_data:
    for word in text.split():
        word_list.add(word)

word2idx = {'<PAD>' : 0,'<UNK>' : 1}
for i,word in enumerate(word_list):
    word2idx[word] = i + 2
    
print(f"词表大小:{len(word2idx)}")

# 3.自定义dataset
class ReviewDataset(Dataset):
    def __init__(self, data, word_dict):
        super().__init__()
        self.word_dict = word_dict
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        text, label = self.data[idx]
        # 编码：文本 ->数字列表(id)
        ids = [self.word_dict.get(w,self.word_dict['<UNK>']) for w in text.split()]
        return torch.tensor(ids), torch.tensor(label, dtype=torch.long)

# 4. Collate Function(打包员)        
# 把DataLoader中取出来的数据打包
def collate_fn(batch):
    
    # 1. 把文本和标签分开
    texts,labels = zip(*batch)

    # 2.核心: pad_sequence
    # 自动补齐长短不一的tensor
    padded_texts = pad_sequence(texts,batch_first=True,padding_value=0)

    # 3.标签转成tensor
    labels = torch.stack(labels)
    return padded_texts, labels

# 5.定义模型
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        # padding_idx=0: 告诉模型，不学习padding，不学习0的特征
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        # output : [Batch, Seq, Hidden]
        # hiddem : [1, Batch, Hidden]
        _, hidden = self.rnn(embedded)
        
        return self.fc(hidden.squeeze(0))
# 6.训练循环
def main():
    EMBED_DIM = 8
    HIDDEN_DIM = 16
    BATCH_SIZE = 2
    LR = 0.01
    EPOCHS = 20
    
    # 准备DataLoader
    train_ds = ReviewDataset(raw_data, word2idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)

    model = SentimentRNN(len(word2idx), EMBED_DIM, HIDDEN_DIM, 2)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    critertion = nn.CrossEntropyLoss()

    print("开始训练")
    for epoch in range(EPOCHS):
        total_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            output = model(texts)
            loss = critertion(output,labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        if not(epoch + 1)%5:
            print(f"Epoch : {epoch + 1}\tLoss : {total_loss / len(train_loader.dataset):.4f}")

    # 7.测试模型
    test_sentences = ["i love it", "garbage movie", "terrible acting"]
    model.eval()
    
    for sent in test_sentences:
        # 手动处理单句话
        ids = [word2idx.get(w, word2idx["<UNK>"]) for w in sent.split()]    
        tensor_in = torch.tensor([ids])

        with torch.no_grad():
            prediction = model(tensor_in)

            pred_label = prediction.argmax(dim=1).item()

        result = "好评" if pred_label else "差评"
        print(f"句子 : {sent}\t标签 : {result}")

if __name__ == "__main__":
    main()