import torch
from torch.utils.data import Dataset, DataLoader

# 1. 准备数据 (这里我们用一个字符串模拟一个大文本文件)
# 在实际项目中，这里通常是读取一个 .txt 文件
raw_text = "Hello world! This is a test for GPT training at XJTU."

# 2. 构建字典(Tokenizer的雏形) 
# 把字符转数字(Embeddin的前一步)
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

#print(chars)
print(f"词表大小: {vocab_size}")
print(f"字符映射: {char_to_ix}")

# 3.定义核心Dataset类
class CharDateset(Dataset):
    def __init__(self, text, block_size):
        """
        text : 原始文本数据
        block_size : 也就是context length, 一次看多长的序列
        """
        super().__init__()
        self.text = text
        self.block_size = block_size
        self.vocab = char_to_ix

    def __len__(self):
        # 我们可以生成的样本数 = 总长度 - 块长度
        return len(self.text) - self.block_size

    def __getitem__(self, idx):
        # 取出一段长度为 block_size + 1 的文本
        # 为什么要 +1？因为我们要切分成 input 和 target
        chunk = self.text[idx : idx + self.block_size + 1]
        
        # 将字符转为数字索引(Encoding)
        encoded = [self.vocab[c] for c in  chunk]

        # 转换为Tensor
        data = torch.tensor(encoded, dtype=torch.long)
        
        # x 是前 block_size 个字符
        # y 是后 block_size 个字符 (相当于 x 向后移一位)
        x = data[:-1]
        y = data[1:]
        
        return x, y

# 测试代码

BLOCK_SIZE = 8

train_dataset = CharDateset(raw_text, BLOCK_SIZE)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

#模拟训练循环取一次数据看看
print("\n--- 查看一个 Batch 的数据 ---")
for x, y in train_loader:
    print(f"Input shape: {x.shape}") # [Batch_Size, Block_Size]
    print(f"Target shape: {y.shape}")
    
    print("\n举例第一条数据:")
    print("Input (x):", x[0].tolist(), "->", "".join([ix_to_char[i] for i in x[0].tolist()]))
    print("Target (y):", y[0].tolist(), "->", "".join([ix_to_char[i] for i in y[0].tolist()]))
    break