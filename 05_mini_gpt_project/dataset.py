import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class BPEDataset(Dataset):
    def __init__(self, config, data_path):
        """
        初始化数据集: 读取文件， 构建字典， 将文本转为Tensor
        """
        super().__init__()
        self.config = config

        # 1. 加载GPT-2分词器
        print(f"Loading GPT-2 tokenzier...")
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

        # 2.读取文本文件
        print(f"Loading data from {data_path} ...")
        with open(data_path, encoding='utf-8') as f:
            text = f.read()

        # 3.编码(把几十万个字符变成几万个Token)
        print(f"Tokenizing data...")
        tokens = self.enc.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)

        print(f"Data loaded. Tokens length: {len(self.data)}")

    def __len__(self):
        # 能够切分出的样本数量
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # 这一步是DataLoader核心调用的部分
        # 给定一个索引idx, 返回(x, y)

        chunk = self.data[idx : idx + self.config.block_size + 1]
        
        x = chunk[:-1]
        y = chunk[1:]

        return x, y

    # 辅助函数：直接调用tiktoken
    def encode(self, s):
        return self.enc.encode(s)

    def decode(self, l):
        return self.enc.decode(l)

def get_data_loaders(config, data_path):
    """
    创建Dataset并分割出Train/Val DataLoader
    """
    dataset = BPEDataset(config, data_path)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, dataset
