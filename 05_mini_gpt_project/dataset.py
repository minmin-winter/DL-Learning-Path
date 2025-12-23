import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class CharDataset(Dataset):
    def __init__(self, config, data_path):
        """
        初始化数据集: 读取文件， 构建字典， 将文本转为Tensor
        """
        super().__init__()
        self.config = config

        # 1.读取文本文件
        print(f"Loading data from {data_path} ...")
        with open(data_path, encoding='utf-8') as f:
            text = f.read()

        # 2.构建字符级词表(Vocabulary)
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(f"Data loaded. Text length: {len(text)}, Vocab size: {self.vocab_size}")

        # 构建stoi和itos的词表
        self.stoi = {ch : i for i, ch in enumerate(chars)}
        self.itos = {i : ch for i, ch in enumerate(chars)}

        # 3.将整个文本编码为整数Tensor(整数？)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

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

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])

def get_data_loaders(config, data_path):
    """
    创建Dataset并分割出Train/Val DataLoader
    """
    # 1.实例化Dataset
    dataset = CharDataset(config, data_path)

    # 2.划分数据集为训练集与验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 3.创建DataLoader
    # num_workers : 使用多少个子进程来加载数据
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
