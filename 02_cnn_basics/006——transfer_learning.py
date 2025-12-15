import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
import time

def main():
    BATCH_SIZE = 64
    
    # 2.准备数据
    # Grayscale()复制成3通道
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3801,))
    ])
    
    train_dataset = datasets.MNIST("./data",train=True,download=True,transform=transform)
    test_dataset = datasets.MNIST("./data",train=False,transform=transform)

    train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

    # 3.【核心】加载预训练模型
    print(f"正在下载/加载欲训练模型")
    # weights='DEFAULT' 下载官方在ImageNet上训练好的参数
    model = models.resnet18(weights='DEFAULT')

    # 4.【核心】冻结参数(freeze)
    # 将前面预训练好的参数冻结，不被更改
    for param in model.parameters():
        param.requires_grad = False
        
    # 5.【核心】修改最后一层
    # 预训练的fc 输出1000类
    num_fits = model.fc.in_features
    model.fc = nn.Linear(num_fits,10)

    # 6.定义最后一层的规划器
    optimizer = optim.Adam(model.fc.parameters(),lr = 0.001)

    # 7.训练循环
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output,target)
        loss.backward()
        optimizer.step()

        if not batch_idx%100:
            print(f"Batch : {batch_idx}\tLoss : {loss:.4f}")

    # 8.测试
    model.eval()
    correct = 0
    for i in range(2):
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        print(f"Accuracy : {correct / len(test_loader.dataset) : .2%}")

if __name__ == "__main__":
    main()

        