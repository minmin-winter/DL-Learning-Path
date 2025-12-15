import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# 1. 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 特征提取部分(卷积 + 激活 + 池化)
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # 第二层卷积块
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )   

        # 分类通道(全连接)
        self.classifier = nn.Linear(32 * 7 * 7, 10)
        
        
    def forward(self,x):
        # x: [B, 1, 28, 28]
        x = self.features(x)
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
        return out
    
# 2.训练函数
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if not batch_idx%300 :
            print(f"Epoch : {epoch}\tLoss : {loss.item()}")
            
# 3.测试函数
def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output,target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    print(f"Average Loss : {test_loss:.4f}\tAccuracy : {accuracy : .2%}")
    return accuracy

# 4.主程序
def main():
    
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    transform =transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3801,))
    ])
    
    # 下载/加载数据
    train_dataset = datasets.MNIST("./data",train=True,download=True,transform=transform)
    test_dataset = datasets.MNIST("./data",train=False,transform=transform)
    
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=1000,shuffle=False)
    
    # 初始化模型
    model = SimpleCNN()

    # 定义优化器
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    # 开始跑
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train(model,train_loader,optimizer,epoch)
        test(model,test_loader)
        
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds")
    
    # 保存模型
    torch.save(model.state_dict(),"cnn_model.pth")
    
if __name__ == "__main__":
    main()