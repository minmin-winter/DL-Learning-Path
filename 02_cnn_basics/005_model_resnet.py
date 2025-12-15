import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import time

# 1. 定义残差块
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channnels,out_channels,stride=1):
        super().__init__()

        # 主干道
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channnels,out_channels,kernel_size=3,padding=1,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels * self.expansion,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        
        # 捷径
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channnels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channnels,out_channels * self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
            
    def forward(self,x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out += identity
        return nn.ReLU(inplace=True)(out)

# 2.Resnet模型：整体架构
class Resnet(nn.Module):
    def __init__(self,block,block_counts,num_classes=10):
        super().__init__()

        self.in_channels = 64
        
        # 1.初始输入层: 将1通道图片转为 64 通道，为后续残差块做准备
        self.conv1 = nn.Sequential(
            # [B, 1, 28, 28] -> [B, 64, 14, 14]
            nn.Conv2d(in_channels=1,out_channels=64,stride=2,padding=3,kernel_size=7,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 2.[B, 64, 14, 14] -> [B, 64 , 7, 7]
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)   
        )

        # 3.四大残差块定义
        # ResNet-18 定义：Stage1(2个块), Stage2(2个块), Stage3(2个块), Stage4(2个块)
        self.layer1 = self._make_layer(block, 64, block_counts[0], stride=1)  # 64 进 64 出，尺寸不变
        self.layer2 = self._make_layer(block, 128, block_counts[1], stride=2) # 64 进 128 出，尺寸减半
        self.layer3 = self._make_layer(block, 256, block_counts[2], stride=2) # 128 进 256 出，尺寸减半
        self.layer4 = self._make_layer(block, 512, block_counts[3], stride=2) # 256 进 512 出，尺寸减半

        # 4.平均池化和全连接分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 自适应池化到 1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    # 搭建一个阶段的残差块，例如 Stage 2
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 确定第一个块是否需要降采样（stride=2）或通道升维
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
    
# 实例化 Resnet18:
def ResNet18():
    return Resnet(ResidualBlock,[2,2,2,2])
    
# 3.训练与测试

def train(model, train_loader,optimzier, epoch):
    model.train()
    for batch_ind, (data, target) in enumerate(train_loader):
        optimzier.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output,target)
        loss.backward()
        optimzier.step()
        if not batch_ind%100 :
            print(f"Epoch : {epoch}\tloss : {loss:.4f}")

def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output,target).item()
            pred = output.argmax(dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"Average Loss : {test_loss:.4f}\tAccuracy : {accuracy:.2%}")
    return accuracy

# 4.主程序
def main():
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # 数据加载
    transfrom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3801,))
    ])
    train_dataset = datasets.MNIST("./data",train=True,download=True,transform=transfrom)
    test_dataset = datasets.MNIST("./data",train=False,transform=transfrom)
    train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

    # 初始化模型
    model = ResNet18()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    # 开始跑了
    start_time = time.time()
    for epoch in range(1,EPOCHS+1):
        train(model,train_loader,optimizer,epoch)
        test(model,test_loader)
        
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()