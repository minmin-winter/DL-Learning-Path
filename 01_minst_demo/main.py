#-------------第一步：搞定数据 -------------

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 定义超参数

#类比实验前的参数设置
BATCH_SIZE = 64       #一次让模型看64张图
LEARNING_RATE = 0.01

# 2. 数据预处理

# 输入(原始图片)是PIL Image 格式（0 - 255的像素点）
# 神经网络只接受Tensor（0.0 - 1.0的浮点数）
# *类比Embedding - ToTensor()会把图片变为Tensor,并自动归一化到【0,1】
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3801,))#标准化：减均值，除标准差（让数据分布在0附近，训练更快）
])

# 3.下载并加载数据(Dataset & DataLoader)

print("正在下载数据请稍候...")

#A. 训练集(training dataset)
train_dataset = datasets.MNIST(
    root='./data',             #root : 数据存储的地方
    train=True,                #train : 是否为训练集
    download=True,             #download : 没数据时是否自动下载
    transform=transform        #transform : 处理数据的方法
)

# B. 测试集(testing dataset)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,               #不是训练集（是数据集）
    download=True,
    transform=transform
)

# C. 装载机(DataLoader)
# 类别： Dataset 只是书架上的书 ， DataLoader 是把书打包给模型的搬运工
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True                #*每次训练要打乱数据(洗牌)，防止模型死记硬背   
)

test_loader = DataLoader(
    dataset=test_dataset,       #dataset : 此dataloader加载/装载的数据集 
    batch_size=BATCH_SIZE,        
    shuffle=False               #测试不需要打乱
)


# 4. 检查一下数据长啥样（Sanity Check）
#next(iter()) 是获取迭代器第一个元素的写法
#iter()将可迭代对象转为可以抽取的迭代器
#next()抽取迭代器的下一个
images, labels = next(iter(train_loader))

print("-" * 30)
print(f"这一批图片(Batch)的形状: {images.shape}")
print(f"这一批标签(Labels)的形状: {labels.shape}")
print("-" * 30)
print("数据加载测试通过！✅")


# ----------- 小插曲： 图片可视化 ------------
import matplotlib.pyplot as plt
import numpy as np
import torchvision

#1.获取一批数据
#images, labels 已经在上面写过了

#定义一个简易的函数来展示图片
def imshow(img):
    #反标准化
    img = img * 0.3081 + 0.1307

    #把Tesnsor转化为numpy数组
    npimg = img.numpy()

    # 维度转化 
    plt.imshow(np.transpose(npimg,(1,2,0)),cmap='gray')
    plt.show()

#2.展示前4张图片

#print("标签：",' '.join(f"{labels[j].item()}" for j in range(4)))
#imshow(torchvision.utils.make_grid(images[:4]))


# ------ 第二步：搭建神经网络模型 ------
import torch
import torch.nn as nn
import torch.nn.functional as F
# 1.定义网络结构
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet,self).__init__()
        #第一层 ： 输入图片的像素点（28*28） -> 输出128（隐藏层节点数）
        self.fc1 = nn.Linear(28*28,128)
        #第二层 :  输入隐藏层节点数（128） -> 输出标签数(0-9 10个数)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        # x 的形状一开始是[batch_size,1,28,28]
        # view(-1,  )把图片变成[batch_size,28*28]的数据形式
        x = x.view(-1,28*28)

        # 通过第一层全连接，再通过ReLU非线性函数
        x = F.relu(self.fc1(x))

        # 再通过第二层输入层
        x = self.fc2(x)
        #注意： 不需要加Softmax，因为后面的Loss函数会自带
        return x
        
# 2. 实例化模型
model = SimpleNet()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

#print(f"模型结构:{model}")


#---------- 第三步：训练循环 ---------
def train():

    model.train()

    #跑epochs遍整个数据集
    epochs = 3

    #
    loss_history = []

    for epoch in range(epochs):
        for batch_idx, (data,target) in enumerate(train_loader):
            # A. 梯度清零
            optimizer.zero_grad()

            # B. 前向传播
            output = model(data)

            # C. 算损失
            loss = criterion(output,target)

            # D. 后向传播
            loss.backward()

            # E. 更新参数
            optimizer.step()

            loss_history.append(loss.item())

            # F. 打印一下进度
            if not batch_idx% 100:
                print(f"Epoch : {epoch+1} | Batch : {batch_idx} | Loss : {loss : .4f}")
    
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.show() # 这会弹出一个窗口显示折线图

def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data,target in test_loader:
             output = model(data)

             test_loss += criterion(output,target).item()

             pred = output.argmax(dim=1,keepdim=True)

             correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        print(f"\n测试集结果: Loss:{test_loss:.4f} 准确率：{correct} /{len(test_loader.dataset)} (accuracy : {accuracy :.2%})\n")


if __name__ == "__main__":
    train()
    test()