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

print("标签：",' '.join(f"{labels[j].item()}" for j in range(4)))
imshow(torchvision.utils.make_grid(images[:4]))