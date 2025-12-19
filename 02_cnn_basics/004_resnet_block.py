import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 1.主干道:正常的卷积 - BN - ReLU - 卷积 - BN
        self.main_path = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 第二层
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 2. 捷径 (Shortcut / Skip Connection)
        # 如果输入和输出的形状不一样(比如通道变了，或者长宽变了)，捷径也得做个变换才能相加
        self.shorctcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shorctcut = nn.Sequential(
                # 用1*1的卷积核来调整厚度，纯粹凑形状
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self,x):
        # 1.主干道
        out = self.main_path(x)
        
        # 2.走捷径，得原始数据
        identity = self.shorctcut(x)
        
        # 3.核心！主干+原始
        out += identity
        
        # 4.最后激活
        out = nn.ReLU(inplace=True)(out)

        return out