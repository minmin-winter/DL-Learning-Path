import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def test_convolution():
    # 1. 模拟一张图片 (1个通道, 5x5 的大小)
    # 想象这是一个黑色的背景，中间有一条亮白色的竖线
    # 0=黑, 10=白
    img = torch.tensor([
        [0, 0, 10, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 10, 0, 0]
    ], dtype=torch.float32)
    
    # 变换维数
    # Pytroch要求图像4维格式:[batch,channel,height,width]
    # unsqueeze(0)加维
    input_tensor = img.unsqueeze(0).unsqueeze(0)
    print(f"输入形状:{input_tensor.shape}")
    
    # 2. 定义一个卷积层
    conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,bias=False)
    
    # 3.手工设置将将卷积核的权重(Weight)
    # 我们搞一个“竖线检测器” (Sobel 算子的一种变体)
    # 左边是负数，右边是正数。如果它扫到竖线，左右差异会很大，结果就会很大。
    sobel_kernel = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32)
    
    # 把这个权重加维，以给卷积层
    conv.weight.data = sobel_kernel.unsqueeze(0).unsqueeze(0)
    
    # 4. 手电筒扫描图片(前向传播)
    output = conv(input_tensor)
    
    print(f"输出形状: {output.shape}") 
    # 思考：5x5 的图，用 3x3 的核扫，不填充，结果会是几乘几？
    
    print("\n卷积后的结果 (特征图):\n", output.squeeze().detach().numpy())

    
    # ------可视化---------
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Feature Map (Result)")
    plt.imshow(output.squeeze().detach().numpy(), cmap='gray')
    
    plt.show()

if __name__ == "__main__":
    test_convolution()