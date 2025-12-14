#----------- 小插曲： 图片可视化 ------------
import matplotlib.pyplot as plt
import numpy as np
import torchvision

#定义一个简易的函数来展示图片
def imshow(img,labels):
    #反标准化
    img = img * 0.3081 + 0.1307

    #把Tesnsor转化为numpy数组
    npimg = img.numpy()

    # 维度转化 
    plt.imshow(np.transpose(npimg,(1,2,0)),cmap='gray')
    plt.show()

    #展示前4张图片
    print("标签：",' '.join(f"{labels[j].item()}" for j in range(4)))