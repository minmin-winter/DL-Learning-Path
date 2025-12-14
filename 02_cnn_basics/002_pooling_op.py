import torch
import torch.nn as nn

def test_max_pooling():

    # 1. 手工模拟一个 4x4 的特征图 (Feature Map)
    # 想象这是上一层卷积出来的结果
    feature_map = torch.tensor([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [0, 1, 0, 0],
        [2, 1, 3, 2]
    ], dtype=torch.float32)

    # 升维
    input_tensor = feature_map.unsqueeze(0).unsqueeze(0)
    print(f"原图形状:{input_tensor.shape}")
    print(f"原图内容:\n{feature_map.numpy()}")
    
    # 2.定义最大化池层
    pool = nn.MaxPool2d(kernel_size=2,stride=2)
    
    # 3.开始压缩
    output = pool(input_tensor)
    
    # 4.查看结果
    print(f"\n池化后形状:{output.shape}")
    print(f"池化后内容:\n{output.squeeze().detach().numpy()}")
    
if __name__ == "__main__":
    test_max_pooling()
    