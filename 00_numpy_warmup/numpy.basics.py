import numpy as np

# 1. 基础：创建一个 2x3 的矩阵
# 目标：[[1, 2, 3], [4, 5, 6]]
data = np.array([[1,2,3],[4,5,6]]) 
print("形状:", data.shape)

# 2. 维度变换 (最重要！回忆 view/reshape)
# 目标：把它变成 3x2 的
reshaped_data = np.reshape(data,(3,2))
print(f"维度变换后：\n{reshaped_data}")

# 3. 广播机制 (Broadcasting) - 那个让很多新手晕菜的概念
# 创建一个 [3, 1] 的向量和 [3] 的向量相加
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
# 思考：结果是 [3, 3] 吗？
c = a + b 
print("广播结果:\n", c)

# 4. 矩阵乘法 (点积)
# 模拟一下全连接层：输入 [1, 784] 乘 权重 [784, 128]
# 这里用随机数模拟
input_vec = np.random.randn(1, 784)
weight = np.random.randn(784, 128)
output = input_vec @ weight
print("输出层形状:", output.shape)