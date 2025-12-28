import os
# 镜像设置（习惯性加上，防止连不上）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from transformers import GPT2Tokenizer

# 1. 加载一个真实的数据集
# 这是一个非常经典的中文情感分析数据集 (好评/差评)
print("正在加载数据集...")
dataset = load_dataset("XiangPan/waimai_10k", split="train")

# 2. 看看数据长啥样
print("\n数据集加载成功！样本示例：")
print(dataset[0]) 
# 输出应该是类似：{'label': 1, 'review': '很快，好吃，味道足，量大'}

# 3. 数据预处理 (Tokenization)
print("\n正在进行批量分词...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT-2 没有默认的 pad_token，我们需要手动指定一下
tokenizer.pad_token = tokenizer.eos_token 

def process_function(examples):
    # 这里演示了如何一次性处理一堆数据
    # padding='max_length': 保证每句话长度一样，不够的补 0
    # truncation=True: 太长的切掉
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)

# map 是 datasets 库最强大的函数，它会多线程并行处理所有数据
tokenized_dataset = dataset.map(process_function, batched=True)

print("\n处理完成！看看处理后的数据结构：")
print(tokenized_dataset[0])
# 你会看到 'input_ids' 变成了一堆数字，这就是喂给模型的饲料