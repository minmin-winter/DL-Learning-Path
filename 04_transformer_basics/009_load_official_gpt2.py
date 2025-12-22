import os
# 这一行设置环境变量，告诉 transformers 库去国内镜像站下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. 加载GPT-2的分词器
print("正在加载 Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 2. 造一个text
text = "Hello, world! This is a simple test."

# 3,看看BPE的切分
encoded_input = tokenizer(text, return_tensors='pt')
input_ids = encoded_input["input_ids"]

print("-" * 30)
print(f"原始文本: {text}")
print(f"Token IDs: {input_ids}")

# 再把ID变回文字
tokens = [tokenizer.decode([i]) for i in input_ids[0]]
print(f"切分细节: {tokens}")
print(f"词表总大小: {tokenizer.vocab_size}") 
print("-" * 30)

# 4.加载GPT-2模型(预训练好的权重)
print("正在加载模型权重...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# 5.打印出模型里的所有层的名字和形状
print("\nGPT-2 官方权重结构：")
for name, param in model.named_parameters():
    # 我们只打印最外层的形状，不刷屏
    if name.endswith(".weight") and ("wte" in name or "wpe" in name or "c_attn" in name or "ln_1" in name):
        print(f"{name: <40} | shape: {param.shape}")

print("-" * 30)
print("任务完成！这就是你要复现的标准答案。")