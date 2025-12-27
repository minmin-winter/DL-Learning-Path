import os
# 设置 HF 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ==========================================
# 1. 加载“军火”：Tokenizer 和 Model
# ==========================================
print("正在连接 Hugging Face Hub 下载/加载 GPT-2...")

# from_pretrained 是 Hugging Face 最核心的魔法
# 它会自动去官网找叫 "gpt2" 的模型，下载配置文件、词表和权重
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("模型加载完毕！准备起飞！🚀")

# ==========================================
# 2. 准备输入
# ==========================================
# 我们可以试试给它一个更难的 Prompt，看它懂不懂常识
text = "The scientist discovered a new planet that"
print(f"\n[用户输入]: {text}")

# 编码 (这就相当于你写的 dataset.encode + unsqueeze)
# return_tensors='pt' 表示直接返回 PyTorch 的 Tensor
inputs = tokenizer(text, return_tensors="pt")

# ==========================================
# 3. 生成 (Inference)
# ==========================================
# 设为评估模式 (不计算梯度，省内存)
model.eval()

print("正在生成中...")
with torch.no_grad():
    # 这里的参数是不是很眼熟？
    # do_sample=True: 开启随机采样 (Temperature才有效)
    # temperature=0.7: 稍微有点创造力，但不要太疯
    # top_k=50: 只看前50个概率最高的词
    # max_length=100: 生成的总长度
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=100, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50,
        pad_token_id=tokenizer.eos_token_id # 避免警告
    )

# ==========================================
# 4. 解码
# ==========================================
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("-" * 30)
print("[GPT-2 生成结果]:")
print(generated_text)
print("-" * 30)