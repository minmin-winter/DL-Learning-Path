import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. 自动找到最新的模型存档 (Checkpoint)
output_dir = "./models"

# 检查文件夹是否存在
if not os.path.exists(output_dir):
    raise FileNotFoundError(f"找不到目录: {output_dir}，请确认你已经运行过训练脚本！")

# 找到里面叫 'checkpoint-xxx' 的文件夹
checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
if not checkpoints:
    raise FileNotFoundError("在 waimai_model 里没找到 checkpoint 文件夹！")

# 排序，取数字最大的那个（也就是训练最久的）
latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
model_path = os.path.join(output_dir, latest_checkpoint)

print(f"🌟 正在加载训练好的模型: {model_path} ...")

# 2. 加载模型和分词器
# 注意：我们加载的是“微调后”的权重
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # 别忘了这一步，保持一致

# 3. 准备测试输入
# 我们用中文开头，看看它能不能接中文
prompt = "味道" 
print(f"\n[用户输入]: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")

# 4. 生成 (Inference)
model.eval()
print("正在生成中...")

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=50, 
        do_sample=True, 
        temperature=0.7, # 温度低一点，让它背诵得准一点
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

# 5. 解码
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("-" * 30)
print("[外卖 GPT 生成结果]:")
print(generated_text)
print("-" * 30)