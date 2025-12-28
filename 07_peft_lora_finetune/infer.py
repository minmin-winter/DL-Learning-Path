import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel # 👈 关键角色：负责拼装模型

# 1. 路径设置
# 指向原本的 GPT-2
base_model_path = "gpt2" 
# 指向你刚才训好的 LoRA 权重文件夹, 自动寻找最新的 checkpoint 
checkpoints = [d for d in os.listdir("./models") if d.startswith("checkpoint")]
if checkpoints:
    latest = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    lora_path = os.path.join("./models", latest)
    print(f"自动定位到最新权重: {lora_path}")
else:
    print("❌ 没找到训练好的 LoRA 权重！请检查路径。")
    exit()

# 2. 加载“素体”模型 (Base Model)
print(f"正在加载素体模型 GPT-2...")
base_model = GPT2LMHeadModel.from_pretrained(base_model_path)
tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# 3. 💥 合体！加载 LoRA 外挂
print(f"正在挂载 LoRA 外挂...")
# PeftModel.from_pretrained 会自动把 LoRA 权重“加”到 base_model 上
model = PeftModel.from_pretrained(base_model, lora_path)

# 4. 看看效果
prompt = "外卖"
print(f"\n[用户输入]: {prompt}")
inputs = tokenizer(prompt, return_tensors="pt")

model.eval()
print("正在生成中...")

with torch.no_grad():
    outputs = model.generate(
        input_ids = inputs["input_ids"], 
        attention_mask = inputs["attention_mask"],
        max_length=50, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("-" * 30)
print("[LoRA GPT-2 生成结果]:")
print(generated_text)
print("-" * 30)