"""
[Colab 专用脚本]
注意：此脚本需要 NVIDIA 显卡 + bitsandbytes 库支持，建议在 Colab T4 环境运行。

核心功能：
1. 使用 bitsandbytes 进行 4-bit 量化加载 (QLoRA 基础)。
2. 使用 apply_chat_template 处理对话格式。
3. 体验 T4 GPU 的推理速度。
"""
# ... 下面粘贴你的代码 ...
# 1. 安装必要的库 (Colab里要加感叹号运行命令)
# bitsandbytes 是量化核心库，accelerate 是加速库
# !pip install -q transformers accelerate bitsandbytes

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 2. 设置模型 ID
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 3. 🔥 核心科技：4-bit 量化配置 (QLoRA 的基础)
# 这会让模型体积缩小 4 倍，速度飞快，且显存占用极低
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f"正在下载并以 4-bit 量化加载模型: {model_id} ...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, # 👈 应用量化配置
    device_map="auto"               # 自动丢给 GPU
)

# 4. 准备对话 (Chat Template)
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain Quantum Mechanics to a 5-year-old in simple English."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 5. 生成
print("\n正在思考中... (感受一下 GPU 的速度)")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # 👈 丢到 CUDA (GPU) 上

model.eval()
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

response = outputs[0][inputs.input_ids.shape[1]:]
print("\n🤖 [TinyLlama @ GPU]:")
print(tokenizer.decode(response, skip_special_tokens=True))