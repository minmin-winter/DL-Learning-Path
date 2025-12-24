import torch
from config import Config
from model import GPTLanguageModel
from dataset import CharDataset

# 1.加载环境
device = 'cpu'
print(f"Loading model on {device}...")

# 2. 拿到Dataset里的解编码函数
cfg = Config()
data_path = "./data/mini_gpt/input.txt"
dataset = CharDataset(cfg, data_path)
cfg.vocab_size = dataset.vocab_size

# 3.初始化模型
model = GPTLanguageModel(cfg)
model.to(device)

# 4,关键：加载训练好的权重
checkpoint_path = 'models/mini_gpt_step_5000.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Model loaded from {checkpoint_path}")

# 5.开始生成
model.eval()
start_str = "\n"
context = torch.tensor([dataset.encode(start_str)], dtype=torch.long, device=device)

print("--- Generating text ---")
generated_idx = model.generate(context, max_new_tokens=500)

# 解码成人类能看的文字
txt = dataset.decode(generated_idx[0].tolist())
print(txt)
print("--- End ---")


