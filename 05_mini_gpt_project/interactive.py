import os
import torch 
from model import GPTLanguageModel
from config import Config
from dataset import BPEDataset

# 1.设置配置
device = 'cpu'
config = Config()
# 自动寻找模型路径
model_path = os.path.join(os.path.dirname(__file__), 'models', 'mini_gpt_step_5000.pth')

# 2.加载数据处理类
train_data_path = os.path.join(os.path.dirname(__file__), "data", "mini_gpt", "input.txt")
dataset = BPEDataset(config, train_data_path)

# 3.初始化模型
model = GPTLanguageModel(config)
# 4.加载模型权重
check_point = torch.load(model_path, map_location=device)
if isinstance(check_point, dict) and 'model_state_dict' in check_point:
    state_dict = check_point["model_state_dict"]
elif isinstance(check_point, dict) and "model" in check_point :
    state_dict = check_point["model"]
else:
    state_dict = check_point
# 若训练连用了多卡(DataParallel),key前面会有module
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("\n" + "="*40)
print("🤖 Mini-GPT Interactive Session")
print("Type 'quit' to exit.")
print("Format: [temperature] [prompt]")
print("Example: 0.8 The king said")
print("="*40 + "\n")

while True:
    user_input = input("User >>")

    if user_input.lower() in ["exit", "quit"]:
        break

    # 解析输入：分离温度和提示词
    try:
        parts = user_input.split(" ", 1)
        temp = float(parts[0])
        prompt = parts[1]
    except (ValueError, IndexError):
        # 没写温度, 默认为0
        temp = 1.0
        prompt = user_input

    if not prompt.strip():
        prompt = " "

    context = torch.tensor(dataset.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    print(f"\nGeneratring with Temp={temp} ...\n")
    print("-" * 20)

    # 生成
    # top_k = 50是经典参数
    generated_ids = model.generate(context, max_new_tokens=200, temperature=temp, top_k=50)

    # 解码
    output_text = dataset.decode(generated_ids[0].tolist())
    print(output_text)
    print("-" * 20 + "\n")

    
