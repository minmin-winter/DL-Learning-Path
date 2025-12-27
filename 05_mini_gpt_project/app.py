import gradio as gr
import torch
import os
from model import GPTLanguageModel
from config import Config
from dataset import BPEDataset

# -----------------------------------------------------------------------------
# 1. 准备工作：加载模型 (只在启动时运行一次)
# -----------------------------------------------------------------------------
print("正在启动服务，请稍候...")
device = 'cpu' # 如果你有 GPU 可以改成 'cuda'
config = Config()

# 路径设置 (使用相对路径，确保在任何地方跑都不会错)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'mini_gpt_step_5000.pth') # 确保文件名对
data_path = os.path.join(base_dir, 'data', 'mini_gpt', 'input.txt')

# 加载分词器 (Tokenizer)
print("Loading Tokenizer...")
dataset = BPEDataset(config, data_path)

# 初始化模型骨架
print("Loading Model...")
model = GPTLanguageModel(config)

# 加载权重 (使用之前修复过的万能加载逻辑)
checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint
    
# 去掉多卡训练的 module. 前缀
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()
print("Model loaded successfully!")

# -----------------------------------------------------------------------------
# 2. 定义核心功能函数
# -----------------------------------------------------------------------------
def generate_text(prompt, temperature, top_k):
    """
    这是 Gradio 按钮背后的逻辑：
    输入：提示词, 温度, Top-K
    输出：生成的文本
    """
    if not prompt.strip():
        return "⚠️ 请输入一点提示词..."

    # 1. 编码 (记得加 unsqueeze 变成 2D)
    context = torch.tensor(dataset.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # 2. 生成 (复用你写好的 generate)
    # 这里的 max_new_tokens 可以调，比如生成长一点
    generated_ids = model.generate(context, max_new_tokens=200, temperature=temperature, top_k=int(top_k))
    
    # 3. 解码
    output_text = dataset.decode(generated_ids[0].tolist())
    return output_text

# -----------------------------------------------------------------------------
# 3. 搭建网页界面 (UI Layout)
# -----------------------------------------------------------------------------
with gr.Blocks(title="Mini-GPT Playground") as demo:
    gr.Markdown("# 🤖 Mini-GPT: Shakespeare Edition")
    gr.Markdown("这是一个基于 Transformer 架构从零训练的微型 GPT 模型，经过了 5000 步的莎士比亚全集训练。")
    gr.Markdown("Created by minmin-winter")
    
    with gr.Row():
        with gr.Column():
            # 左边：输入区
            input_box = gr.Textbox(label="输入提示词 (Prompt)", placeholder="例如: The King said", lines=2)
            
            # 两个滑块：控制参数
            temp_slider = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature (温度 - 创造力)")
            topk_slider = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K (采样范围)")
            
            # 按钮
            generate_btn = gr.Button("🚀 开始生成", variant="primary")
            
        with gr.Column():
            # 右边：输出区
            output_box = gr.Textbox(label="模型生成的续写", lines=10, interactive=False)

    # 绑定事件：点按钮 -> 运行函数 -> 更新输出
    generate_btn.click(fn=generate_text, inputs=[input_box, temp_slider, topk_slider], outputs=output_box)

# -----------------------------------------------------------------------------
# 4. 启动！
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True)