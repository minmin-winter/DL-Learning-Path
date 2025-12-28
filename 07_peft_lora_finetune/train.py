from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# 👇 新引入的库：PEFT (Parameter-Efficient Fine-Tuning)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 准备数据 (跟昨天一样，只取前100条做演示)
print("正在加载数据...")
dataset = load_dataset("XiangPan/waimai_10k", split="train[:100]")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def process_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(process_function, batched=True)

# 2. 加载模型
print("正在加载模型...")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ====================================================
# 🔥🔥🔥 LoRA 核心配置 (魔法发生的地方) 🔥🔥🔥
# ====================================================
print("\n正在给模型挂载 LoRA 外挂...")

# 定义配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型：因果语言模型 (GPT系列)
    inference_mode=False,         # 训练模式
    r=8,                          # Rank (秩)：这个数越小，参数越少。通常 8, 16, 32
    lora_alpha=32,                # 缩放系数，通常是 r 的 2 倍或 4 倍
    lora_dropout=0.1,             # 防止过拟合
    # ⚠️ 关键：我们要去影响模型里的哪一层？
    # 对于 GPT-2，注意力层的名字通常叫 'c_attn'。
    # 对于 Llama，通常是 'q_proj', 'v_proj'。
    target_modules=["c_attn"]     
)

# 🪄 变身！把普通模型变成 LoRA 模型
model = get_peft_model(model, peft_config)

# 🖨️ 打印一下，看看我们省了多少参数！
print("="*50)
model.print_trainable_parameters()
print("="*50)

# ====================================================

# 3. 训练参数 (跟昨天一样)
training_args = TrainingArguments(
    output_dir="./models/",
    per_device_train_batch_size=4,
    num_train_epochs=50,          # LoRA 收敛稍慢，或者需要多跑几轮，这里设50保证学会
    logging_steps=10,
    save_steps=100,
    learning_rate=1e-3,           # ⚠️ 注意：LoRA 的学习率通常比全量微调要大 (1e-3 vs 1e-5)
    use_cpu=True,
)

# 4. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("开始 LoRA 微调...")
trainer.train()