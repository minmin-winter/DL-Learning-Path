import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. 准备数据 (刚才那一套流程)
print("正在加载数据...")
dataset = load_dataset("XiangPan/waimai_10k", split="train[:20]") # 为了演示快一点，我们只取前200条
# ⚠️ 进阶技巧：真正训练时去掉 [:200]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token 

def process_function(examples):
    # 这里我们把 input_ids 也作为 labels，因为我们要让模型学会“续写”
    # 模型会自己计算 input 和 label 的错位 (Shift)
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    tokens["labels"] = tokens["input_ids"].copy() # 监督信号就是它自己
    return tokens

tokenized_datasets = dataset.map(process_function, batched=True)

# 2. 准备模型
print("正在加载模型...")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 3. 设置训练参数 (TrainingArguments)
# 这里相当于 config.py，但功能多出 100 倍
training_args = TrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=4, # 显存小就调小
    num_train_epochs=30,            # 训练几轮
    logging_steps=10,              # 多少步打印一次日志
    save_steps=50,                 # 多少步保存一次
    use_cpu=True,                  # ⚠️ 只有你有 NVIDIA 显卡并装了驱动，才能设为 False
)

# 4. 召唤 Trainer (训练器)
# 它自动帮你管：Optimizer, Scheduler, Loss, Checkpoint, Logging...
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 5. 开始微调！
print("开始训练... (CPU可能会比较慢，耐心等待几步看看效果)")
trainer.train()