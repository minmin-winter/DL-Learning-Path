import matplotlib.pyplot as plt
import re

# 1. 读取日志文件 (假设你把 tmux 里的输出保存到了 training.log)
# 如果你没有保存文件，可以去 tmux 里复制一部分出来粘贴到一个新文件 log.txt 里
log_file = "./05_mini_gpt_project/logs/training_log.txt" 

steps = []
losses = []

# 2. 正则表达式：用来提取 "Step 100" 和 "Loss: 3.54" 里的数字
# 模式：Step [数字] | Loss: [数字]
# print(f"Epoch: {epoch + 1}\t Step: {step}\t Loss: {loss.item():.4f}") 
pattern = re.compile(r"Epoch:\s+(\d+).*?Step:\s+(\d+)\s+Loss:\s+([\d\.]+)")

print("Reading log file...")
try:
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(2)))
                losses.append(float(match.group(3)))

    # 3. 画图
    if steps:
        plt.figure(figsize=(10, 6))
        
        # ★ 修改这里：加上 marker='o' (画圆点) 和 linestyle='--' (虚线)
        # 这样即使只有几个点，看起来也很专业
        plt.plot(steps, losses, label='Training Loss', marker='o', linestyle='--', color='b')
        
        # 标出每个点的具体数值
        for i, loss in enumerate(losses):
            plt.text(steps[i], loss + 0.05, f"{loss:.2f}", ha='center')

        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Mini-GPT Training Progress (Key Milestones)')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.savefig("./05_mini_gpt_project/loss_curve.png")
    else:
        print("No data found in log file. Check the format!")

except FileNotFoundError:
    print(f"File {log_file} not found. Did you save the tmux output?")