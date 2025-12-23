import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# 导入自己写的模块
from config import Config
from model import GPTLanguageModel
from dataset import get_data_loaders

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    保存模型检查点
    """
    checkpoint = {
        'model_state_dict' : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "epoch" : epoch,
        "loss" : loss
    }
    torch.save(checkpoint, filename)
    print(f"--> Model saved to {filename}")

@torch.no_grad()
def estimate_loss(model, dataloader, device, eval_iters=10):
    """
    评估函数:计算验证集上的平均Loss
    """
    model.eval()
    losses = []

    data_iter = iter(dataloader)
    for _ in range(eval_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def train():
    # 1.准备配置
    cfg = Config()
    # 强制使用CPU
    device = 'cpu'
    print(f"Using device : {device}")

    # 2.准备数据
    # 注意：这里需根据实际情况调整
    data_path = "../data/train_data/input.txt"
    train_loader, val_loader, dataset = get_data_loaders(cfg, data_path)

    # 工程化关键点: 从数据集中动态获取真实的vocab_size
    cfg.vocab_size = dataset.vocab_size
    print(f"Vocab size updated to: {cfg.vocab_size}")

    # 3.初始化数据和优化器
    model = GPTLanguageModel(cfg)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # 4. 开始训练循环
    print(f"Start Training Loop...")
    start_time = time.time()

    for epoch in range(cfg.epoch):
        for step, (x, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not step%100 :
                print(f"Epoch: {epoch + 1}\tStep: {step}\tLoss: {loss.item():.4f}")

        # 每个Epoch结束后的操作

        # 1.验证集测试
        val_loss = estimate_loss(model, val_loader, device)
        print(f"End of Epoch: {epoch + 1}\tVal Loss: {val_loss:.4f}")

        # 2.保存模型(文档操作)
        os.makedirs('models', exist_ok=True)
        save_path = f"models/gpt_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch, val_loss, save_path)

if __name__ == "__main__":
    train()
