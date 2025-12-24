# Mini-GPT: 从零构建的字符级生成模型 🚀

这是一个基于 PyTorch 从零实现的 GPT (Generative Pre-trained Transformer) 模型。本项目不依赖高级封装库（如 HuggingFace），旨在通过手写每一行代码，深入理解 Transformer 的底层原理（Attention, LayerNorm, Residual Connections）。

目前模型在 **Tiny Shakespeare** 数据集上训练，能够生成类似莎士比亚风格的古英语文本。

## 📂 项目结构 (Project Structure)

本项目采用了标准的工程化分层结构：

* **`model.py`**: 模型核心定义 (含 Multi-Head Attention, FeedForward, Transformer Block)。
* **`train.py`**: 训练脚本 (包含训练循环、Loss 监控、模型定期保存)。
* **`dataset.py`**: 数据处理模块 (自定义 PyTorch Dataset，处理字符级 Tokenization)。
* **`config.py`**: 配置中心 (集中管理超参数，如 learning_rate, batch_size 等)。
* **`inference.py`**: 推理脚本 (加载训练好的权重并生成文本)。
* **`data/`**: 存放训练数据 (如 `input.txt`)。

## 🛠️ 快速开始 (Quick Start)

### 1. 安装依赖

确保你的环境中有 PyTorch 和 NumPy：

    pip install -r requirements.txt

### 2. 开始训练

运行训练脚本。脚本会自动加载 `data/` 下的数据并开始训练。

    python 05_transformer_project/train.py

### 3. 生成文本 (模型推理)

训练完成后（默认 5000 步），运行推理脚本来查看效果：

    python 05_transformer_project/inference.py

## 📊 训练效果

经过 **5000 step** 的训练，模型在验证集上达到了 **1.78** 的 Loss。
生成的文本示例：

> **ORCULIO:**  
> Yen mystry; peasince  
> To pron the dudgeth Rombeash so?  
> ...

## 🧠 核心知识点

* **从零手写**：手动实现了 Causal Self-Attention 和 Multi-Head Attention。
* **工程化重构**：将学习阶段的单文件脚本重构为模块化的工程项目。
* **训练管理**：实现了 checkpoint 保存与加载机制，支持断点续练。

---
Created by Minmin-winter | DL Learning Path
