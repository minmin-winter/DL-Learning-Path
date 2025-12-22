import torch
import torch.nn as nn

# 1.模拟训练数据
sentences = [
    "I love deep learning",
    "I like nutural language processing"
]

# --- Step1:构建词表(Vocabulary)-----------
# 把每个独特的单词，映射成唯一的数字ID

# 简单的拆分单词(Tokenization)
word_list = " ".join(sentences).split() # 
# 去重并排序
vocab = list(set(word_list))
# <PAD>:padding,这里补零用的，句子长度不一样，短的要补齐
# <UNK>:unknown,没见过的词就用他替换
word2idx = {'<PAD>' : 0, '<UNK>' : 1}

for idx,word in enumerate(vocab):
    word2idx[word] = idx + 2 # 从2开始，0和1被占了

print(f"词表字典:{word2idx}")
print(f"词表大小:{len(word2idx)}")

# ----Step2 : 句子转数字(Encoding) ------
# 把"i love deep learning" 变成 [Tensor] 

def make_sequence(sentence,word_dict):
    # 把句子拆开，查字典
    return [word_dict.get(word,word_dict['<UNK>']) for word in sentence.split()]

# 测试一下
input_seq = make_sequence('i love deep learning',word2idx)
print(f"\n原始句子:'i love deep learning'")
print(f"数字序列:{input_seq}")

# 转成 Tensor (Batch Size = 1)
input_tensor = torch.tensor([input_seq],dtype=torch.long)

# ---Step3 : Embedding层(核心魔法) ---
embed_layer = nn.Embedding(num_embeddings=len(word2idx),embedding_dim=5)


output = embed_layer(input_tensor)

print(f"输入形状:{input_tensor.shape}")
print(f"输出形状:{output.shape}")
print(f"具体的向量值:\n{output}")