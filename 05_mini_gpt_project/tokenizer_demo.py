import tiktoken

# 1. 加载 GPT-2 的分词器
# (GPT-2 和 GPT-3 大部分用的是同一个词表，有 50257 个 token)
enc = tiktoken.get_encoding("gpt2")

print(f"Vocab size: {enc.n_vocab}") # 应该是 50257

# 2. 准备一段测试文本
text = "Hello, world! This is a Mini-GPT project."

# 3. 编码 (Encode): 文本 -> 数字列表
tokens = enc.encode(text)
print(f"\n[Raw Text]: {text}")
print(f"[Tokens]  : {tokens}")
print(f"[Length]  : {len(text)} chars -> {len(tokens)} tokens")

# 4. 解码 (Decode): 数字列表 -> 文本
decoded_text = enc.decode(tokens)
print(f"[Decoded] : {decoded_text}")

# 5. 看看每个 Token 到底长什么样？
print("\n--- Token Breakdown ---")
for t in tokens:
    # enc.decode_single_token_bytes(t) 返回的是字节，如果有特殊符号需要处理一下显示
    token_str = enc.decode_single_token_bytes(t).decode('utf-8', errors='replace')
    print(f"ID: {t:<6} | String: '{token_str}'")